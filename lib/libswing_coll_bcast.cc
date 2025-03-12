#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <unistd.h>

#include "libswing_common.h"
#include "libswing_coll.h"
#include <climits>
#ifdef FUGAKU
#include "fugaku/swing_utofu.h"
#endif

// TODO: Rely on trees as for the other collectives.

static void dfs_reversed(int* coord_rank, size_t step, size_t num_steps, uint32_t* reached_at_step, uint32_t* parent, uint port, swing_algo_family_t algo, SwingCoordConverter* scc, bool allgather_schedule){
    for(size_t i = step; i < num_steps; i++){
        int real_step;
        if(allgather_schedule){
            real_step = num_steps - 1 - i; // We consider allgather schedule
        }else{
            real_step = i;
        }
        int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        get_peer_c(coord_rank, real_step, peer_rank, port, scc->dimensions_num, scc->dimensions, algo);
        
        uint32_t rank = scc->getIdFromCoord(peer_rank);
        if(parent[rank] == UINT32_MAX || i < reached_at_step[rank]){
            parent[rank] = scc->getIdFromCoord(coord_rank);
            reached_at_step[rank] = i;
        }
        dfs_reversed(peer_rank, i + 1, num_steps, reached_at_step, parent, port, algo, scc, allgather_schedule);
    }
}

static void get_step_from_root(int* coord_root, uint32_t* reached_at_step, uint32_t* parent, size_t num_steps, uint port, uint dimensions_num, uint* dimensions, swing_algo_family_t algo, bool allgather_schedule){
    SwingCoordConverter scc(dimensions, dimensions_num);
    dfs_reversed(coord_root, 0, num_steps, reached_at_step, parent, port, algo, &scc, allgather_schedule);
    parent[scc.getIdFromCoord(coord_root)] = UINT32_MAX;
    reached_at_step[scc.getIdFromCoord(coord_root)] = 0; // To avoid sending the step for myself at a wrong value
}

int SwingCommon::swing_bcast_l(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.bcast_config.algo_family == SWING_ALGO_FAMILY_SWING || env.bcast_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.bcast_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.bcast_config.algo == SWING_BCAST_ALGO_BINOMIAL_TREE);
#endif
#ifdef FUGAKU
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_bcast_l (init)");
    Timer timer("swing_bcast_l (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    
    size_t max_count;
    if(env.segment_size){
        max_count = floor(env.segment_size / dtsize);
    }else{
        max_count = floor(MAX_PUTGET_SIZE / dtsize);
    }    

    char* tmpbuf;
    bool free_tmpbuf = false;
    size_t tmpbuf_size = count*dtsize;
    char use_tmpbuf = 0;
    // For small messages we do everything in the known temp buffer to avoid exchanging information
    // about STADDs and to avoid registering the buffer.
    if(count*dtsize <= env.bcast_config.tmp_threshold){ // TODO: Do it in the same way we do for the other collectives.
        assert(tmpbuf_size <= env.prealloc_size); // I do not want to complicate the code too much so I assume the preallocated buffer is large enough
        tmpbuf = env.prealloc_buf;
        use_tmpbuf = 1;
    }
    
    timer.reset("= swing_bcast_l (utofu buf reg)"); 
    if(this->rank == root){
        swing_utofu_reg_buf(this->utofu_descriptor, buffer, count*dtsize, NULL, 0, NULL, 0, env.num_ports); 
    }else{
        if(use_tmpbuf){
            swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, 0, NULL, NULL, 0, env.num_ports); 
        }else{
            swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, buffer, count*dtsize, NULL, 0, env.num_ports); 
        }                
    }

    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    if(use_tmpbuf){
        // Store the lcl_recv_stadd and rmt_recv_buffer STADD of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_recv_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_recv_stadd = lcl_temp_stadd[i];
        }
    }else{
        timer.reset("= swing_bcast_l (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.bcast_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.bcast_config.algo_family, this->scc_real, peers[mp]);
                swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }        
    }

    uint partition_size = count / env.num_ports;
    uint remaining = count % env.num_ports;        
    int res = MPI_SUCCESS; 

#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t p = 0; p < env.num_ports; p++){
        // Compute the peers of this port if I did not do it yet
        if(peers[p] == NULL){
            peers[p] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, p, env.bcast_config.algo_family, this->scc_real, peers[p]);
        }        
        timer.reset("= swing_bcast_l (computing trees)");
        int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        this->scc_real->retrieve_coord_mapping(this->rank, coord);

        int coord_root[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        this->scc_real->getCoordFromId(root, coord_root);

        uint32_t* reached_at_step = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
        uint32_t* parent = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
        for(size_t i = 0; i < this->size; i++){
            reached_at_step[i] = this->num_steps;
            parent[i] = UINT32_MAX;
        }
        get_step_from_root(coord_root, reached_at_step, parent, this->num_steps, p, env.dimensions_num, env.dimensions, env.bcast_config.algo_family, true);
        int receiving_step = reached_at_step[this->rank];
        int peer;        

        size_t count_port = partition_size + (p < remaining ? 1 : 0);
        size_t offset_port = 0;
        for(size_t j = 0; j < p; j++){
            offset_port += partition_size + (j < remaining ? 1 : 0);
        }
        offset_port *= dtsize;

        // Pipeline the recv with the first send
        size_t remaining = count_port;
        size_t bytes_to_send = 0, count_segment, offset_segment = 0;
        size_t issued_sends = 0, issued_recvs = 0;
        while(remaining){
            count_segment = remaining < max_count ? remaining : max_count;
            bytes_to_send = count_segment*dtsize;

            timer.reset("= swing_bcast_l (waiting recv)");
            if(root != this->rank){        
                // Receive the data from the root    
                swing_utofu_wait_recv(utofu_descriptor, p, 0, issued_recvs);
                issued_recvs++;
            }else{
                receiving_step = -1;
            }
        
            // Now perform all the subsequent steps            
            issued_sends = 0;
            for(size_t step = receiving_step + 1; step < (uint) this->num_steps; step++){
                peer = peers[p][(this->num_steps - step - 1)]; // Consider the allgather peers since they start from the distant ones and then get closer.
                if(parent[peer] == this->rank){
                    utofu_stadd_t lcl_addr;
                    if(this->rank == root){
                        lcl_addr = utofu_descriptor->port_info[p].lcl_send_stadd + offset_port + offset_segment;
                    }else{
                        lcl_addr = utofu_descriptor->port_info[p].lcl_recv_stadd + offset_port + offset_segment;
                    }
                    utofu_stadd_t rmt_addr = utofu_descriptor->port_info[p].rmt_recv_stadd[peer] + offset_port + offset_segment;                                        
                    swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[p][peer]), p, peer, lcl_addr, bytes_to_send, rmt_addr, 0);                                         
                    ++issued_sends;                    
                }
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_bcast_l (waiting all sends)");
            swing_utofu_wait_sends(utofu_descriptor, p, issued_sends);

            offset_segment += bytes_to_send;
            remaining -= count_segment;            
        }

        free(reached_at_step);
        free(parent);
        free(peers[p]);
    }

    if(use_tmpbuf && root != this->rank){
        timer.reset("= swing_bcast_l (final memcpy)");
        memcpy(buffer, tmpbuf, count*dtsize);
    }
    if(free_tmpbuf){
        free(tmpbuf);
    }
    
    timer.reset("= swing_bcast_l (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif
}

int SwingCommon::swing_bcast_b(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    return MPI_ERR_OTHER;
}

// TODO: Pipeline
int SwingCommon::swing_bcast_l_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.bcast_config.algo_family == SWING_ALGO_FAMILY_SWING || env.bcast_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.bcast_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.bcast_config.algo == SWING_BCAST_ALGO_BINOMIAL_TREE);
#endif
    assert(env.num_ports == 1); // Hard to do without being able to call MPI from multiple threads at the same time
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_bcast_l_mpi (init)");
    Timer timer("swing_bcast_l_mpi (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    
    
    uint* peers = (uint*) malloc(sizeof(uint)*this->num_steps);
    compute_peers(this->rank, 0, env.bcast_config.algo_family, this->scc_real, peers);

    // Not actually needed, is just to use the get_peer, should be refactored
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    this->scc_real->retrieve_coord_mapping(this->rank, coord);
    int res = MPI_SUCCESS; 

    int coord_root[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    this->scc_real->getCoordFromId(root, coord_root);

    timer.reset("= swing_bcast_l_mpi (actual sendrecvs)");
    uint32_t* reached_at_step = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
    uint32_t* parent = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
    for(size_t i = 0; i < this->size; i++){
        reached_at_step[i] = this->num_steps;
        parent[i] = UINT32_MAX;
    }
    get_step_from_root(coord_root, reached_at_step, parent, this->num_steps, 0, env.dimensions_num, env.dimensions, env.bcast_config.algo_family, true);
    int receiving_step = reached_at_step[this->rank];
    int peer;
    if(root != this->rank){        
        // Receive the data from the root    
        peer = peers[(this->num_steps - receiving_step - 1)]; // Consider the allgather peers since they start from the distant ones and then get closer.
        assert(peer == parent[this->rank]); 
        DPRINTF("[%d] Receiving from %d\n", rank, peer);
        res = MPI_Recv(buffer, count, datatype, peer, TAG_SWING_BCAST, comm, MPI_STATUS_IGNORE);                    
        if(res != MPI_SUCCESS){DPRINTF("[%d] Error on recv\n", rank); return res;}                
    }else{
        receiving_step = -1;
    }

    // Now perform all the subsequent steps
    MPI_Request requests_s[LIBSWING_MAX_STEPS];
    size_t posted_send = 0;
    for(size_t step = receiving_step + 1; step < (uint) this->num_steps; step++){
        peer = peers[(this->num_steps - step - 1)]; // Consider the allgather peers since they start from the distant ones and then get closer.
        if(parent[peer] == this->rank){
            DPRINTF("[%d] Sending to %d\n", rank, peer);
            res = MPI_Isend(buffer, count, datatype, peer, TAG_SWING_BCAST, comm, &(requests_s[posted_send]));
            if(res != MPI_SUCCESS){DPRINTF("[%d] Error on isend\n", rank); return res;}
            ++posted_send;
        }
    }
    MPI_Waitall(posted_send, requests_s, MPI_STATUSES_IGNORE);
    
    timer.reset("= swing_bcast_l_mpi (writing profile data to file)");
    free(reached_at_step);
    free(parent);
    free(peers);
    return res;
}

int SwingCommon::swing_bcast_b_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    return MPI_ERR_OTHER;
}
