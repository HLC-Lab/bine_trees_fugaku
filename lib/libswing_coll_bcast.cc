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

    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);    
    
    timer.reset("= swing_bcast_l (utofu buf reg)"); 
    if(this->rank == root){
        swing_utofu_reg_buf(this->utofu_descriptor, buffer, count*dtsize, NULL, 0, NULL, 0, env.num_ports); 
    }else{
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, buffer, count*dtsize, NULL, 0, env.num_ports); 
    }

    
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

    uint partition_size = count / env.num_ports;
    uint remaining = count % env.num_ports;        
    int res = MPI_SUCCESS; 

#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t p = 0; p < env.num_ports; p++){
        swing_tree_t tree = get_tree(root, p, env.bcast_config.algo_family, env.bcast_config.distance_type, this->scc_real);
        // Compute the peers of this port if I did not do it yet
        if(peers[p] == NULL){
            peers[p] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, p, env.bcast_config.algo_family, this->scc_real, peers[p]);
        }        
        timer.reset("= swing_bcast_l (computing trees)");

        uint peer;        

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
            int receiving_step;
            if(root != this->rank){        
                receiving_step = tree.reached_at_step[this->rank];
                // Receive the data from the root    
                swing_utofu_wait_recv(utofu_descriptor, p, 0, issued_recvs);
                issued_recvs++;
            }else{
                receiving_step = -1;
            }
        
            // Now perform all the subsequent steps            
            issued_sends = 0;
            for(size_t step = receiving_step + 1; step < (uint) this->num_steps; step++){
                // Send to peer
                if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[p][this->num_steps - step - 1];         
                }else{  
                    peer = peers[p][step];
                }
                if(tree.parent[peer] == this->rank){
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

        destroy_tree(&tree);
        free(peers[p]);
    }

    timer.reset("= swing_bcast_l (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif
}

int SwingCommon::swing_bcast_l_tmpbuf(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.bcast_config.algo_family == SWING_ALGO_FAMILY_SWING || env.bcast_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.bcast_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.bcast_config.algo == SWING_BCAST_ALGO_BINOMIAL_TREE_TMPBUF);
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
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);    
    size_t tmpbuf_size = count*dtsize;
    
    timer.reset("= swing_bcast_utofu (utofu buf reg)"); 
    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        if(this->rank == root){
            swing_utofu_reg_buf(this->utofu_descriptor, buffer, count*dtsize, NULL, 0, NULL, 0, env.num_ports); 
        }else{            
            posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
            free_tmpbuf = true;    
            // Use tmpbuf as recvbuf
            swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, tmpbuf, tmpbuf_size, NULL, 0, env.num_ports);                         
        }

        timer.reset("= swing_bcast_utofu (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
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
    }else{
        if(this->rank == root){
            swing_utofu_reg_buf(this->utofu_descriptor, buffer, count*dtsize, NULL, 0, NULL, 0, env.num_ports); 
        }else{
            swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, NULL, 0, env.num_ports); 
        }
        tmpbuf = env.prealloc_buf;
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_recv_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_recv_stadd = lcl_temp_stadd[i];
        }
    }

    uint partition_size = count / env.num_ports;
    uint remaining = count % env.num_ports;        
    int res = MPI_SUCCESS; 

#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t p = 0; p < env.num_ports; p++){
        swing_tree_t tree = get_tree(root, p, env.bcast_config.algo_family, env.bcast_config.distance_type, this->scc_real);
        // Compute the peers of this port if I did not do it yet
        if(peers[p] == NULL){
            peers[p] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, p, env.bcast_config.algo_family, this->scc_real, peers[p]);
        }        
        timer.reset("= swing_bcast_l (computing trees)");

        uint peer;        

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
            int receiving_step;
            if(root != this->rank){        
                receiving_step = tree.reached_at_step[this->rank];
                // Receive the data from the root    
                swing_utofu_wait_recv(utofu_descriptor, p, 0, issued_recvs);
                issued_recvs++;
            }else{
                receiving_step = -1;
            }
        
            // Now perform all the subsequent steps            
            issued_sends = 0;
            for(size_t step = receiving_step + 1; step < (uint) this->num_steps; step++){
                // Send to peer
                if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[p][this->num_steps - step - 1];         
                }else{  
                    peer = peers[p][step];
                }
                if(tree.parent[peer] == this->rank){
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

        destroy_tree(&tree);
        free(peers[p]);
        if(free_tmpbuf){
            swing_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, p);
        }
    }

    if(root != this->rank){
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

    size_t port = 0;
    swing_tree_t tree = get_tree(root, port, env.bcast_config.algo_family, env.bcast_config.distance_type, this->scc_real);

    timer.reset("= swing_bcast_l_mpi (actual sendrecvs)");

    int receiving_step;
    if(root != this->rank){        
        // Receive the data from the parent    
        receiving_step = tree.reached_at_step[this->rank];
        DPRINTF("[%d] Receiving from %d\n", rank, tree.parent[this->rank]);
        int res = MPI_Recv(buffer, count, datatype, tree.parent[this->rank], TAG_SWING_BCAST, comm, MPI_STATUS_IGNORE);                    
        if(res != MPI_SUCCESS){DPRINTF("[%d] Error on recv\n", rank); return res;}                
    }else{
        receiving_step = -1;
    }

    // Now perform all the subsequent steps
    MPI_Request requests_s[LIBSWING_MAX_STEPS];
    size_t posted_send = 0;
    for(size_t step = receiving_step + 1; step < (uint) this->num_steps; step++){
        // Send to peer
        uint peer;
        if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[this->num_steps - step - 1];         
        }else{  
            peer = peers[step];
        }
        if(tree.parent[peer] == this->rank){
            DPRINTF("[%d] Sending to %d\n", rank, peer);
            int res = MPI_Isend(buffer, count, datatype, peer, TAG_SWING_BCAST, comm, &(requests_s[posted_send]));
            if(res != MPI_SUCCESS){DPRINTF("[%d] Error on isend\n", rank); return res;}
            ++posted_send;
        }
    }
    MPI_Waitall(posted_send, requests_s, MPI_STATUSES_IGNORE);
    
    timer.reset("= swing_bcast_l_mpi (writing profile data to file)");
    destroy_tree(&tree);
    free(peers);
    return MPI_SUCCESS;
}

int SwingCommon::swing_bcast_b_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    return MPI_ERR_OTHER;
}



int SwingCommon::swing_bcast_scatter_allgather(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
#ifdef FUGAKU
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.bcast_config.algo_family == SWING_ALGO_FAMILY_SWING || env.bcast_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.bcast_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.bcast_config.algo == SWING_BCAST_ALGO_SCATTER_ALLGATHER);
    assert(env.bcast_config.distance_type == SWING_DISTANCE_INCREASING);
#endif
    assert(count / env.num_ports >= this->size);

    // Extra goes to last block (makes things simpler even if it might add a bit of unbalance)

    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_bcast_scatter_allgather (init)");
    Timer timer("swing_bcast_scatter_allgather (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
   
    timer.reset("= swing_bcast_scatter_allgather (utofu buf reg)"); 
    // Register buffer as recvbuf
    swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, buffer, count*dtsize, NULL, 0, env.num_ports);   

    timer.reset("= swing_bcast_scatter_allgather (utofu buf exch)");           
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

    int res = MPI_SUCCESS; 
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.bcast_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_bcast_scatter_allgather (computing trees)");
        swing_tree_t tree = get_tree(root, port, env.bcast_config.algo_family, env.bcast_config.distance_type, this->scc_real);
        
        /*************************/
        /*        Scatter        */
        /*************************/
        int receiving_step;
        if(root == this->rank){
            receiving_step = -1;
        }else{
            receiving_step = tree.reached_at_step[this->rank];
        }

        DPRINTF("[%d] Step from root: %d\n", this->rank, receiving_step);
        timer.reset("= swing_bcast_scatter_allgather (waiting recv)");

        size_t sendcount = count / (env.num_ports * this->size);
        size_t recvbuf_offset_port = sendcount * this->size * port * dtsize;
        // The last port of the last block might be bigger.
        // This might create unbalance, but simplifies the logic/code

        // Now perform all the subsequent steps       
        size_t issued_sends = 0;
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            if(root != this->rank && step == receiving_step){       
                size_t min_block_r = tree.remapped_ranks[this->rank];
                size_t max_block_r = tree.remapped_ranks_max[this->rank];            
                size_t blocks_to_recv = (max_block_r - min_block_r) + 1; 
                size_t bytes_to_recv = sendcount*blocks_to_recv*dtsize; // All blocks for this port have the same size
                if(max_block_r == this->size - 1 && port == env.num_ports - 1){
                    bytes_to_recv += (count % (env.num_ports * this->size))*dtsize;
                }
                size_t segments_max_put_size = ceil(bytes_to_recv / ((float) MAX_PUTGET_SIZE));
                swing_utofu_wait_recv(utofu_descriptor, port, 0, segments_max_put_size - 1);
            }

            if(step >= receiving_step + 1){
                uint peer;
                if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][this->num_steps - step - 1];
                }else{  
                    peer = peers[port][step];
                }
                if(tree.parent[peer] == this->rank){
                    size_t min_block_s = tree.remapped_ranks[peer];
                    size_t max_block_s = tree.remapped_ranks_max[peer];            
                    size_t blocks_to_send = (max_block_s - min_block_s) + 1;                 
                    utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + recvbuf_offset_port + min_block_s*sendcount*dtsize;
                    utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + recvbuf_offset_port + min_block_s*sendcount*dtsize;
                    size_t recvcnt = sendcount*blocks_to_send;
                    if(max_block_s == this->size - 1 && port == env.num_ports - 1){
                        recvcnt += (count % (env.num_ports * this->size));
                    }
                    DPRINTF("Rank %d sending %d elems to %d at step %d\n", this->rank, recvcnt, peer, step);
                    issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, recvcnt*dtsize, rmt_addr, 0);
                }
            }
        }
        
        /*************************/
        /*        Allgather      */
        /*************************/
        size_t num_blocks = 1;
        size_t min_block_resident = tree.remapped_ranks[this->rank];
        size_t min_block_r;
        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            uint peer;
            // This is ok, we need to do the opposite of the scatter. I.e., if scatter was done in decreasing phase, this must be done in increasing phase, and viceversa
            if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][step];
            }else{              
                peer = peers[port][this->num_steps - step - 1];                               
            }

            size_t count_to_sendrecv = num_blocks*sendcount;
            if(min_block_resident + num_blocks - 1 == this->size - 1 && port == env.num_ports - 1){
                count_to_sendrecv += (count % (env.num_ports * this->size));
            }
            // The data I am going to receive contains the block
            // with id equal to the remapped rank of my peer,
            // and is aligned to a power of 2^step
            // Thus, I need to do proper masking to get the block id
            // i.e., I need to set to 0 the least significant step bits
            min_block_r = tree.remapped_ranks[peer] & ~((1 << step) - 1);

            //printf("Rank %d Step %d peer %d min_block_r %d min_block_resident %d\n", this->rank, step, peer, min_block_r, min_block_resident);
            // Always send from the beginning of the buffer
            // and receive in the remaining part.
            timer.reset("= swing_bcast_scatter_allgather (sendrecv)");        
            utofu_stadd_t lcl_addr, rmt_addr;
            lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + recvbuf_offset_port + min_block_resident*sendcount*dtsize;
            rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + recvbuf_offset_port + min_block_resident*sendcount*dtsize;
            
            DPRINTF("[%d] Sending/receiving %d bytes from %d\n", this->rank, count_to_sendrecv*dtsize, peer);
            // In the notifications for isend/recv we do step+1 rather than step because one receive step was already done in the scatter phase
            // Send only if this is something that the peer does not already have
            if(min_block_resident < tree.remapped_ranks[peer] || min_block_resident + num_blocks - 1 > tree.remapped_ranks_max[peer]){
                issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step + 1);
            }
            
            // Receive only if this is something I did not have already
            if(min_block_r < tree.remapped_ranks[this->rank] || min_block_r + num_blocks - 1 > tree.remapped_ranks_max[this->rank]){
                size_t segments_max_put_size = ceil((count_to_sendrecv*dtsize) / ((float) MAX_PUTGET_SIZE));
                swing_utofu_wait_recv(utofu_descriptor, port, step + 1, segments_max_put_size - 1);
            }

            utofu_descriptor->port_info[port].completed_send = 0;                               

            min_block_resident = std::min(min_block_resident, min_block_r);
            num_blocks *= 2;
        }
        timer.reset("= swing_bcast_scatter_allgather (waiting all sends)");
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);

        free(peers[port]);    
        destroy_tree(&tree);
    }
    timer.reset("= swing_bcast_scatter_allgather (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif
}

// We do not need to reorder the data since it is not just a scatter or not just an allgather
// Thus, for the scatter we can just use the continuous version. For the allgather also (unless it is not power of 2 -- TODO)
int SwingCommon::swing_bcast_scatter_allgather_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.bcast_config.algo_family == SWING_ALGO_FAMILY_SWING || env.bcast_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.bcast_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.bcast_config.algo == SWING_BCAST_ALGO_SCATTER_ALLGATHER);
    assert(env.bcast_config.distance_type == SWING_DISTANCE_INCREASING);
#endif
    assert(env.num_ports == 1);
    assert(count / env.num_ports >= this->size);

    size_t sendcount = count / this->size;
    // Extra goes to last block (makes things simpler even if it might add a bit of unbalance)

    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_bcast_scatter_allgather_mpi (init)");
    Timer timer("swing_bcast_scatter_allgather_mpi (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
   
    timer.reset("= swing_bcast_scatter_allgather_mpi (utofu buf reg)"); 

    int res = MPI_SUCCESS; 

    size_t port = 0;
    // Compute the peers of this port if I did not do it yet
    if(peers[port] == NULL){
        peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, port, env.bcast_config.algo_family, this->scc_real, peers[port]);
    }        
    timer.reset("= swing_bcast_scatter_allgather_mpi (computing trees)");
    swing_tree_t tree = get_tree(root, port, env.bcast_config.algo_family, env.bcast_config.distance_type, this->scc_real);

    /*************************/
    /*        Scatter        */
    /*************************/
    int receiving_step;
    if(root == this->rank){
        receiving_step = -1;
    }else{
        receiving_step = tree.reached_at_step[this->rank];
    }

    DPRINTF("[%d] Step from root: %d\n", this->rank, receiving_step);
    timer.reset("= swing_bcast_scatter_allgather_mpi (waiting recv)");

    size_t recvbuf_offset_port = ((count*dtsize) / env.num_ports) * port;

    // Now perform all the subsequent steps            
    for(size_t step = 0; step < (uint) this->num_steps; step++){
        if(root != this->rank && step == receiving_step){       
            uint peer = tree.parent[this->rank];
            size_t min_block_r = tree.remapped_ranks[this->rank];
            size_t max_block_r = tree.remapped_ranks_max[this->rank];            
            size_t num_blocks = (max_block_r - min_block_r) + 1; 
            size_t elems_to_recv = sendcount*num_blocks; // All blocks for this port have the same size
            if(max_block_r == this->size - 1){
                elems_to_recv += (count % this->size);
            }
            DPRINTF("Rank %d receiving %d elems from %d at step %d\n", this->rank, elems_to_recv, peer, step);
            MPI_Recv((char*) buffer + recvbuf_offset_port + min_block_r*sendcount*dtsize, elems_to_recv, datatype, peer, TAG_SWING_BCAST, comm, MPI_STATUS_IGNORE);
        }

        if(step >= receiving_step + 1){
            uint peer;
            if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{  
                peer = peers[port][step];
            }
            if(tree.parent[peer] == this->rank){
                size_t min_block_s = tree.remapped_ranks[peer];
                size_t max_block_s = tree.remapped_ranks_max[peer];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                size_t elems_to_send = num_blocks*sendcount;
                if(max_block_s == this->size - 1){
                    elems_to_send += (count % this->size);
                }
                DPRINTF("Rank %d sending %d elems to %d at step %d\n", this->rank, elems_to_send, peer, step);
                MPI_Send((char*) buffer + recvbuf_offset_port + min_block_s*sendcount*dtsize, elems_to_send, datatype, peer, TAG_SWING_BCAST, comm);
            }
        }
        // Wait all the sends for this segment before moving to the next one
        timer.reset("= swing_bcast_scatter_allgather_mpi (waiting all sends)");
    }
    
    /*************************/
    /*        Allgather      */
    /*************************/
    size_t num_blocks = 1;
    size_t min_block_resident = tree.remapped_ranks[this->rank];
    size_t min_block_r;
    for(size_t step = 0; step < (uint) this->num_steps; step++){        
        uint peer;
        if(env.bcast_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][step];
        }else{              
            peer = peers[port][this->num_steps - step - 1];                               
        }

        size_t count_to_sendrecv = num_blocks*sendcount;
        if(min_block_resident + num_blocks - 1 == this->size - 1){
            count_to_sendrecv += (count % this->size);
        }
        // The data I am going to receive contains the block
        // with id equal to the remapped rank of my peer,
        // and is aligned to a power of 2^step
        // Thus, I need to do proper masking to get the block id
        // i.e., I need to set to 0 the least significant step bits
        min_block_r = tree.remapped_ranks[peer] & ~((1 << step) - 1);

        //printf("Rank %d Step %d peer %d min_block_r %d min_block_resident %d\n", this->rank, step, peer, min_block_r, min_block_resident);

        // Always send from the beginning of the buffer
        // and receive in the remaining part.
        timer.reset("= swing_bcast_scatter_allgather_mpi (sendrecv)");        
        //MPI_Sendrecv((char*) buffer + recvbuf_offset_port + min_block_resident*sendcount*dtsize, count_to_sendrecv, datatype, peer, TAG_SWING_BCAST, 
        //             (char*) buffer + recvbuf_offset_port + min_block_r*sendcount*dtsize       , count_to_sendrecv, datatype, peer, TAG_SWING_BCAST, 
        //             comm, MPI_STATUS_IGNORE);                                   
        // Do Irecv + send instead of sendrecv
        MPI_Request req;
        int wait = 0;
        // Parts of the blocks I want to send might be already in receiver buffer (since it scattered those blocks)
        // Receive only if this is something I did not have already
        if(min_block_r < tree.remapped_ranks[this->rank] || min_block_r + num_blocks - 1 > tree.remapped_ranks_max[this->rank]){
            MPI_Irecv((char*) buffer + recvbuf_offset_port + min_block_r*sendcount*dtsize, count_to_sendrecv, datatype, peer, TAG_SWING_BCAST, comm, &req);
            wait = 1;
        }

        // Send only if this is something that the peer does not already have
        if(min_block_resident < tree.remapped_ranks[peer] || min_block_resident + num_blocks - 1 > tree.remapped_ranks_max[peer]){
            MPI_Send((char*) buffer + recvbuf_offset_port + min_block_resident*sendcount*dtsize, count_to_sendrecv, datatype, peer, TAG_SWING_BCAST, comm);
        }

        if(wait){
            MPI_Wait(&req, MPI_STATUS_IGNORE);        
        }

        min_block_resident = std::min(min_block_resident, min_block_r);
        num_blocks *= 2;
    }

    free(peers[port]);    
    destroy_tree(&tree);
    timer.reset("= swing_bcast_scatter_allgather_mpi (writing profile data to file)");
    return res;
}
