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

int SwingCommon::swing_gather_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.gather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.gather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.gather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.gather_config.algo == SWING_GATHER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
#ifdef FUGAKU
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_gather_utofu (init)");
    Timer timer("swing_gather_utofu (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= swing_gather_utofu (utofu buf reg)"); 

    // Everyone sends from tempbuf because they need to put the sendbuf in the correct position
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_gather_utofu (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.gather_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.gather_config.algo_family, this->scc_real, peers[mp]);
                swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        // Everything to 0/NULL just to initialize the internal status.
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, NULL, 0, env.num_ports); 
        tmpbuf = env.prealloc_buf;
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_temp_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }
    DPRINTF("tmpbuf allocated\n");    

    int res = MPI_SUCCESS; 
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.gather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_gather_utofu (computing trees)");
        swing_tree_t tree = get_tree(root, port, env.gather_config.algo_family, env.gather_config.distance_type, this->scc_real);

        // I do a bunch of receives (unless I am a leaf), and then I send the data to the parent
        // To understand at which step I must send the data, I need to check at which step I am 
        // reached by the root.
        // If this is step s, then I start sending at (num_steps - 1 - s)
        // To check if I am a leaf, I can just check if I am reached by the root in the last step
        int sending_step;
        if(root == this->rank){
            sending_step = this->num_steps;
        }else{
            sending_step = this->num_steps - 1 - tree.reached_at_step[this->rank];
        }

        DPRINTF("[%d] Sending step: %d\n", this->rank, sending_step);        
        timer.reset("= swing_gather_utofu (waiting recv)");

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * port;

        // Put sendbuf in the correct positions (at index of remapped rank) in tempbuf
        DPRINTF("[%d] Copying sendbuf from offset 0 to %d\n", this->rank, tmpbuf_offset_port + blocks_info[port][0].count*tree.remapped_ranks[this->rank]*dtsize);
        memcpy(tmpbuf + tmpbuf_offset_port + blocks_info[port][0].count*tree.remapped_ranks[this->rank]*dtsize, ((char*) sendbuf) + blocks_info[port][0].offset, blocks_info[port][0].count*dtsize);       
               
        for(size_t step = 0; step < (uint) this->num_steps; step++){ 
            size_t issued_sends = 0;
            size_t issued_recvs = 0;       
            if(step < sending_step){
                // Receive from peer                
                uint peer;
                if(env.gather_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][step];                               
                }else{  
                    peer = peers[port][this->num_steps - step - 1];
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    swing_utofu_wait_recv(utofu_descriptor, port, step, issued_recvs);
                    issued_recvs++;
                }
            }else if(step == sending_step){
                // Send to parent
                uint peer = tree.parent[this->rank];            
                size_t min_block_s = tree.remapped_ranks[this->rank];
                size_t max_block_s = tree.remapped_ranks_max[this->rank];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                DPRINTF("[%d] sending %d elems to %d at step %d [offset %d, count %d]\n", this->rank, num_blocks*blocks_info[port][0].count, peer, step, min_block_s*blocks_info[port][0].count*dtsize, num_blocks*blocks_info[port][0].count);
                utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd       + tmpbuf_offset_port + min_block_s*blocks_info[port][0].count*dtsize;
                utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + tmpbuf_offset_port + min_block_s*blocks_info[port][0].count*dtsize;
                size_t tmpcnt = num_blocks*blocks_info[port][0].count; // All blocks for this port have the same size
                swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, tmpcnt*dtsize, rmt_addr, step);                                                             
                ++issued_sends;
                swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_gather_utofu (waiting all sends)");
        }

        // For each port we need to permute back the data in the correct position
        if(this->rank == root){
            timer.reset("= swing_gather_utofu (permute)");    
            for(size_t i = 0; i < size; i++){      
                DPRINTF("[%d] Moving block %d to %d\n", this->rank, i, tree.remapped_ranks[i]);          
                // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
                // E.g., with two ports and 4 ranks
                // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
                // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
                size_t pos_in_recvbuf = blocks_info[port][i].offset;
                size_t pos_in_tmpbuf_port = tree.remapped_ranks[i]*blocks_info[port][0].count*dtsize;
                DPRINTF("[%d] Copying %d bytes from %d to %d\n", this->rank, blocks_info[port][i].count*dtsize, tmpbuf_offset_port + pos_in_tmpbuf_port, pos_in_recvbuf);
                memcpy(((char*) recvbuf) + pos_in_recvbuf, tmpbuf + tmpbuf_offset_port + pos_in_tmpbuf_port, blocks_info[port][i].count*dtsize);
            }        
        }
        
        free(peers[port]);
        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_gather_utofu (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif
}

int SwingCommon::swing_gather_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.gather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.gather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.gather_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.gather_config.algo == SWING_GATHER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_gather_mpi (init)");
    Timer timer("swing_gather_mpi (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= swing_gather_mpi (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }

    int res = MPI_SUCCESS; 

    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.gather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_gather_mpi (computing trees)");
        swing_tree_t tree = get_tree(root, port, env.gather_config.algo_family, env.gather_config.distance_type, this->scc_real);

        // I do a bunch of receives (unless I am a leaf), and then I send the data to the parent
        // To understand at which step I must send the data, I need to check at which step I am 
        // reached by the root.
        // If this is step s, then I start sending at (num_steps - 1 - s)
        // To check if I am a leaf, I can just check if I am reached by the root in the last step
        int sending_step;
        if(root == this->rank){
            sending_step = this->num_steps;
        }else{
            sending_step = this->num_steps - 1 - tree.reached_at_step[this->rank];
        }

        DPRINTF("[%d] Sending step: %d\n", this->rank, sending_step);
        timer.reset("= swing_gather_mpi (waiting recv)");

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * port;
        // Put sendbuf in the correct positions (at index of remapped rank) in tempbuf
        DPRINTF("[%d] Copying sendbuf from offset 0 to %d\n", this->rank, tmpbuf_offset_port + blocks_info[port][0].count*tree.remapped_ranks[this->rank]*dtsize);
        memcpy(tmpbuf + tmpbuf_offset_port + blocks_info[port][0].count*tree.remapped_ranks[this->rank]*dtsize, ((char*) sendbuf) + blocks_info[port][0].offset, blocks_info[port][0].count*dtsize);
                
        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            if(step < sending_step){
                // Receive from peer
                uint peer;
                if(env.gather_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][step];                               
                }else{  
                    peer = peers[port][this->num_steps - step - 1];
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    size_t min_block_r = tree.remapped_ranks[peer];
                    size_t max_block_r = tree.remapped_ranks_max[peer];            
                    size_t num_blocks = (max_block_r - min_block_r) + 1; 
                    DPRINTF("[%d] receiving %d elems from %d at step %d [offset %d, count %d]\n", this->rank, num_blocks*blocks_info[port][0].count, peer, step, min_block_r*blocks_info[port][0].count*dtsize, num_blocks*blocks_info[port][0].count);
                    MPI_Recv(tmpbuf + tmpbuf_offset_port + min_block_r*blocks_info[port][0].count*dtsize, num_blocks*blocks_info[port][0].count, sendtype, peer, TAG_SWING_GATHER, comm, MPI_STATUS_IGNORE);
                }
            }else if(step == sending_step){
                // Send to parent
                uint peer = tree.parent[this->rank];            
                size_t min_block_s = tree.remapped_ranks[this->rank];
                size_t max_block_s = tree.remapped_ranks_max[this->rank];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                DPRINTF("[%d] sending %d elems to %d at step %d [offset %d, count %d]\n", this->rank, num_blocks*blocks_info[port][0].count, peer, step, min_block_s*blocks_info[port][0].count*dtsize, num_blocks*blocks_info[port][0].count);
                MPI_Send(tmpbuf + tmpbuf_offset_port + min_block_s*blocks_info[port][0].count*dtsize, num_blocks*blocks_info[port][0].count, sendtype, peer, TAG_SWING_GATHER, comm);
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_gather_mpi (waiting all sends)");
        }

        // For each port we need to permute back the data in the correct position
        if(this->rank == root){
            timer.reset("= swing_gather_mpi (permute)");    
            for(size_t i = 0; i < size; i++){      
                DPRINTF("[%d] Moving block %d to %d\n", this->rank, i, tree.remapped_ranks[i]);          
                // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
                // E.g., with two ports and 4 ranks
                // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
                // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
                size_t pos_in_recvbuf = blocks_info[port][i].offset;
                size_t pos_in_tmpbuf_port = tree.remapped_ranks[i]*blocks_info[port][0].count*dtsize;
                DPRINTF("[%d] Copying %d bytes from %d to %d\n", this->rank, blocks_info[port][i].count*dtsize, tmpbuf_offset_port + pos_in_tmpbuf_port, pos_in_recvbuf);
                memcpy(((char*) recvbuf) + pos_in_recvbuf, (char*) tmpbuf + tmpbuf_offset_port + pos_in_tmpbuf_port, blocks_info[port][i].count*dtsize);
            }        
        }
        
        free(peers[port]);
        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_gather_mpi (writing profile data to file)");
    return res;
}
