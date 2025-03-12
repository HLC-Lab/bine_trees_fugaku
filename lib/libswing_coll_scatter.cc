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

int SwingCommon::swing_scatter_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.scatter_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.scatter_config.algo == SWING_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
#ifdef FUGAKU
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_scatter_utofu (init)");
    Timer timer("swing_scatter_utofu (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= swing_scatter_utofu (utofu buf reg)"); 
    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_scatter_utofu (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.algo_family, this->scc_real, peers[mp]);
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
    for(size_t p = 0; p < env.num_ports; p++){
        DPRINTF("Computing peers\n");
        // Compute the peers of this port if I did not do it yet
        if(peers[p] == NULL){
            peers[p] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, p, env.algo_family, this->scc_real, peers[p]);
        }        
        timer.reset("= swing_scatter_utofu (computing trees)");
        swing_tree_t tree = get_tree(root, p, env.algo_family, env.scatter_config.distance_type, this->scc_real);

        int receiving_step = tree.reached_at_step[this->rank];

        DPRINTF("Step from root: %d\n", receiving_step);
        size_t issued_sends = 0, issued_recvs = 0;
        timer.reset("= swing_scatter_utofu (waiting recv)");
        if(root != this->rank){        
            // Receive the data from the root    
            swing_utofu_wait_recv(utofu_descriptor, p, 0, issued_recvs);
            issued_recvs++;
        }else{
            receiving_step = -1;
        }

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * p;
        DPRINTF("Port %d tmpbuf_offset_port %d tmpbuf size %d ceil %f (%d/%d)\n", p, tmpbuf_offset_port, tmpbuf_size, ceil((float) sendcount / env.num_ports), sendcount, env.num_ports);
        uint my_remapped_rank;
        // Blocks remapping. The root permutes the array so that it can send contiguous blocks
        if(this->rank == root){
            timer.reset("= swing_scatter_utofu (permute blocks)");                
            for(size_t i = 0; i < size; i++){      
                DPRINTF("Moving block %d to %d\n", i, tree.remapped_ranks[i]);          
                // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
                // E.g., with two ports and 4 ranks
                // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
                // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 4 5 8 9 12 13 and port 1 will work on 2 3 6 7 10 11 14 15
                // For a given port, all the sub-blocks have the same size, so we can just consider the count of any block (e.g., 0)
                memcpy(tmpbuf + tmpbuf_offset_port + blocks_info[p][0].count*tree.remapped_ranks[i]*dtsize, (char*) sendbuf + blocks_info[p][i].offset, blocks_info[p][i].count*dtsize);
            }
        }
        my_remapped_rank = tree.remapped_ranks[this->rank];
        DPRINTF("My remapped rank is %d\n", my_remapped_rank);

        // Now perform all the subsequent steps                    
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            issued_sends = 0;
            // Compute the range to send/recv
            if(step >= receiving_step + 1){
                uint peer;
                if(env.scatter_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[p][this->num_steps - step - 1];
                }else{
                    peer = peers[p][step];
                }                
                if(tree.parent[peer] == this->rank){
                    size_t min_block_s = tree.remapped_ranks[peer];
                    size_t max_block_s = tree.remapped_ranks_max[peer];
                    size_t blocks_to_send = (max_block_s - min_block_s) + 1;
                    utofu_stadd_t lcl_addr = utofu_descriptor->port_info[p].lcl_temp_stadd       + tmpbuf_offset_port + min_block_s*blocks_info[p][0].count*dtsize;
                    utofu_stadd_t rmt_addr = utofu_descriptor->port_info[p].rmt_temp_stadd[peer] + tmpbuf_offset_port + min_block_s*blocks_info[p][0].count*dtsize;
                    size_t tmpcnt = blocks_info[p][0].count*blocks_to_send; // All blocks for this port have the same size
                    swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[p][peer]), p, peer, lcl_addr, tmpcnt*dtsize, rmt_addr, 0);                                                             
                    ++issued_sends;
                    swing_utofu_wait_sends(utofu_descriptor, p, issued_sends);
                }
            }
        }    
        free(peers[p]);
        destroy_tree(&tree);

        timer.reset("= swing_scatter_utofu (final memcpy)"); // TODO: Can be avoided if the last put is done in recvbuf rather than tmpbuf
        // Consider offsets of block 0 since everything must go "in the first block"
        // Last ports have one element less.
        DPRINTF("p=%d Copying %d bytes from %d to %d\n", p, blocks_info[p][my_remapped_rank].count*dtsize, tmpbuf_offset_port + (blocks_info[p][0].count)*my_remapped_rank*dtsize, blocks_info[p][0].offset); 
        memcpy((char*) recvbuf + blocks_info[p][0].offset, 
               tmpbuf + tmpbuf_offset_port + blocks_info[p][0].count*my_remapped_rank*dtsize, 
               blocks_info[p][my_remapped_rank].count*dtsize);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }
    
    timer.reset("= swing_scatter_utofu (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif
}

int SwingCommon::swing_scatter_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.scatter_config.algo == SWING_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    assert(env.num_ports == 1);
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_scatter_mpi (init)");
    Timer timer("swing_scatter_mpi (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= swing_scatter_mpi (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }

    int res = MPI_SUCCESS; 

    size_t port = 0;
    // Compute the peers of this port if I did not do it yet
    if(peers[port] == NULL){
        peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, port, env.scatter_config.algo_family, this->scc_real, peers[port]);
    }        
    timer.reset("= swing_scatter_mpi (computing trees)");
    swing_tree_t tree = get_tree(root, port, env.scatter_config.algo_family, env.scatter_config.distance_type, this->scc_real);

    int receiving_step;
    if(root == this->rank){
        receiving_step = -1;
    }else{
        receiving_step = tree.reached_at_step[this->rank];
    }

    DPRINTF("[%d] Step from root: %d\n", this->rank, receiving_step);
    timer.reset("= swing_scatter_mpi (waiting recv)");
#ifdef DEBUG
    printf("[%d] Parents: ", this->rank);
    for(size_t i = 0; i < this->size; i++){
        printf("%d ", tree.parent[i]);
    }
    printf("\n");
    fflush(stdout);
#endif

    size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * port;
    uint my_remapped_rank;
    // Blocks remapping. The root permutes the array so that it can send contiguous blocks
    if(this->rank == root){
        timer.reset("= swing_scatter_mpi (permute)");    
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving block %d to %d\n", this->rank, i, tree.remapped_ranks[i]);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            // For a given port, all the sub-blocks have the same size, so we can just consider the count of any block (e.g., 0)
            memcpy(tmpbuf + tmpbuf_offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize, (char*) sendbuf + blocks_info[port][i].offset, blocks_info[port][i].count*dtsize);
        }        
    }
    my_remapped_rank = tree.remapped_ranks[rank];

    DPRINTF("[%d] My remapped rank is %d\n", this->rank, my_remapped_rank);

    // Now perform all the subsequent steps            
    for(size_t step = 0; step < (uint) this->num_steps; step++){
        if(root != this->rank && step == receiving_step){       
            uint peer = tree.parent[this->rank];
            size_t min_block_r = tree.remapped_ranks[peer];
            size_t max_block_r = tree.remapped_ranks_max[peer];            
            size_t num_blocks = (max_block_r - min_block_r) + 1; 
            DPRINTF("Rank %d receiving %d elems from %d at step %d\n", this->rank, num_blocks*recvcount, peer, step);
            MPI_Recv(tmpbuf + min_block_r*recvcount*dtsize, num_blocks*recvcount, sendtype, peer, TAG_SWING_SCATTER, comm, MPI_STATUS_IGNORE);
        }

        if(step >= receiving_step + 1){
            uint peer;
            if(env.scatter_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{  
                peer = peers[port][step];
            }
            if(tree.parent[peer] == this->rank){
                size_t min_block_s = tree.remapped_ranks[this->rank];
                size_t max_block_s = tree.remapped_ranks_max[this->rank];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                DPRINTF("Rank %d sending %d elems to %d at step %d\n", this->rank, num_blocks*recvcount, peer, step);
                MPI_Send(tmpbuf + min_block_s*recvcount*dtsize, num_blocks*recvcount, sendtype, peer, TAG_SWING_SCATTER, comm);
            }
        }
        // Wait all the sends for this segment before moving to the next one
        timer.reset("= swing_scatter_mpi (waiting all sends)");
    }
    
    free(peers[port]);

    timer.reset("= swing_scatter_mpi (final memcpy)"); // TODO: Can be avoided if the last put is done in recvbuf rather than tmpbuf
    // Consider offsets of block 0 since everything must go "in the first block"
    DPRINTF("[%d] p=%d Copying %d bytes from %d to %d\n", this->rank, port, blocks_info[port][my_remapped_rank].count*dtsize, tmpbuf_offset_port + blocks_info[port][0].count*my_remapped_rank*dtsize, blocks_info[port][my_remapped_rank].offset); 
    memcpy((char*) recvbuf + blocks_info[port][0].offset, tmpbuf + tmpbuf_offset_port + blocks_info[port][0].count*my_remapped_rank*dtsize, blocks_info[port][my_remapped_rank].count*dtsize);

    if(free_tmpbuf){
        free(tmpbuf);
    }
    destroy_tree(&tree);
    timer.reset("= swing_scatter_mpi (writing profile data to file)");
    return res;
}
