#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <unistd.h>

#include "libbine_common.h"
#include "libbine_coll.h"
#include <climits>
#ifdef FUGAKU
#include "fugaku/bine_utofu.h"
#endif

int BineCommon::bine_scatter_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.scatter_config.algo_family == BINE_ALGO_FAMILY_BINE || env.scatter_config.algo_family == BINE_ALGO_FAMILY_RECDOUB);
    assert(env.scatter_config.algo_layer == BINE_ALGO_LAYER_UTOFU);
    assert(env.scatter_config.algo == BINE_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
#ifdef FUGAKU
    assert(sendcount >= env.num_ports);
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= bine_scatter_utofu (init)");
    Timer timer("bine_scatter_utofu (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBBINE_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= bine_scatter_utofu (utofu buf reg)"); 
    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBBINE_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
        bine_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= bine_scatter_utofu (utofu buf exch)");           
        if(env.utofu_add_ag){
            bine_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.scatter_config.algo_family, this->scc_real, peers[0]);
            bine_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.scatter_config.algo_family, this->scc_real, peers[mp]);
                bine_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        // Everything to 0/NULL just to initialize the internal status.
        bine_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, NULL, 0, env.num_ports); 
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
            compute_peers(this->rank, p, env.scatter_config.algo_family, this->scc_real, peers[p]);
        }        
        timer.reset("= bine_scatter_utofu (computing trees)");
        bine_tree_t tree = get_tree(root, p, env.scatter_config.algo_family, env.scatter_config.distance_type, this->scc_real);

        int receiving_step = tree.reached_at_step[this->rank];

        DPRINTF("Step from root: %d\n", receiving_step);
        size_t issued_sends = 0;
        timer.reset("= bine_scatter_utofu (waiting recv)");
        if(root != this->rank){        
            // Receive the data from the root    
            size_t min_block_r = tree.remapped_ranks[rank];
            size_t max_block_r = tree.remapped_ranks_max[rank];
            size_t blocks_to_recv = (max_block_r - min_block_r) + 1;
            size_t bytes_to_recv = blocks_info[p][0].count*blocks_to_recv*dtsize; // All blocks for this port have the same size
            size_t segments_max_put_size = ceil(bytes_to_recv / ((float) MAX_PUTGET_SIZE));
            bine_utofu_wait_recv(utofu_descriptor, p, 0, segments_max_put_size - 1);
        }else{
            receiving_step = -1;
        }

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * p;
        DPRINTF("Port %d tmpbuf_offset_port %d tmpbuf size %d ceil %f (%d/%d)\n", p, tmpbuf_offset_port, tmpbuf_size, ceil((float) sendcount / env.num_ports), sendcount, env.num_ports);
        uint my_remapped_rank;
        // Blocks remapping. The root permutes the array so that it can send contiguous blocks
        if(this->rank == root){
            timer.reset("= bine_scatter_utofu (permute blocks)");                
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
        issued_sends = 0;
        for(size_t step = 0; step < (uint) this->num_steps; step++){            
            // Compute the range to send/recv
            if(step >= receiving_step + 1){
                uint peer;
                if(env.scatter_config.distance_type == BINE_DISTANCE_DECREASING){
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
                    issued_sends += bine_utofu_isend(utofu_descriptor, &(this->vcq_ids[p][peer]), p, peer, lcl_addr, tmpcnt*dtsize, rmt_addr, 0);                                                             
                }
            }
        }    
        bine_utofu_wait_sends(utofu_descriptor, p, issued_sends);
        free(peers[p]);
        destroy_tree(&tree);

        timer.reset("= bine_scatter_utofu (final memcpy)"); // TODO: Can be avoided if the last put is done in recvbuf rather than tmpbuf
        // Consider offsets of block 0 since everything must go "in the first block"
        // Last ports have one element less.
        DPRINTF("p=%d Copying %d bytes from %d to %d\n", p, blocks_info[p][my_remapped_rank].count*dtsize, tmpbuf_offset_port + (blocks_info[p][0].count)*my_remapped_rank*dtsize, blocks_info[p][0].offset); 
        memcpy((char*) recvbuf + blocks_info[p][0].offset, 
               tmpbuf + tmpbuf_offset_port + blocks_info[p][0].count*my_remapped_rank*dtsize, 
               blocks_info[p][my_remapped_rank].count*dtsize);
        if(free_tmpbuf){
            bine_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, p);
        }        
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }
    
    timer.reset("= bine_scatter_utofu (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif
}


#if 1
static uint32_t btonb(int32_t bin) {
    if (bin > 0x55555555) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}

static int32_t nbtob(uint32_t neg) {
    //const int32_t even = 0x2AAAAAAA, odd = 0x55555555;
    //if ((neg & even) > (neg & odd)) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask ^ neg) - mask;
}

int BineCommon::bine_scatter_mpi(const void *sendbuf, int sendcount, MPI_Datatype dt, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.scatter_config.algo_family == BINE_ALGO_FAMILY_BINE || env.scatter_config.algo_family == BINE_ALGO_FAMILY_RECDOUB);
    assert(env.scatter_config.algo_layer == BINE_ALGO_LAYER_MPI);
    assert(env.scatter_config.algo == BINE_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(dt == recvtype); // TODO: Implement the case where sendtype != recvtype

    int size, rank, dtsize;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(dt, &dtsize);
    // Tempbuffer
    char *tmpbuf = NULL, *sbuf, *rbuf;

    int vrank = mod(rank - root, size); // mod computes math modulo rather than reminder
    int halving_direction = 1; // Down -- send bottom half
    if(rank % 2){
      halving_direction = -1; // Up -- send top half
    }
    // The gather started with these directions. Thus this will
    // be the direction they ended up with if we have an odd number
    // of steps. If not, invert.
    if((int) ceil(log2(size)) % 2 == 0){
        halving_direction *= -1;
    }
    
    // I need to do the opposite of what I did in the gather.
    // Thus, I need to know where min_resident_block and max_resident_block
    // ended up after the last step.
    // Even ranks added 2^0, 2^2, 2^4, ... to max_resident_block
    //   and subtracted 2^1, 2^3, 2^5, ... from min_resident_block
    // Odd ranks subtracted 2^0, 2^2, 2^4, ... from min_resident_block
    //            and added 2^1, 2^3, 2^5, ... to max_resident_block
    size_t min_resident_block, max_resident_block;
    if(rank % 2 == 0){        
        max_resident_block = mod(rank + 0x55555555 & ((0x1 << (int) ceil(log2(size))) - 1), size);
        min_resident_block = mod(rank - 0xAAAAAAAA & ((0x1 << (int) ceil(log2(size))) - 1), size);
    }else{
        min_resident_block = mod(rank - 0x55555555 & ((0x1 << (int) ceil(log2(size))) - 1), size);
        max_resident_block = mod(rank + 0xAAAAAAAA & ((0x1 << (int) ceil(log2(size))) - 1), size);        
    }
    
    int mask = 0x1 << (int) ceil(log2(size)) - 1;
    int recvd = 0;
    int sbuf_offset = rank;
    int is_leaf = 0;
    if(root == rank){
        recvd = 1;
        sbuf = (char*) sendbuf;
    }
    int vrank_nb = btonb(vrank);
    while(mask > 0){
      int partner = vrank_nb ^ ((mask << 1) - 1);
      partner = mod(nbtob(partner) + root, size);      
      int mask_lsbs = (mask << 1) - 1; // Mask with num_steps - step + 1 LSBs set to 1
      int lsbs = vrank_nb & mask_lsbs; // Extract k LSBs
      int equal_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

      size_t top_start, top_end, bottom_start, bottom_end;
      top_start = min_resident_block;
      top_end = mod(min_resident_block + mask - 1, size);
      bottom_start = mod(top_end + 1, size);
      bottom_end = max_resident_block;
      size_t send_start, send_end, recv_start, recv_end;
      if(halving_direction == 1){
        // Send bottom half [..., size - 1]
        send_start = bottom_start;
        send_end = bottom_end;
        recv_start = top_start;
        recv_end = top_end;
        max_resident_block = mod(max_resident_block - mask, size);
      }else{
        // Send top half [0, ...]
        send_start = top_start;
        send_end = top_end;
        recv_start = bottom_start;
        recv_end = bottom_end;
        min_resident_block = mod(min_resident_block + mask, size);
      }
      if(recvd){
        //printf("[%d] Sending [%d, %d] to %d\n", rank, send_start, send_end, partner);
        if(send_end >= send_start){
          // Single send
          MPI_Send((char*) sbuf + send_start*sendcount*dtsize, sendcount*(send_end - send_start + 1), dt, partner, 0, comm);
        }else{
          // Wrapped send
          MPI_Send((char*) sbuf + send_start*sendcount*dtsize, sendcount*((size - 1) - send_start + 1), dt, partner, 0, comm);
          MPI_Send((char*) sbuf                              , sendcount*(send_end + 1)               , dt, partner, 0, comm);
        }
      }else if(equal_lsbs){
        // Setup the buffers to be used from now on
        // How large should the tmpbuf be?
        // It must be large enough to hold a number of blocks 
        // equal to the number of children in the tree rooted in me.
        size_t num_blocks = mod((recv_end - recv_start + 1), size);

        if(recv_start == recv_end){
            // I am a leaf and this is the last step, I do not need a tmpbuf
            rbuf = (char*) recvbuf;
            is_leaf = 1;
        }else{
            tmpbuf = (char*) malloc(recvcount*num_blocks*dtsize);
            sbuf = (char*) tmpbuf;
            rbuf = (char*) tmpbuf;

            // Adjust min and max resident blocks
            min_resident_block = 0;
            max_resident_block = num_blocks - 1;
            
            sbuf_offset = mod(rank - recv_start, size);
        }
        
        //printf("[%d] Receiving [%d, %d] from %d\n", rank, recv_start, recv_end, partner);

        if(recv_end >= recv_start){ 
          // Single recv
          MPI_Recv((char*) rbuf, recvcount*num_blocks, dt, partner, 0, comm, MPI_STATUS_IGNORE);
        }else{
          // Wrapped recv
          MPI_Recv((char*) rbuf                                                   , recvcount*((size - 1) - recv_start + 1), dt, partner, 0, comm, MPI_STATUS_IGNORE);
          MPI_Recv((char*) rbuf + (recvcount*((size - 1) - recv_start + 1))*dtsize, recvcount*(recv_end + 1)               , dt, partner, 0, comm, MPI_STATUS_IGNORE);
        }    
        recvd = 1;
      }
      mask >>= 1;
      halving_direction *= -1;
    }
    if(!is_leaf){
        memcpy(recvbuf, (char*) sbuf + sbuf_offset*recvcount*dtsize, recvcount*dtsize);
    }
    if(tmpbuf != NULL){
        free(tmpbuf);
    }
    return MPI_SUCCESS;
}

#else
int BineCommon::bine_scatter_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.scatter_config.algo_family == BINE_ALGO_FAMILY_BINE || env.scatter_config.algo_family == BINE_ALGO_FAMILY_RECDOUB);
    assert(env.scatter_config.algo_layer == BINE_ALGO_LAYER_MPI);
    assert(env.scatter_config.algo == BINE_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    assert(env.num_ports == 1);
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= bine_scatter_mpi (init)");
    Timer timer("bine_scatter_mpi (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBBINE_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= bine_scatter_mpi (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBBINE_TMPBUF_ALIGNMENT, tmpbuf_size);
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
    timer.reset("= bine_scatter_mpi (computing trees)");
    bine_tree_t tree = get_tree(root, port, env.scatter_config.algo_family, env.scatter_config.distance_type, this->scc_real);

    int receiving_step;
    if(root == this->rank){
        receiving_step = -1;
    }else{
        receiving_step = tree.reached_at_step[this->rank];
    }

    DPRINTF("[%d] Step from root: %d\n", this->rank, receiving_step);
    timer.reset("= bine_scatter_mpi (waiting recv)");
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
        timer.reset("= bine_scatter_mpi (permute)");    
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
            size_t min_block_r = tree.remapped_ranks[this->rank];
            size_t max_block_r = tree.remapped_ranks_max[this->rank];            
            size_t num_blocks = (max_block_r - min_block_r) + 1; 
            DPRINTF("Rank %d receiving %d elems from %d at step %d\n", this->rank, num_blocks*recvcount, peer, step);
            MPI_Recv(tmpbuf + min_block_r*recvcount*dtsize, num_blocks*recvcount, sendtype, peer, TAG_BINE_SCATTER, comm, MPI_STATUS_IGNORE);
        }

        if(step >= receiving_step + 1){
            uint peer;
            if(env.scatter_config.distance_type == BINE_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{  
                peer = peers[port][step];
            }
            if(tree.parent[peer] == this->rank){
                size_t min_block_s = tree.remapped_ranks[peer];
                size_t max_block_s = tree.remapped_ranks_max[peer];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                DPRINTF("Rank %d sending %d elems to %d at step %d\n", this->rank, num_blocks*recvcount, peer, step);
                MPI_Send(tmpbuf + min_block_s*recvcount*dtsize, num_blocks*recvcount, sendtype, peer, TAG_BINE_SCATTER, comm);
            }
        }
        // Wait all the sends for this segment before moving to the next one
        timer.reset("= bine_scatter_mpi (waiting all sends)");
    }
    
    free(peers[port]);

    timer.reset("= bine_scatter_mpi (final memcpy)"); // TODO: Can be avoided if the last put is done in recvbuf rather than tmpbuf
    // Consider offsets of block 0 since everything must go "in the first block"
    DPRINTF("[%d] p=%d Copying %d bytes from %d to %d\n", this->rank, port, blocks_info[port][my_remapped_rank].count*dtsize, tmpbuf_offset_port + blocks_info[port][0].count*my_remapped_rank*dtsize, blocks_info[port][my_remapped_rank].offset); 
    memcpy((char*) recvbuf + blocks_info[port][0].offset, tmpbuf + tmpbuf_offset_port + blocks_info[port][0].count*my_remapped_rank*dtsize, blocks_info[port][my_remapped_rank].count*dtsize);

    if(free_tmpbuf){
        free(tmpbuf);
    }
    destroy_tree(&tree);
    timer.reset("= bine_scatter_mpi (writing profile data to file)");
    return res;
}
#endif
