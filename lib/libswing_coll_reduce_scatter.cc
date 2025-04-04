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


int SwingCommon::swing_reduce_scatter_utofu_blocks(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    return MPI_ERR_OTHER;
}

int SwingCommon::swing_reduce_scatter_utofu_contiguous(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
#ifdef FUGAKU
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_scatter_utofu_contiguous (init)");
    Timer timer("swing_reduce_scatter_utofu_contiguous (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    
    // sendbuf is read-only, and recvbuf has space only for one (final) block.
    // Thus, I need two larger temporary buffers: one for receiving the new data, and the other one to aggregate the data in (and to send the data from).
    // To avoid registering two different pointers, I will allocate a single buffer and then I will use two pointers to point to the two different parts of the buffer.
    // I call these two buffers tmpbuf_send and tmpbuf_recv.
    // How big should they be? I should consider the largest count possible
    // In case count not divisible by num ports, the first port will for sure have the largest blocks.
    // Thus, it is enough to check the count of the first block of the first port to know the largest block.
    // TODO: This work for reduce_scatter_block, generalize it to reduce_scatter
    size_t tmpbuf_send_size = blocks_info[0][0].count*env.num_ports*dtsize*this->size;
    // For tmpbuf_recv_size I have two options:
    // 1. I can either make it as large as the tmpbuf_send. In this case, at each step each rank can write in a different part of the buffer.
    // 2. I can make it half of that, but this would require explicit synchronization between rank before doing the PUTs
    //
    // I chose the first option. // TODO: Switch between the two to save some memory
    size_t tmpbuf_recv_size = tmpbuf_send_size;
    
    size_t tmpbuf_size = tmpbuf_send_size + tmpbuf_recv_size;    
    timer.reset("= swing_reduce_scatter_utofu_contiguous (utofu buf reg)"); 
    // Also the root sends from tmpbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        assert(posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size) == 0);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_reduce_scatter_utofu_contiguous (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.reduce_scatter_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.reduce_scatter_config.algo_family, this->scc_real, peers[mp]);
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
    char* tmpbuf_send = tmpbuf;
    char* tmpbuf_recv = tmpbuf + tmpbuf_send_size;

    int res = MPI_SUCCESS; 
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_scatter_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_scatter_utofu_contiguous (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.reduce_scatter_config.algo_family, env.reduce_scatter_config.distance_type, this->scc_real);

        size_t offset_port = tmpbuf_send_size / env.num_ports * port;
        size_t offset_port_recv = tmpbuf_recv_size / env.num_ports * port;

        DPRINTF("Offset_port_recv %d: %d\n", port, offset_port_recv);
        DPRINTF("Offset_port %d: %d\n", port, offset_port);

        char* tmpbuf_send_port = tmpbuf_send + offset_port;
        char* tmpbuf_recv_port = tmpbuf_recv + offset_port_recv;

        timer.reset("= swing_reduce_scatter_utofu_contiguous (permute)");    
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving %d bytes from %d to %d\n", this->rank, blocks_info[port][i].count*dtsize, blocks_info[port][i].offset, offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            // For a given port, all the sub-blocks have the same size, so we can just consider the count of any block (e.g., 0)
            memcpy(tmpbuf_send_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize, (char*) sendbuf + blocks_info[port][i].offset, blocks_info[port][i].count*dtsize);
        }        

        // Now perform all the subsequent steps            
        size_t offset_step_recv = 0; // This is used so that each step writes at a different location in tmpbuf_recv
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            uint peer;
            if(env.reduce_scatter_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{
                peer = peers[port][step];
            }

            // Always send from the end of the buffer
            // and receive in the beginning of the buffer
            timer.reset("= swing_reduce_scatter_utofu_contiguous (sendrecv)");            
            size_t num_blocks = this->size / (pow(2, step + 1));                        
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;
            
            utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd                          + offset_port      + count_to_sendrecv*dtsize;
            utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + tmpbuf_send_size + offset_port_recv + offset_step_recv; 

            DPRINTF("Port %d sending from %d to %d\n", port, lcl_addr - utofu_descriptor->port_info[port].lcl_temp_stadd, rmt_addr - utofu_descriptor->port_info[port].rmt_temp_stadd[peer]);
            DPRINTF("tmpbuf[0] (port %d) at step %d before send: %d \n", port, step, ((char*) tmpbuf_send_port)[0]);

            size_t issued_sends = 0;
            issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step); 
            swing_utofu_wait_recv(utofu_descriptor, port, step, issued_sends - 1);
            swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);

            DPRINTF("tmpbuf_recv[0] (port %d) at step %d after send: %d \n", port, step, ((char*) tmpbuf_recv_port)[0]);

#pragma omp critical
{
            if(step == this->num_steps - 1){
                // To avoid doing a memcpy at the end
                reduce_local(tmpbuf_recv_port + offset_step_recv, tmpbuf_send_port, (char*) recvbuf + blocks_info[port][0].offset, count_to_sendrecv, datatype, op);
                DPRINTF("tmpbuf_send[0] (port %d) at step %d after aggr:  %d \n", port, step, ((char*) recvbuf + blocks_info[port][0].offset)[0]);
            }else{
                reduce_local(tmpbuf_recv_port + offset_step_recv, tmpbuf_send_port, count_to_sendrecv, datatype, op);
                DPRINTF("tmpbuf_send[0] (port %d) at step %d after aggr:  %d \n", port, step, ((char*) tmpbuf_send_port)[0]);
            }
}
            offset_step_recv += count_to_sendrecv*dtsize;
        }        
        free(peers[port]);
        destroy_tree(&tree);
        if(free_tmpbuf){
            swing_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, port);
        }        
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_scatter_utofu_contiguous (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif 
}

int SwingCommon::swing_reduce_scatter_utofu(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    if(is_power_of_two(this->size)){
        return swing_reduce_scatter_utofu_contiguous(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }else{
        return swing_reduce_scatter_utofu_blocks(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }
}

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

static inline uint32_t reverse(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}


static inline int in_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline uint32_t get_rank_negabinary_representation(uint32_t rank, uint32_t size){
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
    size_t num_bits = ceil_log2(size);
    if(rank % 2){
        if(in_range(rank, num_bits)){
            nba = btonb(rank);
        }
        if(in_range(rank - size, num_bits)){
            nbb = btonb(rank - size);
        }
    }else{
        if(in_range(-rank, num_bits)){
            nba = btonb(-rank);
        }
        if(in_range(-rank + size, num_bits)){
            nbb = btonb(-rank + size);
        }
    }
    assert(nba != UINT32_MAX || nbb != UINT32_MAX);

    if(nba == UINT32_MAX && nbb != UINT32_MAX){
        return nbb;
    }else if(nba != UINT32_MAX && nbb == UINT32_MAX){
        return nba;
    }else{ // Check MSB
        if(nba & (80000000 >> (32 - num_bits))){
            return nba;
        }else{
            return nbb;
        }
    }
}

static inline uint32_t remap_rank(uint32_t rank, uint32_t size){
    uint32_t remap_rank = get_rank_negabinary_representation(rank, size);    
    remap_rank = remap_rank ^ (remap_rank >> 1);
    size_t num_bits = ceil_log2(size);
    remap_rank = reverse(remap_rank) >> (32 - num_bits);
    return remap_rank;
}

#if 0
// Version with send at the end
int SwingCommon::swing_reduce_scatter_mpi_new(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype dt, MPI_Op op, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
  int size, rank, dtsize;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int* displs = (int*) malloc(size*sizeof(int));
  int* step_to_send = (int*) malloc(size*sizeof(int));
  for(int i = 0; i < size; i++){
    displs[i] = count;
    count += recvcounts[i];    
  }
  
  void* tmpbuf = malloc(count*dtsize);
  void* resbuf = malloc(count*dtsize);
  memcpy(resbuf, sendbuf, count*dtsize);
  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (ceil(log2(size)) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(rank, size);
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
        partner = mod(rank + nbtob((mask << 1) - 1), size); 
    }else{
        partner = mod(rank - nbtob((mask << 1) - 1), size); 
    }

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to 
    // the power of two
    int send_block_first = remap_rank(partner, size) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
    MPI_Sendrecv((char*) resbuf + displs[send_block_first]*dtsize, send_count, dt, partner, 0,
                 (char*) tmpbuf + displs[recv_block_first]*dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
    MPI_Reduce_local((char*) tmpbuf + displs[recv_block_first]*dtsize, (char*) resbuf + displs[recv_block_first]*dtsize, recv_count, dt, op);

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }
  
  // Final send
  // Whom I have been remapped to? I.e., who is going to send me my data? Just do a recv from any
  MPI_Status status;
  MPI_Sendrecv((char*) resbuf + displs[remapped_rank]*dtsize, recvcounts[remapped_rank], dt, remapped_rank , 0,
               (char*) recvbuf                              , recvcounts[rank]         , dt, MPI_ANY_SOURCE, 0, 
               comm, &status);

  free(tmpbuf);
  free(resbuf);
  free(displs);
  return MPI_SUCCESS;
}
#endif

#if 0
// Version with permute at the beginning
int SwingCommon::swing_reduce_scatter_mpi_new(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype dt, MPI_Op op, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
  int size, rank, dtsize;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int* displs = (int*) malloc(size*sizeof(int));
  int* step_to_send = (int*) malloc(size*sizeof(int));
  for(int i = 0; i < size; i++){
    displs[i] = count;
    count += recvcounts[i];    
  }
  
  void* tmpbuf = malloc(count*dtsize);
  void* resbuf = malloc(count*dtsize);

  // Permute memcpy
  for(int i = 0; i < size; i++){
    int remapped_rank = remap_rank(i, size);
    memcpy((char*) resbuf + displs[remapped_rank]*dtsize, (char*) sendbuf + displs[i]*dtsize, recvcounts[i]*dtsize);
  }

  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (ceil(log2(size)) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(rank, size);
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
        partner = mod(rank + nbtob((mask << 1) - 1), size); 
    }else{
        partner = mod(rank - nbtob((mask << 1) - 1), size); 
    } 

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to 
    // the power of two
    int send_block_first = remap_rank(partner, size) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];
    
    MPI_Sendrecv((char*) resbuf + displs[send_block_first]*dtsize, send_count, dt, partner, 0,
                 (char*) tmpbuf + displs[recv_block_first]*dtsize, recv_count, dt, partner, 0, comm, MPI_STATUS_IGNORE);
    MPI_Reduce_local((char*) tmpbuf + displs[recv_block_first]*dtsize, (char*) resbuf + displs[recv_block_first]*dtsize, recv_count, dt, op);

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }

  // Final memcpy
  memcpy(recvbuf, (char*) resbuf + displs[remapped_rank]*dtsize, recvcounts[rank]*dtsize);

  free(tmpbuf);
  free(resbuf);
  free(displs);
  return MPI_SUCCESS;
}
#endif

#if 0
// Version with send block-by-block
int SwingCommon::swing_reduce_scatter_mpi_new(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype dt, MPI_Op op, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
  int size, rank, dtsize;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int* displs = (int*) malloc(size*sizeof(int));
  int* step_to_send = (int*) malloc(size*sizeof(int));
  int* inverse_remapping = (int*) malloc(size*sizeof(int));
  for(int i = 0; i < size; i++){
    displs[i] = count;
    count += recvcounts[i];    

    inverse_remapping[remap_rank(i, size)] = i;
  }
  
  void* tmpbuf = malloc(count*dtsize);
  void* resbuf = malloc(count*dtsize);
  memcpy(resbuf, sendbuf, count*dtsize);
  
  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (ceil(log2(size)) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  int remapped_rank = remap_rank(rank, size);
  MPI_Request* reqs = (MPI_Request*) malloc(size*sizeof(MPI_Request));  
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
        partner = mod(rank + nbtob((mask << 1) - 1), size); 
    }else{
        partner = mod(rank - nbtob((mask << 1) - 1), size); 
    }   

    // For sure I need to send my (remapped) partner's data
    // the actual start block however must be aligned to 
    // the power of two
    int send_block_first = remap_rank(partner, size) & block_first_mask;
    int send_block_last = send_block_first + inverse_mask - 1;
    int send_count = displs[send_block_last] - displs[send_block_first] + recvcounts[send_block_last];
    // Something similar for the block to recv.
    // I receive my block, but aligned to the power of two
    int recv_block_first = remapped_rank & block_first_mask;
    int recv_block_last = recv_block_first + inverse_mask - 1;
    int recv_count = displs[recv_block_last] - displs[recv_block_first] + recvcounts[recv_block_last];

    int next_req = 0;
    for(size_t block = recv_block_first; block <= recv_block_last; block++){
        if(mask << 1 >= size){
            // Last step, receiving in recvbuf
            MPI_Irecv((char*) recvbuf, recvcounts[inverse_remapping[block]], dt, partner, 0,
                      comm, &reqs[next_req]);
        }else{
            MPI_Irecv((char*) tmpbuf + displs[inverse_remapping[block]]*dtsize, recvcounts[inverse_remapping[block]], dt, partner, 0,
                      comm, &reqs[next_req]);
        }
        ++next_req;
    }

    for(size_t block = send_block_first; block <= send_block_last; block++){
        MPI_Isend((char*) resbuf + displs[inverse_remapping[block]]*dtsize, recvcounts[inverse_remapping[block]], dt, partner, 0,
                  comm, &reqs[next_req]);
        ++next_req;
    }

    int w_req = 0;
    for(size_t block = recv_block_first; block <= recv_block_last; block++){
        MPI_Wait(&reqs[w_req], MPI_STATUS_IGNORE);
        if(mask << 1 >= size){
            // Last step, received in recvbuf, aggregating from resbuf
            MPI_Reduce_local((char*) resbuf + displs[inverse_remapping[block]]*dtsize, (char*) recvbuf, recvcounts[inverse_remapping[block]], dt, op);
        }else{
            MPI_Reduce_local((char*) tmpbuf + displs[inverse_remapping[block]]*dtsize, (char*) resbuf + displs[inverse_remapping[block]]*dtsize, recvcounts[inverse_remapping[block]], dt, op);
        }
        ++w_req;
    }
    MPI_Waitall(next_req - w_req, &reqs[w_req], MPI_STATUSES_IGNORE);

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
  }
  free(reqs);
  free(tmpbuf);
  free(resbuf);
  free(inverse_remapping);
  free(displs);
  return MPI_SUCCESS;
}
#endif

#if 1
static inline uint32_t nb_to_nu(uint32_t nb, uint32_t size){
    return reverse(nb ^ (nb >> 1)) >> 32 - ceil_log2(size);
}

static inline uint32_t get_nu(uint32_t rank, uint32_t size){
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
    size_t num_bits = ceil_log2(size);
    if(rank % 2){
        if(in_range(rank, num_bits)){
            nba = btonb(rank);
        }
        if(in_range(rank - size, num_bits)){
            nbb = btonb(rank - size);
        }
    }else{
        if(in_range(-rank, num_bits)){
            nba = btonb(-rank);
        }
        if(in_range(-rank + size, num_bits)){
            nbb = btonb(-rank + size);
        }
    }
    assert(nba != UINT32_MAX || nbb != UINT32_MAX);

    if(nba == UINT32_MAX && nbb != UINT32_MAX){
        return nb_to_nu(nbb, size);
    }else if(nba != UINT32_MAX && nbb == UINT32_MAX){
        return nb_to_nu(nba, size);
    }else{ // Check MSB
        int nu_a = nb_to_nu(nba, size);
        int nu_b = nb_to_nu(nbb, size);
        if(nu_a < nu_b){
            return nu_a;
        }else{
            return nu_b;
        }
    }
}

// Version with send block-by-block that also works for non-power of two
int SwingCommon::swing_reduce_scatter_mpi_new(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype dt, MPI_Op op, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
  int size, rank, dtsize;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(dt, &dtsize);
  int count = 0;
  int* displs = (int*) malloc(size*sizeof(int));
  for(int i = 0; i < size; i++){
    displs[i] = count;
    count += recvcounts[i];    
  }
  
  void* tmpbuf = malloc(count*dtsize);
  void* resbuf = malloc(count*dtsize);
  memcpy(resbuf, sendbuf, count*dtsize);
  
  int mask = 0x1;
  int inverse_mask = 0x1 << (int) (ceil(log2(size)) - 1);
  int block_first_mask = ~(inverse_mask - 1);
  MPI_Request* reqs_s = (MPI_Request*) malloc(size*sizeof(MPI_Request));  
  MPI_Request* reqs_r = (MPI_Request*) malloc(size*sizeof(MPI_Request));  
  int* blocks_to_recv = (int*) malloc(size*sizeof(int));
  int next_req_s = 0, next_req_r = 0;
  int reverse_step = ceil_log2(size) - 1;
  int last_recv_done = 0;
  while(mask < size){
    int partner;
    if(rank % 2 == 0){
        partner = mod(rank + nbtob((mask << 1) - 1), size); 
    }else{
        partner = mod(rank - nbtob((mask << 1) - 1), size); 
    }   

    next_req_r = 0;
    next_req_s = 0;

    // We start from 1 because 0 never sends block 0
    for(size_t block = 1; block < size; block++){
        // Get the position of the highest set bit using clz
        // That gives us the first at which block departs from 0
        int k = 31 - __builtin_clz(get_nu(block, size));
        // Check if this must be sent
        if(k == reverse_step){
            // 0 would send this block
            size_t block_to_send, block_to_recv;
            if(rank % 2 == 0){
                // I am even, thus I need to shift by rank position to the right
                block_to_send = mod(block + rank, size);
                // What to receive? What my partner is sending
                // Since I am even, my partner is odd, thus I need to mirror it and then shift
                block_to_recv = mod(partner - block, size);
            }else{
                // I am odd, thus I need to mirror it
                block_to_send = mod(rank - block, size);
                // What to receive? What my partner is sending
                // Since I am odd, my partner is even, thus I need to mirror it and then shift   
                block_to_recv = mod(block + partner, size);
            }

            if(block_to_send != rank){
                MPI_Isend((char*) resbuf + displs[block_to_send]*dtsize, recvcounts[block_to_send], dt, partner, 0,
                           comm, &reqs_s[next_req_s]);
                ++next_req_s;
            }

            if(block_to_recv != partner){
                blocks_to_recv[next_req_r] = block_to_recv;
                if(mask << 1 >= size){
                    // Last step, receiving in recvbuf                       
                    MPI_Irecv((char*) recvbuf, recvcounts[block_to_recv], dt, partner, 0,
                            comm, &reqs_r[next_req_r]);       
                    last_recv_done = 1;                             
                }else{
                    MPI_Irecv((char*) tmpbuf + displs[block_to_recv]*dtsize, recvcounts[block_to_recv], dt, partner, 0,
                            comm, &reqs_r[next_req_r]);                
                }                            
                ++next_req_r;
            }
        }        
    }

    for(size_t block = 0; block < next_req_r; block++){
        MPI_Wait(&reqs_r[block], MPI_STATUS_IGNORE);
        if(mask << 1 >= size){
            // Last step, received in recvbuf, aggregating from resbuf
            MPI_Reduce_local((char*) resbuf + displs[blocks_to_recv[block]]*dtsize, (char*) recvbuf                                      , recvcounts[blocks_to_recv[block]], dt, op);
        }else{
            MPI_Reduce_local((char*) tmpbuf + displs[blocks_to_recv[block]]*dtsize, (char*) resbuf + displs[blocks_to_recv[block]]*dtsize, recvcounts[blocks_to_recv[block]], dt, op);
        }
    }
    MPI_Waitall(next_req_s, reqs_s, MPI_STATUSES_IGNORE);

    mask <<= 1;
    inverse_mask >>= 1;
    block_first_mask >>= 1;
    reverse_step--;
  }
  if(!last_recv_done){
    memcpy(recvbuf, (char*) resbuf + displs[rank]*dtsize, recvcounts[rank]*dtsize);
  }
  free(tmpbuf);
  free(resbuf);
  free(displs);
  free(blocks_to_recv);
  return MPI_SUCCESS;
}
#endif

int SwingCommon::swing_reduce_scatter_mpi_contiguous(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_scatter_mpi_contiguous (init)");
    Timer timer("swing_reduce_scatter_mpi_contiguous (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    
    // How big should the buffer be? I should consider the largest count possible
    // In case count not divisible by num ports, the first port will for sure have the largest blocks.
    // Thus, it is enough to check the count of the first block of the first port to know the largest block.
    // TODO: This work for reduce_scatter_block, generalize it to reduce_scatter
    size_t tmpbuf_size = blocks_info[0][0].count * dtsize * env.num_ports * this->size;
    
    timer.reset("= swing_reduce_scatter_mpi_contiguous (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }

    char* tmpbuf_send = tmpbuf;

    int res = MPI_SUCCESS; 
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_scatter_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_scatter_mpi_contiguous (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.reduce_scatter_config.algo_family, env.reduce_scatter_config.distance_type, this->scc_real);

        size_t offset_port = tmpbuf_size / env.num_ports * port;
        DPRINTF("Offset_port %d: %d\n", port, offset_port);
        char* tmpbuf_send_port = tmpbuf_send + offset_port;

        timer.reset("= swing_reduce_scatter_mpi_contiguous (permute)");    
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving %d bytes from %d to %d\n", this->rank, blocks_info[port][i].count*dtsize, blocks_info[port][i].offset, offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            // For a given port, all the sub-blocks have the same size, so we can just consider the count of any block (e.g., 0)
            memcpy(tmpbuf + offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize, (char*) sendbuf + blocks_info[port][i].offset, blocks_info[port][i].count*dtsize);
        }        

        // Now perform all the subsequent steps            
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            uint peer;
            if(env.reduce_scatter_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{
                peer = peers[port][step];
            }

            // Always send from the end of the buffer
            // and receive in the beginning of the buffer
            timer.reset("= swing_reduce_scatter_mpi_contiguous (sendrecv)");
            size_t num_blocks = this->size / (pow(2, step + 1));                        
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;

            MPI_Sendrecv_replace(tmpbuf_send_port + count_to_sendrecv*dtsize, count_to_sendrecv, datatype, 
                                 peer, TAG_SWING_REDUCESCATTER, 
                                 peer, TAG_SWING_REDUCESCATTER, comm, MPI_STATUS_IGNORE); 

            if(step == this->num_steps - 1){
                // To avoid doing a memcpy at the end
                reduce_local(tmpbuf_send_port + count_to_sendrecv*dtsize, tmpbuf_send_port, (char*) recvbuf + offset_port, count_to_sendrecv, datatype, op);
            }else{
                MPI_Reduce_local(tmpbuf_send_port + count_to_sendrecv*dtsize, tmpbuf_send_port, count_to_sendrecv, datatype, op);
            }
        }        
        free(peers[port]);

        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_scatter_mpi_contiguous (writing profile data to file)");
    return res;
}

int SwingCommon::swing_reduce_scatter_mpi_blocks(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    return MPI_ERR_OTHER;
}

int SwingCommon::swing_reduce_scatter_mpi(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    if(is_power_of_two(this->size)){
        return swing_reduce_scatter_mpi_contiguous(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }else{
        return swing_reduce_scatter_mpi_blocks(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }
}
