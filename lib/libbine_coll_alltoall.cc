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

#define BINE_ALLTOALL_NOSYNC_THRESHOLD 0 //6710886400

// Adapted from https://github.com/harp-lab/bruck-alltoallv/blob/main/src/padded_bruck.cpp
int BineCommon::bruck_alltoall(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm) {
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.alltoall_config.algo_family == BINE_ALGO_FAMILY_BRUCK);
    assert(env.alltoall_config.algo_layer == BINE_ALGO_LAYER_MPI);
    assert(env.alltoall_config.algo == BINE_ALLTOALL_ALGO_LOG);
#endif
    Timer timer("bruck_alltoall (init)");

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(datatype, &typesize);

    char *temp_send_buffer, *temp_buffer, *temp_recv_buffer;

    bool free_tmpbuf = false;
    size_t tmpbuf_size = count*nprocs*typesize + count*typesize*((nprocs+1)/2) + count*typesize*((nprocs+1)/2);
    if(tmpbuf_size > env.prealloc_size){
        temp_send_buffer = (char*)malloc(count*nprocs*typesize);
        temp_buffer = (char*)malloc(count*typesize*((nprocs+1)/2));
        temp_recv_buffer = (char*)malloc(count*typesize*((nprocs+1)/2));
        free_tmpbuf = true;
    }else{
        temp_send_buffer = env.prealloc_buf;
        temp_buffer = env.prealloc_buf + count*nprocs*typesize;
        temp_recv_buffer = env.prealloc_buf + count*nprocs*typesize + count*typesize*((nprocs+1)/2);
    }

    // 2. local rotation	
    timer.reset("= bruck_alltoall (rotation)");
	memset(temp_send_buffer, 0, count*nprocs*typesize);
	int offset = 0;
	for (int i = 0; i < nprocs; i++) {
		int index = (i - rank + nprocs) % nprocs;
		memcpy(&temp_send_buffer[index*count*typesize], &((char*) sendbuf)[offset], count*typesize);
		offset += count*typesize;
	}

	// 3. exchange data with log(P) steps
	long long unit_size = count * typesize;
	for (int k = 1; k < nprocs; k <<= 1) {
		// 1) find which data blocks to send
        timer.reset("= bruck_alltoall (send_indexes calc)");
		int send_indexes[(nprocs+1)/2];
		int sendb_num = 0;
		for (int i = k; i < nprocs; i++) {
			if (i & k)
				send_indexes[sendb_num++] = i;
		}

		// 2) copy blocks which need to be sent at this step
        timer.reset("= bruck_alltoall (memcpys)");
		for (int i = 0; i < sendb_num; i++) {
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_buffer+(i*unit_size), temp_send_buffer+offset, unit_size);
		}

		// 3) send and receive
        timer.reset("= bruck_alltoall (sendrecv)");
		int recv_proc = (rank - k + nprocs) % nprocs; // receive data from rank - 2^step process
		int send_proc = (rank + k) % nprocs; // send data from rank + 2^k process

		long long comm_size = sendb_num * unit_size;
		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, temp_recv_buffer, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

		// 4) replace with received data
        timer.reset("= bruck_alltoall (replace)");
		for (int i = 0; i < sendb_num; i++) {
			long long offset = send_indexes[i] * unit_size;
			memcpy(temp_send_buffer+offset, temp_recv_buffer+(i*unit_size), unit_size);
		}
	}

	// 4. second rotation
    timer.reset("= bruck_alltoall (final rotation)");
	offset = 0;
	for (int i = 0; i < nprocs; i++) {
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&((char*)recvbuf)[index*count*typesize], &temp_send_buffer[i*unit_size], count*typesize);
	}
    if(free_tmpbuf){
        free(temp_buffer);
	    free(temp_recv_buffer);
	    free(temp_send_buffer);
    }
    return MPI_SUCCESS;
}

int BineCommon::bine_alltoall_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm) {
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.alltoall_config.algo_family == BINE_ALGO_FAMILY_BINE);
    assert(env.alltoall_config.algo_layer == BINE_ALGO_LAYER_UTOFU);
    assert(env.alltoall_config.algo == BINE_ALLTOALL_ALGO_LOG);
#endif
#ifdef FUGAKU
    assert(env.num_ports == 1); // TODO: Support multiport
    Timer timer("bine_alltoall_utofu (init)");
    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(datatype, &dtsize);
    char* tmpbuf;
    char* scratch = NULL;

    uint* resident_block;  // resident_block[i] contains the id of a block that is resident in the current rank (for i < num_resident_blocks)
    uint* resident_block_next;  // resident_block_next[i] contains the id of a block that is resident in the current rank in the next step (for i < num_resident_blocks_next)
    size_t num_resident_blocks = size;
    size_t num_resident_blocks_next = 0;
    size_t tmpbuf_size = count*dtsize*size;
    if(count*dtsize <= BINE_ALLTOALL_NOSYNC_THRESHOLD){
        tmpbuf_size += (size/2)*count*dtsize*(this->num_steps - 1);
    }
    size_t scratch_size = sizeof(uint)*size + sizeof(uint)*size + tmpbuf_size;
    size_t tmpbuf_offset = 0;

    uint* peers[LIBBINE_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    timer.reset("= bine_alltoall_utofu (utofu buf reg)"); 
    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(scratch_size > env.prealloc_size){
        posix_memalign((void**) &scratch, LIBBINE_TMPBUF_ALIGNMENT, scratch_size);
        resident_block = (uint*) (scratch);
        resident_block_next = (uint*) (scratch + sizeof(uint)*size);
        tmpbuf = scratch + sizeof(uint)*size + sizeof(uint)*size;

        bine_utofu_reg_buf(this->utofu_descriptor, NULL, 0, recvbuf, count*dtsize*size, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= bine_alltoall_utofu (utofu buf exch)");           
        if(env.utofu_add_ag){
            bine_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.alltoall_config.algo_family, this->scc_real, peers[0]);
            bine_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.alltoall_config.algo_family, this->scc_real, peers[mp]);
                bine_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        // Everything to 0/NULL just to initialize the internal status.
        bine_utofu_reg_buf(this->utofu_descriptor, NULL, 0, recvbuf, count*dtsize*size, NULL, 0, env.num_ports); 
        resident_block = (uint*) (env.prealloc_buf);
        resident_block_next = (uint*) (env.prealloc_buf + sizeof(uint)*size);
        tmpbuf_offset = sizeof(uint)*size + sizeof(uint)*size;
        tmpbuf = env.prealloc_buf + tmpbuf_offset;

        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_temp_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }

    // At the beginning I only have my blocks
    uint port = 0; // TODO: Support multiport
    for(size_t i = 0; i < size; i++){
        resident_block[i] = i;
    }

    // Compute the tree
    bine_tree_t tree = get_tree(rank, port, env.alltoall_config.algo_family, env.alltoall_config.distance_type, this->scc_real);
    if(peers[port] == NULL){
        peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(rank, port, env.alltoall_config.algo_family, this->scc_real, peers[port]);
    }

    timer.reset("= bine_alltoall_utofu (remap)");    
    memcpy(tmpbuf, sendbuf, count*dtsize*size);
    size_t min_block_s, max_block_s;

    // We use recvbuf to receive/send the data, and tmpbuf to organize the data to send at the next step
    // By doing so, we avoid a copy form tmpbuf to recvbuf at the end
    size_t tmpbuf_step_offset = 0;
    for(size_t step = 0; step < this->num_steps; step++){
        timer.reset("= bine_alltoall_utofu (bookeeping and copies)");
        uint peer;
        if(env.alltoall_config.distance_type == BINE_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - step - 1];
        }else{          
            peer = peers[port][step];
        }
        min_block_s = tree.remapped_ranks[peer];
        max_block_s = tree.remapped_ranks_max[peer];

        size_t block_recvd_cnt = 0, block_send_cnt = 0;
        size_t offset_send = 0, offset_keep = 0;
        num_resident_blocks_next = 0;
        for(size_t i = 0; i < this->size; i++){
            uint block = resident_block[i % num_resident_blocks];
            // Shall I send this block? Check the negabinary thing    
            uint remap_block = tree.remapped_ranks[block];
            size_t offset = i*count*dtsize;
            
            if(i >= this->size / 2){
                offset += tmpbuf_step_offset;
            }

            // I move to the beginning of tmpbuf the blocks I want to keep,
            // and I move to recvbuf the blocks I want to send.
            if(remap_block >= min_block_s && remap_block <= max_block_s){                
                memcpy((char*) recvbuf + offset_send, tmpbuf + offset, count*dtsize);
                offset_send += count*dtsize;                
                block_send_cnt++;
            }else{
                // Copy the blocks we are not sending to the second half of recvbuf
                if(offset != offset_keep){
                    memcpy(tmpbuf + offset_keep, tmpbuf + offset, count*dtsize);
                }
                offset_keep += count*dtsize;                
                block_recvd_cnt++;

                resident_block_next[num_resident_blocks_next] = block;
                num_resident_blocks_next++;
            }
        }
        assert(block_recvd_cnt == size / 2);
        assert(block_send_cnt == size / 2);
        
        num_resident_blocks /= 2;

        timer.reset("= bine_alltoall_utofu (sendrecv)");
        utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd;
        utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + tmpbuf_offset + (size/2)*count*dtsize + tmpbuf_step_offset;

        size_t issued_sends = 0;
        if(count*dtsize > BINE_ALLTOALL_NOSYNC_THRESHOLD){ 
            // Do a 0-byte put to notify I am ready to recv
            issued_sends += bine_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, 0, rmt_addr, step);                    

            // Do a 0-byte recv to check if the peer is ready to recv
            bine_utofu_wait_recv(utofu_descriptor, port, step, issued_sends - 1);
        }else{
            tmpbuf_step_offset += (size/2)*count*dtsize;            
        }
        

        issued_sends += bine_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count*block_send_cnt*dtsize, rmt_addr, step); 
        
        bine_utofu_wait_recv(utofu_descriptor, port, step, issued_sends - 1);   
        bine_utofu_wait_sends(utofu_descriptor, port, issued_sends); 

        // Update resident blocks
        memcpy(resident_block, resident_block_next, sizeof(uint)*num_resident_blocks);
        /**
        for(size_t i = 0; i < num_resident_blocks; i++){
            resident_block[i] = resident_block_next[i];
        }
        **/
    }

    timer.reset("= bine_alltoall_utofu (final permutation)");
    // Now I need to permute tmpbuf into recvbuf
    // Since I always received the new block on the right, and moved the blocks
    // I wanted to keep to the left, they are now sorted in the same order they reached this
    // rank from their corresponding source ranks. 
    // I.e., I should consider the "reverse tree" (with this rank at the bottom and all the other ranks on top),
    // which represent how the data arrived here.
    // This tree is basically the opposite that I used to send the data (i.e., if LIBBINE_BIN_TREE_DISTANCE=INCREASING)
    // I should consider the decreasing tree, and viceversa.
    // The DFS order (i.e., the remapping) of that tree gives me the permutation.
    // TODO: For multiport I should start from the last port I received from.
    bine_tree_t perm_tree = get_tree(rank, port, env.alltoall_config.algo_family, env.alltoall_config.distance_type == BINE_DISTANCE_DECREASING ? BINE_DISTANCE_INCREASING : BINE_DISTANCE_DECREASING, this->scc_real);
    for(size_t i = 0; i < size; i++){
        size_t index = perm_tree.remapped_ranks[i];
        size_t offset_src = index*count*dtsize;
        size_t offset_dst = i*count*dtsize;
        memcpy((char*) recvbuf + offset_dst, ((char*) tmpbuf) + offset_src, count*dtsize);
    }
    destroy_tree(&perm_tree);
    
    timer.reset("= bine_alltoall_utofu (dealloc)");
    if(scratch){
        bine_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, port);
        free(scratch);
    }
    free(peers[port]);
    destroy_tree(&tree);
    return MPI_SUCCESS;
#else
    assert("uTofu not supported");
    return -1;
#endif
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

// Function to calculate a Mersenne number (2^n - 1)
uint32_t mersenne(int n) {
    return (1UL << n + 1) - 1;
}

// Function that computes the remapped rank when using distance-doubling
// (e.g., in a bcast tree or reduce tree).
// We know that in distance-doubling trees each rank does
// r XOR a Mersenne number (i.e., 2^k - 1 -- like 111, 11, 1111, etc...).
// Thus, if we decompose a number into the XOR of Mersenne numbers,
// then we know the sequence of steps that a rank took to reach that rank.
// We remap the bine tree to a standard distance-halving binary tree
// i.e., if we know a number is decomposed as 11 XOR 1111, we know that 
// it has been reached through the second and fourth step. 
// Thus, we remap it to 10 XOR 1000, since those would be the second and
// fourth steps in a traditional distance-halving binomial tree.
//
// To compute the Mersenne decomposition we simply XOR with the largest
// Mersenne number we can (e.g., if decomposing 101, we XOR with 111).
// Then, we keep applying the same process to the remaining number,
// until we reach 0.
int remap_distance_doubling(uint32_t num) {
    int remapped = 0;
    while (num > 0) {
        int k = 31 - __builtin_clz(num); // Find the position of the highest set bit
        remapped ^= (0x1 << k); // Set the k-th bit in the remapped number
        num ^= mersenne(k); // XOR the Mersenne number with the remaining number
    }
    return remapped;
}

int BineCommon::bine_alltoall_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm) {    
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.alltoall_config.algo_family == BINE_ALGO_FAMILY_BINE);
    assert(env.alltoall_config.algo_layer == BINE_ALGO_LAYER_MPI);
    assert(env.alltoall_config.algo == BINE_ALLTOALL_ALGO_LOG);
#endif
    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(datatype, &dtsize);    

    uint* resident_block;  // resident_block[i] contains the id of a block that is resident in the current rank (for i < num_resident_blocks)
    uint* resident_block_next;  // resident_block_next[i] contains the id of a block that is resident in the current rank in the next step (for i < num_resident_blocks_next)
    size_t num_resident_blocks = size;
    size_t num_resident_blocks_next = 0;
    size_t tmpbuf_size = count*dtsize*size + sizeof(uint)*size + sizeof(uint)*size;
    char* tmpbuf = (char*) malloc(tmpbuf_size);
    resident_block = (uint*) (tmpbuf + count*dtsize*size);
    resident_block_next = (uint*) (tmpbuf + count*dtsize*size + sizeof(uint)*size);

    // At the beginning I only have my blocks
    for(size_t i = 0; i < size; i++){
        resident_block[i] = i;
    }

    memcpy(tmpbuf, sendbuf, count*dtsize*size);
    size_t min_block_s, max_block_s;

    // We use recvbuf to receive/send the data, and tmpbuf to organize the data to send at the next step
    // By doing so, we avoid a copy form tmpbuf to recvbuf at the end
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
        min_block_s = remap_rank(partner, size) & block_first_mask;
        max_block_s = min_block_s + inverse_mask - 1;

        size_t block_recvd_cnt = 0, block_send_cnt = 0;
        size_t offset_send = 0, offset_keep = 0;
        num_resident_blocks_next = 0;
        for(size_t i = 0; i < this->size; i++){
            uint block = resident_block[i % num_resident_blocks];
            // Shall I send this block? Check the negabinary thing    
            uint remap_block = remap_rank(block, size);
            size_t offset = i*count*dtsize;
            
            // I move to the beginning of tmpbuf the blocks I want to keep,
            // and I move to recvbuf the blocks I want to send.
            if(remap_block >= min_block_s && remap_block <= max_block_s){                
                memcpy((char*) recvbuf + offset_send, tmpbuf + offset, count*dtsize);
                offset_send += count*dtsize;                
                block_send_cnt++;
            }else{
                // Copy the blocks we are not sending to the second half of recvbuf
                if(offset != offset_keep){
                    memcpy(tmpbuf + offset_keep, tmpbuf + offset, count*dtsize);
                }
                offset_keep += count*dtsize;                
                block_recvd_cnt++;

                resident_block_next[num_resident_blocks_next] = block;
                num_resident_blocks_next++;
            }
        }
        assert(block_recvd_cnt == size/2);
        assert(block_send_cnt == size/2);
        
        num_resident_blocks /= 2;

        // I receive data in the second half of tmpbuf (the first half contains the blocks I am keeping from previous iteration)
        int r = MPI_Sendrecv((char*) recvbuf, count*block_send_cnt, datatype,
                             partner, 0,
                             tmpbuf + (size / 2)*count*dtsize, count*block_send_cnt, datatype,
                             partner, 0, 
                             comm, MPI_STATUS_IGNORE);
        
        if(r != MPI_SUCCESS){
            return r;
        }      

        // Update resident blocks
        uint* tmp = resident_block_next;
        resident_block_next = resident_block;
        resident_block = tmp;

        mask <<= 1;
        inverse_mask >>= 1;
        block_first_mask >>= 1;        
    }

    // Now I need to permute tmpbuf into recvbuf
    // Since I always received the new block on the right, and moved the blocks
    // I wanted to keep to the left, they are now sorted in the same order they reached this
    // rank from their corresponding source ranks. 
    // I.e., I should consider the "reverse tree" (with this rank at the bottom and all the other ranks on top),
    // which represent how the data arrived here.
    // This tree is basically the opposite that I used to send the data
    // I should consider the decreasing tree, and viceversa.
    for(size_t i = 0; i < size; i++){
        int rotated_i = 0;
        if((rank % 2) == 0){            
            rotated_i = mod(i - rank, size);
        }else{
            rotated_i = mod(rank - i, size);
        }
        int repr = 0;
        if(in_range(rotated_i, ceil(log2(size)))){
            repr = btonb(rotated_i);
        }else{
            repr = btonb(rotated_i - size);
        }                            
        int index = remap_distance_doubling(repr);
        //printf("[%d] i=%d index: %d vs %d\n", rank, i, index, x);

        size_t offset_src = index*count*dtsize;
        size_t offset_dst = i*count*dtsize;
        memcpy((char*) recvbuf + offset_dst, ((char*) tmpbuf) + offset_src, count*dtsize);
    }    
    free(tmpbuf);
    return MPI_SUCCESS;
}


#if 0
int BineCommon::bine_alltoall_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm) {
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.alltoall_config.algo_family == BINE_ALGO_FAMILY_BINE);
    assert(env.alltoall_config.algo_layer == BINE_ALGO_LAYER_MPI);
    assert(env.alltoall_config.algo == BINE_ALLTOALL_ALGO_LOG);
#endif
    assert(env.num_ports == 1); // TODO: Support multiport
    Timer timer("bine_alltoall_mpi (init)");
    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(datatype, &dtsize);
    char* tmpbuf;
    bool free_tmpbuf = false;

    uint* resident_block;  // resident_block[i] contains the id of a block that is resident in the current rank (for i < num_resident_blocks)
    uint* resident_block_next;  // resident_block_next[i] contains the id of a block that is resident in the current rank in the next step (for i < num_resident_blocks_next)
    size_t num_resident_blocks = size;
    size_t num_resident_blocks_next = 0;
    size_t tmpbuf_size = count*dtsize*size + sizeof(uint)*size + sizeof(uint)*size;
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBBINE_TMPBUF_ALIGNMENT, tmpbuf_size);        
        resident_block = (uint*) malloc(sizeof(uint)*size);
        resident_block_next = (uint*) malloc(sizeof(uint)*size);
        free_tmpbuf = true;
    }else{
        tmpbuf = env.prealloc_buf;
        resident_block = (uint*) (tmpbuf + count*dtsize*size);
        resident_block_next = (uint*) (tmpbuf + count*dtsize*size + sizeof(uint)*size);
    }

    // At the beginning I only have my blocks
    uint port = 0; // TODO: Support multiport
    for(size_t i = 0; i < size; i++){
        resident_block[i] = i;
    }

    // Compute the tree
    bine_tree_t tree = get_tree(rank, port, env.alltoall_config.algo_family, env.alltoall_config.distance_type, this->scc_real);
    uint* peers = (uint*) malloc(sizeof(uint)*this->size);
    compute_peers(rank, port, env.alltoall_config.algo_family, this->scc_real, peers);

    timer.reset("= bine_alltoall_mpi (remap)");    
    memcpy(tmpbuf, sendbuf, count*dtsize*size);
    size_t min_block_s, max_block_s;

    // We use recvbuf to receive/send the data, and tmpbuf to organize the data to send at the next step
    // By doing so, we avoid a copy form tmpbuf to recvbuf at the end
    for(size_t step = 0; step < this->num_steps; step++){
        timer.reset("= bine_alltoall_mpi (bookeeping and copies)");
        uint peer;
        if(env.alltoall_config.distance_type == BINE_DISTANCE_DECREASING){
            peer = peers[this->num_steps - step - 1];
        }else{          
            peer = peers[step];
        }
        min_block_s = tree.remapped_ranks[peer];
        max_block_s = tree.remapped_ranks_max[peer];

        size_t block_recvd_cnt = 0, block_send_cnt = 0;
        size_t offset_send = 0, offset_keep = 0;
        num_resident_blocks_next = 0;
        for(size_t i = 0; i < this->size; i++){
            uint block = resident_block[i % num_resident_blocks];
            // Shall I send this block? Check the negabinary thing    
            uint remap_block = tree.remapped_ranks[block];
            size_t offset = i*count*dtsize;
            
            // I move to the beginning of tmpbuf the blocks I want to keep,
            // and I move to recvbuf the blocks I want to send.
            if(remap_block >= min_block_s && remap_block <= max_block_s){                
                DPRINTF("[%d] Step %d (send) copying from offset %d to %d [%u]\n", rank, step, offset, offset_send, ((char*) tmpbuf)[offset]);
                memcpy((char*) recvbuf + offset_send, tmpbuf + offset, count*dtsize);
                offset_send += count*dtsize;                
                block_send_cnt++;
            }else{
                // Copy the blocks we are not sending to the second half of recvbuf
                DPRINTF("[%d] Step %d (keep) copying from offset %d to %d [%u]\n", rank, step, offset, offset_keep, ((char*) tmpbuf)[offset]);
                if(offset != offset_keep){
                    memcpy(tmpbuf + offset_keep, tmpbuf + offset, count*dtsize);
                }
                offset_keep += count*dtsize;                
                block_recvd_cnt++;

                resident_block_next[num_resident_blocks_next] = block;
                num_resident_blocks_next++;
            }
        }
        assert(block_recvd_cnt == size/2);
        assert(block_send_cnt == size/2);
        
        num_resident_blocks /= 2;

        timer.reset("= bine_alltoall_mpi (sendrecv)");
        // I receive data in the second half of tmpbuf (the first half contains the blocks I am keeping from previous iteration)
        int r = MPI_Sendrecv((char*) recvbuf, count*block_send_cnt, datatype,
                             peer, TAG_BINE_ALLTOALL,
                             tmpbuf + (size / 2)*count*dtsize, count*block_send_cnt, datatype,
                             peer, TAG_BINE_ALLTOALL, 
                             comm, MPI_STATUS_IGNORE);
        
        //memcpy(tmpbuf, recvbuf, block_send_cnt*dtsize*count);// TODO: I can avoid this memcpy with double buffering
        if(r != MPI_SUCCESS){
            return r;
        }      

        // Update resident blocks
        memcpy(resident_block, resident_block_next, sizeof(uint)*num_resident_blocks);
        /*
        for(size_t i = 0; i < num_resident_blocks; i++){
            resident_block[i] = resident_block_next[i];
        }
        */
    }

    timer.reset("= bine_alltoall_mpi (final permutation)");
    // Now I need to permute tmpbuf into recvbuf
    // Since I always received the new block on the right, and moved the blocks
    // I wanted to keep to the left, they are now sorted in the same order they reached this
    // rank from their corresponding source ranks. 
    // I.e., I should consider the "reverse tree" (with this rank at the bottom and all the other ranks on top),
    // which represent how the data arrived here.
    // This tree is basically the opposite that I used to send the data (i.e., if LIBBINE_BIN_TREE_DISTANCE=INCREASING)
    // I should consider the decreasing tree, and viceversa.
    // The DFS order (i.e., the remapping) of that tree gives me the permutation.
    // TODO: For multiport I should start from the last port I received from.
    bine_tree_t perm_tree = get_tree(rank, port, env.alltoall_config.algo_family, env.alltoall_config.distance_type == BINE_DISTANCE_DECREASING ? BINE_DISTANCE_INCREASING : BINE_DISTANCE_DECREASING, this->scc_real);
    for(size_t i = 0; i < size; i++){
        size_t index = perm_tree.remapped_ranks[i];
        size_t offset_src = index*count*dtsize;
        size_t offset_dst = i*count*dtsize;
        memcpy((char*) recvbuf + offset_dst, ((char*) tmpbuf) + offset_src, count*dtsize);
    }
    destroy_tree(&perm_tree);
    
    timer.reset("= bine_alltoall_mpi (dealloc)");
    if(free_tmpbuf){
        free(tmpbuf);
        free(resident_block);
        free(resident_block_next);
    }
    free(peers);
    destroy_tree(&tree);
    return MPI_SUCCESS;
}
#endif
