#include <assert.h>
#include <inttypes.h>

#include "libswing_common.h"
#ifdef FUGAKU
#include "fugaku/swing_utofu.h"
#endif

Timer::Timer() {
#ifdef PROFILE
    start_time_point = std::chrono::high_resolution_clock::now();
#endif
}

Timer::~Timer() {
    stop("destructor");
}

void Timer::stop(std::string name) {
#ifdef PROFILE
    auto end_time_point = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_time_point).time_since_epoch().count();
    auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time_point).time_since_epoch().count();
    auto duration = end - start;
    std::cout << "Timer [" << name << "]: " << duration << " us" << std::endl;
#endif
}

void Timer::reset(std::string name){
#ifdef PROFILE
    stop(name);
    start_time_point = std::chrono::high_resolution_clock::now();
#endif        
}

#ifdef FUGAKU
static void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    if(datatype == MPI_INT){
        const int *in = (const int *)inbuf;
        int *inout = (int *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction operation\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_CHAR){
        const char *in = (const char *)inbuf;
        char *inout = (char *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction datatype\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_FLOAT){
        const float *in = (const float *)inbuf;
        float *inout = (float *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction datatype\n");
            exit(EXIT_FAILURE);
        }
    }
}
#endif

static inline int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

int is_odd(int x){
    return x & 1;
}


#ifdef DEBUG
static std::string vectostr(const std::vector<int>& v){
    std::string s = "";
    for(auto i : v){
        s += std::to_string(i) + ",";
    }
    return s;
}
#endif


// https://stackoverflow.com/questions/37637781/calculating-the-negabinary-representation-of-a-given-number-without-loops
static uint32_t binary_to_negabinary(int32_t bin) {
    if (bin > 0x55555555) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}


static int32_t negabinary_to_binary(uint32_t neg) {
    //const int32_t even = 0x2AAAAAAA, odd = 0x55555555;
    //if ((neg & even) > (neg & odd)) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask ^ neg) - mask;
}

/*
static int32_t smallest_negabinary(uint32_t nbits){
    return -2 * ((pow(2, (nbits / 2)*2) - 1) / 3);
}

static int32_t largest_negabinary(uint32_t nbits){
    int32_t tmp = ((pow(2, (nbits / 2)*2) - 1) / 3);
    if(!is_odd(nbits)){
        return tmp;
    }else{
        return tmp + pow(2, nbits - 1);
    }
}
*/

// Checks if a given number can be represented as a negabinary number with nbits bits
// TODO: Check this directly on binary representation by comparing with the largest number on nbits!
static inline int in_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline int is_power_of_two(int x){
    return (x != 0) && ((x & (x - 1)) == 0);
}

SwingCommon::SwingCommon(MPI_Comm comm, uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, Algo algo, uint num_ports, uint max_size): algo(algo), num_ports(num_ports), max_size(max_size), all_p2_dimensions(true), num_steps(0), peers_computed(false), reference_distances_computed(false){
    this->size = 1;
    for (uint i = 0; i < dimensions_num; i++) {
        this->dimensions[i] = dimensions[i];
        this->size *= dimensions[i];        
        this->num_steps_per_dim[i] = (int) ceil(log2(dimensions[i]));
        if(!is_power_of_two(dimensions[i])){
            this->all_p2_dimensions = false;
            this->dimensions_virtual[i] = pow(2, this->num_steps_per_dim[i] - 1);
        }else{
            this->dimensions_virtual[i] = this->dimensions[i];
        }
        this->num_steps += this->num_steps_per_dim[i];
    }
    if(this->num_steps > LIBSWING_MAX_STEPS){
        assert("Max steps limit must be increased and constants updated.");
    }
    int size;
    MPI_Comm_size(comm, &size);
    assert(size == this->size);
    MPI_Comm_rank(comm, &this->rank);
    this->dimensions_num = dimensions_num;
    this->coordinates = (int*) malloc(sizeof(int)*this->size*this->dimensions_num);
    this->coordinates_virtual = (int*) malloc(sizeof(int)*this->size*this->dimensions_num);
    memset(this->coordinates        , -1, sizeof(int)*this->size*dimensions_num);
    memset(this->coordinates_virtual, -1, sizeof(int)*this->size*dimensions_num);
    memset(this->remapping_per_port, 0, sizeof(uint*)*LIBSWING_MAX_SUPPORTED_PORTS);
}

SwingCommon::~SwingCommon(){
    if(this->peers_computed){
        for(size_t p = 0; p < this->num_ports; p++){
            free(this->peers_per_port[p]);
        }
    }
    if(this->reference_distances_computed){
        for(size_t d = 0; d < this->dimensions_num; d++){
            for(size_t i = 0; i < this->num_steps_per_dim[d]; i++){
                free(reference_valid_distances[d][i]);
            }
            free(reference_valid_distances[d]);
            free(num_valid_distances[d]);
        }
    }
    free(this->peers_per_port);
    free(this->coordinates_virtual);
    free(this->coordinates);

    for(size_t p = 0; p < this->num_ports; p++){
        if(this->remapping_per_port[p] != 0){
            free(this->remapping_per_port[p]); 
        }
    }
}

// Adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c
void SwingCommon::getCoordFromId(int id, bool virt, int* coord){
    int nnodes = 1;
    uint* dimensions = virt ? this->dimensions_virtual : this->dimensions;
    for(size_t i = 0; i < dimensions_num; i++){
        nnodes *= dimensions[i];
    }
    for (uint i = 0; i < dimensions_num; i++) {
        nnodes = nnodes / dimensions[i];
        coord[i] = id / nnodes;
        id = id % nnodes;
    }
}

// Adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c)
int SwingCommon::getIdFromCoord(int* coords, bool virt){
    int rank = 0;
    int multiplier = 1;
    int coord;
    uint* dimensions = virt ? this->dimensions_virtual : this->dimensions;
    for (int i = dimensions_num - 1; i >= 0; i--) {
        coord = coords[i];
        if (/*cart_ptr->topo.cart.periodic[i]*/ 1) {
            if (coord >= dimensions[i])
                coord = coord % dimensions[i]; 
            else if (coord < 0) {
                coord = coord % dimensions[i];
                if (coord)
                    coord = dimensions[i] + coord;
            }
        }
        rank += multiplier * coord;
        multiplier *= dimensions[i];
    }
    return rank;
}

// Cache coordinates
void SwingCommon::retrieve_coord_mapping(uint rank, bool virt, int* coord){
    int* coordinates = virt ? this->coordinates_virtual : this->coordinates;
    if(coordinates[rank*dimensions_num] == -1){
        getCoordFromId(rank, virt, &(coordinates[rank*dimensions_num]));
    }
    memcpy(coord, &(coordinates[rank*dimensions_num]), sizeof(uint)*dimensions_num);
}

void SwingCommon::compute_peers(int port, uint rank, bool virt, uint* peers){
    bool terminated_dimensions_bitmap[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int num_steps_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    uint8_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    memset(next_step_per_dim, 0, sizeof(uint8_t)*LIBSWING_MAX_SUPPORTED_DIMENSIONS);
    uint* dimensions = virt ? this->dimensions_virtual : this->dimensions;
    
    int num_steps = 0;
    for(size_t i = 0; i < dimensions_num; i++){
        num_steps_per_dim[i] = ceil(log2(dimensions[i]));
        num_steps += num_steps_per_dim[i];
    }

    // Compute default directions
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(rank, virt, coord);
    for(size_t i = 0; i < dimensions_num; i++){
        terminated_dimensions_bitmap[i] = false;            
    }
    
    int target_dim, relative_step, distance, last_dim = port - 1;
    uint terminated_dimensions = 0, o = 0;
    
    // Generate peers
    for(size_t i = 0; i < (uint) num_steps; ){            
        if(dimensions_num > 1){
            retrieve_coord_mapping(rank, virt, coord); // Regenerate rank coord
            o = 0;
            do{
                target_dim = (last_dim + 1 + o) % (dimensions_num);            
                o++;
            }while(terminated_dimensions_bitmap[target_dim]);
            relative_step = next_step_per_dim[target_dim];
            ++next_step_per_dim[target_dim];
            last_dim = target_dim;
        }else{
            target_dim = 0;
            relative_step = i;
            coord[0] = rank;
        }
        
        distance = rhos[relative_step];
        // Flip the sign for odd nodes
        if(is_odd(coord[target_dim])){distance *= -1;}
        // Mirrored collectives
        if((uint) port >= dimensions_num){distance *= -1;}

        if(relative_step < num_steps_per_dim[target_dim]){
            coord[target_dim] = mod((coord[target_dim] + distance), dimensions[target_dim]); // We need to use mod to avoid negative coordinates
            if(dimensions_num > 1){
                peers[i] = getIdFromCoord(coord, virt);
            }else{
                peers[i] = coord[0];
            }
    
            i += 1;
        }else{
            terminated_dimensions_bitmap[target_dim] = true;
            terminated_dimensions++;                
        }        

    }        
}


// Sends the data from nodes outside of the power-of-two boundary to nodes within the boundary.
// This is done one dimension at a time.
// Returns the new rank.
int SwingCommon::shrink_non_power_of_two(const void *sendbuf, void *recvbuf, int count, 
                                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, 
                                         char* tmpbuf, int* idle, int* rank_virtual){    
    int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    
    int dtsize, rank;
    MPI_Type_size(datatype, &dtsize);    
    MPI_Comm_rank(comm, &rank);    
    
    retrieve_coord_mapping(rank, false, coord);

    // We do the swapping between buffers to avoid additional
    // explicit memcopies.
    // Contains the data aggregated so far.
    void* aggregated_buf = (void*) sendbuf;
    // Contains the buffer into which the data must be received.
    void* recvbuf_real = recvbuf;
    // Contains the data to aggregate to recvbuf.
    void* aggregation_source = (void*) sendbuf;
    bool buf_copied = false;
    for(size_t i = 0; i < this->dimensions_num; i++){
        // This dimensions is not a power of two, shrink it
        if(!is_power_of_two(dimensions[i])){
            memcpy(coord_peer, coord, sizeof(uint)*this->dimensions_num);
            int extra = dimensions[i] - this->dimensions_virtual[i];
            if(coord[i] >= this->dimensions_virtual[i]){            
                coord_peer[i] = coord[i] - extra;
                int peer = getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Shr) Sending to %d\n", rank, peer);
                int res = MPI_Send(aggregated_buf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm);                
                if(res != MPI_SUCCESS){return res;}
                *idle = 1;
                break;
            }else if(coord[i] + extra >= this->dimensions_virtual[i]){
                coord_peer[i] = coord[i] + extra;
                int peer = getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Shr) Receiving from %d\n", rank, peer);
                int res = MPI_Recv(recvbuf_real, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, NULL);                
                if(res != MPI_SUCCESS){return res;}
                DPRINTF("[%d] (Shr) Recvbuf (%p) before aggr %d\n", rank, recvbuf, ((char*) recvbuf)[0]);
                MPI_Reduce_local(aggregation_source, recvbuf, count, datatype, op);
                DPRINTF("[%d] (Shr) Recvbuf (%p) after aggr %d\n", rank, recvbuf, ((char*) recvbuf)[0]);
                aggregated_buf = recvbuf;
                recvbuf_real = tmpbuf;
                aggregation_source = tmpbuf;
            }else{
                // I am neither sending or receiving, thus I copy my sendbuf to recvbuf
                // so that later I can aggregate directly rbuf with recvbuf
                // I only need to do it once.
                if(!buf_copied){
                    memcpy(recvbuf, sendbuf, count*dtsize);
                    aggregated_buf = recvbuf;
                    recvbuf_real = tmpbuf;
                    aggregation_source = tmpbuf;                    
                }
            }
            buf_copied = true; // Do not move it from here (ugly)
        }
    }
    *rank_virtual = getIdFromCoord(coord, true);
    return MPI_SUCCESS;
}

int SwingCommon::enlarge_non_power_of_two(void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm){
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int dtsize, rank;
    MPI_Type_size(datatype, &dtsize);    
    MPI_Comm_rank(comm, &rank);    
    
    retrieve_coord_mapping(rank, false, coord);    
    //for(size_t d = 0; d < dimensions_num; d++){
    for(int d = dimensions_num - 1; d >= 0; d--){
        // This dimensions was a non-power of two, enlarge it
        if(!is_power_of_two(dimensions[d])){
            memcpy(coord_peer, coord, sizeof(uint)*this->dimensions_num);
            int extra = dimensions[d] - this->dimensions_virtual[d];
            if(coord[d] >= (uint) this->dimensions_virtual[d]){                
                coord_peer[d] = coord[d] - extra;
                int peer = getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Enl) Receiving from %d\n", rank, peer);
                // I can overwrite the recvbuf and don't need to aggregate, since 
                // I was an extra node and did not participate to the actual allreduce
                int r = MPI_Recv(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, NULL);
                if(r != MPI_SUCCESS){return r;}
            }else if(coord[d] + extra >= (uint) this->dimensions_virtual[d]){
                coord_peer[d] = coord[d] + extra;
                int peer = getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Enl) Sending to %d\n", rank, peer);
                int r = MPI_Send(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm);                
                if(r != MPI_SUCCESS){return r;}
            }
        }
    }
    return MPI_SUCCESS;
}

// This works the following way:
// 1. Shrink the topology by sending ranks that are outside the power of two boundary, to ranks within the boundary
// 2. Run allreduce on a configuration where each dimension has a power of two size
// 3. Enlarge the topology by sending data to ranks outside the boundary.
//
// Steps 1. and 3. are done once on the entire buffer.
// Step 2. is done for each chunk and each port.
int SwingCommon::MPI_Allreduce_lat_optimal(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    
    
    char* tmpbuf = (char*) malloc(count*dtsize); // Temporary buffer (to avoid overwriting sendbuf)
    // To avoid memcpying, the first recv+aggregation and the subsequent ones use different buffers
    // (see MPI_Allreduce_lat_optimal_swing_sendrecv). This variable keeps track of that.
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(this->rank, false, coord);

    // Compute data offset per port
    BlockInfo buf_info[LIBSWING_MAX_SUPPORTED_PORTS];
    uint partition_size = count / this->num_ports;
    uint remaining = count % this->num_ports;
    uint count_so_far = 0;
    for(size_t p = 0; p < this->num_ports; p++){
        size_t count_port = partition_size + (p < remaining ? 1 : 0);
        size_t offset_port = count_so_far*dtsize;
        count_so_far += count_port;
        buf_info[p].count = count_port;
        buf_info[p].offset = offset_port;
    }

    int idle = 0;
    int rank_virtual = rank;
    int res;
    res = shrink_non_power_of_two(sendbuf, recvbuf, count, datatype, op, comm, tmpbuf, &idle, &rank_virtual);
    if(res != MPI_SUCCESS){return res;}
    int size_virtual = 1, num_steps_virtual = 0;
    for(size_t i = 0; i < this->dimensions_num; i++){
        size_virtual *= this->dimensions_virtual[i];
        num_steps_virtual += ceil(log2(this->dimensions_virtual[i]));
    }
    DPRINTF("[%d] Virtual steps: %d Virtual dimensions (%d, %d, %d)\n", rank, num_steps_virtual, this->dimensions_virtual[0], this->dimensions_virtual[1], this->dimensions_virtual[2]);

    if(!idle){
        // If the topology has some dimension which was not a power of 2,
        // then we can receive in tmpbuf and aggregate into recvbuf.
        // Compute the number of steps on the shrunk topology

        // Computes the peer sequence on each port.
        DPRINTF("[%d] Computing peers\n", rank);  
        uint** peers_per_port = (uint**) malloc(sizeof(uint*)*this->num_ports);
        for(size_t p = 0; p < this->num_ports; p++){
            peers_per_port[p] = (uint*) malloc(sizeof(uint)*num_steps_virtual);
            compute_peers(p, rank_virtual, true, peers_per_port[p]);
        }
        DPRINTF("[%d] Peers computed\n", rank);

        // Do the step-by-step communication on the shrunk topology.
        MPI_Request requests_s[LIBSWING_MAX_SUPPORTED_PORTS];
        MPI_Request requests_r[LIBSWING_MAX_SUPPORTED_PORTS];
        const void *sendbuf_real, *aggbuff_a;
        void *aggbuff_b, *recvbuf_real;
         
        for(size_t step = 0; step < (uint) num_steps_virtual; step++){     
            // Isend/Irecv requests
            memset(requests_s, 0, sizeof(requests_s));                
            memset(requests_r, 0, sizeof(requests_r));
            
            // Schedule all the send and recv
            for(size_t p = 0; p < this->num_ports; p++){
                // Get the peer
                int virtual_peer = peers_per_port[p][step]; 
                int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
                retrieve_coord_mapping(virtual_peer, true, coord_peer);
                int peer = getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] Sending to %d\n", rank, peer);
                // Get the buffers
                // If all dimensions are powers of two, the ranks
                // did not send, recv, and aggregate anything and thus
                // the trick with the buffers swap has not been done.
                // We need to do it here (once per port and chunk -- i.e.,
                // always at the first step)
                if(all_p2_dimensions && step == 0){
                    sendbuf_real = ((char*) sendbuf) + buf_info[p].offset;
                    recvbuf_real = ((char*) recvbuf) + buf_info[p].offset;
                }else{
                    sendbuf_real = ((char*) recvbuf) + buf_info[p].offset;
                    recvbuf_real = ((char*) tmpbuf) + buf_info[p].offset;
                }
                DPRINTF("[%d] Count: %d\n", rank, buf_info[p].count);
                // Schedule the sends and recvs
                res = MPI_Isend(sendbuf_real, buf_info[p].count, datatype, peer, TAG_SWING_ALLREDUCE + p, comm, &(requests_s[p]));
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on isend\n", rank); return res;}

                res = MPI_Irecv(recvbuf_real, buf_info[p].count, datatype, peer, TAG_SWING_ALLREDUCE + p, comm, &(requests_r[p]));
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on irecv\n", rank); return res;}                       
            }  
            DPRINTF("[%d] Send/Recv issued, going to wait\n", rank);
            // Wait and aggregate
            for(size_t p = 0; p < this->num_ports; p++){
                int terminated_port;
                res = MPI_Waitany(this->num_ports, requests_r, &terminated_port, MPI_STATUS_IGNORE);
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on waitany\n", rank); return res;}     
                // Now wait also for the send. If we wait for all the sends only at the end,
                // we might overwrite recvbuf while still being used by a send. // TODO
                res = MPI_Wait(&(requests_s[terminated_port]), MPI_STATUS_IGNORE);
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on wait\n", rank); return res;}     
                DPRINTF("[%d] Irecv/Isend on port %d completed\n", rank, terminated_port);
                
                // See comment above about buffers trick
                if(all_p2_dimensions && step == 0){
                    aggbuff_a = ((char*) sendbuf) + buf_info[terminated_port].offset;
                }else{
                    aggbuff_a = ((char*) tmpbuf) + buf_info[terminated_port].offset;
                }                    
                aggbuff_b = ((char*) recvbuf) + buf_info[terminated_port].offset;   
                
                //DPRINTF("[%d] sendbuf %p recvbuf %p sendbuf_real %p recvbuf_real %p tmpbuf %p aggbuff_a %p aggbuff_b %p\n", rank, sendbuf, recvbuf, sendbuf_real, recvbuf_real, tmpbuf, aggbuff_a, aggbuff_b);


                DPRINTF("[%d] (m) Recvbuf (%p) before aggr %d\n", rank, aggbuff_b, ((char*) aggbuff_b)[0]);
                res = MPI_Reduce_local(aggbuff_a, aggbuff_b, buf_info[terminated_port].count, datatype, op); 
                DPRINTF("[%d] (m) Recvbuf (%p) after aggr %d\n", rank, aggbuff_b, ((char*) aggbuff_b)[0]);
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on reduce_local\n", rank); return res;}
            }
            // Wait for all the send to finish
            DPRINTF("[%d] All sends and receive completed\n", rank);
        }
    }

    DPRINTF("[%d] Propagating data to extra nodes\n", rank);
    res = enlarge_non_power_of_two(recvbuf, count, datatype, comm);
    if(res != MPI_SUCCESS){return res;}
    DPRINTF("[%d] Data propagated\n", rank);
    
    free(tmpbuf);
    return res;
}

/*
static inline int check_last_n_bits_equal(uint32_t a, uint32_t b, uint32_t n){
    uint32_t mask = (1 << n) - 1;
    return (a & mask) == (b & mask);
}
*/

/*
static int reverse_bits(int block, int num_steps){
    int res = 0;
    for(int i = 0; i < num_steps; i++){
        if ((block & (1 << i)))
            res |= 1 << ((num_steps - 1) - i);
    }
    return res;
}
*/

#ifdef __GNUC__
#define ctz(x) __builtin_ctz(x)
#define clz(x) __builtin_clz(x)
#else
// Counts leading zeros. x MUST be different from zero.
// https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
static inline int ctz(uint32_t x){
    unsigned int v;      // 32-bit word input to count zero bits on right
    unsigned int c = 32; // c will be the number of zero bits on the right
    v &= -signed(v);
    if (v) c--;
    if (v & 0x0000FFFF) c -= 16;
    if (v & 0x00FF00FF) c -= 8;
    if (v & 0x0F0F0F0F) c -= 4;
    if (v & 0x33333333) c -= 2;
    if (v & 0x55555555) c -= 1;
    return c;
}

// https://stackoverflow.com/questions/23856596/how-to-count-leading-zeros-in-a-32-bit-unsigned-integer
static inline int clz(uint32_t x){
    static const char debruijn32[32] = {
        0, 31, 9, 30, 3, 8, 13, 29, 2, 5, 7, 21, 12, 24, 28, 19,
        1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18
    };
    x |= x>>1;
    x |= x>>2;
    x |= x>>4;
    x |= x>>8;
    x |= x>>16;
    x++;
    return debruijn32[x*0x076be629>>27];
}
#endif

static inline int get_first_step(uint32_t block_distance){
    // If I have something like 0000111 or 1111000, the first step is 2.
    // Find the position where we have the first switch from bit 1 to 0 or viceversa.
    if(block_distance & 0x1){
        return ctz(~block_distance) - 1;
    }else{
        return ctz(block_distance) - 1;
    }
}

/*
static inline int get_last_step(uint32_t block_distance){
    // The last step is the position of the most significant bit.
    return 32 - clz(block_distance) - 1;
}
*/

int SwingCommon::swing_coll_step_b(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                             
                                   MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                   CollType coll_type, char*** bitmap_send, char*** bitmap_recv){    
    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    memset(requests_s, 0, sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    memset(requests_r, 0, sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    BlockInfo* req_idx_to_block_idx = (BlockInfo*) malloc(sizeof(BlockInfo)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    uint num_requests_s = 0, num_requests_r = 0;    
    assert(sendtype == recvtype);
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    int tag, res = MPI_SUCCESS;
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);            
    for(size_t port = 0; port < this->num_ports; port++){
        uint peer = (this->peers_per_port)[port][block_step];
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", this->rank, step, port, this->num_ports, peer);                

        if(coll_type == SWING_REDUCE_SCATTER){
            tag = TAG_SWING_REDUCESCATTER + port;
        }else{
            tag = TAG_SWING_ALLGATHER + port;
        }

        // Sendrecv + aggregate
        // Search for the blocks that must be sent.
        for(size_t i = 0; i < (uint) size; i++){
            int send_block, recv_block;
            if(coll_type == SWING_REDUCE_SCATTER){
                send_block = bitmap_send[port][block_step][i];
                recv_block = bitmap_recv[port][block_step][i];
            }else{                
                send_block = bitmap_recv[port][block_step][i];
                recv_block = bitmap_send[port][block_step][i];
            }
            
            size_t block_count = blocks_info[port][i].count;
            size_t block_offset = blocks_info[port][i].offset;

            //DPRINTF("[%d] Block %d (send %d recv %d)\n", rank, i, send_block, recv_block);
            if(send_block){              
                DPRINTF("[%d] Sending block %d to %d at step %d (coll %d)\n", rank, i, peer, step, coll_type);
                res = MPI_Isend(((char*) buf) + block_offset, block_count, sendtype, peer, tag, comm, &(requests_s[num_requests_s]));
                if(res != MPI_SUCCESS){return res;}
                ++(num_requests_s);
            }
            if(recv_block){
                DPRINTF("[%d] Receiving block %d from %d at step %d (coll %d)\n", rank, i, peer, step, coll_type);
                res = MPI_Irecv(((char*) rbuf) + block_offset, block_count, recvtype, peer, tag, comm, &(requests_r[num_requests_r]));
                if(res != MPI_SUCCESS){return res;}
                req_idx_to_block_idx[(num_requests_r)].offset = block_offset;
                req_idx_to_block_idx[(num_requests_r)].count = block_count;
                ++(num_requests_r);
            }
        }
    }
    DPRINTF("[%d] Issued %d send requests and %d receive requests\n", rank, num_requests_s, num_requests_r);

    // Wait for all the recvs to be over
    if(coll_type == SWING_REDUCE_SCATTER){
//#define ALWAYS_WAITALL
#ifdef ALWAYS_WAITALL     
      res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
#endif
      
        int index;
        for(size_t i = 0; i < num_requests_r; i++){
#ifndef ALWAYS_WAITALL	  
            res = MPI_Waitany(num_requests_r, requests_r, &index, MPI_STATUS_IGNORE);	    
            if(res != MPI_SUCCESS){return res;}
#else
	        index = i;
#endif	    
            void* rbuf_block = (void*) (((char*) rbuf) + req_idx_to_block_idx[index].offset);
            void* buf_block = (void*) (((char*) buf) + req_idx_to_block_idx[index].offset);  
            DPRINTF("[%d] Aggregating from %p to %p (i %d index %d offset %d count %d)\n", this->rank, rbuf_block, buf_block, i, index, req_idx_to_block_idx[index].offset, req_idx_to_block_idx[index].count);
            MPI_Reduce_local(rbuf_block, buf_block, req_idx_to_block_idx[index].count, sendtype, op); 
        }
    }else{
        res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
        if(res != MPI_SUCCESS){return res;}            
    }


    // Wait for all the sends to be over    
    res = MPI_Waitall(num_requests_s, requests_s, MPI_STATUSES_IGNORE);

    free(requests_s);
    free(requests_r);
    free(req_idx_to_block_idx);
    return res;
}

int SwingCommon::swing_coll_step_cont(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
                                MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                CollType coll_type, char*** bitmap_send, char*** bitmap_recv){
    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    memset(requests_s, 0, sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    memset(requests_r, 0, sizeof(MPI_Request)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    BlockInfo* req_idx_to_block_idx = (BlockInfo*) malloc(sizeof(BlockInfo)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);
    uint num_requests_s = 0, num_requests_r = 0;    
    assert(sendtype == recvtype);
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    int tag, res = MPI_SUCCESS;
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(this->num_steps - step - 1); 

    for(size_t port = 0; port < this->num_ports; port++){
        uint peer = peers_per_port[port][block_step];
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", this->rank, step, port, this->num_ports, peer);                

        if(coll_type == SWING_REDUCE_SCATTER){
            tag = TAG_SWING_REDUCESCATTER + port;
        }else{
            tag = TAG_SWING_ALLGATHER + port;
        }

        // Sendrecv + aggregate
        // Search for the blocks that must be sent.
        bool start_found_s = false, start_found_r = false;
        size_t offset_s, offset_r, count_s = 0, count_r = 0;
        for(size_t i = 0; i < (uint) this->size; i++){
            int send_block, recv_block;
            if(coll_type == SWING_REDUCE_SCATTER){
                send_block = bitmap_send[port][block_step][i];
                recv_block = bitmap_recv[port][block_step][i];
            }else{                
                send_block = bitmap_recv[port][block_step][i];
                recv_block = bitmap_send[port][block_step][i];
            }
            
            size_t block_count = blocks_info[port][i].count;
            size_t block_offset = blocks_info[port][i].offset;

            if(send_block){
                if(!start_found_s){
                    start_found_s = true;
                    offset_s = block_offset;
                }
                count_s += block_count;
            }
            if(start_found_s && (!send_block || i == (size_t) this->size - 1)){ // The train of consecutive blocks is over
                DPRINTF("[%d] Sending offset %d count %d at step %d (coll %d)\n", this->rank, offset_s, count_s, step, coll_type);            
                res = MPI_Isend(((char*) buf) + offset_s, count_s, sendtype, peer, tag, comm, &(requests_s[num_requests_s]));
                if(res != MPI_SUCCESS){return res;}
                (num_requests_s)++;

                // In some rare cases (e.g., for 10 nodes), I might have not one but two consecutive trains of blocks
                // Reset everything in case we need to send another train of blocks //TODO: Fix it for this case so to have one single consecutive train of blocks
                count_s = 0;
                offset_s = 0;
                start_found_s = false;
            }

            if(recv_block){
                if(!start_found_r){
                    start_found_r = true;
                    offset_r = block_offset;
                }
                count_r += block_count;
            }            
            if(start_found_r && (!recv_block || i == (size_t) this->size - 1)){ // The train of consecutive blocks is over
                DPRINTF("[%d] Receiving offset %d count %d at step %d (coll %d)\n", this->rank, offset_r, count_r, step, coll_type);
                req_idx_to_block_idx[num_requests_r].offset = offset_r;
                req_idx_to_block_idx[num_requests_r].count = count_r;
                res = MPI_Irecv(((char*) rbuf) + offset_r, count_r, recvtype, peer, tag, comm, &(requests_r[num_requests_r]));
                if(res != MPI_SUCCESS){return res;}
                (num_requests_r)++;
                // In some rare cases (e.g., for 10 nodes), I might have not one but two consecutive trains of blocks
                // Reset everything in case we need to send another train of blocks
                count_r = 0;
                offset_r = 0;
                start_found_r = false;
            }
        }
    }
    DPRINTF("[%d] Issued %d send requests and %d receive requests\n", this->rank, num_requests_s, num_requests_r);

    if(coll_type == SWING_NULL){return MPI_SUCCESS;}
    // Wait for all the recvs to be over
    if(coll_type == SWING_REDUCE_SCATTER){
//#define ALWAYS_WAITALL
#ifdef ALWAYS_WAITALL     
      res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
#endif
      
        int index;
        for(size_t i = 0; i < num_requests_r; i++){
#ifndef ALWAYS_WAITALL	  
            res = MPI_Waitany(num_requests_r, requests_r, &index, MPI_STATUS_IGNORE);	    
            if(res != MPI_SUCCESS){return res;}
#else
	        index = i;
#endif	    
            void* rbuf_block = (void*) (((char*) rbuf) + req_idx_to_block_idx[index].offset);
            void* buf_block = (void*) (((char*) buf) + req_idx_to_block_idx[index].offset);  
            DPRINTF("[%d] Aggregating from %p to %p (i %d index %d offset %d count %d)\n", this->rank, rbuf_block, buf_block, i, index, req_idx_to_block_idx[index].offset, req_idx_to_block_idx[index].count);
            MPI_Reduce_local(rbuf_block, buf_block, req_idx_to_block_idx[index].count, sendtype, op); 
        }
    }else{
        res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
        if(res != MPI_SUCCESS){return res;}            
    }


    // Wait for all the sends to be over    
    res = MPI_Waitall(num_requests_s, requests_s, MPI_STATUSES_IGNORE);

    free(requests_s);
    free(requests_r);
    free(req_idx_to_block_idx);
    return res;
}

int SwingCommon::swing_coll_step(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
                           MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type, char*** bitmap_send, char*** bitmap_recv){
    if(coll_type == SWING_NULL){return MPI_SUCCESS;}

    if(algo == ALGO_SWING_B){
        return swing_coll_step_b(buf, rbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type, bitmap_send, bitmap_recv);
    }else if(algo == ALGO_SWING_B_CONT || algo == ALGO_SWING_B_COALESCE){
        return swing_coll_step_cont(buf, rbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type, bitmap_send, bitmap_recv);
    }else{
        assert("Unknown algo" == 0);
        return MPI_ERR_OTHER;
    }
}

#ifdef FUGAKU
int SwingCommon::swing_coll_step_utofu(size_t port, swing_utofu_comm_descriptor* utofu_descriptor, void *buf, void* rbuf, BlockInfo** blocks_info, size_t step, 
                                       MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                       CollType coll_type, char*** bitmap_send, char*** bitmap_recv){
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(this->num_steps - step - 1); 
    size_t offsets_s[LIBSWING_MAX_SUPPORTED_PORTS], counts_s[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t offsets_r[LIBSWING_MAX_SUPPORTED_PORTS], counts_r[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(offsets_s, 0, sizeof(offsets_s));
    memset(counts_s, 0, sizeof(counts_s));
    memset(offsets_r, 0, sizeof(offsets_r));
    memset(counts_r, 0, sizeof(counts_r));
    BlockInfo* req_idx_to_block_idx = (BlockInfo*) malloc(sizeof(BlockInfo)*this->size*LIBSWING_MAX_SUPPORTED_PORTS);

    // Sendrecv + aggregate
    // Search for the blocks that must be sent.
    bool start_found_s = false, start_found_r = false, sent_on_port = false, recvd_from_port = false;
    size_t offset_s, offset_r, count_s = 0, count_r = 0;
    for(size_t i = 0; i < (uint) this->size; i++){
        int send_block, recv_block;
        if(coll_type == SWING_REDUCE_SCATTER){
            send_block = bitmap_send[port][block_step][i];
            recv_block = bitmap_recv[port][block_step][i];
        }else{                
            send_block = bitmap_recv[port][block_step][i];
            recv_block = bitmap_send[port][block_step][i];
        }
        
        size_t block_count = blocks_info[port][i].count;
        size_t block_offset = blocks_info[port][i].offset;

        if(send_block){
            if(!start_found_s){
                start_found_s = true;
                offset_s = block_offset;
            }
            count_s += block_count;
        }
        if(start_found_s && (!send_block || i == (size_t) this->size - 1)){ // The train of consecutive blocks is over
            if(sent_on_port){
                fprintf(stderr, "With uTofu we support at most one send/recv per port\n");
                exit(-1);
            }
            sent_on_port = true;
            DPRINTF("[%d] Port %d Sending offset %d count %d at step %d (coll %d)\n", this->rank, port, offset_s, count_s, step, coll_type);            
            offsets_s[port] = offset_s;
            counts_s[port] = count_s;

            // In some rare cases (e.g., for 10 nodes), I might have not one but two consecutive trains of blocks
            // Reset everything in case we need to send another train of blocks //TODO: Fix it for this case so to have one single consecutive train of blocks
            count_s = 0;
            offset_s = 0;
            start_found_s = false;
        }

        if(recv_block){
            if(!start_found_r){
                start_found_r = true;
                offset_r = block_offset;
            }
            count_r += block_count;
        }            
        if(start_found_r && (!recv_block || i == (size_t) this->size - 1)){ // The train of consecutive blocks is over
            if(recvd_from_port){
                fprintf(stderr, "With uTofu we support at most one send/recv per port\n");
                exit(-1);
            }
            recvd_from_port = true;
            DPRINTF("[%d] Port %d Receiving offset %d count %d at step %d (coll %d)\n", this->rank, port, offset_r, count_r, step, coll_type);
            req_idx_to_block_idx[port].offset = offset_r;
            req_idx_to_block_idx[port].count = count_r;
            offsets_r[port] = offset_r;
            counts_r[port] = count_r;
            // In some rare cases (e.g., for 10 nodes), I might have not one but two consecutive trains of blocks
            // Reset everything in case we need to send another train of blocks
            count_r = 0;
            offset_r = 0;
            start_found_r = false;
        }
    }

    int dtsize;
    MPI_Type_size(sendtype, &dtsize);  
    size_t max_count = floor(MAX_PUTGET_SIZE / dtsize);
    char issued_sends = 0, issued_recvs = 0;
    size_t offset = offsets_s[port];        
    size_t count = 0;     
    assert(counts_s[port] == counts_r[port]); // TODO: Not sure if everything works when this is not true (e.g., chunking, offset_rs_r, etc.)
    size_t utofu_offset_r = offset;

    
    // In reduce-scatter, if some rank are faster than others,
    // one executing a later step might write the data in the destination
    // rank earlier than a rank executing a previous step.
    // As a result, the first put would be lost. To avoid that, 
    // instead of writing in the actual offset, we force the offsets
    // to be different, since anyway the data must be moved from the receive
    // buffer when aggregating it.
    // 
    // To do that, the intuition is the following:
    // - During the allgather, the rank receives always in different parts of the array
    // - Thus, I should do a put in the offset where he receives in the corresponding step of the allgather
    // - So, I should do it at the offset from where he sends in the step of the reduce_scatter
    // - i.e., he sends where I receive, so I must do the put at the same offset where I receive
    //
    // This is why we do in the following
    //      -  char* rbuf_block = ((char*) rbuf) + offsets_s[port]; 
    if(coll_type == SWING_REDUCE_SCATTER){
        utofu_offset_r = req_idx_to_block_idx[port].offset;
    }

    double starttt = MPI_Wtime();
    // We first enqueue all the send. Then, we receive and aggregate
    // Aggregation and reception of next block should be overlapped
    // Issue isends for all the blocks
    if(counts_s[port]){ 
        size_t remaining = counts_s[port];
        size_t bytes_to_send = 0;
        // Split the transmission into chunks < MAX_PUTGET_SIZE
        while(remaining){
            assert(issued_sends <= MAX_NUM_CHUNKS);
            count = remaining < max_count ? remaining : max_count;
            bytes_to_send = count*dtsize;
            swing_utofu_isend(utofu_descriptor, port, block_step, issued_sends, offset, utofu_offset_r, bytes_to_send, coll_type == SWING_ALLGATHER); 
            offset += bytes_to_send;
            utofu_offset_r += bytes_to_send;
            remaining -= count;
            ++issued_sends;
        }
    }
    // Receive and aggregate
    offset = 0;
    if(counts_r[port]){ 
        char* buf_block = ((char*) buf) + req_idx_to_block_idx[port].offset;
        char* rbuf_block = ((char*) rbuf) + offsets_s[port]; 
        size_t remaining = counts_r[port];
        size_t bytes_to_recv = 0;
        // Split the transmission into chunks < MAX_PUTGET_SIZE
        while(remaining){
            assert(issued_recvs <= MAX_NUM_CHUNKS);
            count = remaining < max_count ? remaining : max_count;
            bytes_to_recv = count*dtsize;
            swing_utofu_wait_recv(utofu_descriptor, port);

            if(coll_type == SWING_REDUCE_SCATTER){
                reduce_local(rbuf_block + offset, buf_block + offset, count, sendtype, op);
            }

            offset += bytes_to_recv;
            remaining -= count;
            ++issued_recvs;
        }            
    }
    //printf("[%d] Issued %d sends and %d recvs (%d and %d elems) in %lf us\n", omp_get_thread_num(), issued_sends, issued_recvs, counts_s[port], counts_r[port], (MPI_Wtime() - starttt)*1000000.0);
    // Wait for send completion
    if(counts_s[port]){
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
    }
    free(req_idx_to_block_idx);
    return MPI_SUCCESS;
}
#else
int SwingCommon::swing_coll_step_utofu(size_t port, swing_utofu_comm_descriptor* utofu_descriptor, void *buf, void* rbuf, BlockInfo** blocks_info, size_t step, 
                                       MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                       CollType coll_type, char*** bitmap_send, char*** bitmap_recv){
    fprintf(stderr, "uTofu can only be used on Fugaku.\n");
    exit(-1);
}
#endif


// Returns an array of valid distances (considering a plain collective on an even node)
void SwingCommon::compute_valid_distances(uint d, int step){
    int max_steps = this->num_steps_per_dim[d];
    int size = this->dimensions[d];
    int* valid_distances = this->reference_valid_distances[d][step];
    size_t max_num_blocks = pow(2, (max_steps - step - 1));
    this->num_valid_distances[d][step] = 0;
    // Generate all binary strings with a 0 in position step+1 and a 1 in all the bits in position j (j <= step)
    for(size_t i = 0; i < max_num_blocks; i++){
        if(!(i & 0x1)){ // Only if it has a 0 as LSB (we generate in one shot bot the string with LSB=0 and that with LSB=1)         
            int nbin[2]; // At position 0, we have the numbers with 0 as LSB, at position 1, the numbers with 1 as LSB
            nbin[1] = (i << (step + 1)) | ((1 << (step + 1)) - 1); // LSB=1
            nbin[0] = ~nbin[1]          & ((1 <<  max_steps) - 1); // LSB=0
            
            int distance[2] = {negabinary_to_binary(nbin[0]),  // At position 0, we have the numbers with 0 as LSB, 
                               negabinary_to_binary(nbin[1])}; // at position 1, the numbers with 1 as LSB.
            char distance_valid_tmp[2] = {1, 1};
            if(!is_power_of_two(size)){ // If size is not a power of two, I need to check for alternative ways of reaching a given distance, to avoid reaching it twice
                for(size_t q = 0; q < 2; q++){                                     
                    // We know that the distance, when size is not a power of two, can be in the range (-2*size, 2*size)
                    // Thus, there are 4 different negabinary numbers that, modulo size, could give the same distance.
                    // We need to check all the other three alternatives.
                    int alternatives[3];
                    if(distance[q] > size && distance[q] < 2*size){
                        alternatives[0] = distance[q] - size;
                        alternatives[1] = distance[q] - 2*size;
                        alternatives[2] = distance[q] - 3*size;
                    }else if(distance[q] > 0 && distance[q] < size){
                        alternatives[0] = distance[q] + size;
                        alternatives[1] = distance[q] - size;
                        alternatives[2] = distance[q] - 2*size;
                    }else if(distance[q] < 0 && distance[q] > -size){
                        alternatives[0] = distance[q] + 2*size;
                        alternatives[1] = distance[q] + size;
                        alternatives[2] = distance[q] - size;
                    }else if(distance[q] < -size && distance[q] > -2*size){
                        alternatives[0] = distance[q] + 3*size;
                        alternatives[1] = distance[q] + 2*size;
                        alternatives[2] = distance[q] + size;
                    }else{
                         // distance[q] % size == 0
#ifdef DEBUG
                        assert(distance[q] >= -2*size && distance[q] <= 2*size);
                        assert(distance[q] % size == 0);
#endif                        
                        distance_valid_tmp[q] = 0;
                        continue;
                    }

                    for(size_t k = 0; k < 3; k++){
                        if(in_range(alternatives[k], max_steps)){ // First of all, check if the corresponding negabinary number can be represented with the given number of bits
                            int nbin_alt = binary_to_negabinary(alternatives[k]);
                            int first_step_alt = get_first_step(nbin_alt);
                            if(first_step_alt != step && first_step_alt > get_first_step(nbin[q])){
                                distance_valid_tmp[q] = 0;
                                break;
                            }
                        }
                    }
                }
            }else{
                // I still need to check that distance[q] % size != 0
                for(size_t q = 0; q < 2; q++){ 
                    if(mod(distance[q], size) == 0){
                        distance_valid_tmp[q] = 0;
                    }
                }
            }

            if(distance_valid_tmp[0]){
                this->reference_valid_distances[d][step][this->num_valid_distances[d][step]] = -distance[0]; // Subtract form r all the strings with LSB=0
                this->num_valid_distances[d][step]++;
            }

            if(distance_valid_tmp[1]){
                this->reference_valid_distances[d][step][this->num_valid_distances[d][step]] = +distance[1]; // Sum to r all the strings with LSB=1
                this->num_valid_distances[d][step]++;
            }            
        }
    }
}

int SwingCommon::get_distance_sign(size_t rank, size_t port){
    int multiplier = 1;
    if(is_odd(rank)){ // Invert sign if odd rank
        multiplier *= -1;
    }
    if(port >= this->dimensions_num){ // Invert sign if mirrored collective
        multiplier *= -1;     
    }
    return multiplier;
}

void SwingCommon::get_blocks_bitmaps_multid(size_t* next_step_per_dim, size_t* current_d, size_t step,
                                            size_t port, int* coord_peer, char** bitmap_send, 
                                            char** bitmap_recv, char* bitmap_send_merged, char* bitmap_recv_merged, 
                                            int* coord_mine){    

    /*************************/
    /* Distances calculation */
    /*************************/
    if(!reference_distances_computed){
        // For each dimension, and each step (relative to that dimension), 
        // we compute all the valid distances at that step (considering a plain collective on an even node)
        for(size_t d = 0; d < dimensions_num; d++){
            this->num_valid_distances[d] = (uint*) malloc(sizeof(uint)*this->num_steps);
            this->reference_valid_distances[d] = (int**) malloc(sizeof(int*)*this->num_steps);
            memset(this->reference_valid_distances[d], 0, sizeof(int*)*this->num_steps);            
            for(size_t sk = 0; sk < (uint) this->num_steps; sk++){
                this->reference_valid_distances[d][sk] = (int*) malloc(sizeof(int)*this->size);
                compute_valid_distances(d, sk);
            }
        }
        reference_distances_computed = true;
    }

    // Compute the bitmap for each dimension
    for(size_t k = 0; k < this->dimensions_num; k++){
        size_t d = (k + current_d[port]) % this->dimensions_num;

        memset(bitmap_send[d], 0, sizeof(char)*this->dimensions[d]);
        memset(bitmap_recv[d], 0, sizeof(char)*this->dimensions[d]);

        // To deal with the case where I don't move in that dimension.
        // e.g. if I send on the row dimension to a node with ID 0001,
        // (i.e., I send on the first step on the row, and never on the column (00))
        if(k){
            bitmap_send[d][coord_mine[d]] = 1;
            bitmap_recv[d][coord_peer[d]] = 1;
        }
                        
        // We skip dimension d if we are done with that dimension.
        if(next_step_per_dim[d] >= this->num_steps_per_dim[d]){
            continue;
        }     
                    
        size_t last_step;
        if(k == 0){
            last_step = next_step_per_dim[d] + 1;
        }else{
            last_step = this->num_steps_per_dim[d];
        }

        //DPRINTF("[%d] step %d port %d target dim %d rel step %d last step %d\n", this->rank, step, p, d, rel_step, last_step);                
        int sign_mine = get_distance_sign(coord_mine[d], port);
        int sign_peer = get_distance_sign(coord_peer[d], port);
        for(size_t sk = next_step_per_dim[d]; sk < last_step; sk++){
            // Compute bitmaps for coord_mine and coord_peer. 
            // I just need to adjust the distance according to the port 
            // and oddness of the node (reflected in the sign variables).
            for(size_t i = 0; i < (uint) this->num_valid_distances[d][sk]; i++){
                int distance = this->reference_valid_distances[d][sk][i];
                int distance_mine = sign_mine*distance;
                int distance_peer = sign_peer*distance;
                bitmap_send[d][mod(coord_mine[d] + distance_mine, dimensions[d])] = 1;
                bitmap_recv[d][mod(coord_peer[d] + distance_peer, dimensions[d])] = 1;
            }
        }
    }

    // Combine the per-dimension bitmaps
    int coord_block[LIBSWING_MAX_SUPPORTED_DIMENSIONS];    
    for(size_t i = 0; i < (uint) this->size; i++){
        retrieve_coord_mapping(i, false, coord_block);
        //DPRINTF("[%d] Step %d Peer (%d, %d) Block %d coord (%d,%d) bitmap send (%d,%d)\n", this->rank, step, coord_peer[0], coord_peer[1], i, coord_block[0], coord_block[1], bitmap_send[0][coord_block[0]], bitmap_send[1][coord_block[1]]);
        char set_s = 1, set_r = 1;
        for(size_t d = 0; d < dimensions_num; d++){
            set_s &= bitmap_send[d][coord_block[d]];
            set_r &= bitmap_recv[d][coord_block[d]];
        }

        if(bitmap_send_merged){ // I can decide to retrieve only the send or only the recv bitmap
            bitmap_send_merged[i] = set_s;
        }

        if(bitmap_recv_merged){ // I can decide to retrieve only the send or only the recv bitmap
            bitmap_recv_merged[i] = set_r;
        }
    }

    // Move to the next dimension for the next step
    if(step < (size_t) this->num_steps - 1){
        size_t d = current_d[port];              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < this->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d[port] = (current_d[port] + 1) % dimensions_num;
            d = current_d[port];
        }while(next_step_per_dim[d] >= this->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
}


// Same as the one above, but we compute next_step_per_dim and current_d on the fly so that we do not need to do bookeping
void SwingCommon::get_blocks_bitmaps_multid(size_t step, size_t port, int* coord_peer, 
                                             char* bitmap_send_merged, char* bitmap_recv_merged, 
                                             int* coord_mine){
    size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    size_t current_d[LIBSWING_MAX_SUPPORTED_PORTS];
    char* bitmap_send[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    char* bitmap_recv[LIBSWING_MAX_SUPPORTED_DIMENSIONS];    
    for(size_t d = 0; d < dimensions_num; d++){
        bitmap_send[d] = (char*) malloc(sizeof(char)*dimensions[d]);
        bitmap_recv[d] = (char*) malloc(sizeof(char)*dimensions[d]);   
    }

    current_d[port] = port % dimensions_num;
    memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);

    for(size_t i = 0; i < step; i++){
        // Move to the next dimension for the next step
        size_t d = current_d[port];              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < this->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d[port] = (current_d[port] + 1) % dimensions_num;
            d = current_d[port];
        }while(next_step_per_dim[d] >= this->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
    //DPRINTF("[%d] Going to do step %d on dim %d rel %d\n", info->rank, step, current_d[port], next_step_per_dim[current_d[port]]);    
    get_blocks_bitmaps_multid(next_step_per_dim, current_d, step, port, coord_peer, bitmap_send, bitmap_recv, bitmap_send_merged, bitmap_recv_merged, coord_mine);
    for(size_t d = 0; d < dimensions_num; d++){
        free(bitmap_send[d]);
        free(bitmap_recv[d]);   
    }
}


#ifdef DEBUG
/*
static void print_bitmaps(SwingInfo* info, size_t step, char* bitmap_send_merged, char* bitmap_recv_merged){          
    DPRINTF("[%d] Step %d Bitmap send: ", info->rank, step);
    for(size_t i = 0; i < (uint) info->size; i++){
        if(bitmap_send_merged[i]){
            DPRINTF("1");
        }else{
            DPRINTF("0");
        }
    }
    DPRINTF("\n");

    DPRINTF("[%d] Step %d Bitmap recv: ", info->rank, step);
    for(size_t i = 0; i < (uint) info->size; i++){
        if(bitmap_recv_merged[i]){
            DPRINTF("1");
        }else{
            DPRINTF("0");
        }
    }
    DPRINTF("\n");
}
*/
#endif

void SwingCommon::get_peer(int* coord_rank, size_t step, size_t port, int* coord_peer){
    size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    size_t current_d = port % dimensions_num;
    memcpy(coord_peer, coord_rank, sizeof(uint)*dimensions_num);
    memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);
    for(size_t i = 0; i < step; i++){
        // Move to the next dimension for the next step
        size_t d = current_d;              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < this->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d = (current_d + 1) % dimensions_num;
            d = current_d;
        }while(next_step_per_dim[d] >= this->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
    size_t distance = rhos[next_step_per_dim[current_d]];
    distance *= get_distance_sign(coord_rank[current_d], port);
    coord_peer[current_d] = mod(coord_peer[current_d] + distance, dimensions[current_d]);
}

// We pass bitmap_tmp as a parameter only to avoid allocating/deallocating it at each call.
// Otherwise, it could be local to a call (it is just needed to transform the bitmap into the vector of corresponding nodes).
void SwingCommon::remap(const std::vector<int>& nodes, uint start_range, uint end_range, uint* blocks_remapping,
                        int step, size_t port, int* coord_rank, char* bitmap_tmp){
    if(nodes.size() < 2){ // Needed for cases with non-power of two sizes (e.g., 6x6)
        return;
    }else if(nodes.size() == 2){
        blocks_remapping[nodes[0]] = start_range;
        blocks_remapping[nodes[1]] = end_range - 1;
        assert(end_range == start_range + 2);
    }else{
        // Find two partitions of node that talk with each other. If I have n nodes, 
        // if I see what happens in next step, I have two disjoint sets of nodes.        
        int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];   
        get_peer(coord_rank, step, port, coord_peer);
        get_blocks_bitmaps_multid(step, port, coord_peer, bitmap_tmp, NULL, coord_rank);

        std::vector<int> left, right;
        left.reserve(this->size);
        right.reserve(this->size);
        for(auto n : nodes){
            if(bitmap_tmp[n] == 0){
                left.push_back(n);
            }else{
                right.push_back(n);
            }
        }

        DPRINTF("[%d] step %d NODESIZE %d %d\n", this->rank, step, left.size(), right.size());        

        remap(left , start_range             , start_range + left.size(), blocks_remapping, step + 1, port, coord_rank, bitmap_tmp);
        remap(right, end_range - right.size(), end_range                , blocks_remapping, step + 1, port, coord_peer, bitmap_tmp);
    }
}

void SwingCommon::compute_bitmaps(size_t step, char** bitmap_ready, char** bitmap_recv,
                                  size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_PORTS][LIBSWING_MAX_SUPPORTED_DIMENSIONS], size_t* current_d, int* coord_mine,
                                  char*** bitmap_send_merged, char*** bitmap_recv_merged){
    assert(peers_computed);
    char* tmp_s = (char*) malloc(sizeof(char)*this->size);
    char* tmp_r = (char*) malloc(sizeof(char)*this->size);
    char* bitmap_tmp = (char*) malloc(sizeof(char)*this->size);   
    
    std::vector<int> nodes(this->size);
    for(size_t n = 0; n < (size_t) this->size; n++){nodes[n] = n;}     
    
    char* bitmap_send[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    for(size_t d = 0; d < this->dimensions_num; d++){
        bitmap_send[d] = (char*) malloc(sizeof(char)*dimensions[d]);
    }        
    for(size_t p = 0; p < this->num_ports; p++){                
        // Compute bitmaps of blocks to send and receive (we do not need to do this for allgather since bitmap_ready would always be 1)
        if(!bitmap_ready[p][step]){
            bitmap_send_merged[p][step] = (char*) malloc(sizeof(char)*this->size);
            bitmap_recv_merged[p][step] = (char*) malloc(sizeof(char)*this->size);
            
            int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];   
            retrieve_coord_mapping(this->peers_per_port[p][step], false, coord_peer);

            get_blocks_bitmaps_multid(next_step_per_dim[p], current_d, step, 
                                      p, coord_peer, bitmap_send, 
                                      bitmap_recv, bitmap_send_merged[p][step], bitmap_recv_merged[p][step], 
                                      coord_mine);
            bitmap_ready[p][step] = 1;

            /*************/
            /* REMAPPING */
            /*************/
            if(algo == ALGO_SWING_B_CONT || algo == ALGO_SWING_B_UTOFU){
                DPRINTF("[%d] Remapping\n", this->rank);

                // If the remapping_per_port for this port has not yet been computed, compute it
                if(remapping_per_port[p] == 0){
                    int coord_zero[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
                    memset(coord_zero, 0, sizeof(int)*this->dimensions_num);
                    remapping_per_port[p] = (uint*) malloc(sizeof(uint)*this->size);                          
                    remap(nodes, 0, this->size, remapping_per_port[p], 0, p, coord_zero, bitmap_tmp);            
                }

                memcpy(tmp_s, bitmap_send_merged[p][step], sizeof(char)*this->size);
                memcpy(tmp_r, bitmap_recv_merged[p][step], sizeof(char)*this->size);
                for(size_t i = 0; i < (uint) this->size; i++){
                    DPRINTF("[%d] Remapping %d to %d\n", this->rank, i, this->remapping_per_port[p][i]);
                    bitmap_send_merged[p][step][this->remapping_per_port[p][i]] = tmp_s[i];
                    bitmap_recv_merged[p][step][this->remapping_per_port[p][i]] = tmp_r[i];
                }
            }
#ifdef DEBUG
            //print_bitmaps(info, step, bitmap_send_merged[p][step], bitmap_recv_merged[p][step]);
#endif
        }
    }
    free(tmp_s);
    free(tmp_r);
    free(bitmap_tmp);
    for(size_t d = 0; d < this->dimensions_num; d++){
        free(bitmap_send[d]);
    }
}

/**
 * A generic collective operation sending/transmitting blocks rather than the entire buffer.
 * @param blocks_info: For each chunk, port, and block, the count and offset of the block
*/
int SwingCommon::swing_coll_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type){    
    Timer timer;
    int res = MPI_SUCCESS;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);  

    if(!peers_computed){
        DPRINTF("[%d] Computing peers\n", this->rank);
        this->peers_per_port = (uint**) malloc(sizeof(uint*)*this->num_ports);
        for(size_t p = 0; p < this->num_ports; p++){
            this->peers_per_port[p] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(p, this->rank, false, this->peers_per_port[p]);
        }
        DPRINTF("[%d] Peers computed\n", this->rank);
        peers_computed = true;
    }

    char* rbuf = NULL;
    if(coll_type == SWING_REDUCE_SCATTER || coll_type == SWING_ALLREDUCE){
        size_t total_size_bytes = count*dtsize;
        rbuf = (char*) malloc(total_size_bytes);
    }
    int coord_mine[LIBSWING_MAX_SUPPORTED_DIMENSIONS];    
    getCoordFromId(this->rank, false, coord_mine);

    char* bitmap_recv[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    char** bitmap_send_merged[LIBSWING_MAX_SUPPORTED_PORTS];
    char** bitmap_recv_merged[LIBSWING_MAX_SUPPORTED_PORTS];
    char* bitmap_ready[LIBSWING_MAX_SUPPORTED_PORTS]; // For each port and step, if 1, the send/recv bitmaps have been already computed.
    size_t current_d[LIBSWING_MAX_SUPPORTED_PORTS]; // For each port, what's the current dimension we are sending in.
    size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_PORTS][LIBSWING_MAX_SUPPORTED_DIMENSIONS]; // For each port and for each dimension, what's the next step to execute in that dimension.

    for(size_t d = 0; d < this->dimensions_num; d++){
        bitmap_recv[d] = (char*) malloc(sizeof(char)*dimensions[d]);   
    }

    for(size_t p = 0; p < this->num_ports; p++){
        bitmap_ready[p] = (char*) malloc(sizeof(char)*this->num_steps);
        memset(bitmap_ready[p], 0, sizeof(char)*this->num_steps);        
        bitmap_send_merged[p] = (char**) malloc(sizeof(char*)*this->num_steps);
        bitmap_recv_merged[p] = (char**) malloc(sizeof(char*)*this->num_steps);                                
    }

    size_t collectives_to_run_num = 0;
    CollType collectives_to_run[LIBSWING_MAX_COLLECTIVE_SEQUENCE]; 
    void *buf_s[LIBSWING_MAX_COLLECTIVE_SEQUENCE];
    void *buf_r[LIBSWING_MAX_COLLECTIVE_SEQUENCE];
    switch(coll_type){
        case SWING_ALLREDUCE:{
            collectives_to_run[0] = SWING_REDUCE_SCATTER;
            collectives_to_run[1] = SWING_ALLGATHER;
            buf_s[0] = recvbuf;
            buf_r[0] = rbuf;
            buf_s[1] = recvbuf;
            buf_r[1] = recvbuf;
            collectives_to_run_num = 2;
            break;
        }
        case SWING_REDUCE_SCATTER:{
            collectives_to_run[0] = SWING_REDUCE_SCATTER;
            collectives_to_run_num = 1;
            buf_s[0] = recvbuf;
            buf_r[0] = rbuf;
            break;
        }
        case SWING_ALLGATHER:{
            // We first run a "NULL" collective (to force bookeping data computation), and then the actual allgather 
            // TODO: Find a better way, avoid the NULL collective
            collectives_to_run[0] = SWING_NULL;
            collectives_to_run[1] = SWING_ALLGATHER;
            collectives_to_run_num = 2;
            buf_s[0] = recvbuf; // Just needed for utofu
            buf_r[0] = recvbuf; // Just needed for utofu
            buf_s[1] = recvbuf;
            buf_r[1] = recvbuf;
            break;
        }
        default:{
            assert("Unknown collective" == 0);
            return MPI_ERR_OTHER;
        }
    }

    if(algo == ALGO_SWING_B_UTOFU){     
#ifdef FUGAKU   
        // Setup all the communications        
        for(size_t port = 0; port < this->num_ports; port++){            
            memset(next_step_per_dim[port], 0, sizeof(size_t)*dimensions_num);
            current_d[port] = port % this->dimensions_num;
        }        
        // Compute all bitmaps // TODO: Refactor to do this step by step while waiting the data (overlap)
        for(size_t step = 0; step < this->num_steps; step++){
            compute_bitmaps(step, bitmap_ready, bitmap_recv, next_step_per_dim, current_d, coord_mine, bitmap_send_merged, bitmap_recv_merged);
        }   
        timer.reset("Bitmaps computation");
        swing_utofu_comm_descriptor* utofu_descriptor = swing_utofu_setup(buf_s[0], count*dtsize, buf_r[0], count*dtsize, this->num_ports, this->num_steps, peers_per_port);
        timer.reset("uTofu setup");
        swing_utofu_setup_wait(utofu_descriptor, this->num_steps);
        timer.reset("uTofu wait");
        // Needed to be sure everyone registered the buffers
        MPI_Barrier(MPI_COMM_WORLD);
        timer.reset("uTofu barrier");

#pragma omp parallel for num_threads(this->num_ports) private(res)
        for(size_t port = 0; port < this->num_ports; port++){
            // For reduce-scatter and allreduce we need to copy the data from sendbuf to recvbuf.
            if(coll_type == SWING_REDUCE_SCATTER || coll_type == SWING_ALLREDUCE){
                size_t offset = blocks_info[port][0].offset;
                size_t bytes_on_port = 0;                
                for(size_t i = 0; i < this->size; i++){
                    bytes_on_port += blocks_info[port][i].count*dtsize;
                }
                memcpy(((char*) recvbuf) + offset, ((char*) sendbuf) + offset, bytes_on_port);    
            }

            for(size_t collective = 0; collective < collectives_to_run_num; collective++){        
                // Reset info for the next series of steps        
                memset(next_step_per_dim[port], 0, sizeof(size_t)*this->dimensions_num);
                current_d[port] = port % this->dimensions_num;
                for(size_t step = 0; step < this->num_steps; step++){       
                    res = swing_coll_step_utofu(port, utofu_descriptor, buf_s[collective], buf_r[collective], blocks_info, step, 
                                                op, comm, datatype, datatype, 
                                                collectives_to_run[collective], bitmap_send_merged, bitmap_recv_merged);                                                    
                    assert(res == MPI_SUCCESS);
                }
            }
        }
        timer.reset("main loop");
        // Cleanup utofu resources
        swing_utofu_teardown(utofu_descriptor);
        timer.reset("uTofu teardown");
#else
        fprintf(stderr, "uTofu can only be used on Fugaku.\n");
        exit(-1);
#endif
    }else{
        if(coll_type == SWING_REDUCE_SCATTER || coll_type == SWING_ALLREDUCE){
            size_t total_size_bytes = count*dtsize;
            memcpy(recvbuf, sendbuf, total_size_bytes);    
        }
        for(size_t collective = 0; collective < collectives_to_run_num; collective++){ 
            // Reset info for the next series of steps        
            for(size_t p = 0; p < this->num_ports; p++){                
                memset(next_step_per_dim[p], 0, sizeof(size_t)*dimensions_num);
                current_d[p] = p % dimensions_num;
            }           
        
            for(size_t step = 0; step < this->num_steps; step++){        
                compute_bitmaps(step, bitmap_ready, bitmap_recv, next_step_per_dim, current_d, coord_mine, bitmap_send_merged, bitmap_recv_merged);
                res = swing_coll_step(buf_s[collective], buf_r[collective], blocks_info, step,                                 
                                      op, comm, datatype, datatype,  
                                      collectives_to_run[collective], bitmap_send_merged, bitmap_recv_merged);
                if(res != MPI_SUCCESS){return res;} 
            }
        }
    }
   
    /********/
    /* Free */
    /********/
    if(rbuf){
        free(rbuf);
    }
    for(size_t p = 0; p < this->num_ports; p++){
        for(size_t s = 0; s < (uint) this->num_steps; s++){
            free(bitmap_send_merged[p][s]);
            free(bitmap_recv_merged[p][s]);
        }
        free(bitmap_send_merged[p]);
        free(bitmap_recv_merged[p]);
    }
    for(size_t d = 0; d < dimensions_num; d++){
        free(bitmap_recv[d]);
    }
    return res;
}

