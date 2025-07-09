#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>

#include "libbine_common.h"
#ifdef FUGAKU
#include "fugaku/bine_utofu.h"
#endif

Timer::Timer(std::string fname, std::string name):_name(name), _timer_stopped(false){
#ifdef PROFILE
    _start_time_point = std::chrono::PROFILE_TIMER_TYPE::now();
    _fname = fname;  
#endif
}

Timer::Timer(std::string name):_name(name), _timer_stopped(false){
#ifdef PROFILE
    _start_time_point = std::chrono::PROFILE_TIMER_TYPE::now();
    _fname = "";  
#endif
}

Timer::~Timer() {
#ifdef PROFILE    
    if(!_timer_stopped){
        stop();
    }
    if(_fname == ""){
        std::cout << _ss.str();
    }else{
        std::ofstream out;
        out.open(_fname);
        out << _ss.str();
        out.close();
    }
#endif
}

void Timer::stop() {
#ifdef PROFILE
    _end_time_point = std::chrono::PROFILE_TIMER_TYPE::now();
    auto start = std::chrono::time_point_cast<std::chrono::microseconds>(_start_time_point).time_since_epoch().count();
    auto end = std::chrono::time_point_cast<std::chrono::microseconds>(_end_time_point).time_since_epoch().count();
    auto duration = end - start;
    _ss << "Timer [" << _name << "]: " << duration << " us | Start: " << start << " End: " << end << std::endl;
    _timer_stopped = true;
#endif
}

void Timer::reset(std::string name){
#ifdef PROFILE
    stop();
    _timer_stopped = false;
    _start_time_point = _end_time_point; //std::chrono::high_resolution_clock::now();
    _name = name;
#endif        
}

#ifdef FUGAKU
static inline void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    if(datatype == MPI_INT){
        const int *in = (const int *)inbuf;
        int *inout = (int *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
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
            fprintf(stderr, "Unknown reduction op\n");
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
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else{
        fprintf(stderr, "Unknown reduction datatype\n");
        exit(EXIT_FAILURE);
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


BineCommon::BineCommon(MPI_Comm comm, uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, Algo algo, uint num_ports, uint segment_size, size_t prealloc_size, char* prealloc_buf): 
            algo(algo), num_ports(num_ports), segment_size(segment_size), all_p2_dimensions(true), num_steps(0), scc(dimensions, dimensions_num), prealloc_size(prealloc_size), prealloc_buf(prealloc_buf){
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
    if(this->num_steps > LIBBINE_MAX_STEPS){
        assert("Max steps limit must be increased and constants updated.");
    }
    int size;
    MPI_Comm_size(comm, &size);
    assert(size == this->size);
    MPI_Comm_rank(comm, &this->rank);
    this->dimensions_num = dimensions_num;
    for(size_t i = 0; i < this->num_ports; i++){
        this->sbc[i] = NULL;
    }

    // Compute the number of steps on the virtual shrunk topology (for latency-optimal)
    size_virtual = 1;
    num_steps_virtual = 0;
    if(all_p2_dimensions){
        this->num_steps_virtual = this->num_steps;
        this->size_virtual = this->size;
    }else{
        for(size_t i = 0; i < this->dimensions_num; i++){
            this->size_virtual *= this->dimensions_virtual[i];
            this->num_steps_virtual += ceil(log2(this->dimensions_virtual[i]));
        }
    }
    for(size_t i = 0; i < this->num_ports; i++){
        virtual_peers[i] = NULL;
    }
}

BineCommon::~BineCommon(){
    for(size_t i = 0; i < this->num_ports; i++){
        if(this->sbc[i] != NULL){
            delete this->sbc[i];
        }
        if(virtual_peers[i] != NULL){
            free(virtual_peers[i]);
        }
    }
}

// Adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c
void BineCoordConverter::getCoordFromId(int id, bool virt, int* coord){
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
int BineCoordConverter::getIdFromCoord(int* coords, bool virt){
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
void BineCoordConverter::retrieve_coord_mapping(uint rank, bool virt, int* coord){
    int* coordinates = virt ? this->coordinates_virtual : this->coordinates;
    if(coordinates[rank*dimensions_num] == -1){
        getCoordFromId(rank, virt, &(coordinates[rank*dimensions_num]));
    }
    memcpy(coord, &(coordinates[rank*dimensions_num]), sizeof(uint)*dimensions_num);
}

BineCoordConverter::BineCoordConverter(uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num): dimensions_num(dimensions_num){
    memcpy(this->dimensions, dimensions, sizeof(uint)*dimensions_num);
    this->size = 1;
    for(size_t d = 0; d < dimensions_num; d++){
        this->size *= dimensions[d];
        if(!is_power_of_two(dimensions[d])){
            this->dimensions_virtual[d] = pow(2, (int) ceil(log2(dimensions[d])) - 1);
        }else{
            this->dimensions_virtual[d] = this->dimensions[d];
        }
    }
    this->coordinates = (int*) malloc(sizeof(int)*this->size*this->dimensions_num);
    this->coordinates_virtual = (int*) malloc(sizeof(int)*this->size*this->dimensions_num);
    memset(this->coordinates        , -1, sizeof(int)*this->size*dimensions_num);
    memset(this->coordinates_virtual, -1, sizeof(int)*this->size*dimensions_num);
}

BineCoordConverter::~BineCoordConverter(){
    free(this->coordinates_virtual);
    free(this->coordinates);
}

    
// Compute the peers of a rank in a torus which start transmitting from a specific port.
// @param port (IN): the port from which the transmission starts
// @param rank (IN): the rank
// @param virt (IN): if true, the virtual coordinates are considered, otherwise the real ones
// @param peers (OUT): the array where the peers are stored (one per step)
static void compute_peers(int port, uint rank, bool virt, uint* peers, uint* dimensions, uint dimensions_num, BineCoordConverter& scc){
    bool terminated_dimensions_bitmap[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    int num_steps_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    uint8_t next_step_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    memset(next_step_per_dim, 0, sizeof(uint8_t)*LIBBINE_MAX_SUPPORTED_DIMENSIONS);
    
    int num_steps = 0;
    for(size_t i = 0; i < dimensions_num; i++){
        num_steps_per_dim[i] = ceil(log2(dimensions[i]));
        num_steps += num_steps_per_dim[i];
    }

    // Compute default directions
    int coord[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    scc.retrieve_coord_mapping(rank, virt, coord);
    for(size_t i = 0; i < dimensions_num; i++){
        terminated_dimensions_bitmap[i] = false;            
    }
    
    int target_dim, relative_step, distance, last_dim = port - 1;
    uint terminated_dimensions = 0, o = 0;
    
    // Generate peers
    for(size_t i = 0; i < (uint) num_steps; ){            
        if(dimensions_num > 1){
            scc.retrieve_coord_mapping(rank, virt, coord); // Regenerate rank coord
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
                peers[i] = scc.getIdFromCoord(coord, virt);
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
int BineCommon::shrink_non_power_of_two(void *recvbuf, int count, 
                                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, 
                                         char* tmpbuf, int* idle, int* rank_virtual){    
    int coord_peer[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    int coord[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    
    int dtsize, rank;
    MPI_Type_size(datatype, &dtsize);    
    MPI_Comm_rank(comm, &rank);    
    
    this->scc.retrieve_coord_mapping(rank, false, coord);

    for(size_t i = 0; i < this->dimensions_num; i++){
        // This dimensions is not a power of two, shrink it
        if(!is_power_of_two(dimensions[i])){
            memcpy(coord_peer, coord, sizeof(uint)*this->dimensions_num);
            int extra = dimensions[i] - this->dimensions_virtual[i];
            if(coord[i] >= this->dimensions_virtual[i]){            
                coord_peer[i] = coord[i] - extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Shr) Sending to %d\n", rank, peer);
                int res = MPI_Send(recvbuf, count, datatype, peer, TAG_BINE_ALLREDUCE, comm);                
                if(res != MPI_SUCCESS){return res;}
                *idle = 1;
                break;
            }else if(coord[i] + extra >= this->dimensions_virtual[i]){
                coord_peer[i] = coord[i] + extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Shr) Receiving from %d\n", rank, peer);
                int res = MPI_Recv(tmpbuf, count, datatype, peer, TAG_BINE_ALLREDUCE, comm, NULL);                
                if(res != MPI_SUCCESS){return res;}
                DPRINTF("[%d] (Shr) Recvbuf (%p) before aggr %d\n", rank, recvbuf, ((char*) recvbuf)[0]);
                MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
                DPRINTF("[%d] (Shr) Recvbuf (%p) after aggr %d\n", rank, recvbuf, ((char*) recvbuf)[0]);
            }
        }
    }
    *rank_virtual = this->scc.getIdFromCoord(coord, true);
    return MPI_SUCCESS;
}

int BineCommon::enlarge_non_power_of_two(void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm){
    int coord[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    int coord_peer[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    int dtsize, rank;
    MPI_Type_size(datatype, &dtsize);    
    MPI_Comm_rank(comm, &rank);    
    
    this->scc.retrieve_coord_mapping(rank, false, coord);    
    //for(size_t d = 0; d < dimensions_num; d++){
    for(int d = dimensions_num - 1; d >= 0; d--){
        // This dimensions was a non-power of two, enlarge it
        if(!is_power_of_two(dimensions[d])){
            memcpy(coord_peer, coord, sizeof(uint)*this->dimensions_num);
            int extra = dimensions[d] - this->dimensions_virtual[d];
            if(coord[d] >= (uint) this->dimensions_virtual[d]){                
                coord_peer[d] = coord[d] - extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Enl) Receiving from %d\n", rank, peer);
                // I can overwrite the recvbuf and don't need to aggregate, since 
                // I was an extra node and did not participate to the actual allreduce
                int r = MPI_Recv(recvbuf, count, datatype, peer, TAG_BINE_ALLREDUCE, comm, NULL);
                if(r != MPI_SUCCESS){return r;}
            }else if(coord[d] + extra >= (uint) this->dimensions_virtual[d]){
                coord_peer[d] = coord[d] + extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Enl) Sending to %d\n", rank, peer);
                int r = MPI_Send(recvbuf, count, datatype, peer, TAG_BINE_ALLREDUCE, comm);                
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
// TODO: Implement a multiport version? There was a partial one (pre August 2024). In principle it does not have much sense to
// have it since we are in the latency-optimal case. Might help a bit for medium-sized messages, but probably needs 
// to be implemented using uTofu rather than MPI so that transmissions actually happen in parallel.

// TODO: Utofu implementation of latency optimal?
int BineCommon::MPI_Allreduce_lat_optimal(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(this->num_ports) + "/master.profile", "= MPI_Allreduce_lat_optimal (init)");
    Timer timer("MPI_Allreduce_lat_optimal (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    
    
    char* tmpbuf = (char*) malloc(count*dtsize); // Temporary buffer (to avoid overwriting sendbuf)
    memcpy(recvbuf, sendbuf, count*dtsize); // I send from recvbuf, which is where I aggregate the data. I receive data in tmpbuf
    int coord[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    this->scc.retrieve_coord_mapping(this->rank, false, coord);

    int res = MPI_SUCCESS, idle = 0, rank_virtual = rank;        
    if(!all_p2_dimensions){
        timer.reset("= MPI_Allreduce_lat_optimal (shrink)");
        res = shrink_non_power_of_two(recvbuf, count, datatype, op, comm, tmpbuf, &idle, &rank_virtual);
        if(res != MPI_SUCCESS){return res;}
    }    

    DPRINTF("[%d] Virtual steps: %d Virtual dimensions (%d, %d, %d)\n", rank, num_steps_virtual, this->dimensions_virtual[0], this->dimensions_virtual[1], this->dimensions_virtual[2]);

    if(!idle){                
        // Do the step-by-step communication on the shrunk topology.         
        timer.reset("= MPI_Allreduce_lat_optimal (actual sendrecvs)");
        uint partition_size = count / this->num_ports;
        uint remaining = count % this->num_ports;        
        for(size_t step = 0; step < (uint) num_steps_virtual; step++){                 
            // Schedule all the send and recv
            //timer.reset("= MPI_Allreduce_lat_optimal (sendrecv for step " + std::to_string(step) + ")");
            uint count_so_far = 0;
            MPI_Request requests_s[LIBBINE_MAX_SUPPORTED_PORTS];
            MPI_Request requests_r[LIBBINE_MAX_SUPPORTED_PORTS];

            for(size_t p = 0; p < this->num_ports; p++){
                // Get the peer
                if(virtual_peers[p] == NULL){
                    virtual_peers[p] = (uint*) malloc(sizeof(uint)*num_steps_virtual);
                    compute_peers(p, rank_virtual, true, virtual_peers[p], this->dimensions_virtual, this->dimensions_num, this->scc);
                }
                int coord_peer[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
                int virtual_peer = virtual_peers[p][step];                 
                this->scc.retrieve_coord_mapping(virtual_peer, true, coord_peer); // Get the virtual coordinates of the peer
                int peer = this->scc.getIdFromCoord(coord_peer, false); // Convert the virtual coordinates to the real rank
                DPRINTF("[%d] Sending to %d count %d\n", rank, peer, count);

                size_t count_port = partition_size + (p < remaining ? 1 : 0);
                size_t offset_port = count_so_far * dtsize;
                count_so_far += count_port;
                res = MPI_Isend(((char*) recvbuf) + offset_port, count_port, datatype, peer, TAG_BINE_ALLREDUCE, comm, &(requests_s[p]));
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on isend\n", rank); return res;}
                res = MPI_Irecv(((char*) tmpbuf) + offset_port, count_port, datatype, peer, TAG_BINE_ALLREDUCE, comm, &(requests_r[p]));                    
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on irecv\n", rank); return res;}                
            }
            MPI_Waitall(this->num_ports, requests_s, MPI_STATUSES_IGNORE);
            MPI_Waitall(this->num_ports, requests_r, MPI_STATUSES_IGNORE);
            res = MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op); 
            if(res != MPI_SUCCESS){DPRINTF("[%d] Error on reduce_local\n", rank); return res;}
            DPRINTF("[%d] Step %d completed\n", rank, step);            
        }
    }

    if(!all_p2_dimensions){
        timer.reset("= MPI_Allreduce_lat_optimal (enlarge)");
        DPRINTF("[%d] Propagating data to extra nodes\n", rank);
        res = enlarge_non_power_of_two(recvbuf, count, datatype, comm);
        if(res != MPI_SUCCESS){return res;}
        DPRINTF("[%d] Data propagated\n", rank);
    }    
    timer.reset("= MPI_Allreduce_lat_optimal (writing profile data to file)");
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

int BineCommon::bine_coll_step_b(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                             
                                   MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                   CollType coll_type){    
    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    memset(requests_s, 0, sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    memset(requests_r, 0, sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    BlockInfo* req_idx_to_block_idx = (BlockInfo*) malloc(sizeof(BlockInfo)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    uint num_requests_s = 0, num_requests_r = 0;    
    assert(sendtype == recvtype);
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    int tag, res = MPI_SUCCESS;
    for(size_t port = 0; port < this->num_ports; port++){
        if(step == 0){
            sbc[port]->compute_bitmaps(0, coll_type);
        }
        uint peer = sbc[port]->get_peer(step, coll_type);
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", this->rank, step, port, this->num_ports, peer);                

        if(coll_type == BINE_REDUCE_SCATTER){
            tag = TAG_BINE_REDUCESCATTER + port;
        }else{
            tag = TAG_BINE_ALLGATHER + port;
        }

        // Sendrecv + aggregate
        // Search for the blocks that must be sent.
        for(size_t i = 0; i < (uint) size; i++){
            bool send_block = sbc[port]->block_must_be_sent(step, coll_type, i);
            bool recv_block = sbc[port]->block_must_be_recvd(step, coll_type, i);
            
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
                res = MPI_Irecv(((char*) tmpbuf) + block_offset, block_count, recvtype, peer, tag, comm, &(requests_r[num_requests_r]));
                if(res != MPI_SUCCESS){return res;}
                req_idx_to_block_idx[(num_requests_r)].offset = block_offset;
                req_idx_to_block_idx[(num_requests_r)].count = block_count;
                ++(num_requests_r);
            }
        }
    }
    DPRINTF("[%d] Issued %d send requests and %d receive requests\n", rank, num_requests_s, num_requests_r);

    if(step < this->num_steps - 1){
        for(size_t port = 0; port < this->num_ports; port++){
            sbc[port]->compute_bitmaps(step + 1, coll_type);
        }
    }

    // Wait for all the recvs to be over
    if(coll_type == BINE_REDUCE_SCATTER){
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
            void* tmpbuf_block = (void*) (((char*) tmpbuf) + req_idx_to_block_idx[index].offset);
            void* buf_block = (void*) (((char*) buf) + req_idx_to_block_idx[index].offset);  
            DPRINTF("[%d] Aggregating from %p to %p (i %d index %d offset %d count %d)\n", this->rank, tmpbuf_block, buf_block, i, index, req_idx_to_block_idx[index].offset, req_idx_to_block_idx[index].count);
            MPI_Reduce_local(tmpbuf_block, buf_block, req_idx_to_block_idx[index].count, sendtype, op); 
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

int BineCommon::bine_coll_step_cont(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                                 
                                MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                CollType coll_type){
    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    memset(requests_s, 0, sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    memset(requests_r, 0, sizeof(MPI_Request)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    BlockInfo* req_idx_to_block_idx = (BlockInfo*) malloc(sizeof(BlockInfo)*this->size*LIBBINE_MAX_SUPPORTED_PORTS);
    uint num_requests_s = 0, num_requests_r = 0;    
    assert(sendtype == recvtype);
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    int tag, res = MPI_SUCCESS;

    for(size_t port = 0; port < this->num_ports; port++){
        if(step == 0){
            sbc[port]->compute_bitmaps(0, coll_type);
        }
        uint peer = sbc[port]->get_peer(step, coll_type);
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", this->rank, step, port, this->num_ports, peer);                

        if(coll_type == BINE_REDUCE_SCATTER){
            tag = TAG_BINE_REDUCESCATTER + port;
        }else{
            tag = TAG_BINE_ALLGATHER + port;
        }

        // Sendrecv + aggregate
        // Search for the blocks that must be sent.
        bool start_found_s = false, start_found_r = false;
        size_t offset_s, offset_r, count_s = 0, count_r = 0;
        for(size_t i = 0; i < (uint) this->size; i++){
            bool send_block = sbc[port]->block_must_be_sent(step, coll_type, i);
            bool recv_block = sbc[port]->block_must_be_recvd(step, coll_type, i);
            
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
                res = MPI_Irecv(((char*) tmpbuf) + offset_r, count_r, recvtype, peer, tag, comm, &(requests_r[num_requests_r]));
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
    if(step < this->num_steps - 1){
        for(size_t port = 0; port < this->num_ports; port++){
            sbc[port]->compute_bitmaps(step + 1, coll_type);
        }
    }

    // Wait for all the recvs to be over
    if(coll_type == BINE_REDUCE_SCATTER){
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
            void* tmpbuf_block = (void*) (((char*) tmpbuf) + req_idx_to_block_idx[index].offset);
            void* buf_block = (void*) (((char*) buf) + req_idx_to_block_idx[index].offset);  
            DPRINTF("[%d] Aggregating from %p to %p (i %d index %d offset %d count %d)\n", this->rank, tmpbuf_block, buf_block, i, index, req_idx_to_block_idx[index].offset, req_idx_to_block_idx[index].count);
            MPI_Reduce_local(tmpbuf_block, buf_block, req_idx_to_block_idx[index].count, sendtype, op); 
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

int BineCommon::bine_coll_step(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                                 
                                MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                CollType coll_type){
    if(algo == ALGO_BINE_B){
        return bine_coll_step_b(buf, tmpbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type);
    }else if(algo == ALGO_BINE_B_CONT || algo == ALGO_BINE_B_COALESCE){
        return bine_coll_step_cont(buf, tmpbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type);
    }else{
        assert("Unknown algo" == 0);
        return MPI_ERR_OTHER;
    }
}

#ifdef FUGAKU
int BineCommon::bine_coll_step_utofu(size_t port, bine_utofu_comm_descriptor* utofu_descriptor, const void* sendbuf, void *recvbuf, void* tempbuf, size_t tmpbuf_size, const BlockInfo *const *const blocks_info, size_t step, 
                                       MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                       CollType coll_type, bool is_first_coll){                                        
    size_t offsets_s, counts_s;
    size_t offsets_r, counts_r;
    
    Timer timer("== bine_coll_step_utofu (compute bitmaps 0)");

    if(step == 0){
        sbc[port]->compute_bitmaps(step, coll_type);
    }    

    timer.reset("== bine_coll_step_utofu (indexes calc)");
    ChunkParams cp = sbc[port]->get_chunk_params(step, coll_type, blocks_info);
    offsets_s = cp.send_offset;
    counts_s = cp.send_count;
    offsets_r = cp.recv_offset;
    counts_r = cp.recv_count;

    timer.reset("== bine_coll_step_utofu (misc)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);  
    size_t max_count;
    if(coll_type == BINE_REDUCE_SCATTER && this->segment_size){
        max_count = floor(this->segment_size / dtsize);
    }else{
        max_count = floor(MAX_PUTGET_SIZE / dtsize);
    }
    char issued_sends = 0, issued_recvs = 0;
    size_t count = 0;     
    size_t offset = offsets_s;
    size_t utofu_offset_r = offsets_s;
    size_t utofu_offset_r_start = offsets_s;
    
    // In reduce-scatter, if some rank are faster than others,
    // one executing a later step might write the data in the destination
    // rank earlier than a rank executing a previous step.
    // As a result, the first put would be lost. To avoid that, 
    // instead of writing in the actual offset, we force the offsets
    // to be different, since anyway the data must be moved from the receive
    // buffer when aggregating it.
    if(coll_type == BINE_REDUCE_SCATTER && step != 0){ // For first step we do not need to do it (we write directly in user_recvbuf rather than tmpbuf)
        utofu_offset_r = (tmpbuf_size / this->num_ports) * port; // Start from the beginning of the buffer
        for(size_t i = 0; i < step; i++){
            utofu_offset_r += (tmpbuf_size / this->num_ports) / pow(2, (i + 1)); // TODO: Does not work if number of ranks is not a power of 2
        }
        utofu_offset_r_start = utofu_offset_r;
    }

    timer.reset("== bine_coll_step_utofu (sends)");
    // We first enqueue all the send. Then, we receive and aggregate
    // Aggregation and reception of next block should be overlapped
    // Issue isends for all the blocks
    if(counts_s){ 
        size_t remaining = counts_s;
        size_t bytes_to_send = 0;
        int peer = sbc[port]->get_peer(step, coll_type);

        // Segment the transmission
        while(remaining){
            count = remaining < max_count ? remaining : max_count;
            bytes_to_send = count*dtsize;
            DPRINTF("[%d] Sending %d bytes to %d at step %d (coll %d)\n", this->rank, bytes_to_send, sbc[port]->get_peer(step, coll_type), step, coll_type);
            
            utofu_stadd_t lcl_addr, rmt_addr;
            if(coll_type == BINE_REDUCE_SCATTER){
                if(step == 0){
                    // To avoid memcpy from sendbuf to recvbuf in the first step I need to (in the first step):
                    // - Send from local sendbuf to remote recvbuf, then aggregate from remote sendbuf to remote recvbuf
                    lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd + offset;
                    rmt_addr = (*(utofu_descriptor->port_info[port].rmt_info))[peer].recv_stadd + utofu_offset_r;
                }else{
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd + offset;
                    rmt_addr = (*(utofu_descriptor->port_info[port].rmt_info))[peer].temp_stadd + utofu_offset_r;
                }
            }else if(coll_type == BINE_ALLGATHER){
                if(is_first_coll && step == 0){
                    lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd + offset;
                }else{ // If I executed a collective before (i.e., allgather), the data to send is already in recvbuf
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd + offset;                    
                }
                rmt_addr = (*(utofu_descriptor->port_info[port].rmt_info))[peer].recv_stadd + utofu_offset_r;
            }else{
                assert("Unknown collective type" == 0);
            }        
   
#if !(UTOFU_THREAD_SAFE)
            #pragma omp critical // To remove this we should put BINE_UTOFU_VCQ_FLAGS to UTOFU_VCQ_FLAG_EXCLUSIVE. However, this adds crazy overhead when creating/destroying the VCQs
#endif
            bine_utofu_isend(utofu_descriptor, port, peer, lcl_addr, bytes_to_send, rmt_addr); 
            
            offset += bytes_to_send;
            utofu_offset_r += bytes_to_send;
            remaining -= count;
            ++issued_sends;
        }
    }

    // Here I can overlap computation to the reception of the data

    timer.reset("== bine_coll_step_utofu (compute bitmaps i)");
    // We issued the sends, while we wait for transmission we compute the bitmaps for the next step  
    if(step < this->num_steps - 1){
        sbc[port]->compute_bitmaps(step + 1, coll_type);
    }
    timer.reset("== bine_coll_step_utofu (recv + aggregate (init))");

    //double start = MPI_Wtime();
    // Receive and aggregate
    offset = 0;
    if(counts_r){
        size_t remaining = counts_r;
        size_t bytes_to_recv = 0;
        // Segment the transmission
        while(remaining){
            count = remaining < max_count ? remaining : max_count;
            bytes_to_recv = count*dtsize;
            DPRINTF("[%d] Receiving %d bytes at step %d (coll %d)\n", this->rank, bytes_to_recv, step, coll_type);

            utofu_stadd_t end_addr; // = stadd + offset + length;
            size_t recvbuf_offset = offsets_r + offset;
            char* recvbuf_block = (char*) recvbuf + recvbuf_offset;

            if(coll_type == BINE_REDUCE_SCATTER){
                if(step == 0){
                    // In the first step I receive in recvbuf and I aggregate from sendbuf and recvbuf in recvbuf (at same offsets)
                    size_t sendbuf_offset = recvbuf_offset;
                    end_addr = utofu_descriptor->port_info[port].lcl_recv_stadd + recvbuf_offset + bytes_to_recv;
                    timer.reset("== bine_coll_step_utofu (recv)");
                    bine_utofu_wait_recv(utofu_descriptor, port, end_addr);
                    timer.reset("== bine_coll_step_utofu (aggr)");
                    reduce_local((char*) sendbuf + sendbuf_offset, recvbuf_block, count, sendtype, op);
                }else{
                    // In the other steps I receive in tmpbuf and I aggregate from tmpbuf and recvbuf in recvbuf (for tmbuf we use adjusted offset)
                    size_t tempbuf_offset = utofu_offset_r_start + offset;
                    end_addr = utofu_descriptor->port_info[port].lcl_temp_stadd + tempbuf_offset + bytes_to_recv;
                    timer.reset("== bine_coll_step_utofu (recv)");
                    bine_utofu_wait_recv(utofu_descriptor, port, end_addr);                    
                    timer.reset("== bine_coll_step_utofu (aggr)");
                    reduce_local((char*) tempbuf + tempbuf_offset, recvbuf_block, count, sendtype, op);
                    //MPI_Reduce_local(tmpbuf_block + offset, buf_block + offset, count, sendtype, op); // TODO: Try to replace again with MPI_Reduce_local ?
                }
            }else{
                end_addr = utofu_descriptor->port_info[port].lcl_recv_stadd + recvbuf_offset + bytes_to_recv;
                timer.reset("== bine_coll_step_utofu (recv)");
                bine_utofu_wait_recv(utofu_descriptor, port, end_addr);
            }
            

            offset += bytes_to_recv;
            remaining -= count;
            ++issued_recvs;
        }            
    }
    /*
    if(omp_get_thread_num() == 0){
        std::cout << "recv+aggr: " << (MPI_Wtime() - start)*1000000.0 << std::endl;
    }
    */
    timer.reset("== bine_coll_step_utofu (wait isends)");
    DPRINTF("[%d] Issued %d sends and %d recvs\n", rank, issued_sends, issued_recvs);
    // Wait for send completion
    if(counts_s){
        bine_utofu_wait_sends(utofu_descriptor, port, issued_sends);
    }
    timer.reset("== bine_coll_step_utofu (profile writing)");
    DPRINTF("[%d] Sends completed\n", rank);
    return MPI_SUCCESS;
}
#else
int BineCommon::bine_coll_step_utofu(size_t port, bine_utofu_comm_descriptor* utofu_descriptor, const void* sendbuf, void *recvbuf, void* tempbuf, size_t tmpbuf_size, const BlockInfo *const *const blocks_info, size_t step, 
                                       MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,
                                       CollType coll_type, bool is_first_coll){
    fprintf(stderr, "uTofu can only be used on Fugaku.\n");
    exit(-1);
}
#endif


// Returns an array of valid distances (considering a plain collective on an even node)
void BineBitmapCalculator::compute_valid_distances(uint d, int step){
    int max_steps = this->num_steps_per_dim[d];
    int size = this->dimensions[d];
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

int BineBitmapCalculator::get_distance_sign(size_t rank, size_t port){
    int multiplier = 1;
    if(is_odd(rank)){ // Invert sign if odd rank
        multiplier *= -1;
    }
    if(port >= this->dimensions_num){ // Invert sign if mirrored collective
        multiplier *= -1;     
    }
    return multiplier;
}

void BineBitmapCalculator::get_blocks_bitmaps_multid(size_t step,
                                                      int* coord_peer, char** bitmap_send, 
                                                      char** bitmap_recv, char* bitmap_send_merged, char* bitmap_recv_merged, 
                                                      int* coord_mine, size_t* next_step_per_dim, size_t *current_d){
    if(next_step_per_dim == NULL){
        next_step_per_dim = this->next_step_per_dim;
    }    
    if(current_d == NULL){
        current_d = &(this->current_d);
    }
    // Compute the bitmap for each dimension
    for(size_t k = 0; k < this->dimensions_num; k++){
        size_t d = (k + *current_d) % this->dimensions_num;

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
    int coord_block[LIBBINE_MAX_SUPPORTED_DIMENSIONS];    
    for(size_t i = 0; i < (uint) this->size; i++){
        this->scc.retrieve_coord_mapping(i, false, coord_block);
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
        size_t d = *current_d;              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < this->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            *current_d = (*current_d + 1) % dimensions_num;
            d = *current_d;
        }while(next_step_per_dim[d] >= this->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
}


// Same as the one above, but we compute next_step_per_dim and current_d on the fly so that we do not need to do bookeping
void BineBitmapCalculator::get_blocks_bitmaps_multid(size_t step, int* coord_peer, 
                                                      char* bitmap_send_merged, char* bitmap_recv_merged, 
                                                      int* coord_mine){
    size_t next_step_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    size_t current_d;
    char* bitmap_send[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    char* bitmap_recv[LIBBINE_MAX_SUPPORTED_DIMENSIONS];    
    for(size_t d = 0; d < dimensions_num; d++){
        bitmap_send[d] = (char*) malloc(sizeof(char)*dimensions[d]);
        bitmap_recv[d] = (char*) malloc(sizeof(char)*dimensions[d]);   
    }

    current_d = port % dimensions_num;
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
    get_blocks_bitmaps_multid(step, coord_peer, bitmap_send, bitmap_recv, bitmap_send_merged, bitmap_recv_merged, coord_mine, next_step_per_dim, &current_d);
    for(size_t d = 0; d < dimensions_num; d++){
        free(bitmap_send[d]);
        free(bitmap_recv[d]);   
    }
}


#ifdef DEBUG
/*
static void print_bitmaps(BineInfo* info, size_t step, char* bitmap_send_merged, char* bitmap_recv_merged){          
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

void BineBitmapCalculator::get_peer(int* coord_rank, size_t step, int* coord_peer){
    size_t next_step_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
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

void BineBitmapCalculator::compute_bitmaps(uint step, CollType coll_type){
    size_t block_step = (coll_type == BINE_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);
    while(block_step >= this->next_step){
        compute_next_bitmaps();
    }
}

void BineBitmapCalculator::compute_block_step(int* coord_rank, size_t starting_step, size_t step, size_t num_steps, uint32_t* block_step){
    if(step < num_steps){
        for(size_t i = step + 1; i < num_steps; i++){
            int peer_rank[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
            get_peer(coord_rank, i, peer_rank);
            compute_block_step(peer_rank, starting_step, i, num_steps, block_step);
        }
        uint rank = scc.getIdFromCoord(coord_rank, false);
        if(block_step[rank] == num_steps ||   // If I have not reached this node yet
           starting_step > block_step[rank]){ // Or if I can reach it with a starting step higher than what previously found
            if(block_step[rank] == num_steps - 2){
                assert(step == num_steps - 1);
            }
            //if(this->rank == 0){
            //    printf("Setting block_step[%d] to %d and arrival_step[%d] to %d\n", rank, starting_step, rank, step);
            //}
            block_step[rank] = starting_step;
        }
    }
}

void BineBitmapCalculator::compute_next_bitmaps(){
    if(this->next_step >= this->num_steps){
        return;
    }                       
    bitmap_send_merged[this->next_step] = (char*) malloc(sizeof(char)*this->size);
    bitmap_recv_merged[this->next_step] = (char*) malloc(sizeof(char)*this->size);
    memset(bitmap_send_merged[this->next_step], 0, sizeof(char)*this->size);
    memset(bitmap_recv_merged[this->next_step], 0, sizeof(char)*this->size);

    /*************/
    /* REMAPPING */
    /*************/
    if(remap_blocks){       
        // We always assume reduce_scatter
        this->min_block_s = this->min_block_r;
        this->max_block_s = this->max_block_r;
        size_t middle = (this->min_block_r + this->max_block_r + 1) / 2; // == min + (max - min) / 2
        if(this->remapped_rank < middle){
            this->min_block_s = middle;
            this->max_block_r = middle;
        }else{
            this->max_block_s = middle;
            this->min_block_r = middle;
        }
        for(size_t i = this->min_block_s; i < this->max_block_s; i++){
            bitmap_send_merged[this->next_step][i] = 1;            
        }        
        for(size_t i = this->min_block_r; i < this->max_block_r; i++){
            bitmap_recv_merged[this->next_step][i] = 1;
        }
    }else{
        for(size_t i = 0; i < this->size; i++){
            // I am gonna send the blocks that I need to send at this step...
            if(block_step[i] == this->next_step){
                bitmap_send_merged[this->next_step][i] = 1;
            }

            uint32_t* peer_block_step = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
            for(size_t i = 0; i < this->size; i++){
                peer_block_step[i] = this->num_steps;
            }
            uint peer = get_peer(this->next_step, BINE_REDUCE_SCATTER);
            int peer_coord[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
            scc.getCoordFromId(peer, false, peer_coord);

            for(size_t j = this->next_step; j < this->num_steps; j++){
                int peer_peer_rank[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
                get_peer(peer_coord, j, peer_peer_rank);
                compute_block_step(peer_peer_rank, j, j, this->num_steps, peer_block_step);
            }

            peer_block_step[peer] = this->num_steps; // Disconnect myself (there might be loops, e.g., for 10 nodes)
            if(peer_block_step[i] == this->next_step){
                bitmap_recv_merged[this->next_step][i] = 1;
            }
            free(peer_block_step);
#if 0
            // ... and receive all those that my peer is going to send at this step.
            // To know which are those without computing them from scratch everytime,
            // I can make the following observations:
            // - If rank 0 sends block i, rank 1 sends block (1-i) % num_ranks
            // - If rank 1 sends block i, rank 0 sends block (1-i) % num_ranks
            // - If rank 0 sends a block i, rank 0+k sends block (i+k) % num_ranks (for k even)
            // - If rank 1 sends a block i, rank 1+k sends block (i+k) % num_ranks (for k even)
            //
            // Recall that even ranks only communicate with odd ranks (and vice-versa).
            // Let's assume r sends to q. If r is even, we first try to understand which block 0 sends
            // and from the blocks 0 sends, we can determine the blocks 1 sends, and thus the blocks q sends
            size_t block_sent_by_peer;
            if(rank % 2 == 0){
                // I receive block i if my peer (r) sends block i
                // r (odd) sends block i, iff 1 sends block (i - (r - 1)) % num_ranks
                // 1 sends block j, iff 0 sends block (1-j) % num_ranks
                // 0 sends block k, iff I send block (k + rank) % num_ranks
                size_t block_sent_by_1 = mod((i - (get_peer(this->next_step, BINE_REDUCE_SCATTER) - 1)), this->size);
                size_t block_sent_by_0 = mod((1 - block_sent_by_1), this->size);
                block_sent_by_peer = mod((block_sent_by_0 + rank), this->size);
            }else{
                // I receive block i if my peer (r) sends block i
                // r (even) sends block i, iff 0 sends block (i - r) % num_ranks
                // 0 sends block j, iff 1 sends block (1-j) % num_ranks
                // 1 sends block k, iff I send block (k + rank - 1) % num_ranks                
                size_t block_sent_by_0 = mod((i - get_peer(this->next_step, BINE_REDUCE_SCATTER)), this->size);
                size_t block_sent_by_1 = mod((1 - block_sent_by_0), this->size);
                block_sent_by_peer = mod((block_sent_by_1 + rank - 1), this->size);
            }

            if(block_step[block_sent_by_peer] == this->next_step){
                bitmap_recv_merged[this->next_step][i] = 1;
            }
#endif
        }
    }
    ++this->next_step;
}

/**
 * A generic collective operation sending/transmitting blocks rather than the entire buffer.
 * @param blocks_info: For each chunk, port, and block, the count and offset of the block
*/
int BineCommon::bine_coll_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type){    
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(this->num_ports) + "/master.profile", "= bine_coll_b (init)");
    Timer timer("= bine_coll_b (init)");
    int res = MPI_SUCCESS;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);  
    timer.reset("= bine_coll_b (tmpbuf alloc)");
    // Receive into tmpbuf and aggregate into recvbuf
    char* tmpbuf = NULL;
    size_t tmpbuf_size = 0;
    bool free_tmpbuf = false;
    if(coll_type == BINE_REDUCE_SCATTER || coll_type == BINE_ALLREDUCE){        
        if(algo == ALGO_BINE_B_UTOFU){
            // We can't write in the actual blocks positions since writes
            // might be executed in a different order than the one in which they were issued.
            // Thus, we must enforce writes to do not overlap. However, this means a rank must 
            // know how many blocks have been already written. Because blocks might have uneven size
            // (e.g., if the buffer size is not divisible by the number of ranks), it is hard to know
            // where exactly to write the data so that it does not overlap.
            // For this reason, we allocate a buffer so that it is a multiple of num_ports*num_blocks,
            // so that we can assume all the blocks have the same size.
            size_t fixed_count = count;
            if(count % (this->num_ports * this->size)){
                // Set fixed_count to the next multiple of this->num_ports * this->size
                fixed_count = count + (this->num_ports * this->size - count % (this->num_ports * this->size));
            }
            tmpbuf_size = fixed_count*dtsize;  
        }else{
            tmpbuf_size = count*dtsize;        
        }        
        if(tmpbuf_size > prealloc_size){
            tmpbuf = (char*) malloc(tmpbuf_size);
            free_tmpbuf = true;
        }else{
            tmpbuf = prealloc_buf;
        }
    }   

    timer.reset("= bine_coll_b (sbc alloc)");
    // Create bitmap calculators if not already created
    for(size_t p = 0; p < this->num_ports; p++){
        if(this->sbc[p] == NULL){
            this->sbc[p] = new BineBitmapCalculator(this->rank, this->dimensions, this->dimensions_num, p, (algo == ALGO_BINE_B_UTOFU || algo == ALGO_BINE_B_CONT) && is_power_of_two(this->size));
        }
    }
    
    timer.reset("= bine_coll_b (coll to run)");
    size_t collectives_to_run_num = 0;
    CollType collectives_to_run[LIBBINE_MAX_COLLECTIVE_SEQUENCE]; 
    void *buf_s[LIBBINE_MAX_COLLECTIVE_SEQUENCE];
    void *buf_r[LIBBINE_MAX_COLLECTIVE_SEQUENCE];
    switch(coll_type){
        case BINE_ALLREDUCE:{
            collectives_to_run[0] = BINE_REDUCE_SCATTER;            
            buf_s[0] = recvbuf;
            buf_r[0] = tmpbuf;
            collectives_to_run[1] = BINE_ALLGATHER;
            buf_s[1] = recvbuf;
            buf_r[1] = recvbuf;
            collectives_to_run_num = 2;
            break;
        }
        case BINE_REDUCE_SCATTER:{
            collectives_to_run[0] = BINE_REDUCE_SCATTER;
            collectives_to_run_num = 1;
            buf_s[0] = recvbuf;
            buf_r[0] = tmpbuf;
            break;
        }
        case BINE_ALLGATHER:{
            collectives_to_run[0] = BINE_ALLGATHER;
            collectives_to_run_num = 1;
            buf_s[0] = recvbuf;
            buf_r[0] = recvbuf;
            break;
        }
        default:{
            assert("Unknown collective" == 0);
            return MPI_ERR_OTHER;
        }
    }
    if(algo == ALGO_BINE_B_UTOFU){     
#ifdef FUGAKU   
        // Setup all the communications        
        // TODO: Cache also utofu descriptors to avoid exchanging pointers at each allreduce?
        timer.reset("= bine_coll_b (utofu setup)");        
        bine_utofu_comm_descriptor* utofu_descriptor = bine_utofu_setup((void*) sendbuf, count*dtsize, recvbuf, count*dtsize, tmpbuf, tmpbuf_size, 
                                                                          this->num_ports, this->num_steps, this->sbc[0]);
        
        timer.reset("= bine_coll_b (utofu wait)");            
        bine_utofu_setup_wait(utofu_descriptor, this->num_steps);
        
        // Needed to be sure everyone registered the buffers
        timer.reset("= bine_coll_b (utofu barrier)");
        MPI_Barrier(MPI_COMM_WORLD);
        
        timer.reset("= bine_coll_b (utofu main loop)");

#pragma omp parallel for num_threads(this->num_ports) schedule(static, 1) collapse(1)
        for(size_t port = 0; port < this->num_ports; port++){
            /*
            int thread_num = omp_get_thread_num();
            int cpu_num = sched_getcpu();
            printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
            */
            for(size_t collective = 0; collective < collectives_to_run_num; collective++){        
                for(size_t step = 0; step < this->num_steps; step++){       
                    int r = bine_coll_step_utofu(port, utofu_descriptor, sendbuf, recvbuf, tmpbuf, tmpbuf_size, blocks_info, step, 
                                                op, comm, datatype, datatype, 
                                                collectives_to_run[collective], collective == 0);                                                    
                    assert(r == MPI_SUCCESS);
                }
            }
        }
        timer.reset("= bine_coll_b (utofu teardown)");
        // Cleanup utofu resources
        bine_utofu_teardown(utofu_descriptor);
#else
        fprintf(stderr, "uTofu can only be used on Fugaku.\n");
        exit(-1);
#endif
    }else{
        if(coll_type == BINE_REDUCE_SCATTER || coll_type == BINE_ALLREDUCE){
            size_t total_size_bytes = count*dtsize;
            memcpy(recvbuf, sendbuf, total_size_bytes);    
        }
        for(size_t collective = 0; collective < collectives_to_run_num; collective++){        
            for(size_t step = 0; step < this->num_steps; step++){                        
                DPRINTF("[%d] Bitmap computed for step %d\n", this->rank, step);
                res = bine_coll_step(buf_s[collective], buf_r[collective], blocks_info, step,                                 
                                      op, comm, datatype, datatype,  
                                      collectives_to_run[collective]);
                if(res != MPI_SUCCESS){return res;} 
            }
        }
    }
   
    /********/
    /* Free */
    /********/
    if(free_tmpbuf){
        free(tmpbuf);
    }

    timer.reset("= bine_coll_b (profile writing)");
    return res;
}

uint BineBitmapCalculator::get_peer(uint step, CollType coll_type){
    size_t block_step = (coll_type == BINE_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);            
    return (this->peers)[block_step];
}

bool BineBitmapCalculator::block_must_be_sent(uint step, CollType coll_type, uint block_id){
    compute_bitmaps(step, coll_type); // This is going to be a nop if compute_bitmaps was already called before
    if(coll_type == BINE_REDUCE_SCATTER){
        return bitmap_send_merged[step][block_id];
    }else{                
        return bitmap_recv_merged[(this->num_steps - step - 1)][block_id];
    }
}

bool BineBitmapCalculator::block_must_be_recvd(uint step, CollType coll_type, uint block_id){
    compute_bitmaps(step, coll_type); // This is going to be a nop if compute_bitmaps was already called before
    if(coll_type == BINE_REDUCE_SCATTER){
        return bitmap_recv_merged[step][block_id];
    }else{                
        return bitmap_send_merged[(this->num_steps - step - 1)][block_id];
    }
}

ChunkParams BineBitmapCalculator::get_chunk_params(uint step, CollType coll_type, const BlockInfo *const *const blocks_info){
    uint block_step = (coll_type == BINE_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);
    
    if(!valid_chunk_params[block_step]){
        ChunkParams cp;
        bool start_found_s = false, start_found_r = false, chunk_sent = false, chunk_recvd = false;
        size_t offset_s, offset_r, count_s = 0, count_r = 0;
        for(size_t i = 0; i < (uint) this->size; i++){
            bool send_block = block_must_be_sent(step, BINE_REDUCE_SCATTER, i);
            bool recv_block = block_must_be_recvd(step, BINE_REDUCE_SCATTER, i);      
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
                if(chunk_sent){
                    fprintf(stderr, "With uTofu we support at most one send/recv per port\n");
                    exit(-1);
                }
                chunk_sent = true;
                cp.send_offset = offset_s;
                cp.send_count = count_s;

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
                if(chunk_recvd){
                    fprintf(stderr, "With uTofu we support at most one send/recv per port\n");
                    exit(-1);
                }
                chunk_recvd = true;
                cp.recv_offset = offset_r;
                cp.recv_count = count_r;
                
                // In some rare cases (e.g., for 10 nodes), I might have not one but two consecutive trains of blocks
                // Reset everything in case we need to send another train of blocks
                count_r = 0;
                offset_r = 0;
                start_found_r = false;
            }
        }
        chunk_params[block_step] = cp;
        valid_chunk_params[block_step] = true;
    }

    ChunkParams r;
    if(coll_type == BINE_REDUCE_SCATTER){
        r = chunk_params[block_step];
    }else{
        r.send_count = chunk_params[block_step].recv_count;
        r.send_offset = chunk_params[block_step].recv_offset;
        r.recv_count = chunk_params[block_step].send_count;
        r.recv_offset = chunk_params[block_step].send_offset;
    }
    return r;
}


static inline uint32_t reverse(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

static inline uint32_t get_rank_negabinary_representation(uint32_t num_ranks, uint32_t rank, uint port, uint dimensions_num){
    // For the other ports reflect along the axis
    if(port >= dimensions_num){
        rank = -rank + num_ranks;
    }
    binary_to_negabinary(rank);
    uint32_t nba = -1, nbb = -1;
    size_t num_bits = ceil(log2(num_ranks));
    if(rank % 2){
        if(in_range(rank, num_bits)){
            nba = binary_to_negabinary(rank);
        }
        if(in_range(rank - num_ranks, num_bits)){
            nbb = binary_to_negabinary(rank - num_ranks);
        }
    }else{
        if(in_range(-rank, num_bits)){
            nba = binary_to_negabinary(-rank);
        }
        if(in_range(-rank + num_ranks, num_bits)){
            nbb = binary_to_negabinary(-rank + num_ranks);
        }
    }
    assert(nba != -1 || nbb != -1);

    if(nba == -1 && nbb != -1){
        return nbb;
    }else if(nba != -1 && nbb == -1){
        return nba;
    }else{ // Check MSB
        if(nba & (80000000 >> (32 - num_bits))){
            return nba;
        }else{
            return nbb;
        }
    }
}

static inline uint32_t remap_rank(uint32_t num_ranks, uint32_t rank, uint port, uint dimensions_num){
    uint32_t remap_rank = get_rank_negabinary_representation(num_ranks, rank, port, dimensions_num);    
    remap_rank = remap_rank ^ (remap_rank >> 1);
    size_t num_bits = ceil(log2(num_ranks));
    remap_rank = reverse(remap_rank) >> (32 - num_bits);
    return remap_rank;
}

void BineBitmapCalculator::dfs(int* coord_rank, size_t step, size_t num_steps, int* target_rank, uint32_t* remap_rank, bool* found){
    for(size_t i = step; i < num_steps; i++){
        int peer_rank[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        get_peer(coord_rank, i, peer_rank);
        dfs(peer_rank, i + 1, num_steps, target_rank, remap_rank, found);
    }
    if(*found){
        (*remap_rank)--;
    }
    if(memcmp(coord_rank, target_rank, sizeof(int)*dimensions_num) == 0){
        *found = true;
    }
}

BineBitmapCalculator::BineBitmapCalculator(uint rank, uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, uint port, bool remap_blocks):
         scc(dimensions, dimensions_num), dimensions_num(dimensions_num), port(port), remap_blocks(remap_blocks), next_step(0), rank(rank){
    this->size = 1;
    this->num_steps = 0;
    for (uint i = 0; i < dimensions_num; i++) {
        this->dimensions[i] = dimensions[i];
        this->size *= dimensions[i];        
        this->num_steps_per_dim[i] = (int) ceil(log2(dimensions[i]));
        this->num_steps += this->num_steps_per_dim[i];
    }
    if(this->num_steps > LIBBINE_MAX_STEPS){
        assert("Max steps limit must be increased and constants updated.");
    }
    this->dimensions_num = dimensions_num;

    this->peers = (uint*) malloc(sizeof(uint)*this->num_steps);
    bitmap_send_merged = (char**) malloc(sizeof(char*)*this->num_steps);
    bitmap_recv_merged = (char**) malloc(sizeof(char*)*this->num_steps);                                          
    memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);
    current_d = this->port % dimensions_num;
    compute_peers(this->port, rank, false, this->peers, this->dimensions, this->dimensions_num, this->scc);
    this->scc.getCoordFromId(rank, false, coord_mine);

    /* Decide when each block must be sent. */
    if(remap_blocks){
        // Compute the remapped rank
        this->remapped_rank = this->size - 1;
        int coord_rank[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        this->scc.getCoordFromId(0, false, coord_rank);
        int my_rank[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        this->scc.getCoordFromId(rank, false, my_rank);
        bool found = false;
        dfs(coord_rank, 0, this->num_steps, my_rank, &(this->remapped_rank), &found);
        assert(found);
        this->min_block_r = this->min_block_s = 0;
        this->max_block_r = this->max_block_s = this->size;
    }else{
        block_step = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
        for(size_t i = 0; i < this->size; i++){
            block_step[i] = this->num_steps;
        }
        for(size_t i = 0; i < this->num_steps; i++){
            int peer_rank[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
            get_peer(coord_mine, i, peer_rank);
            compute_block_step(peer_rank, i, i, this->num_steps, block_step);
        }
        block_step[rank] = this->num_steps; // Disconnect myself (there might be loops, e.g., for 10 nodes)
    }

    /*************************/
    /* Distances calculation */
    /*************************/
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
    memset(valid_chunk_params, 0, sizeof(valid_chunk_params));
}

BineBitmapCalculator::~BineBitmapCalculator(){
    free(this->peers);

    for(size_t d = 0; d < this->dimensions_num; d++){
        for(size_t i = 0; i < this->num_steps_per_dim[d]; i++){
            free(reference_valid_distances[d][i]);
        }
        free(reference_valid_distances[d]);
        free(num_valid_distances[d]);
    }

    for(size_t s = 0; s < (uint) this->num_steps; s++){
        free(bitmap_send_merged[s]);
        free(bitmap_recv_merged[s]);
    }
    free(bitmap_send_merged);
    free(bitmap_recv_merged);
    if(block_step){
        free(block_step);
    }
}

