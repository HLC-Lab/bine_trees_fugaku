#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>

#include "libswing_common.h"
#ifdef FUGAKU
#include "fugaku/swing_utofu.h"
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
    _ss << " Timer [" << _name << "]: " << duration << " us | Start: " << start << " End: " << end << std::endl;
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


SwingCommon::SwingCommon(MPI_Comm comm, uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, Algo algo, uint num_ports, uint segment_size, size_t prealloc_size, char* prealloc_buf): 
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
    if(this->num_steps > LIBSWING_MAX_STEPS){
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
    this->utofu_descriptor = NULL;
}

SwingCommon::~SwingCommon(){
    // Cleanup utofu resources
    if(this->utofu_descriptor != NULL){
        swing_utofu_teardown(this->utofu_descriptor);
    }
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
void SwingCoordConverter::getCoordFromId(int id, bool virt, int* coord){
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
int SwingCoordConverter::getIdFromCoord(int* coords, bool virt){
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
void SwingCoordConverter::retrieve_coord_mapping(uint rank, bool virt, int* coord){
    int* coordinates = virt ? this->coordinates_virtual : this->coordinates;
    if(coordinates[rank*dimensions_num] == -1){
        getCoordFromId(rank, virt, &(coordinates[rank*dimensions_num]));
    }
    memcpy(coord, &(coordinates[rank*dimensions_num]), sizeof(uint)*dimensions_num);
}

SwingCoordConverter::SwingCoordConverter(uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num): dimensions_num(dimensions_num){
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

SwingCoordConverter::~SwingCoordConverter(){
    free(this->coordinates_virtual);
    free(this->coordinates);
}

    
// Compute the peers of a rank in a torus which start transmitting from a specific port.
// @param port (IN): the port from which the transmission starts
// @param rank (IN): the rank
// @param virt (IN): if true, the virtual coordinates are considered, otherwise the real ones
// @param peers (OUT): the array where the peers are stored (one per step)
static void compute_peers(int port, uint rank, bool virt, uint* peers, uint* dimensions, uint dimensions_num, SwingCoordConverter& scc){
    bool terminated_dimensions_bitmap[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int num_steps_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    uint8_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    memset(next_step_per_dim, 0, sizeof(uint8_t)*LIBSWING_MAX_SUPPORTED_DIMENSIONS);
    
    int num_steps = 0;
    for(size_t i = 0; i < dimensions_num; i++){
        num_steps_per_dim[i] = ceil(log2(dimensions[i]));
        num_steps += num_steps_per_dim[i];
    }

    // Compute default directions
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
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
int SwingCommon::shrink_non_power_of_two(const void *sendbuf, void* recvbuf, void* tempbuf, int count, 
                                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, 
                                         int* idle, int* rank_virtual,
                                         int* first_copy_done){    
    int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    
    int dtsize, rank;
    MPI_Type_size(datatype, &dtsize);    
    MPI_Comm_rank(comm, &rank);    
    
    this->scc.retrieve_coord_mapping(rank, false, coord);

    const void* real_sendbuf;
    void* real_recvbuf;
    const void* real_aggbuf;

    for(size_t i = 0; i < this->dimensions_num; i++){
        // This dimensions is not a power of two, shrink it
        if(!is_power_of_two(dimensions[i])){
            memcpy(coord_peer, coord, sizeof(uint)*this->dimensions_num);
            int extra = dimensions[i] - this->dimensions_virtual[i];
            if(!*first_copy_done){
                real_sendbuf = sendbuf;
                real_recvbuf = recvbuf;
                real_aggbuf = sendbuf;
            }else{
                real_sendbuf = recvbuf;
                real_recvbuf = tempbuf;
                real_aggbuf = tempbuf;
            }
            if(coord[i] >= this->dimensions_virtual[i]){            
                coord_peer[i] = coord[i] - extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                int res = MPI_Send(real_sendbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm);                
                if(res != MPI_SUCCESS){return res;}
                *idle = 1;
                break;
            }else if(coord[i] + extra >= this->dimensions_virtual[i]){
                coord_peer[i] = coord[i] + extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                int res = MPI_Recv(real_recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, NULL);                
                if(res != MPI_SUCCESS){return res;}
                MPI_Reduce_local(real_aggbuf, recvbuf, count, datatype, op);
                *first_copy_done = 1;
            }
        }
    }
    *rank_virtual = this->scc.getIdFromCoord(coord, true);
    return MPI_SUCCESS;
}

int SwingCommon::enlarge_non_power_of_two(void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm){
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
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
                int r = MPI_Recv(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, NULL);
                if(r != MPI_SUCCESS){return r;}
            }else if(coord[d] + extra >= (uint) this->dimensions_virtual[d]){
                coord_peer[d] = coord[d] + extra;
                int peer = this->scc.getIdFromCoord(coord_peer, false);
                DPRINTF("[%d] (Enl) Sending to %d\n", rank, peer);
                int r = MPI_Send(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm);                
                if(r != MPI_SUCCESS){return r;}
            }
        }
    }
    return MPI_SUCCESS;
}

int SwingCommon::swing_coll_l_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
#ifdef FUGAKU
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(this->num_ports) + "/master.profile", "= swing_coll_l_utofu (init)");
    Timer timer("swing_coll_l_utofu (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    
    
    // Temporary buffer (to avoid overwriting sendbuf)
    char* tmpbuf;
    bool free_tmpbuf = false;
    // Since data might be written to a given rank in a different order 
    // (e.g., rank p-1 might write to rank 0 memory before rank 1 writes)
    // we allocate a buffer which is num_steps time larger than the original buffer
    // so that at each step the source rank can write at a different offset.
    size_t tmpbuf_size = count*dtsize*num_steps_virtual;
    if(tmpbuf_size > prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;

        // TODO: This is because we register the tempbuffer only once and reuse it across the calls
        // must be fixed in swing_utofu.cc
        fprintf(stderr, "For the moment, this only works if the preallocd buffer is large enough\n");
        exit(-1); 
    }else{
        tmpbuf = prealloc_buf;
    }

    // The non-idle ranks (i.e., those that lie within a power of two and
    // execute most of the collective):
    // - At the first step (either the shrink or the first step of the collective if we are in the power of two case)
    //   they send the data from sendbuf, and receive in recvbuf. Then they aggregate recvbuf = recvbuf + sendbuf
    // - In all the other steps, they send data from recvbuf, and receive in tempbuf.
    //   Then they aggregate recvbuf = recvbuf + tempbuf.
    // By doing so we can avoid doing a memcpy at the beginning of the execution.

    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    this->scc.retrieve_coord_mapping(this->rank, false, coord);

    int res = MPI_SUCCESS, idle = 0, rank_virtual = rank;        
    int first_copy_done = 0;
    if(!all_p2_dimensions){
        timer.reset("= swing_coll_l_utofu (shrink)");
        res = shrink_non_power_of_two(sendbuf, recvbuf, tmpbuf, count, datatype, op, comm, &idle, &rank_virtual, &first_copy_done);
        if(res != MPI_SUCCESS){return res;}
    }    

    DPRINTF("[%d] Virtual steps: %d Virtual dimensions (%d, %d, %d)\n", rank, num_steps_virtual, this->dimensions_virtual[0], this->dimensions_virtual[1], this->dimensions_virtual[2]);

    if(!idle){                
        // Do the step-by-step communication on the shrunk topology.         
        if(utofu_descriptor == NULL){
            timer.reset("= swing_coll_l_utofu (utofu setup)");        
            uint* peers = (uint*) malloc(sizeof(uint)*num_steps_virtual);
            compute_peers(0, rank_virtual, true, peers, this->dimensions_virtual, this->dimensions_num, this->scc);
            utofu_descriptor = swing_utofu_setup((void*) sendbuf, count*dtsize, recvbuf, count*dtsize, tmpbuf, tmpbuf_size, 
                                                                            this->num_ports, num_steps_virtual, peers);
            timer.reset("= swing_coll_l_utofu (utofu wait)");            
            swing_utofu_setup_wait(utofu_descriptor, num_steps_virtual);
            // Needed to be sure everyone registered the buffers
            timer.reset("= swing_coll_l_utofu (utofu barrier)");
            MPI_Barrier(MPI_COMM_WORLD);
            free(peers);
        }

        timer.reset("= swing_coll_l_utofu (actual sendrecvs)");
        uint partition_size = count / this->num_ports;
        uint remaining = count % this->num_ports;        

#pragma omp parallel for num_threads(this->num_ports) schedule(static, 1) collapse(1)
        for(size_t p = 0; p < this->num_ports; p++){
            // Get the peer
            if(virtual_peers[p] == NULL){
                virtual_peers[p] = (uint*) malloc(sizeof(uint)*num_steps_virtual);
                compute_peers(p, rank_virtual, true, virtual_peers[p], this->dimensions_virtual, this->dimensions_num, this->scc);
            }
            for(size_t step = 0; step < (uint) num_steps_virtual; step++){                 
                // Schedule all the send and recv
                //timer.reset("= swing_coll_l_utofu (sendrecv for step " + std::to_string(step) + ")");
                int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
                int virtual_peer = virtual_peers[p][step];                 
                this->scc.retrieve_coord_mapping(virtual_peer, true, coord_peer); // Get the virtual coordinates of the peer
                int peer = this->scc.getIdFromCoord(coord_peer, false); // Convert the virtual coordinates to the real rank
                DPRINTF("[%d] Sending to %d count %d\n", rank, peer, count);

                // Compute the count and the offset of the piece of buffer that is aggregated on this port
                size_t count_port = partition_size + (p < remaining ? 1 : 0);
                size_t offset_port = 0;
                for(size_t j = 0; j < p; j++){
                    offset_port += partition_size + (j < remaining ? 1 : 0);
                }
                offset_port *= dtsize;

                utofu_stadd_t base_lcl_stadd, base_rmt_stadd;
                const void* base_agg_buf;
                // If I did not copied during the shrink, in the first step I must send
                // from sendbuf, receive in recvbuf, and aggregate recvbuf = recvbuf + sendbuf
                if(step == 0 && !first_copy_done){
                    base_lcl_stadd = utofu_descriptor->port_info[p].lcl_send_stadd;
                    base_rmt_stadd = (*(utofu_descriptor->port_info[p].rmt_info))[peer].recv_stadd;
                    base_agg_buf = sendbuf;
                }else{
                    // Otherwise, I send from recvbuf, receive in tmpbuf, and aggregate recvbuf = recvbuf + tmpbuf
                    base_lcl_stadd = utofu_descriptor->port_info[p].lcl_recv_stadd;
                    base_rmt_stadd = (*(utofu_descriptor->port_info[p].rmt_info))[peer].temp_stadd;
                    base_agg_buf = tmpbuf;
                }

                utofu_stadd_t lcl_addr = base_lcl_stadd + offset_port;
                utofu_stadd_t rmt_addr = base_rmt_stadd + count*dtsize*step + offset_port; // We need to add an additional offset because at each step we write on a different offset (see comment above)

#if !(UTOFU_THREAD_SAFE) // TODO: critical can probably be moved within isend before put
#pragma omp critical // To remove this we should put SWING_UTOFU_VCQ_FLAGS to UTOFU_VCQ_FLAG_EXCLUSIVE. However, this adds crazy overhead when creating/destroying the VCQs
#endif
                {
                    swing_utofu_isend(utofu_descriptor, p, peer, lcl_addr, count_port*dtsize, rmt_addr, step); 
                }
                swing_utofu_wait_recv(utofu_descriptor, p, step, 0);                
                // I need to wait for the send to complete locally before doing the aggregation, otherwise I could modify the buffer that is being sent
                swing_utofu_wait_sends(utofu_descriptor, p, 1); 
                reduce_local((char*) base_agg_buf + count*dtsize*step + offset_port, (char*) recvbuf + offset_port, count_port, datatype, op); // TODO: Try to replace again with MPI_Reduce_local ?                
                DPRINTF("[%d] Step %d completed\n", rank, step);    
            }        
        }
    }else{
        if(utofu_descriptor == NULL){
            MPI_Barrier(MPI_COMM_WORLD); // For the uTofu correctness
            utofu_descriptor = (swing_utofu_comm_descriptor*) 0xBADF00D;
        }
    }

    if(!all_p2_dimensions){
        timer.reset("= swing_coll_l_utofu (enlarge)");
        DPRINTF("[%d] Propagating data to extra nodes\n", rank);
        res = enlarge_non_power_of_two(recvbuf, count, datatype, comm);
        if(res != MPI_SUCCESS){return res;}
        DPRINTF("[%d] Data propagated\n", rank);
    }    
    timer.reset("= swing_coll_l_utofu (writing profile data to file)");
    if(free_tmpbuf){
        free(tmpbuf);
    }
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif
}

// This works the following way:
// 1. Shrink the topology by sending ranks that are outside the power of two boundary, to ranks within the boundary
// 2. Run allreduce on a configuration where each dimension has a power of two size
// 3. Enlarge the topology by sending data to ranks outside the boundary.
//
// TODO: Implement a multiport version of the shrink/enlarge? 
int SwingCommon::swing_coll_l_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(this->num_ports) + "/master.profile", "= swing_coll_l_mpi (init)");
    Timer timer("swing_coll_l_mpi (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    
    
    char* tmpbuf;
    bool free_tmpbuf = false;
    size_t tmpbuf_size = count*dtsize;
    if(tmpbuf_size > prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
    }else{
        tmpbuf = prealloc_buf;
    }


    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    this->scc.retrieve_coord_mapping(this->rank, false, coord);

    // The non-idle ranks (i.e., those that lie within a power of two and
    // execute most of the collective):
    // - At the first step (either the shrink or the first step of the collective if we are in the power of two case)
    //   they send the data from sendbuf, and receive in recvbuf. Then they aggregate recvbuf = recvbuf + sendbuf
    // - In all the other steps, they send data from recvbuf, and receive in tempbuf.
    //   Then they aggregate recvbuf = recvbuf + tempbuf.
    // By doing so we can avoid doing a memcpy at the beginning of the execution.

    int res = MPI_SUCCESS, idle = 0, rank_virtual = rank; 
    int first_copy_done = 0;       
    if(!all_p2_dimensions){
        timer.reset("= swing_coll_l_mpi (shrink)");
        res = shrink_non_power_of_two(sendbuf, recvbuf, tmpbuf, count, datatype, op, comm, &idle, &rank_virtual, &first_copy_done);
        if(res != MPI_SUCCESS){return res;}
    }    

    DPRINTF("[%d] Virtual steps: %d Virtual dimensions (%d, %d, %d)\n", rank, num_steps_virtual, this->dimensions_virtual[0], this->dimensions_virtual[1], this->dimensions_virtual[2]);

    if(!idle){                
        // Do the step-by-step communication on the shrunk topology.         
        timer.reset("= swing_coll_l_mpi (actual sendrecvs)");
        uint partition_size = count / this->num_ports;
        uint remaining = count % this->num_ports;        
        for(size_t step = 0; step < (uint) num_steps_virtual; step++){                 
            // Schedule all the send and recv
            //timer.reset("= swing_coll_l_mpi (sendrecv for step " + std::to_string(step) + ")");
            uint count_so_far = 0;
            MPI_Request requests_s[LIBSWING_MAX_SUPPORTED_PORTS];
            MPI_Request requests_r[LIBSWING_MAX_SUPPORTED_PORTS];

            const void *base_send_buf;
            void *base_recv_buf;
            const void *base_agg_buf;
            // If I did not do the shrink, in the first step I must send
            // from sendbuf, receive in recvbuf, and aggregate recvbuf = recvbuf + sendbuf
            if((step == 0) && !first_copy_done){
                base_send_buf = sendbuf;
                base_recv_buf = recvbuf;
                base_agg_buf = sendbuf;
            }else{
                // Otherwise, I send from recvbuf, receive in tmpbuf, and aggregate recvbuf = recvbuf + tmpbuf
                base_send_buf = recvbuf;
                base_recv_buf = tmpbuf;
                base_agg_buf = tmpbuf;
            }

            for(size_t p = 0; p < this->num_ports; p++){
                // Get the peer
                if(virtual_peers[p] == NULL){
                    virtual_peers[p] = (uint*) malloc(sizeof(uint)*num_steps_virtual);
                    compute_peers(p, rank_virtual, true, virtual_peers[p], this->dimensions_virtual, this->dimensions_num, this->scc);
                }
                int coord_peer[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
                int virtual_peer = virtual_peers[p][step];                 
                this->scc.retrieve_coord_mapping(virtual_peer, true, coord_peer); // Get the virtual coordinates of the peer
                int peer = this->scc.getIdFromCoord(coord_peer, false); // Convert the virtual coordinates to the real rank
                DPRINTF("[%d] Sending to %d count %d\n", rank, peer, count);

                size_t count_port = partition_size + (p < remaining ? 1 : 0);
                size_t offset_port = count_so_far * dtsize;
                count_so_far += count_port;

                res = MPI_Isend(((char*) base_send_buf) + offset_port, count_port, datatype, peer, TAG_SWING_ALLREDUCE, comm, &(requests_s[p]));
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on isend\n", rank); return res;}
                res = MPI_Irecv(((char*) base_recv_buf) + offset_port, count_port, datatype, peer, TAG_SWING_ALLREDUCE, comm, &(requests_r[p]));                    
                if(res != MPI_SUCCESS){DPRINTF("[%d] Error on irecv\n", rank); return res;}                
            }
            MPI_Waitall(this->num_ports, requests_s, MPI_STATUSES_IGNORE);
            MPI_Waitall(this->num_ports, requests_r, MPI_STATUSES_IGNORE);
            res = MPI_Reduce_local((char*) base_agg_buf, recvbuf, count, datatype, op); 
            if(res != MPI_SUCCESS){DPRINTF("[%d] Error on reduce_local\n", rank); return res;}
            DPRINTF("[%d] Step %d completed\n", rank, step);            
        }
    }

    if(!all_p2_dimensions){
        timer.reset("= swing_coll_l_mpi (enlarge)");
        DPRINTF("[%d] Propagating data to extra nodes\n", rank);
        res = enlarge_non_power_of_two(recvbuf, count, datatype, comm);
        if(res != MPI_SUCCESS){return res;}
        DPRINTF("[%d] Data propagated\n", rank);
    }    
    timer.reset("= swing_coll_l_mpi (writing profile data to file)");
    if(free_tmpbuf){
        free(tmpbuf);
    }
    return res;
}

int SwingCommon::swing_coll_l(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    if(algo == ALGO_SWING_L){
        return swing_coll_l_mpi(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_SWING_L_UTOFU){
#ifdef FUGAKU
        return swing_coll_l_utofu(sendbuf, recvbuf, count, datatype, op, comm);
#else
        fprintf(stderr, "uTofu can only be used on Fugaku.\n");
        exit(-1);
#endif
    }else{
        assert("Unknown algorithm");
    }
    return -1;
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

int SwingCommon::swing_coll_step_b(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                             
                                   MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                   CollType coll_type){    
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
    for(size_t port = 0; port < this->num_ports; port++){
        if(step == 0){
            sbc[port]->compute_bitmaps(0, coll_type);
        }
        uint peer = sbc[port]->get_peer(step, coll_type);
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", this->rank, step, port, this->num_ports, peer);                

        if(coll_type == SWING_REDUCE_SCATTER){
            tag = TAG_SWING_REDUCESCATTER + port;
        }else{
            tag = TAG_SWING_ALLGATHER + port;
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

int SwingCommon::swing_coll_step_coalesce(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                                 
                                          MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                          CollType coll_type){
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

    for(size_t port = 0; port < this->num_ports; port++){
        if(step == 0){
            sbc[port]->compute_bitmaps(0, coll_type);
        }
        uint peer = sbc[port]->get_peer(step, coll_type);
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


int SwingCommon::swing_coll_step_cont(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                                 
                                MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                CollType coll_type){
    MPI_Request requests_s[LIBSWING_MAX_SUPPORTED_PORTS];
    MPI_Request requests_r[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(requests_s, 0, sizeof(requests_s));
    memset(requests_r, 0, sizeof(requests_r));
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

        if(coll_type == SWING_REDUCE_SCATTER){
            tag = TAG_SWING_REDUCESCATTER + port;
        }else{
            tag = TAG_SWING_ALLGATHER + port;
        }

        ChunkParams cp;
        sbc[port]->get_chunk_params(step, coll_type, &cp);

        // Sendrecv 
        res = MPI_Isend(((char*) buf) + cp.send_offset, cp.send_count, sendtype, peer, tag, comm, &(requests_s[port]));
        if(res != MPI_SUCCESS){return res;}
        res = MPI_Irecv(((char*) tmpbuf) + cp.recv_offset, cp.recv_count, recvtype, peer, tag, comm, &(requests_r[port]));
        if(res != MPI_SUCCESS){return res;}
    }
    if(step < this->num_steps - 1){
        for(size_t port = 0; port < this->num_ports; port++){
            sbc[port]->compute_bitmaps(step + 1, coll_type);
        }
    }

    // Wait for all the recvs to be over
    if(coll_type == SWING_REDUCE_SCATTER){
//#define ALWAYS_WAITALL
#ifdef ALWAYS_WAITALL     
        res = MPI_Waitall(this->num_ports, requests_r, MPI_STATUSES_IGNORE);
#endif
      
        int port;
        for(size_t i = 0; i < this->num_ports; i++){
#ifndef ALWAYS_WAITALL	  
            res = MPI_Waitany(this->num_ports, requests_r, &port, MPI_STATUS_IGNORE);	    
            if(res != MPI_SUCCESS){return res;}
#else
            port = i;
#endif	    
            ChunkParams cp;
            sbc[port]->get_chunk_params(step, coll_type, &cp);
            void* tmpbuf_block = (void*) (((char*) tmpbuf) + cp.recv_offset);
            void* buf_block = (void*) (((char*) buf) + cp.recv_offset);  
            MPI_Reduce_local(tmpbuf_block, buf_block, cp.recv_count, sendtype, op); 
        }
    }else{
        res = MPI_Waitall(this->num_ports, requests_r, MPI_STATUSES_IGNORE);
        if(res != MPI_SUCCESS){return res;}            
    }

    // Wait for all the sends to be over    
    res = MPI_Waitall(this->num_ports, requests_s, MPI_STATUSES_IGNORE);
    return res;
}

int SwingCommon::swing_coll_step(void *buf, void* tmpbuf, BlockInfo** blocks_info, size_t step,                                 
                                MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                CollType coll_type){
    if(algo == ALGO_SWING_B){
        return swing_coll_step_b(buf, tmpbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type);
    }else if(algo == ALGO_SWING_B_COALESCE){
        return swing_coll_step_coalesce(buf, tmpbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type);
    }else if(algo == ALGO_SWING_B_CONT){
        if(is_power_of_two(this->size)){
            return swing_coll_step_cont(buf, tmpbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type);
        }else{
            return swing_coll_step_coalesce(buf, tmpbuf, blocks_info, step, op, comm, sendtype, recvtype, coll_type);
        }
    }else{
        assert("Unknown algo" == 0);
        return MPI_ERR_OTHER;
    }
}

#ifdef FUGAKU
int SwingCommon::swing_coll_step_utofu(size_t port, swing_utofu_comm_descriptor* utofu_descriptor, const void* sendbuf, void *recvbuf, void* tempbuf, size_t tmpbuf_size, const BlockInfo *const *const blocks_info, size_t step, 
                                       MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                       CollType coll_type, bool is_first_coll){                                        
    size_t offsets_s, counts_s;
    size_t offsets_r, counts_r;
    
    Timer timer("== swing_coll_step_utofu (indexes calc)");
    ChunkParams cp;
    sbc[port]->get_chunk_params(step, coll_type, &cp);

    offsets_s = cp.send_offset;
    counts_s = cp.send_count;
    offsets_r = cp.recv_offset;
    counts_r = cp.recv_count;

    timer.reset("== swing_coll_step_utofu (misc)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);  
    size_t max_count;
    if(coll_type == SWING_REDUCE_SCATTER && this->segment_size){
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
    if(coll_type == SWING_REDUCE_SCATTER && step != 0){ // For first step we do not need to do it (we write directly in user_recvbuf rather than tmpbuf)
        utofu_offset_r = (tmpbuf_size / this->num_ports) * port; // Start from the beginning of the buffer
        for(size_t i = 0; i < step; i++){
            utofu_offset_r += (tmpbuf_size / this->num_ports) / pow(2, (i + 1)); // TODO: Does not work if number of ranks is not a power of 2
        }
        utofu_offset_r_start = utofu_offset_r;
    }

    // TODO: At most 256 segments are supported (edata is 8 bits)
    // TODO: Just scale the number of segments so that it is never >= 256
    assert(counts_s / max_count < 256);
    assert(counts_r / max_count < 256);

    timer.reset("== swing_coll_step_utofu (sends)");
    // We first enqueue all the send. Then, we receive and aggregate
    // Aggregation and reception of next block should be overlapped
    // Issue isends for all the blocks
    uint64_t edata = 0;
    if(is_first_coll){
        edata = step; 
    }else{
        edata = this->num_steps + step; // Second collective (e.g., allgather after reduce_scatter)
    }
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
            if(coll_type == SWING_REDUCE_SCATTER){
                if(step == 0){
                    // To avoid memcpy from sendbuf to recvbuf in the first step I need to (in the first step):
                    // - Send from local sendbuf to remote recvbuf, then aggregate from remote sendbuf to remote recvbuf
                    lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd + offset;
                    rmt_addr = (*(utofu_descriptor->port_info[port].rmt_info))[peer].recv_stadd + utofu_offset_r;
                }else{
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd + offset;
                    rmt_addr = (*(utofu_descriptor->port_info[port].rmt_info))[peer].temp_stadd + utofu_offset_r;
                }
            }else if(coll_type == SWING_ALLGATHER){
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
            #pragma omp critical // To remove this we should put SWING_UTOFU_VCQ_FLAGS to UTOFU_VCQ_FLAG_EXCLUSIVE. However, this adds crazy overhead when creating/destroying the VCQs
#endif
            swing_utofu_isend(utofu_descriptor, port, peer, lcl_addr, bytes_to_send, rmt_addr, edata); 

            // Update for the next segment
            offset += bytes_to_send;
            utofu_offset_r += bytes_to_send;
            remaining -= count;
            ++issued_sends;            
        }
    }

    // Here I can overlap computation to the reception of the data

    timer.reset("== swing_coll_step_utofu (compute bitmaps i)");
    // We issued the sends, while we wait for transmission we compute the bitmaps for the next step  
    if(step < this->num_steps - 1){
        sbc[port]->compute_bitmaps(step + 1, coll_type);
    }
    timer.reset("== swing_coll_step_utofu (recv + aggregate (init))");

    //double start = MPI_Wtime();
    // Receive and aggregate
    uint64_t expected_step = 0;
    if(is_first_coll){
        expected_step = step; 
    }else{
        expected_step = this->num_steps + step; // Second collective (e.g., allgather after reduce_scatter)
    }
    offset = 0;
    size_t expected_segment = 0;
    if(counts_r){
        size_t remaining = counts_r;
        size_t bytes_to_recv = 0;
        // Segment the transmission
        timer.reset("== swing_coll_step_utofu (recv/aggr loop)");
        while(remaining){
            count = remaining < max_count ? remaining : max_count;
            bytes_to_recv = count*dtsize;
            DPRINTF("[%d] Receiving %d bytes at step %d (coll %d)\n", this->rank, bytes_to_recv, step, coll_type);
            
            size_t recvbuf_offset = offsets_r + offset;
            char* recvbuf_block = (char*) recvbuf + recvbuf_offset;

            if(coll_type == SWING_REDUCE_SCATTER){
                if(step == 0){
                    // In the first step I receive in recvbuf and I aggregate from sendbuf and recvbuf in recvbuf (at same offsets)
                    size_t sendbuf_offset = recvbuf_offset;
                    swing_utofu_wait_recv(utofu_descriptor, port, expected_step, expected_segment);
                    reduce_local((char*) sendbuf + sendbuf_offset, recvbuf_block, count, sendtype, op);
                }else{
                    // In the other steps I receive in tmpbuf and I aggregate from tmpbuf and recvbuf in recvbuf (for tmbuf we use adjusted offset)
                    size_t tempbuf_offset = utofu_offset_r_start + offset;
                    swing_utofu_wait_recv(utofu_descriptor, port, expected_step, expected_segment);                    
                    reduce_local((char*) tempbuf + tempbuf_offset, recvbuf_block, count, sendtype, op); // TODO: Try to replace again with MPI_Reduce_local ?
                }
            }else{
                swing_utofu_wait_recv(utofu_descriptor, port, expected_step, expected_segment);
            }

            // Update values for next segment
            offset += bytes_to_recv;
            remaining -= count;
            ++issued_recvs;
            ++expected_segment;
        }            
    }
    /*
    if(omp_get_thread_num() == 0){
        std::cout << "recv+aggr: " << (MPI_Wtime() - start)*1000000.0 << std::endl;
    }
    */
    timer.reset("== swing_coll_step_utofu (wait isends)");
    DPRINTF("[%d] Issued %d sends and %d recvs\n", rank, issued_sends, issued_recvs);
    // Wait for send completion
    if(counts_s){
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
    }
    timer.reset("== swing_coll_step_utofu (profile writing)");
    DPRINTF("[%d] Sends completed\n", rank);
    return MPI_SUCCESS;
}
#else
int SwingCommon::swing_coll_step_utofu(size_t port, swing_utofu_comm_descriptor* utofu_descriptor, const void* sendbuf, void *recvbuf, void* tempbuf, size_t tmpbuf_size, const BlockInfo *const *const blocks_info, size_t step, 
                                       MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,
                                       CollType coll_type, bool is_first_coll){
    fprintf(stderr, "uTofu can only be used on Fugaku.\n");
    exit(-1);
}
#endif

int SwingBitmapCalculator::get_distance_sign(size_t rank, size_t port){
    int multiplier = 1;
    if(is_odd(rank)){ // Invert sign if odd rank
        multiplier *= -1;
    }
    if(port >= this->dimensions_num){ // Invert sign if mirrored collective
        multiplier *= -1;     
    }
    return multiplier;
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

void SwingBitmapCalculator::get_peer(int* coord_rank, size_t step, int* coord_peer){
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

void SwingBitmapCalculator::compute_bitmaps(uint step, CollType coll_type){
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);
    while(block_step >= this->next_step){
        compute_next_bitmaps();
    }
}

void SwingBitmapCalculator::compute_block_step(int* coord_rank, size_t starting_step, size_t step, size_t num_steps, uint32_t* block_step){
    if(step < num_steps){
        for(size_t i = step + 1; i < num_steps; i++){
            int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
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

void SwingBitmapCalculator::compute_next_bitmaps(){
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
        
        
        chunk_params_per_step[this->next_step].send_offset = blocks_info[port][this->min_block_s].offset;
        chunk_params_per_step[this->next_step].send_count = 0;
        for(size_t i = this->min_block_s; i < this->max_block_s; i++){
            chunk_params_per_step[this->next_step].send_count += blocks_info[port][i].count;        
        }        

        chunk_params_per_step[this->next_step].recv_offset = blocks_info[port][this->min_block_r].offset;
        chunk_params_per_step[this->next_step].recv_count = 0;
        for(size_t i = this->min_block_r; i < this->max_block_r; i++){
            chunk_params_per_step[this->next_step].recv_count += blocks_info[port][i].count;        
        }

        DPRINTF("[%d] Chunk Params %d %d %d %d\n", rank, chunk_params_per_step[this->next_step].send_count, chunk_params_per_step[this->next_step].send_offset, chunk_params_per_step[this->next_step].recv_count, chunk_params_per_step[this->next_step].recv_offset);
    }else{
        for(size_t i = 0; i < this->size; i++){
            // I am gonna send the blocks that I need to send at this step...
            if(block_step[i] == this->next_step){
                bitmap_send_merged[this->next_step][i] = 1;
            }
        }
#if 1
        // To know what to receive, I must know what my peer is going to send
        uint32_t* peer_block_step = (uint32_t*) malloc(sizeof(uint32_t)*this->size);
        for(size_t i = 0; i < this->size; i++){
            peer_block_step[i] = this->num_steps;
        }
        uint peer = get_peer(this->next_step, SWING_REDUCE_SCATTER);
        int peer_coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        scc.getCoordFromId(peer, false, peer_coord);

        for(size_t j = this->next_step; j < this->num_steps; j++){
            int peer_peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
            get_peer(peer_coord, j, peer_peer_rank);
            compute_block_step(peer_peer_rank, j, j, this->num_steps, peer_block_step);
        }

        peer_block_step[peer] = this->num_steps; // Disconnect myself (there might be loops, e.g., for 10 nodes)
        for(size_t i = 0; i < this->size; i++){
            if(peer_block_step[i] == this->next_step){
                bitmap_recv_merged[this->next_step][i] = 1;
            }            
        }
        free(peer_block_step);
#else
        // TODO: The following only work for 1D. For multidimensional would be more complicated/messy

        // For the way Swing works, communications between nodes never happen "in diagonal", i.e.,
        // the coordinates of two communicating nodes differ at most for the coordinate of one and
        // only one dimension. So we just need to adjust that dimension to know what to send/receive.
        for(size_t i = 0; i < this->size; i++){
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
                size_t block_sent_by_1 = mod((i - (get_peer(this->next_step, SWING_REDUCE_SCATTER) - 1)), this->size);
                size_t block_sent_by_0 = mod((1 - block_sent_by_1), this->size);
                block_sent_by_peer = mod((block_sent_by_0 + rank), this->size);
            }else{
                // I receive block i if my peer (r) sends block i
                // r (even) sends block i, iff 0 sends block (i - r) % num_ranks
                // 0 sends block j, iff 1 sends block (1-j) % num_ranks
                // 1 sends block k, iff I send block (k + rank - 1) % num_ranks                
                size_t block_sent_by_0 = mod((i - get_peer(this->next_step, SWING_REDUCE_SCATTER)), this->size);
                size_t block_sent_by_1 = mod((1 - block_sent_by_0), this->size);
                block_sent_by_peer = mod((block_sent_by_1 + rank - 1), this->size);
            }

            if(block_step[block_sent_by_peer] == this->next_step){
                bitmap_recv_merged[this->next_step][i] = 1;
            }
        }
#endif
    }
    ++this->next_step;
}

/**
 * A generic collective operation sending/transmitting blocks rather than the entire buffer.
 * @param blocks_info: For each chunk, port, and block, the count and offset of the block
*/
int SwingCommon::swing_coll_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type){    
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(this->num_ports) + "/master.profile", "= swing_coll_b (init)");
    Timer timer("= swing_coll_b (init)");
    int res = MPI_SUCCESS;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);  

    timer.reset("= swing_coll_b (tmpbuf alloc)");
    // Receive into tmpbuf and aggregate into recvbuf
    char* tmpbuf = NULL;
    size_t tmpbuf_size = 0;
    bool free_tmpbuf = false;
    if(coll_type == SWING_REDUCE_SCATTER || coll_type == SWING_ALLREDUCE){        
        if(algo == ALGO_SWING_B_UTOFU){
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
            posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
            free_tmpbuf = true;
        }else{
            tmpbuf = prealloc_buf;
        }
    }   

    timer.reset("= swing_coll_b (sbc alloc)");
    // Create bitmap calculators if not already created
    for(size_t p = 0; p < this->num_ports; p++){
        if(this->sbc[p] == NULL){
            this->sbc[p] = new SwingBitmapCalculator(this->rank, this->dimensions, this->dimensions_num, p, blocks_info, (algo == ALGO_SWING_B_UTOFU || algo == ALGO_SWING_B_CONT) && is_power_of_two(this->size));
        }
    }
    
    timer.reset("= swing_coll_b (coll to run)");
    size_t collectives_to_run_num = 0;
    CollType collectives_to_run[LIBSWING_MAX_COLLECTIVE_SEQUENCE]; 
    void *buf_s[LIBSWING_MAX_COLLECTIVE_SEQUENCE];
    void *buf_r[LIBSWING_MAX_COLLECTIVE_SEQUENCE];
    switch(coll_type){
        case SWING_ALLREDUCE:{
            collectives_to_run[0] = SWING_REDUCE_SCATTER;            
            buf_s[0] = recvbuf;
            buf_r[0] = tmpbuf;
            collectives_to_run[1] = SWING_ALLGATHER;
            buf_s[1] = recvbuf;
            buf_r[1] = recvbuf;
            collectives_to_run_num = 2;
            break;
        }
        case SWING_REDUCE_SCATTER:{
            collectives_to_run[0] = SWING_REDUCE_SCATTER;
            collectives_to_run_num = 1;
            buf_s[0] = recvbuf;
            buf_r[0] = tmpbuf;
            break;
        }
        case SWING_ALLGATHER:{
            collectives_to_run[0] = SWING_ALLGATHER;
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
    if(algo == ALGO_SWING_B_UTOFU){     
#ifdef FUGAKU   
        // Setup all the communications        
        // TODO: Cache also utofu descriptors to avoid exchanging pointers at each allreduce?
        if(utofu_descriptor == NULL){
            timer.reset("= swing_coll_b (utofu setup)");        

            utofu_descriptor = swing_utofu_setup((void*) sendbuf, count*dtsize, recvbuf, count*dtsize, tmpbuf, tmpbuf_size, 
                                                                          this->num_ports, this->num_steps, this->sbc[0]->get_peers());
        
            timer.reset("= swing_coll_b (utofu wait)");            
            swing_utofu_setup_wait(utofu_descriptor, this->num_steps);
        
            // Needed to be sure everyone registered the buffers
            timer.reset("= swing_coll_b (utofu barrier)");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        timer.reset("= swing_coll_b (utofu main loop)");

#pragma omp parallel for num_threads(this->num_ports) schedule(static, 1) collapse(1)
        for(size_t port = 0; port < this->num_ports; port++){
            /*
            int thread_num = omp_get_thread_num();
            int cpu_num = sched_getcpu();
            printf("Thread %3d is running on CPU %3d\n", thread_num, cpu_num);
            */
            for(size_t collective = 0; collective < collectives_to_run_num; collective++){        
                for(size_t step = 0; step < this->num_steps; step++){       
                    int r = swing_coll_step_utofu(port, utofu_descriptor, sendbuf, recvbuf, tmpbuf, tmpbuf_size, blocks_info, step, 
                                                op, comm, datatype, datatype, 
                                                collectives_to_run[collective], collective == 0);                                                    
                    assert(r == MPI_SUCCESS);
                }
            }
        }
#else
        fprintf(stderr, "uTofu can only be used on Fugaku.\n");
        exit(-1);
#endif
    }else{
        if(coll_type == SWING_REDUCE_SCATTER || coll_type == SWING_ALLREDUCE){
            size_t total_size_bytes = count*dtsize;
            memcpy(recvbuf, sendbuf, total_size_bytes);  // TODO: If we are going to compare with the non utofu version we should remove this memcpy from here as well.
        }
        for(size_t collective = 0; collective < collectives_to_run_num; collective++){        
            for(size_t step = 0; step < this->num_steps; step++){                        
                DPRINTF("[%d] Bitmap computed for step %d\n", this->rank, step);
                res = swing_coll_step(buf_s[collective], buf_r[collective], blocks_info, step,                                 
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

    timer.reset("= swing_coll_b (profile writing)");
    return res;
}

uint SwingBitmapCalculator::get_peer(uint step, CollType coll_type){
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);            
    return (this->peers)[block_step];
}

bool SwingBitmapCalculator::block_must_be_sent(uint step, CollType coll_type, uint block_id){
    compute_bitmaps(step, coll_type); // This is going to be a nop if compute_bitmaps was already called before
    if(coll_type == SWING_REDUCE_SCATTER){
        return bitmap_send_merged[step][block_id];
    }else{                
        return bitmap_recv_merged[(this->num_steps - step - 1)][block_id];
    }
}

bool SwingBitmapCalculator::block_must_be_recvd(uint step, CollType coll_type, uint block_id){
    compute_bitmaps(step, coll_type); // This is going to be a nop if compute_bitmaps was already called before
    if(coll_type == SWING_REDUCE_SCATTER){
        return bitmap_recv_merged[step][block_id];
    }else{                
        return bitmap_send_merged[(this->num_steps - step - 1)][block_id];
    }
}

void SwingBitmapCalculator::get_chunk_params(uint step, CollType coll_type, ChunkParams* chunk_params){
    compute_bitmaps(step, coll_type); // This is going to be a nop if compute_bitmaps was already called before
    if(coll_type == SWING_REDUCE_SCATTER){
        *chunk_params = this->chunk_params_per_step[step];
    }else{                
        chunk_params->recv_count = this->chunk_params_per_step[this->num_steps - step - 1].send_count;
        chunk_params->recv_offset = this->chunk_params_per_step[this->num_steps - step - 1].send_offset;
        chunk_params->send_count = this->chunk_params_per_step[this->num_steps - step - 1].recv_count;
        chunk_params->send_offset = this->chunk_params_per_step[this->num_steps - step - 1].recv_offset;
    }
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
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
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

// Only works for 1D
static inline uint32_t remap_rank(uint32_t num_ranks, uint32_t rank, uint port, uint dimensions_num){
    uint32_t remap_rank = get_rank_negabinary_representation(num_ranks, rank, port, dimensions_num);    
    remap_rank = remap_rank ^ (remap_rank >> 1);
    size_t num_bits = ceil(log2(num_ranks));
    remap_rank = reverse(remap_rank) >> (32 - num_bits);
    return remap_rank;
}

void SwingBitmapCalculator::dfs(int* coord_rank, size_t step, size_t num_steps, int* target_rank, uint32_t* remap_rank, bool* found){
    for(size_t i = step; i < num_steps; i++){
        int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
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

SwingBitmapCalculator::SwingBitmapCalculator(uint rank, uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, uint port, BlockInfo** blocks_info, bool remap_blocks):
         scc(dimensions, dimensions_num), dimensions_num(dimensions_num), port(port), blocks_info(blocks_info), remap_blocks(remap_blocks), next_step(0), rank(rank){
    this->size = 1;
    this->num_steps = 0;
    for (uint i = 0; i < dimensions_num; i++) {
        this->dimensions[i] = dimensions[i];
        this->size *= dimensions[i];        
        this->num_steps_per_dim[i] = (int) ceil(log2(dimensions[i]));
        this->num_steps += this->num_steps_per_dim[i];
    }
    if(this->num_steps > LIBSWING_MAX_STEPS){
        assert("Max steps limit must be increased and constants updated.");
    }
    this->dimensions_num = dimensions_num;

    this->peers = (uint*) malloc(sizeof(uint)*this->num_steps);
    bitmap_send_merged = (char**) malloc(sizeof(char*)*this->num_steps);
    bitmap_recv_merged = (char**) malloc(sizeof(char*)*this->num_steps);                                          
    compute_peers(this->port, rank, false, this->peers, this->dimensions, this->dimensions_num, this->scc);
    this->scc.getCoordFromId(rank, false, coord_mine);

    chunk_params_per_step = (ChunkParams*) malloc(sizeof(ChunkParams)*this->num_steps);

    /* Decide when each block must be sent. */
    if(remap_blocks){
        // Compute the remapped rank
        this->remapped_rank = this->size - 1;
        int coord_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        this->scc.getCoordFromId(0, false, coord_rank);
        int my_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
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
            int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
            get_peer(coord_mine, i, peer_rank);
            compute_block_step(peer_rank, i, i, this->num_steps, block_step);
        }
        block_step[rank] = this->num_steps; // Disconnect myself (there might be loops, e.g., for 10 nodes)
    }
}

SwingBitmapCalculator::~SwingBitmapCalculator(){
    free(this->peers);

    for(size_t s = 0; s < (uint) this->num_steps; s++){
        free(bitmap_send_merged[s]);
        free(bitmap_recv_merged[s]);
    }
    free(bitmap_send_merged);
    free(bitmap_recv_merged);
    if(block_step){
        free(block_step);
    }
    free(chunk_params_per_step);
}

