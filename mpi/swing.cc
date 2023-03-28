#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <vector>


#define MAX_SUPPORTED_DIMENSIONS 8 // We support up to 8D torus

#define TAG_SWING_REDUCESCATTER ((0x1 << 15) - 1)
#define TAG_SWING_ALLGATHER ((0x1 << 15) - 2)

//#define PERF_DEBUGGING // This breaks the correctness of the algorithm, should only be defined for debugging purposes

//#define DEBUG

#ifdef DEBUG
#define DPRINTF(...) printf(__VA_ARGS__)
#else
#define DPRINTF(...) 
#endif

typedef enum{
    SWING_REDUCE_SCATTER = 0,
    SWING_ALLGATHER,
    SWING_ALLREDUCE
}CollType;

typedef enum{
    ALGO_DEFAULT = 0,
    ALGO_SWING,
    ALGO_RING,
    ALGO_RECDOUB
}Algo;

typedef enum{
    SENDRECV_DT = 0, // Datatypes
    SENDRECV_BBB, // Block-by-block
    SENDRECV_CONT, // Contiguous
    SENDRECV_IDEAL, // Ideal case (just for performance debugging reasons, produces wrong results, never use it in practice)
}SendRecv;

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allgather = 0, disable_allreduce = 0, 
                    dimensions_num = 1, latency_optimal_threshold = 1024, force_env_reload = 1, env_read = 0;
static Algo algo = ALGO_SWING;
static SendRecv srtype = SENDRECV_DT;
static uint dimensions[MAX_SUPPORTED_DIMENSIONS];

void read_env(MPI_Comm comm){
    char* env_str = getenv("LIBSWING_FORCE_ENV_RELOAD");
    if(env_str){
        force_env_reload = atoi(env_str);
    }

    if(!env_read || force_env_reload){
        env_read = 1;
        env_str = getenv("LIBSWING_DISABLE_REDUCESCATTER");
        if(env_str){
            disable_reducescatter = atoi(env_str);
        }

        env_str = getenv("LIBSWING_DISABLE_ALLGATHERV");
        if(env_str){
            disable_allgatherv = atoi(env_str);
        }

        env_str = getenv("LIBSWING_DISABLE_ALLGATHER");
        if(env_str){
            disable_allgather = atoi(env_str);
        }

        env_str = getenv("LIBSWING_DISABLE_ALLREDUCE");
        if(env_str){
            disable_allreduce = atoi(env_str);
        }

        env_str = getenv("LIBSWING_LATENCY_OPTIMAL_THRESHOLD");
        if(env_str){
            latency_optimal_threshold = atoi(env_str);
        }

        env_str = getenv("LIBSWING_ALGO");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                algo = ALGO_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                algo = ALGO_SWING;
            }else if(strcmp(env_str, "RING") == 0){
                algo = ALGO_RING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                algo = ALGO_RECDOUB;
            }else{
                fprintf(stderr, "Unknown LIBSWING_ALGO\n");
                exit(-1);
            }
        }

        env_str = getenv("LIBSWING_SENDRECV_TYPE");
        if(env_str){
            if(strcmp(env_str, "DT") == 0){
                srtype = SENDRECV_DT;
            }else if(strcmp(env_str, "BBB") == 0){
                srtype = SENDRECV_BBB;
            }else if(strcmp(env_str, "CONT") == 0){
                srtype = SENDRECV_CONT;
            }else if(strcmp(env_str, "IDEAL") == 0){
                srtype = SENDRECV_IDEAL;
            }else{
                fprintf(stderr, "Unknown LIBSWING_SENDRECV_TYPE\n");
                exit(-1);
            }
        }

        env_str = getenv("LIBSWING_DIMENSIONS");
        if(env_str){
            char* copy = (char*) malloc(sizeof(char)*(strlen(env_str) + 1));
            strcpy(copy, env_str);
            const char *delim = ",";
            char* rest = NULL;
            char *ptr = strtok_r(copy, delim, &rest);
            uint i = 0;
            while(ptr != NULL){
                dimensions[i] = atoi(ptr);
                ptr = strtok_r(NULL, delim, &rest);
                ++i;
            } 
            free(copy);
            dimensions_num = i;       
        }else{
            int size;
            MPI_Comm_size(comm, &size);
            dimensions[0] = size;
        }

#ifdef DEBUG
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0){
            printf("Libswing called. Environment:\n");
            printf("------------------------------------\n");
            printf("LIBSWING_DISABLE_REDUCESCATTER: %d\n", disable_reducescatter);
            printf("LIBSWING_DISABLE_ALLGATHERV: %d\n", disable_allgatherv);
            printf("LIBSWING_DISABLE_ALLGATHER: %d\n", disable_allgather);
            printf("LIBSWING_DISABLE_ALLREDUCE: %d\n", disable_allreduce);
            printf("LIBSWING_LATENCY_OPTIMAL_THRESHOLD: %d\n", latency_optimal_threshold);
            printf("LIBSWING_ALGO: %d\n", algo);
            printf("LIBSWING_SENDRECV_TYPE: %d\n", srtype);            
            printf("LIBSWING_DIMENSIONS: ");
            for(size_t i = 0; i < dimensions_num; i++){
                printf("%d", dimensions[i]);
                if(i < dimensions_num - 1){
                    printf(",");
                }
            }
            printf("\n");
            printf("------------------------------------\n");
        }
#endif
    }
}

static inline int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

// Convert a rank id into a list of d-dimensional coordinates
static inline void getCoordFromId(int id, int* coord){
    if(dimensions_num == 1){
        coord[0] = id;
    }else if(dimensions_num == 2){
        coord[0] = id / dimensions[1];
        coord[1] = id % dimensions[1];
    }else if(dimensions_num == 3){
        coord[0] = (id / dimensions[1]) % dimensions[0];
        coord[1] = id % dimensions[1];
        coord[2] = id / (dimensions[0]*dimensions[1]);
    }
}

// Convert d-dimensional coordinates into a rank id
static inline int getIdFromCoord(int* coord, uint* dimensions, uint dimensions_num){
    if(dimensions_num == 1){
        return coord[0];
    }else if(dimensions_num == 2){
        return coord[0]*dimensions[1] + coord[1];
    }else if(dimensions_num == 3){    
        return int(coord[2]*(dimensions[0]*dimensions[1]) + getIdFromCoord(coord, dimensions, dimensions_num - 1));
    }else{
        return -1;
    }
}

static inline void compute_peers(uint** peers, int size, int rank, int port, int num_steps){
    int coord[MAX_SUPPORTED_DIMENSIONS];
    bool terminated_dimensions_bitmap[MAX_SUPPORTED_DIMENSIONS];
    int next_directions[MAX_SUPPORTED_DIMENSIONS];
    for(uint rank = 0; rank < size; rank++){
        // Compute default directions
        getCoordFromId(rank, coord);
        for(size_t i = 0; i < dimensions_num; i++){
            next_directions[i] = pow(-1, ((coord[i] % 2) + (port % 2))); // TODO: or XOR?
            terminated_dimensions_bitmap[i] = false;            
        }
        
        int target_dim, relative_step, distance;
        uint terminated_dimensions = 0, o = 0;
        
        // Generate peers
        for(size_t i = 0; i < num_steps; ){
            getCoordFromId(rank, coord); // Regenerate rank coord
            o = 0;
            do{
                target_dim = (port + i + o) % (dimensions_num);            
                o++;
            }while(terminated_dimensions_bitmap[target_dim]);
            relative_step = (i + terminated_dimensions) / dimensions_num;        
            distance = (1 - pow(-2, relative_step + 1)) / 3;
            if(coord[target_dim] % 2){ // Flip the sign for odd nodes
                distance *= -1;
            }
            if(port >= dimensions_num){ // Mirrored collectives
                distance *= -1;
            }

            if(relative_step >= ceil(log2(dimensions[target_dim]))){
                terminated_dimensions_bitmap[target_dim] = true;
                terminated_dimensions++;
            }else{
                coord[target_dim] = mod((coord[target_dim] + distance), dimensions[target_dim]); // We need to use mod to avoid negative coordinates
                peers[rank][i] = getIdFromCoord(coord, dimensions, dimensions_num);
                i += 1;
                next_directions[target_dim] *= -1;
            }        
        }        
    }
}

static void computeBlocksBitmap(int sender, int step, char* blocks_bitmap, uint num_steps, uint** peers){
    if (step >= num_steps){ // Base case
        return;
    }else{
        for(size_t s = step; s < num_steps; s++){
            int peer = peers[sender][s];
            blocks_bitmap[peer] = 1;
            computeBlocksBitmap(peer, s+1, blocks_bitmap, num_steps, peers);
        }
        return;
    }    
}

static void getBitmaps(int rank, int size, char** bitmaps, uint8_t* reached_step, uint num_steps, uint** peers){
    // Bit vector that says if rank reached another node    
    for(size_t step = 0; step < num_steps; step++){
        int dest = peers[rank][step];
        bitmaps[step][dest] = 1; // I'll send its block
        computeBlocksBitmap(dest, step + 1, bitmaps[step], num_steps, peers); // ... plus those it will send in the next steps. 
        bitmaps[step][rank] = 0; // I can never reach myself (this could happen sometimes in the non-power-of-2 case)
        for(size_t i = 0; i < size; i++){
            if(bitmaps[step][i]){
                // Is there any peer I already reached before? If so, delete the earlier reach
                if(reached_step[i] != num_steps){
                    int prev_reached_step = reached_step[i];
                    bitmaps[prev_reached_step][i] = 0;
                }
                reached_step[i] = step;
            }
        }     
    }    
}

static int sendrecv_dt(char* send_bitmap, char* recv_bitmap, 
                    int* array_of_blocklengths_s, int* array_of_displacements_s,
                    int* array_of_blocklengths_r, int* array_of_displacements_r,
                    int dest, int source, int sendtag, int recvtag, 
                    void* buf, void* rbuf, const int *blocks_sizes, 
                    const int *blocks_displs, MPI_Comm comm, int size, int rank,  
                    MPI_Datatype sendtype, MPI_Datatype recvtype, MPI_Op op, CollType coll_type){
#ifdef PERF_DEBUGGING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // Create datatype starting from bitmaps
    // Check how many blocks we have to send/recv
    uint blocks_to_send = 0, blocks_to_recv = 0;
    for(size_t i = 0; i < size; i++){
        if(send_bitmap[i]){
            array_of_blocklengths_s[blocks_to_send] = blocks_sizes[i];
            array_of_displacements_s[blocks_to_send] = blocks_displs[i];
            ++blocks_to_send;
        }
        if(recv_bitmap[i]){
            array_of_blocklengths_r[blocks_to_recv] = blocks_sizes[i];
            array_of_displacements_r[blocks_to_recv] = blocks_displs[i];
            ++blocks_to_recv;
        }
    }
    MPI_Datatype indexed_sendtype, indexed_recvtype;
    MPI_Type_indexed(blocks_to_send, array_of_blocklengths_s, array_of_displacements_s, sendtype, &indexed_sendtype);
    MPI_Type_indexed(blocks_to_recv, array_of_blocklengths_r, array_of_displacements_r, recvtype, &indexed_recvtype);
    MPI_Type_commit(&indexed_sendtype);
    MPI_Type_commit(&indexed_recvtype);
#ifdef PERF_DEBUGGING
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Datatype preparation required: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << "ns\n";
#endif
    int res;
    res = MPI_Sendrecv(buf, 1, indexed_sendtype, dest, sendtag,
                        rbuf, 1, indexed_recvtype, source, recvtag,  
                        comm, MPI_STATUS_IGNORE);

    MPI_Type_free(&indexed_sendtype);
    MPI_Type_free(&indexed_recvtype);

    // Aggregate    
    if(coll_type == SWING_REDUCE_SCATTER){
        int dtsize;
        MPI_Type_size(sendtype, &dtsize);
        for(size_t i = 0; i < size; i++){
            if(recv_bitmap[i]){
                void* rbuf_block = (void*) (((char*) rbuf) + dtsize*blocks_displs[i]);
                void* buf_block = (void*) (((char*) buf) + dtsize*blocks_displs[i]);
                MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[i], sendtype, op);
            }
        }
    }

    return res;
}

// Block-by-block
static int sendrecv_bbb(char* send_bitmap, char* recv_bitmap, 
                    int dest, int source, int sendtag, int recvtag, 
                    void* buf, void* rbuf, const int *blocks_sizes, 
                    const int *blocks_displs, MPI_Comm comm, int size, int rank,  
                    MPI_Datatype sendtype, MPI_Datatype recvtype, MPI_Op op, CollType coll_type){
#ifdef PERF_DEBUGGING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // Create datatype starting from bitmaps
    // Check how many blocks we have to send/recv
    int dtsize_s, dtsize_r;
    MPI_Type_size(sendtype, &dtsize_s);
    MPI_Type_size(recvtype, &dtsize_r);
    uint blocks_to_send = 0, blocks_to_recv = 0;
    int res;
    int num_requests_s = 0, num_requests_r = 0;
    for(size_t i = 0; i < size; i++){
        num_requests_s += send_bitmap[i];
        num_requests_r += recv_bitmap[i];
    }
    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*num_requests_s);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*num_requests_r);
    //memset(requests_s, 0, sizeof(MPI_Request)*num_requests_s);
    //memset(requests_r, 0, sizeof(MPI_Request)*num_requests_r);
    int* req_idx_to_block_idx = (int*) malloc(sizeof(int)*num_requests_r);
    int next_req_s = 0, next_req_r = 0;
    for(size_t i = 0; i < size; i++){
        if(send_bitmap[i]){
            res = MPI_Isend(((char*) buf) + blocks_displs[i]*dtsize_s, blocks_sizes[i], sendtype, dest, sendtag, comm, &(requests_s[next_req_s]));
            ++next_req_s;
            if(res != MPI_SUCCESS){
                return res;
            }
        }
        if(recv_bitmap[i]){
            res = MPI_Irecv(((char*) rbuf) + blocks_displs[i]*dtsize_r, blocks_sizes[i], recvtype, dest, recvtag, comm, &(requests_r[next_req_r]));
            req_idx_to_block_idx[next_req_r] = i;
            ++next_req_r;
            if(res != MPI_SUCCESS){
                return res;
            }
        }
    }

    //assert(next_req_s == num_requests_s);
    //assert(next_req_r == num_requests_r);

    // Aggregate 
    if(coll_type == SWING_REDUCE_SCATTER){
        int index;
        int completed = 0;
        do{
            MPI_Waitany(num_requests_r, requests_r, &index, MPI_STATUS_IGNORE);
            int block_idx = req_idx_to_block_idx[index];
            void* rbuf_block = (void*) (((char*) rbuf) + dtsize_s*blocks_displs[block_idx]);
            void* buf_block = (void*) (((char*) buf) + dtsize_s*blocks_displs[block_idx]);
            MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[block_idx], sendtype, op);
            completed++;
        }while(completed != num_requests_r);
    }else{
        res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
        if(res != MPI_SUCCESS){
            return res;
        }        
    }
    res = MPI_Waitall(num_requests_s, requests_s, MPI_STATUSES_IGNORE);
    if(res != MPI_SUCCESS){
        return res;
    }
    free(requests_s);
    free(requests_r);
    free(req_idx_to_block_idx);
    return res;
}


static int sendrecv_cont(char* send_bitmap, char* recv_bitmap, 
                    int dest, int source, int sendtag, int recvtag, 
                    void* buf, void* rbuf, void* tmpbuf, const int *blocks_sizes, 
                    const int *blocks_displs, MPI_Comm comm, int size, int rank,  
                    MPI_Datatype sendtype, MPI_Datatype recvtype, MPI_Op op, CollType coll_type){
#ifdef PERF_DEBUGGING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // Create datatype starting from bitmaps
    // Check how many blocks we have to send/recv
    int dtsize_s, dtsize_r;
    MPI_Type_size(sendtype, &dtsize_s);
    MPI_Type_size(recvtype, &dtsize_r);
    int res;

    // Copy blocks to send to contiguous memory
    size_t next_good_offset = 0, count_s = 0, count_r = 0;
    for(size_t i = 0; i < size; i++){
        if(send_bitmap[i]){
            size_t block_size_bytes = (blocks_sizes[i])*dtsize_s;
            size_t block_displ_bytes = (blocks_displs[i])*dtsize_s;
            memcpy(((char*) tmpbuf) + next_good_offset, ((char*) buf) + block_displ_bytes, block_size_bytes);
            next_good_offset += block_size_bytes;
            count_s += blocks_sizes[i];
        }
        if(recv_bitmap[i]){
            count_r += blocks_sizes[i];
        }
    }

    res = MPI_Sendrecv(tmpbuf, count_s, sendtype, dest, sendtag,
                  rbuf, count_r, recvtype, source, recvtag,  
                  comm, MPI_STATUS_IGNORE);
    if(res != MPI_SUCCESS){
        return res;
    }

    // Aggregate recvd blocks (or move them to appropriate locations) -- to buf
    size_t next_rbuf_offset = 0;
    for(size_t i = 0; i < size; i++){
        size_t block_size_bytes = (blocks_sizes[i])*dtsize_r;
        size_t block_displ_bytes = (blocks_displs[i])*dtsize_r;
        if(recv_bitmap[i]){
            char* recvd_block_ptr = ((char*) rbuf) + next_rbuf_offset;
            char* buf_block_ptr = ((char*) buf) + block_displ_bytes;
            if(coll_type == SWING_REDUCE_SCATTER){
                MPI_Reduce_local(recvd_block_ptr, buf_block_ptr, blocks_sizes[i], sendtype, op);
            }else{
                memcpy(buf_block_ptr, recvd_block_ptr, block_size_bytes);
            }
            next_rbuf_offset += block_size_bytes;
        }
    }
    return res;
}

// Just to be used for performance debugging reasons
static int sendrecv_ideal(char* send_bitmap, char* recv_bitmap, 
                    int dest, int source, int sendtag, int recvtag, 
                    void* buf, void* rbuf, const int *blocks_sizes, 
                    const int *blocks_displs, MPI_Comm comm, int size, int rank,  
                    MPI_Datatype sendtype, MPI_Datatype recvtype, MPI_Op op, CollType coll_type){
    size_t count_s = 0, count_r = 0;
    for(size_t i = 0; i < size; i++){
        if(send_bitmap[i]){count_s += blocks_sizes[i];}
        if(recv_bitmap[i]){count_r += blocks_sizes[i];}
    }
    return MPI_Sendrecv(buf, count_s, sendtype, dest, sendtag,
                        rbuf, count_r, recvtype, source, recvtag,  
                        comm, MPI_STATUS_IGNORE);
}

static int swing_coll(void *buf, void* rbuf, const int *blocks_sizes, const int *blocks_displs, 
                      MPI_Op op, MPI_Comm comm, int size, int rank,  MPI_Datatype sendtype, MPI_Datatype recvtype,  
                      CollType coll_type){    
    int res, dtsize, tag;    
    char* blocks_bitmap_s;
    char* blocks_bitmap_r;
    char** my_blocks_matrix;
    char** peer_blocks_matrix;
    uint8_t* reached_step;
    uint** peers;
    int *array_of_blocklengths_s, *array_of_displacements_s, *array_of_blocklengths_r, *array_of_displacements_r;

    MPI_Type_size(sendtype, &dtsize);
    uint num_steps = ceil(log2(size));

    if(coll_type == SWING_REDUCE_SCATTER){
        tag = TAG_SWING_REDUCESCATTER;
    }else{
        tag = TAG_SWING_ALLGATHER;
    }

#ifdef PERF_DEBUGGING
    auto start = std::chrono::high_resolution_clock::now();
    long total = 0;
#endif


    // Compute total size of data
    size_t total_size_bytes = 0, total_elements = 0;
    for(size_t i = 0; i < size; i++){
        total_elements += blocks_sizes[i];        
    }
    total_size_bytes = dtsize*total_elements; // TODO: Check for custom datatypes
    int latency_optimal = (total_size_bytes <= latency_optimal_threshold) || (total_elements < size); // TODO Adjust for tori

    my_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);
    peer_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);
    reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
    if(!latency_optimal || coll_type != SWING_ALLREDUCE || srtype == SENDRECV_DT){
        array_of_blocklengths_s = (int*) malloc(sizeof(int)*size);
        array_of_displacements_s = (int*) malloc(sizeof(int)*size);
        array_of_blocklengths_r = (int*) malloc(sizeof(int)*size);
        array_of_displacements_r = (int*) malloc(sizeof(int)*size);
    }
    for(size_t step = 0; step < num_steps; step++){    
        my_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
        peer_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
    }
    char* tmpbuf;  
    if(srtype == SENDRECV_CONT){
        tmpbuf = (char*) malloc(sizeof(char)*total_size_bytes);
    }
    // Compute the peers
    peers = (uint**) malloc(sizeof(uint*)*size);
    for(uint i = 0; i < size; i++){
        peers[i] = (uint*) malloc(sizeof(uint)*num_steps);
    }
#ifdef PERF_DEBUGGING
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Mallocs required: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << "ns\n";
    start = std::chrono::high_resolution_clock::now();
#endif

    DPRINTF("[%d] Computing peers\n", rank);
    compute_peers(peers, size, rank, 0, num_steps); // TODO: For now we assume it is single-ported (and we pass port 0), extending should be trivial

#ifdef PERF_DEBUGGING
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Computing peers required: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << "ns\n";
    start = std::chrono::high_resolution_clock::now();
#endif

    DPRINTF("[%d] Getting bitmaps\n", rank);
    memset(reached_step, num_steps, sizeof(uint8_t)*size); // Init with num_steps to denote it didn't reach
    for(size_t i = 0; i < num_steps; i++){
        memset(my_blocks_matrix[i], 0, sizeof(char)*size);
    }
    getBitmaps(rank, size, my_blocks_matrix, reached_step, num_steps, peers);

#ifdef PERF_DEBUGGING
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Getting bitmaps required: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << "ns\n";
    start = std::chrono::high_resolution_clock::now();
#endif

    // Iterate over steps
    for(size_t step = 0; step < num_steps; step++){
#ifdef PERF_DEBUGGING
        start = std::chrono::high_resolution_clock::now();
#endif
        DPRINTF("[%d] Starting step %d\n", rank, step);
        /*********************************************************************/
        /* Now find which blocks I will receive. These are the block I need, */
        /* plus all of those I'll send starting from next step.              */
        /*********************************************************************/
        uint32_t peer;
        uint reversed_step = int(num_steps - step - 1);
        // Find which blocks I must send and recv.
        if(latency_optimal && coll_type == SWING_ALLREDUCE){
            peer = peers[rank][step];   
            // For latency optimal, I send/recv all the data
            res = MPI_Sendrecv(buf, total_elements, sendtype, peer, tag,
                               rbuf, total_elements, recvtype, peer, tag,  
                               comm, MPI_STATUS_IGNORE);
            if(res != MPI_SUCCESS){
                return res;
            }
            // Aggregate    
            MPI_Reduce_local(rbuf, buf, total_elements, sendtype, op);
        }else{
            DPRINTF("[%d] Computing adjusted blocks\n", rank);            
            memset(reached_step, num_steps, sizeof(uint8_t)*size); // Init with num_steps to denote it didn't reach
            for(size_t i = 0; i < num_steps; i++){
                memset(peer_blocks_matrix[i], 0, sizeof(char)*size);
            }
            if(coll_type == SWING_REDUCE_SCATTER){
                peer = peers[rank][step];                
                getBitmaps(peer, size, peer_blocks_matrix, reached_step, num_steps, peers);
                blocks_bitmap_s = my_blocks_matrix[step];
                blocks_bitmap_r = peer_blocks_matrix[step];
            }else{
                peer = peers[rank][reversed_step];
                getBitmaps(peer, size, peer_blocks_matrix, reached_step, num_steps, peers);
                blocks_bitmap_s = peer_blocks_matrix[reversed_step];
                blocks_bitmap_r = my_blocks_matrix[reversed_step];
            }

#ifdef DEBUG
            DPRINTF("[%d] Blocks Bitmap (Send) at step %d: ", rank, step);
            for(size_t i=0; i < size; i++){
                DPRINTF("%d ", blocks_bitmap_s[i]);
            }
            DPRINTF("\n");

            DPRINTF("[%d] Blocks Bitmap (Recv) at step %d: ", rank, step);
            for(size_t i=0; i < size; i++){
                DPRINTF("%d ", blocks_bitmap_r[i]);
            }
            DPRINTF("\n");
#endif

#ifdef PERF_DEBUGGING
            end = std::chrono::high_resolution_clock::now();
            std::cout << "[" << rank << "] Computing blocks at step " << step << " required: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << "ns\n";
            start = std::chrono::high_resolution_clock::now();
#endif

            // Sendrecv + aggregate
            int res;
            if(srtype == SENDRECV_DT){
                res  = sendrecv_dt(blocks_bitmap_s, blocks_bitmap_r, 
                                  array_of_blocklengths_s, array_of_displacements_s,
                                  array_of_blocklengths_r, array_of_displacements_r,
                                  peer, peer, tag, tag, 
                                  buf, rbuf, blocks_sizes, blocks_displs, 
                                  comm, size, rank, sendtype, recvtype, op, coll_type);
            }else if(srtype == SENDRECV_BBB){
                res  = sendrecv_bbb(blocks_bitmap_s, blocks_bitmap_r, 
                                  peer, peer, tag, tag, 
                                  buf, rbuf, blocks_sizes, blocks_displs, 
                                  comm, size, rank, sendtype, recvtype, op, coll_type);
            }else if(srtype == SENDRECV_CONT){
                res  = sendrecv_cont(blocks_bitmap_s, blocks_bitmap_r, 
                                  peer, peer, tag, tag, 
                                  buf, rbuf, tmpbuf, blocks_sizes, blocks_displs, 
                                  comm, size, rank, sendtype, recvtype, op, coll_type);
            }else{
                res  = sendrecv_ideal(blocks_bitmap_s, blocks_bitmap_r, 
                                  peer, peer, tag, tag, 
                                  buf, rbuf, blocks_sizes, blocks_displs, 
                                  comm, size, rank, sendtype, recvtype, op, coll_type);
            }
            if(res != MPI_SUCCESS){
                return res;
            }
#ifdef PERF_DEBUGGING
            end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
#endif       
        }
    }

#ifdef PERF_DEBUGGING
    end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    std::cout << "[" << rank << "] sendrecv required: " << total << "ns\n";
#endif
    for(size_t step = 0; step < num_steps; step++){    
        free(my_blocks_matrix[step]);
        free(peer_blocks_matrix[step]);
    }
    free(my_blocks_matrix);
    free(peer_blocks_matrix);
    free(reached_step);
    if(!latency_optimal || coll_type != SWING_ALLREDUCE || srtype == SENDRECV_DT){
        free(array_of_blocklengths_s);
        free(array_of_displacements_s);
        free(array_of_blocklengths_r);
        free(array_of_displacements_r);
    }
    if(srtype == SENDRECV_CONT){
        free(tmpbuf);
    }
    return 0;
}

static int MPI_Reduce_scatter_swing(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int res, size, rank, dtsize;    
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(datatype, &dtsize);

    // Create a temporary buffer (to avoid overwriting sendbuf)
    size_t buf_size = 0;
    int* displs = (int*) malloc(sizeof(int)*size);
    for(size_t i = 0; i < size; i++){
        displs[i] = buf_size;
        buf_size += recvcounts[i]; // TODO: If custom datatypes, manage appropriately        
    }

    size_t total_size_bytes = buf_size*dtsize, total_elements = buf_size;
    char* buf = (char*) malloc(buf_size*dtsize);    
    char* rbuf = (char*) malloc(buf_size*dtsize);
    memset(buf, 0, sizeof(char)*buf_size*dtsize);
    size_t my_displ_bytes = displs[rank]*dtsize;
    size_t my_count_bytes = recvcounts[rank]*dtsize;
    memcpy(buf, sendbuf, total_size_bytes);
    res = swing_coll(buf, rbuf, recvcounts, displs, op, comm, size, rank, datatype, datatype, SWING_REDUCE_SCATTER);
    // Copy the block in recvbuf
    memcpy(recvbuf, ((char*) buf) + my_displ_bytes, my_count_bytes);
    free(buf);
    free(rbuf);
    return res;
}

static int MPI_Allreduce_lat_optimal_swing(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int res, size, rank, dtsize;    
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(datatype, &dtsize);

    // Create a temporary buffer (to avoid overwriting sendbuf)
    size_t buf_size = 0;
    int* displs = (int*) malloc(sizeof(int)*size);
    for(size_t i = 0; i < size; i++){
        displs[i] = buf_size;
        buf_size += recvcounts[i]; // TODO: If custom datatypes, manage appropriately        
    }

    size_t total_size_bytes = buf_size*dtsize, total_elements = buf_size;
    char* buf = (char*) malloc(buf_size*dtsize);    
    char* rbuf = (char*) malloc(buf_size*dtsize);
    memset(buf, 0, sizeof(char)*buf_size*dtsize);
    size_t my_displ_bytes = displs[rank]*dtsize;
    size_t my_count_bytes = recvcounts[rank]*dtsize;
    memcpy(buf, sendbuf, total_size_bytes);
    res = swing_coll(buf, rbuf, recvcounts, displs, op, comm, size, rank, datatype, datatype, SWING_ALLREDUCE);
    // Copy the block in recvbuf
    memcpy(recvbuf, buf, total_size_bytes);
    free(buf);
    free(rbuf);
    return res;
}


static int MPI_Allgatherv_swing(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, 
                         const int *displs, MPI_Datatype recvtype, MPI_Comm comm){
    int res, size, rank, dtsize;    
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(recvtype, &dtsize);
    // Create a temporary buffer (to avoid overwriting sendbuf)
    size_t buf_size = 0;
    for(size_t i = 0; i < size; i++){
        buf_size += recvcounts[i]; // TODO: If custom datatypes, manage appropriately        
    }
    memcpy(((char*) recvbuf) + displs[rank]*dtsize, sendbuf, recvcounts[rank]*dtsize);
    if(srtype == SENDRECV_CONT){
        char* tmpbuf = (char*) malloc(buf_size*dtsize);
        res = swing_coll(recvbuf, tmpbuf, recvcounts, displs, MPI_SUM, comm, size, rank, sendtype, recvtype, SWING_ALLGATHER);
        free(tmpbuf);
    }else{
        res = swing_coll(recvbuf, recvbuf, recvcounts, displs, MPI_SUM, comm, size, rank, sendtype, recvtype, SWING_ALLGATHER);
    }
    return res;
}

static int MPI_Allreduce_ring(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int rank;
    int r = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(r != MPI_SUCCESS)
        return r;
    int size;
    r = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(r != MPI_SUCCESS)
        return r;
    int dtsize;
    r = MPI_Type_size(datatype, &dtsize);
    if(r != MPI_SUCCESS)
        return r;

    const size_t segment_size = count / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = count % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[size - 1] == count);

     // Copy your data to the output buffer to avoid modifying the input buffer.
    memcpy(recvbuf, sendbuf, count*dtsize);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    char* buffer = (char*) malloc(segment_sizes[0]*dtsize);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    MPI_Status recv_status;
    MPI_Request recv_req;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).
    for (int i = 0; i < size - 1; i++) {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        char* segment_send = &(((char*)recvbuf)[dtsize*segment_ends[send_chunk] - dtsize*segment_sizes[send_chunk]]);

        MPI_Irecv(buffer, segment_sizes[recv_chunk],
                datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[send_chunk],
                MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);

        char *segment_update = &(((char*)recvbuf)[dtsize*segment_ends[recv_chunk] - dtsize*segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);
        MPI_Reduce_local(buffer, segment_update, segment_sizes[recv_chunk], datatype, op);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + 1 + size) % size;
        int recv_chunk = (rank - i + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        char* segment_send = &(((char*)recvbuf)[dtsize*segment_ends[send_chunk] - dtsize*segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        char* segment_recv = &(((char*)recvbuf)[dtsize*segment_ends[recv_chunk] - dtsize*segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                     datatype, send_to, 0, segment_recv,
                     segment_sizes[recv_chunk], datatype, recv_from,
                     0, MPI_COMM_WORLD, &recv_status);
    }

    // Free temporary memory.
    free(buffer);    
    return 0;
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_reducescatter || algo == ALGO_DEFAULT){
        return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    }else if(algo == ALGO_SWING){
        return MPI_Reduce_scatter_swing(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    }else{
        assert("Only Swing supported for reducescatter." == 0);
    }
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, 
                 MPI_Datatype recvtype, MPI_Comm comm){
    read_env(comm);
    if(disable_allgather || algo == ALGO_DEFAULT){
        return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }else if(algo == ALGO_SWING){
        int size;    
        MPI_Comm_size(comm, &size);    
        int* recvcounts = (int*) malloc(sizeof(int)*size);
        int* displs = (int*) malloc(sizeof(int)*size);
        size_t last = 0;
        for(size_t i = 0; i < size; i++){
            recvcounts[i] = recvcount;
            displs[i] = last;
            last += recvcount;
        }
        int res = MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
        free(recvcounts);
        free(displs);
        return res;
    }else{
        assert("Only Swing supported for Allgather." == 0);
    }
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, 
                   const int *displs, MPI_Datatype recvtype, MPI_Comm comm){
    read_env(comm);
    if(disable_allgatherv || algo == ALGO_DEFAULT){
        return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }else if(algo == ALGO_SWING){
        return MPI_Allgatherv_swing(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }else{
        assert("Only Swing supported for Allgatherv." == 0);
    }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_allreduce || algo == ALGO_DEFAULT){
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_SWING){
        int res, size, rank, dtsize;    
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        MPI_Type_size(datatype, &dtsize);
        int* recvcounts = (int*) malloc(sizeof(int)*size);
        int* displs = (int*) malloc(sizeof(int)*size);
        size_t last = 0;
        for(size_t i = 0; i < size; i++){
            recvcounts[i] = ceil(count / size);
            if(i == size - 1){
                recvcounts[i] = count - ((count / size)*(size - 1));
            } 
            displs[i] = last;
            last += recvcounts[i];
        }
        


        // Compute total size of data
        size_t total_size_bytes = last*dtsize, total_elements = last;
        int latency_optimal = (total_size_bytes <= latency_optimal_threshold) || (total_elements < size); // TODO Adjust for tori
        if(latency_optimal){
            res = MPI_Allreduce_lat_optimal_swing(sendbuf, recvbuf, recvcounts, datatype, op, comm); // Misleading, should call the function in a different way
        }else{
            char* intermediate_buf = (char*) recvbuf + displs[rank];
            res = MPI_Reduce_scatter_swing(sendbuf, intermediate_buf, recvcounts, datatype, op, comm);
            if(res == MPI_SUCCESS){        
                res = MPI_Allgatherv_swing(intermediate_buf, recvcounts[rank], datatype, recvbuf, recvcounts, displs, datatype, comm);
            }
        }
        free(recvcounts);
        free(displs);
        return res;
    }else{
        return MPI_Allreduce_ring(sendbuf, recvbuf, count, datatype, op, comm);
    }
}

// TODO: Don't use Swing for non-continugous non-native datatypes (tedious implementation)
