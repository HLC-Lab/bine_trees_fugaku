#include <mpi.h>
#include <algorithm>
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

#define TAG_SWING_REDUCESCATTER 0x7FFF
#define TAG_SWING_ALLGATHER 0x7FFE
#define TAG_SWING_ALLREDUCE 0x7FFD

//#define PERF_DEBUGGING 
//#define ACTIVE_WAIT

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
    ALGO_RECDOUB_L,
    ALGO_RECDOUB_B,
}Algo;

typedef enum{
    SENDRECV_DT = 0, // Datatypes
    SENDRECV_BBB, // Block-by-block
    SENDRECV_BBBO, // Block-by-block, overlapped
    SENDRECV_BBBN, // Block-by-block, new
    SENDRECV_CONT, // Contiguous
    SENDRECV_IDEAL, // Ideal case (just for performance debugging reasons, produces wrong results, never use it in practice)
}SendRecv;

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allgather = 0, disable_allreduce = 0, 
                    dimensions_num = 1, latency_optimal_threshold = 1024, force_env_reload = 1, env_read = 0, coalesce = 0,
                    fast_bitmaps = 1, cache = 1, cached_p = 0, rdma = 0, switch_point = 0; //2097152;
static char** cached_my_blocks_matrix = NULL;
static uint* cached_blocks_remapping = NULL;
static uint** cached_peers = NULL;
static Algo algo = ALGO_SWING;
static SendRecv srtype = SENDRECV_DT;
static uint dimensions[MAX_SUPPORTED_DIMENSIONS];

static inline void read_env(MPI_Comm comm){
    char* env_str = getenv("LIBSWING_FORCE_ENV_RELOAD");
    if(env_str){
        force_env_reload = 1;
    }else{
        force_env_reload = 0;
    }

    if(!env_read || force_env_reload){
        env_read = 1;

        env_str = getenv("LIBSWING_COALESCE");
        if(env_str){
            coalesce = atoi(env_str);
        }

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

        env_str = getenv("LIBSWING_FAST_BITMAPS");
        if(env_str){
            fast_bitmaps = atoi(env_str);
        }
        
        env_str = getenv("LIBSWING_CACHE");
        if(env_str){
            cache = atoi(env_str);
        }
        
        env_str = getenv("LIBSWING_RDMA");
        if(env_str){
            rdma = atoi(env_str);
        }
        
        env_str = getenv("LIBSWING_SWITCH_POINT");
        if(env_str){
            switch_point = atoi(env_str);
        }

        env_str = getenv("LIBSWING_ALGO");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                algo = ALGO_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                algo = ALGO_SWING;
            }else if(strcmp(env_str, "RING") == 0){
                algo = ALGO_RING;
            }else if(strcmp(env_str, "RECDOUB_L") == 0){
                algo = ALGO_RECDOUB_L;
            }else if(strcmp(env_str, "RECDOUB_B") == 0){
                algo = ALGO_RECDOUB_B;
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
            }else if(strcmp(env_str, "BBBO") == 0){
                srtype = SENDRECV_BBBO;
            }else if(strcmp(env_str, "BBBN") == 0){
                srtype = SENDRECV_BBBN;
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
            printf("LIBSWING_COALESCE: %d\n", coalesce);
            printf("LIBSWING_DISABLE_REDUCESCATTER: %d\n", disable_reducescatter);
            printf("LIBSWING_DISABLE_ALLGATHERV: %d\n", disable_allgatherv);
            printf("LIBSWING_DISABLE_ALLGATHER: %d\n", disable_allgather);
            printf("LIBSWING_DISABLE_ALLREDUCE: %d\n", disable_allreduce);
            printf("LIBSWING_LATENCY_OPTIMAL_THRESHOLD: %d\n", latency_optimal_threshold);
            printf("LIBSWING_FAST_BITMAPS: %d\n", fast_bitmaps);
            printf("LIBSWING_CACHE: %d\n", cache);
            printf("LIBSWING_RDMA: %d\n", rdma);
            printf("LIBSWING_ALGO: %d\n", algo);
            printf("LIBSWING_SENDRECV_TYPE: %d\n", srtype);            
            printf("LIBSWING_SWITCH_POINT: %d\n", switch_point);            
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

//#define mod(a,b)({(a + 3*b) & (b-1);})

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

// With this we are ok up to 2^20 nodes, add other terms if needed.
static int rhos[20] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};

static inline void compute_peers(uint** peers, int port, int num_steps, uint start_rank, uint num_ranks){
    int coord[MAX_SUPPORTED_DIMENSIONS];
    bool terminated_dimensions_bitmap[MAX_SUPPORTED_DIMENSIONS];
    int num_steps_per_dim[MAX_SUPPORTED_DIMENSIONS];
    for(size_t i = 0; i < dimensions_num; i++){
        num_steps_per_dim[i] = ceil(log2(dimensions[i])) - 1;
    }
    
    for(uint rank = start_rank; rank < start_rank + num_ranks; rank++){
        // Compute default directions
        getCoordFromId(rank, coord);
        for(size_t i = 0; i < dimensions_num; i++){
            terminated_dimensions_bitmap[i] = false;            
        }
        
        int target_dim, relative_step, distance;
        uint terminated_dimensions = 0, o = 0;
        
        // Generate peers
        for(size_t i = 0; i < num_steps; ){            
            if(dimensions_num > 1){
                getCoordFromId(rank, coord); // Regenerate rank coord
                o = 0;
                do{
                    target_dim = (port + i + o) % (dimensions_num);            
                    o++;
                }while(terminated_dimensions_bitmap[target_dim]);
                relative_step = (i + terminated_dimensions) / dimensions_num;        
            }else{
                target_dim = 0;
                relative_step = i;
                coord[0] = rank;
            }
            
            distance = rhos[relative_step];
            // Flip the sign for odd nodes
            if(coord[target_dim] & 1){distance *= -1;}
            // Mirrored collectives
            if(port >= dimensions_num){distance *= -1;}

            if(relative_step <= num_steps_per_dim[target_dim]){
                coord[target_dim] = mod((coord[target_dim] + distance), dimensions[target_dim]); // We need to use mod to avoid negative coordinates
                if(dimensions_num > 1){
                    peers[rank][i] = getIdFromCoord(coord, dimensions, dimensions_num);
                }else{
                    peers[rank][i] = coord[0];
                }
                i += 1;
            }else{
                terminated_dimensions_bitmap[target_dim] = true;
                terminated_dimensions++;                
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

static void getBitmapsMatrix(int rank, int size, char** bitmaps, uint8_t* reached_step, uint num_steps, uint** peers,
                             int reference_rank, char** reference_bitmap, int step){
    if(reference_bitmap && fast_bitmaps){
        // Compute from reference bitmap rather than doing it from scratch
        int diff = rank - reference_rank;        
        // How much should I "shift" it to the left or right
        DPRINTF("[%d] Computing bitmap for rank %d by shifting its own by %d positions\n", reference_rank, rank, diff);
        // Elem in pos i in reference_bitmap, should be in pos i+diff in bitmaps
        /*
        for(size_t i = 0; i < size; i++){                
            bitmaps[step][mod(i + diff, size)] = reference_bitmap[step][i];
        }
        */
        if(diff > 0){
            for(int i = 0; i + diff < size; i++){             
                bitmaps[step][i + diff] = reference_bitmap[step][i];
            }
            for(int i = size - diff; i < size; i++){                
                bitmaps[step][i + diff - size] = reference_bitmap[step][i];
            }
        }else{
            for(int i = 0; i + diff < 0; i++){                
                bitmaps[step][i + diff + size] = reference_bitmap[step][i];
            }
            for(int i = -diff; i < size; i++){                
                bitmaps[step][i + diff] = reference_bitmap[step][i];
            }
        }
#ifdef DEBUG
        DPRINTF("[%d] Intermediate bitmap: ", reference_rank);
        for(size_t i = 0; i < size; i++){
            DPRINTF("%d ", bitmaps[step][i]);
        }
        DPRINTF("\n");
#endif
        // If referance_rank is odd and rank even, or vice versa, we need to mirror it (Always true since even talk only with odds and vice versa)
        DPRINTF("[%d] And by flipping it\n", reference_rank);
        // Reflect the array
        for (int i = 1; i < size / 2; i++) {
            //int start_pos = (rank + i) % size;
            //int end_pos = mod((rank - i), size);
            int start_pos = (rank + i) >= size?rank+i-size:rank+i;
            int end_pos = rank-i < 0?rank-i+size:rank-i;
            int temp = bitmaps[step][start_pos];
            bitmaps[step][start_pos] = bitmaps[step][end_pos];
            bitmaps[step][end_pos] = temp;
        }

        /*
        for(size_t i = 0; i < size/2; i++){     
            // First elem is 2*rank - size - diff, from there go backward     
            printf("[%d] (rank %d) Checking if %d -> %d\n", reference_rank, rank, mod(2*rank - size - diff - i, size), i);
            assert(bitmaps[step][i] == reference_bitmap[step][mod(2*rank - size - diff - i, size)]);
        }*/
#ifdef DEBUG
        DPRINTF("[%d] Final bitmap: ", reference_rank);
        for(size_t i = 0; i < size; i++){
            DPRINTF("%d ", bitmaps[step][i]);
        }
        DPRINTF("\n");
#endif
    }else{
        memset(reached_step, num_steps, sizeof(uint8_t)*size); // Init with num_steps to denote it didn't reach
        for(size_t i = 0; i < num_steps; i++){
            memset(bitmaps[i], 0, sizeof(char)*size);
        }
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
}

static void getBitmaps(int rank, int size, uint8_t* reached_step, uint num_steps, uint** peers, 
                       CollType coll_type, uint block_step, char** my_blocks_matrix, char** peer_blocks_matrix, 
                       char** bitmap_s, char** bitmap_r){
    uint32_t peer = peers[rank][block_step];
    getBitmapsMatrix(peer, size, peer_blocks_matrix, reached_step, num_steps, peers,
                    rank, my_blocks_matrix, block_step);
    if(coll_type == SWING_REDUCE_SCATTER){
        *bitmap_s = my_blocks_matrix[block_step];
        *bitmap_r = peer_blocks_matrix[block_step];
    }else{
        *bitmap_s = peer_blocks_matrix[block_step];
        *bitmap_r = my_blocks_matrix[block_step];
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
                    char** send_bitmap_next, char** recv_bitmap_next, 
                    int dest, int source, int sendtag, int recvtag, 
                    void* buf, void* rbuf, void* rbuf_pre, const int *blocks_sizes, 
                    const int *blocks_displs, MPI_Comm comm, int size, int rank,  
                    MPI_Datatype sendtype, MPI_Datatype recvtype, MPI_Op op, CollType coll_type,
                    MPI_Request* requests_s, MPI_Request* requests_r, int* req_idx_to_block_idx, 
                    int* extra_start, int* extra_count, int next_dest,
                    int* aggregate_later, int* aggregate_later_count,
                    uint8_t* reached_step, int step, uint num_steps,
                    uint** peers, char** my_blocks_matrix, char** next_peer_blocks_matrix){
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

    // Move extra requests
    if(extra_count && *extra_count){
        for(size_t i = 0; i < *extra_count; i++){
            requests_s[i] = requests_s[*extra_start + i];
            num_requests_s++;
        }
    }

    for(size_t i = 0; i < size; i++){
        if(send_bitmap[i]){
            res = MPI_Isend(((char*) buf) + blocks_displs[i]*dtsize_s, blocks_sizes[i], sendtype, dest, sendtag, comm, &(requests_s[num_requests_s]));
            ++num_requests_s;
            if(res != MPI_SUCCESS){
                return res;
            }
        }
        if(recv_bitmap[i]){
            res = MPI_Irecv(((char*) rbuf) + blocks_displs[i]*dtsize_r, blocks_sizes[i], recvtype, dest, recvtag, comm, &(requests_r[num_requests_r]));
            req_idx_to_block_idx[num_requests_r] = i;
            ++num_requests_r;
            if(res != MPI_SUCCESS){
                return res;
            }
        }
    }

    // Here I can overlap stuff.
    // Aggregate stuff I didn't aggregate yet
    if(coll_type == SWING_REDUCE_SCATTER && aggregate_later_count && *aggregate_later_count){
        for(size_t i = 0; i < *aggregate_later_count; i++){            
            int block_idx = aggregate_later[i];
            size_t displ_bytes = dtsize_s*blocks_displs[block_idx];
            void* rbuf_block = (void*) (((char*) rbuf_pre) + displ_bytes);
            void* buf_block = (void*) (((char*) buf) + displ_bytes);            
            MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[block_idx], sendtype, op); 
        }
        *aggregate_later_count = 0;
    }

    // Compute bitmaps for next step
    if(*send_bitmap_next){
        if(step != num_steps - 1){
            size_t block_step_next = (coll_type == SWING_REDUCE_SCATTER)?step+1:(num_steps - step - 1 - 1);
            next_dest = peers[rank][block_step_next];
            getBitmaps(rank, size, reached_step, num_steps, peers, 
                        coll_type, block_step_next, my_blocks_matrix, next_peer_blocks_matrix, 
                        send_bitmap_next, recv_bitmap_next);
        }else{
            *send_bitmap_next = NULL;
            *recv_bitmap_next = NULL;
        }
    }

    // Aggregate 
    if(coll_type == SWING_REDUCE_SCATTER){
        int index, completed = 0;
        if(extra_count){
            *extra_count = 0;
            *extra_start = num_requests_s;
        }
        do{
            res = MPI_Waitany(num_requests_r, requests_r, &index, MPI_STATUS_IGNORE);
            if(res != MPI_SUCCESS){
                return res;
            }
            int block_idx = req_idx_to_block_idx[index];
            size_t displ_bytes = dtsize_s*blocks_displs[block_idx];
            void* rbuf_block = (void*) (((char*) rbuf) + displ_bytes);
            void* buf_block = (void*) (((char*) buf) + displ_bytes);            
            if(*send_bitmap_next){ // If BBBO
                // If the block I just receive need to be sent in the next step, I aggregating it immediately won't delay anything
                if((*send_bitmap_next)[block_idx]){ 
                    MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[block_idx], sendtype, op);
                    /*
                    res = MPI_Isend(buf_block, blocks_sizes[block_idx], sendtype, next_dest, sendtag, comm, &(requests_s[num_requests_s + (*extra_count)]));
                    *(extra_count) = *(extra_count) + 1;
                    if(res != MPI_SUCCESS){
                        return res;
                    }
                    send_bitmap_next[block_idx] = 0; // So that it is not sent again later
                    */
                }else{
                    // I can aggregate it later
                    aggregate_later[*aggregate_later_count] = block_idx;
                    *aggregate_later_count = *aggregate_later_count + 1;
                }
            }else{
                MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[block_idx], sendtype, op); 
            }
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
    int res = MPI_Sendrecv(buf, count_s, sendtype, dest, sendtag,
                        rbuf, count_r, recvtype, source, recvtag,  
                        comm, MPI_STATUS_IGNORE);
    if(coll_type == SWING_REDUCE_SCATTER){
        MPI_Reduce_local(rbuf, buf, count_s, sendtype, op);
    }
    return res;
}

static int swing_coll(void *buf, void* rbuf, const int *blocks_sizes, const int *blocks_displs, 
                      MPI_Op op, MPI_Comm comm, int size, int rank, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                      CollType coll_type, uint** peers, char** my_blocks_matrix){    
    int res, dtsize, tag, extra_start = 0, extra_count = 0, aggregate_later_count = 0;    
    char* blocks_bitmap_s;
    char* rbuf_prev = (char*) rbuf;
    char* rbuf_prev_orig;
    char* blocks_bitmap_r;
    char* blocks_bitmap_s_next;
    char* blocks_bitmap_r_next;
    char** peer_blocks_matrix;
    char** next_peer_blocks_matrix;
    uint8_t* reached_step;
    int *array_of_blocklengths_s, *array_of_displacements_s, *array_of_blocklengths_r, *array_of_displacements_r, *aggregate_later;

    MPI_Type_size(sendtype, &dtsize);
    uint num_steps = ceil(log2(size));

    if(coll_type == SWING_REDUCE_SCATTER){
        tag = TAG_SWING_REDUCESCATTER;
    }else{
        tag = TAG_SWING_ALLGATHER;
    }

    // Compute total size of data
    size_t total_size_bytes = 0, total_elements = 0;
    for(size_t i = 0; i < size; i++){
        total_elements += blocks_sizes[i];        
    }
    total_size_bytes = dtsize*total_elements; // TODO: Check for custom datatypes

    peer_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);
    if(srtype == SENDRECV_BBBO){
        next_peer_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);
        aggregate_later = (int*) malloc(sizeof(int)*size);
        if(coll_type == SWING_REDUCE_SCATTER){
            rbuf_prev = (char*) malloc(sizeof(char)*total_size_bytes);
            rbuf_prev_orig = rbuf_prev;
        }
    }
    reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
    if(srtype == SENDRECV_DT){
        array_of_blocklengths_s = (int*) malloc(sizeof(int)*size);
        array_of_displacements_s = (int*) malloc(sizeof(int)*size);
        array_of_blocklengths_r = (int*) malloc(sizeof(int)*size);
        array_of_displacements_r = (int*) malloc(sizeof(int)*size);
    }
    for(size_t step = 0; step < num_steps; step++){    
        peer_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
        if(srtype == SENDRECV_BBBO){
            next_peer_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
        }
    }
    char* tmpbuf;  
    MPI_Request* requests_s;
    MPI_Request* requests_r;
    int* req_idx_to_block_idx;
    if(srtype == SENDRECV_BBB || srtype == SENDRECV_BBBO){
        requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*size);
        requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*size);
        req_idx_to_block_idx = (int*) malloc(sizeof(int)*size);
    }

    if(srtype == SENDRECV_BBBO){
        size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?0:(num_steps - 1);            
        uint32_t peer = peers[rank][block_step];
        getBitmaps(rank, size, reached_step, num_steps, peers, 
                   coll_type, block_step, my_blocks_matrix, peer_blocks_matrix, 
                   &blocks_bitmap_s, &blocks_bitmap_r);
    }

    // Iterate over steps
    for(size_t step = 0; step < num_steps; step++){
        DPRINTF("[%d] Starting step %d\n", rank, step);
        /*********************************************************************/
        /* Now find which blocks I will receive. These are the block I need, */
        /* plus all of those I'll send starting from next step.              */
        /*********************************************************************/        
        // Find which blocks I must send and recv.
#ifdef PERF_DEBUGGING
        start = std::chrono::high_resolution_clock::now();
#endif
        DPRINTF("[%d] Computing adjusted blocks\n", rank);            
        size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(num_steps - step - 1);            
        uint32_t peer = peers[rank][block_step];
        uint32_t next_dest;
        DPRINTF("[%d] Peer %d\n", rank, peer);
        if(srtype != SENDRECV_BBBO){
            getBitmaps(rank, size, reached_step, num_steps, peers, 
                       coll_type, block_step, my_blocks_matrix, peer_blocks_matrix, 
                       &blocks_bitmap_s, &blocks_bitmap_r);
        }
        // Otherwise are computed while waiting for recv
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
            res = sendrecv_dt(blocks_bitmap_s, blocks_bitmap_r, 
                                array_of_blocklengths_s, array_of_displacements_s,
                                array_of_blocklengths_r, array_of_displacements_r,
                                peer, peer, tag, tag, 
                                buf, rbuf, blocks_sizes, blocks_displs, 
                                comm, size, rank, sendtype, recvtype, op, coll_type);
        }else if(srtype == SENDRECV_BBB || srtype == SENDRECV_BBBO){
            if(srtype == SENDRECV_BBB){
                blocks_bitmap_s_next = NULL;
                blocks_bitmap_r_next = NULL;
            }else{
                blocks_bitmap_s_next = (char*) 0x1;
                blocks_bitmap_r_next = (char*) 0x1;
            }
            res = sendrecv_bbb(blocks_bitmap_s, blocks_bitmap_r, 
                                &blocks_bitmap_s_next, &blocks_bitmap_r_next, 
                                peer, peer, tag, tag, 
                                buf, rbuf, rbuf_prev, blocks_sizes, blocks_displs, 
                                comm, size, rank, sendtype, recvtype, op, coll_type,
                                requests_s, requests_r, req_idx_to_block_idx, 
                                &extra_start, &extra_count, next_dest,
                                aggregate_later, &aggregate_later_count,
                                reached_step, step, num_steps,
                                peers, my_blocks_matrix, next_peer_blocks_matrix);
            if(srtype == SENDRECV_BBBO){
                blocks_bitmap_s = blocks_bitmap_s_next;
                blocks_bitmap_r = blocks_bitmap_r_next;
                // Swap bit matrices
                char** tmp;
                tmp = next_peer_blocks_matrix;
                next_peer_blocks_matrix = peer_blocks_matrix;
                peer_blocks_matrix = tmp;
                if(coll_type == SWING_REDUCE_SCATTER){
                    // Swap rbufs
                    char* rt;
                    rt = (char*) rbuf;
                    rbuf = rbuf_prev;
                    rbuf_prev = rt;
                }

            }
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

#ifdef PERF_DEBUGGING
    std::cout << "[" << rank << "] sendrecv required: " << total << "ns\n";
#endif
    for(size_t step = 0; step < num_steps; step++){    
        free(peer_blocks_matrix[step]);
        if(srtype == SENDRECV_BBBO){
            free(next_peer_blocks_matrix[step]);
        }
    }
    free(peer_blocks_matrix);
    free(reached_step);
    if(srtype == SENDRECV_DT){
        free(array_of_blocklengths_s);
        free(array_of_displacements_s);
        free(array_of_blocklengths_r);
        free(array_of_displacements_r);
    }
    if(srtype == SENDRECV_BBB || srtype == SENDRECV_BBBO){
        free(requests_s);
        free(requests_r);
        free(req_idx_to_block_idx);
        if(srtype == SENDRECV_BBBO){
            free(next_peer_blocks_matrix);
            free(aggregate_later);
            if(coll_type == SWING_REDUCE_SCATTER){
                free(rbuf_prev_orig);
            }
        }
    }
    return 0;
}


static int swing_coll_cont(void *buf, void* rbuf, const int *blocks_sizes, const int *blocks_displs, 
                           MPI_Op op, MPI_Comm comm, int size, int rank, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type, uint** peers, char** my_blocks_matrix, uint* blocks_remapping){
    int dtsize, tag, res;  
    MPI_Type_size(sendtype, &dtsize);
    uint num_steps = ceil(log2(size));

    if(coll_type == SWING_REDUCE_SCATTER){
        tag = TAG_SWING_REDUCESCATTER;
    }else{
        tag = TAG_SWING_ALLGATHER;
    }

    MPI_Win win = MPI_WIN_NULL;
    if(rdma){
        int count = 0;
        for(size_t i = 0; i < size; i++){
            count += blocks_sizes[i];
        }
        MPI_Win_create(buf, count*dtsize, dtsize, MPI_INFO_NULL, comm, &win);
        MPI_Win_fence(0, win);
    }

    // Iterate over steps
    for(size_t step = 0; step < num_steps; step++){
        DPRINTF("[%d] Starting step %d\n", rank, step);
        size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(num_steps - step - 1);            
        uint32_t peer = peers[rank][block_step];
        DPRINTF("[%d] Peer %d\n", rank, peer);

        // Sendrecv + aggregate
        int diff = peer - rank;

        size_t contiguous_start_offset_s = -1, contiguous_end_offset_s = 0, contiguous_count_s = 0; // TODO, cache start/end indexes
        size_t contiguous_start_offset_r = -1, contiguous_end_offset_r = 0, contiguous_count_r = 0;
        for(size_t i = 0; i < size; i++){
            // Instead of computing the peer's bitmaps, I considering them as shifted versions of my bitmap.
            // If dist is the distance between me and my peer, I should just shift my bitmap by dist positions, and then reverse it along the x-axis
            // with respect to my peer. However, I can avoid doing that and just play with indexes
            //assert(bitmaps[step][i] == reference_bitmap[step][mod(2*rank - size - diff - i, size)]);

            size_t send_index, recv_index;

            // peer - size + rank - i >= size  ==> peer + rank - i >= 2*size. At most we can have i=0, and thus peer + rank >= 2* size. However this can never be true since peer + rank < 2*size
            // peer - size + rank - i < 0      ==> peer + rank - i < size. At most we can have i=size-1, and thus peer + rank - size + 1 < size ==> peer + rank + 1 < 2*size, which might indeed happen.
            // To see if we can simply sum sum when it's negative, we have to check that:
            // peer - size + rank - i >= -size ==> peer + rank - i >= 0, which might not be true, so we have to loop and add
            //int other = mod(peer - size + rank - i, size); 
            int other = peer - size + rank - i;
            while(other < 0){other += size;}
            if(coll_type == SWING_REDUCE_SCATTER){
                send_index = i;
                recv_index = other;
            }else{
                send_index = other;
                recv_index = i;
            }
            if(my_blocks_matrix[block_step][send_index]){
                size_t k = blocks_remapping[i];
                DPRINTF("[%d] Going to send block %d (displ %d)\n", rank, k, blocks_displs[k]);
                if(blocks_displs[k] < contiguous_start_offset_s){
                    contiguous_start_offset_s = blocks_displs[k];
                }
                if(blocks_displs[k] + blocks_sizes[k] > contiguous_end_offset_s){
                    contiguous_end_offset_s = blocks_displs[k] + blocks_sizes[k];
                }
                contiguous_count_s += blocks_sizes[k];
            }
            if(my_blocks_matrix[block_step][recv_index]){
                size_t k = blocks_remapping[i];
                DPRINTF("[%d] Going to recv block %d (displ %d)\n", rank, k, blocks_displs[k]);
                if(blocks_displs[k] < contiguous_start_offset_r){
                    contiguous_start_offset_r = blocks_displs[k];
                }
                if(blocks_displs[k] + blocks_sizes[k] > contiguous_end_offset_r){
                    contiguous_end_offset_r = blocks_displs[k] + blocks_sizes[k];
                }
                contiguous_count_r += blocks_sizes[k];
            }
        }

        // Overlap here
        if(step != num_steps - 1){
            size_t block_step_next = (coll_type == SWING_REDUCE_SCATTER)?step + 1:(num_steps - step - 1 - 1);            
            __builtin_prefetch(my_blocks_matrix[block_step_next], 0, 0);
        }
        // End overlap

        if(rdma){
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, peer, 0, win);
            if(coll_type == SWING_REDUCE_SCATTER){
                res = MPI_Accumulate(((char*) buf) + contiguous_start_offset_s*dtsize, contiguous_count_s, sendtype, peer, contiguous_start_offset_r, contiguous_count_r, recvtype, op, win);
            }else{
                res = MPI_Put(((char*) buf) + contiguous_start_offset_s*dtsize, contiguous_count_s, sendtype, peer, contiguous_start_offset_r, contiguous_count_r, recvtype, win);
            }
            MPI_Win_unlock(peer, win);
            //MPI_Win_fence(0, win);
        }else{
            assert(contiguous_end_offset_s - contiguous_start_offset_s == contiguous_count_s);
            assert(contiguous_end_offset_r - contiguous_start_offset_r == contiguous_count_r);
            MPI_Sendrecv(((char*) buf) + contiguous_start_offset_s*dtsize, contiguous_count_s, sendtype,
                         peer, tag,
                         ((char*) rbuf) + contiguous_start_offset_r*dtsize, contiguous_count_r, recvtype,
                         peer, tag, comm, MPI_STATUS_IGNORE);
            if(coll_type == SWING_REDUCE_SCATTER){
                void* rbuf_block = (void*) (((char*) rbuf) + contiguous_start_offset_r*dtsize);
                void* buf_block = (void*) (((char*) buf) + contiguous_start_offset_r*dtsize);            
                MPI_Reduce_local(rbuf_block, buf_block, contiguous_count_r, sendtype, op); 
            }
        }     
    }

    if(rdma){
        MPI_Win_free(&win);
    }
    return 0;
}


static int swing_coll_bbbn(void *buf, void* rbuf, const int *blocks_sizes, const int *blocks_displs, 
                           MPI_Op op, MPI_Comm comm, int size, int rank, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type, uint** peers, char** my_blocks_matrix, uint* blocks_remapping){
    int dtsize, tag, res;  
    MPI_Type_size(sendtype, &dtsize);
    uint num_steps = ceil(log2(size));

    if(coll_type == SWING_REDUCE_SCATTER){
        tag = TAG_SWING_REDUCESCATTER;
    }else{
        tag = TAG_SWING_ALLGATHER;
    }

    MPI_Win win = MPI_WIN_NULL;
    if(rdma){
        int count = 0;
        for(size_t i = 0; i < size; i++){
            count += blocks_sizes[i];
        }
        MPI_Win_create(buf, count*dtsize, dtsize, MPI_INFO_NULL, comm, &win);
        MPI_Win_fence(0, win);
    }

    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*size);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*size);;
    int* req_idx_to_block_idx = (int*) malloc(sizeof(int)*size);

    // Iterate over steps
    for(size_t step = 0; step < num_steps; step++){
        DPRINTF("[%d] Starting step %d\n", rank, step);
        size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(num_steps - step - 1);            
        uint32_t peer = peers[rank][block_step];
        DPRINTF("[%d] Peer %d\n", rank, peer);

        // Sendrecv + aggregate
        int num_requests_s = 0, num_requests_r = 0;
        int diff = peer - rank;
        int sendcnt = 0, recvcnt = 0, recvblocks = 0;     
        for(size_t i = 0; i < size; i++){
            // Instead of computing the peer's bitmaps, I considering them as shifted versions of my bitmap.
            // If dist is the distance between me and my peer, I should just shift my bitmap by dist positions, and then reverse it along the x-axis
            // with respect to my peer. However, I can avoid doing that and just play with indexes
            //assert(bitmaps[step][i] == reference_bitmap[step][mod(2*rank - size - diff - i, size)]);

            size_t send_index, recv_index;

            // peer - size + rank - i >= size  ==> peer + rank - i >= 2*size. At most we can have i=0, and thus peer + rank >= 2* size. However this can never be true since peer + rank < 2*size
            // peer - size + rank - i < 0      ==> peer + rank - i < size. At most we can have i=size-1, and thus peer + rank - size + 1 < size ==> peer + rank + 1 < 2*size, which might indeed happen.
            // To see if we can simply sum sum when it's negative, we have to check that:
            // peer - size + rank - i >= -size ==> peer + rank - i >= 0, which might not be true, so we have to loop and add
            //int other = mod(peer - size + rank - i, size); 
            int other = peer - size + rank - i;
            while(other < 0){other += size;}
            if(coll_type == SWING_REDUCE_SCATTER){
                send_index = i;
                //recv_index = mod(2*peer - size - diff - i, size);
                recv_index = other;
            }else{
                //send_index = mod(2*peer - size - diff - i, size);
                send_index = other;
                recv_index = i;
            }
            if(my_blocks_matrix[block_step][send_index]){
                if(srtype == SENDRECV_IDEAL){
                    sendcnt += blocks_sizes[i];
                }else{                    
                    if(rdma){
                        if(coll_type == SWING_REDUCE_SCATTER){
                            res = MPI_Accumulate(((char*) buf) + blocks_displs[i]*dtsize, blocks_sizes[i], sendtype, peer, blocks_displs[i], blocks_sizes[i], recvtype, op, win);
                        }else{
                            res = MPI_Put(((char*) buf) + blocks_displs[i]*dtsize, blocks_sizes[i], sendtype, peer, blocks_displs[i], blocks_sizes[i], recvtype, win);
                        }
                    }else{
                        DPRINTF("[%d] Sending block %d to %d at step %d (coll %d) (i %d)\n", rank, i, peer, step, coll_type, i);
                        res = MPI_Isend(((char*) buf) + blocks_displs[i]*dtsize, blocks_sizes[i], sendtype, peer, tag, comm, &(requests_s[num_requests_s]));
                        if(res != MPI_SUCCESS){return res;}
                        ++num_requests_s;
                    }                
                }
            }
            if(my_blocks_matrix[block_step][recv_index]){
                if(srtype == SENDRECV_IDEAL){
                    recvcnt += blocks_sizes[i];
                    recvblocks++;
                }else{
                    if(!rdma){
                        DPRINTF("[%d] Receiving block %d from %d at step %d (coll %d) (i %d)\n", rank, i, peer, step, coll_type, i);
                        res = MPI_Irecv(((char*) rbuf) + blocks_displs[i]*dtsize, blocks_sizes[i], recvtype, peer, tag, comm, &(requests_r[num_requests_r]));
                        if(res != MPI_SUCCESS){return res;}
                        req_idx_to_block_idx[num_requests_r] = i;
                        ++num_requests_r;
                    }
                }
            }
        }

        // Overlap here
        if(step != num_steps - 1){
            size_t block_step_next = (coll_type == SWING_REDUCE_SCATTER)?step + 1:(num_steps - step - 1 - 1);            
            __builtin_prefetch(my_blocks_matrix[block_step_next], 0, 0);
        }
        // End overlap
        if(srtype == SENDRECV_IDEAL){
            MPI_Sendrecv(buf, sendcnt, sendtype,
                        peer, tag,
                        rbuf, recvcnt, recvtype,
                        peer, tag, comm, MPI_STATUS_IGNORE);
            for(size_t i = 0; i < recvblocks; i++){
                size_t displ_bytes = dtsize*blocks_displs[i];
                void* rbuf_block = (void*) (((char*) rbuf) + displ_bytes);
                void* buf_block = (void*) (((char*) buf) + displ_bytes);            
                MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[i], sendtype, op); 
            }
        }else if(!rdma){
            if(coll_type == SWING_REDUCE_SCATTER){
                int index, completed = 0;
                for(size_t i = 0; i < num_requests_r; i++){
    #ifdef ACTIVE_WAIT
                    int flag = 0;
                    do{
                        MPI_Testany(num_requests_r, requests_r, &index, &flag, MPI_STATUS_IGNORE);
                    }while(!flag);

    #else
                    res = MPI_Waitany(num_requests_r, requests_r, &index, MPI_STATUS_IGNORE);
                    if(res != MPI_SUCCESS){return res;}
    #endif                
                    int block_idx = req_idx_to_block_idx[index];
                    size_t displ_bytes = dtsize*blocks_displs[block_idx];
                    void* rbuf_block = (void*) (((char*) rbuf) + displ_bytes);
                    void* buf_block = (void*) (((char*) buf) + displ_bytes);            
                    MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[block_idx], sendtype, op); 
                }
            }else{
    #ifdef ACTIVE_WAIT
                int index;
                for(size_t i = 0; i < num_requests_r; i++){
                    int flag = 0;
                    do{
                        MPI_Testany(num_requests_r, requests_r, &index, &flag, MPI_STATUS_IGNORE);
                    }while(!flag);
                }
    #else                            
                res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
                if(res != MPI_SUCCESS){return res;}        
    #endif
            }
    #ifdef ACTIVE_WAIT
            int index;
            for(size_t i = 0; i < num_requests_s; i++){
                int flag = 0;
                do{
                    MPI_Testany(num_requests_s, requests_s, &index, &flag, MPI_STATUS_IGNORE);
                }while(!flag);
            }
    #else                            
            res = MPI_Waitall(num_requests_s, requests_s, MPI_STATUSES_IGNORE);
            if(res != MPI_SUCCESS){return res;}        
    #endif
        }else{ // rdma
            MPI_Win_fence(0, win);
        }
    }

    if(rdma){
        MPI_Win_free(&win);
    }

    free(requests_s);
    free(requests_r);
    free(req_idx_to_block_idx);
    return 0;
}

static int MPI_Reduce_scatter_swing(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, uint** peers, char** my_blocks_matrix, uint* block_remapping){
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
    size_t my_displ_bytes;
    size_t my_count_bytes;

    if(block_remapping){
        my_displ_bytes = displs[block_remapping[rank]]*dtsize;
        my_count_bytes = recvcounts[block_remapping[rank]]*dtsize;
    }else{
        my_displ_bytes = displs[rank]*dtsize;
        my_count_bytes = recvcounts[rank]*dtsize;
    }

    memcpy(buf, sendbuf, total_size_bytes);
    if(srtype == SENDRECV_BBBN || srtype == SENDRECV_IDEAL){
        res = swing_coll_bbbn(buf, rbuf, recvcounts, displs, op, comm, size, rank, datatype, datatype, SWING_REDUCE_SCATTER, peers, my_blocks_matrix, block_remapping);
    }else if(srtype == SENDRECV_CONT){
        res = swing_coll_cont(buf, rbuf, recvcounts, displs, op, comm, size, rank, datatype, datatype, SWING_REDUCE_SCATTER, peers, my_blocks_matrix, block_remapping);
    }else{
        res = swing_coll(buf, rbuf, recvcounts, displs, op, comm, size, rank, datatype, datatype, SWING_REDUCE_SCATTER, peers, my_blocks_matrix);
    }
    // Copy the block in recvbuf
    memcpy(recvbuf, ((char*) buf) + my_displ_bytes, my_count_bytes);
    free(buf);
    free(rbuf);
    free(displs);
    return res;
}

static inline int MPI_Allreduce_lat_optimal_swing(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){    
    int res, size, rank, dtsize;    
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(datatype, &dtsize);

    // Create a temporary buffer (to avoid overwriting sendbuf)
    size_t total_size_bytes = count*dtsize;        
    char* rbuf;

    int num_steps = ceil(log2(size));
    uint** peers = (uint**) malloc(sizeof(uint*)*size);    
    peers[rank] = (uint*) malloc(sizeof(uint)*num_steps); // It's stupid but avoids changing too much stuff
    DPRINTF("[%d] Computing peers\n", rank);   
    // Here I need to compute only my peers       
    compute_peers(peers, 0, num_steps, rank, 1);
    
    for(size_t step = 0; step < num_steps; step++){
        DPRINTF("[%d] Starting step %d\n", rank, step);        
        uint32_t peer = peers[rank][step];   
        if(step == 0){
            MPI_Request requests[2];
            res = MPI_Isend(sendbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, &(requests[0]));
            if(res != MPI_SUCCESS){return res;}
            res = MPI_Irecv(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, &(requests[1]));
            if(res != MPI_SUCCESS){return res;}
            // While data is transmitted, we allocate buffer for the next steps
            rbuf = (char*) malloc(count*dtsize);
            res = MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            if(res != MPI_SUCCESS){return res;}
            MPI_Reduce_local(sendbuf, recvbuf, count, datatype, op);
        }else{
            // For latency optimal, I send/recv all the data
            res = MPI_Sendrecv(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE,
                            rbuf, count, datatype, peer, TAG_SWING_ALLREDUCE,  
                            comm, MPI_STATUS_IGNORE);
            if(res != MPI_SUCCESS){return res;}
            // Aggregate    
            MPI_Reduce_local(rbuf, recvbuf, count, datatype, op);
        }
    }    
    free(peers[rank]);
    free(peers);
    free(rbuf);
    return res;
}


static int MPI_Allgatherv_swing(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, 
                         const int *displs, MPI_Datatype recvtype, MPI_Comm comm, uint** peers, char** my_blocks_matrix, uint* blocks_remapping){
    int res, size, rank, dtsize;    
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(recvtype, &dtsize);
    // Create a temporary buffer (to avoid overwriting sendbuf)
    size_t buf_size = 0;
    for(size_t i = 0; i < size; i++){
        buf_size += recvcounts[i]; // TODO: If custom datatypes, manage appropriately        
    }
    int displ, count;
    if(blocks_remapping){
        displ = displs[blocks_remapping[rank]];
        count = recvcounts[blocks_remapping[rank]];        
    }else{
        displ = displs[rank];
        count = recvcounts[rank];
    }
    memcpy(((char*) recvbuf) + displ*dtsize, sendbuf, count*dtsize);
    if(srtype == SENDRECV_BBBN || srtype == SENDRECV_IDEAL){
        res = swing_coll_bbbn(recvbuf, recvbuf, recvcounts, displs, MPI_SUM, comm, size, rank, sendtype, recvtype, SWING_ALLGATHER, peers, my_blocks_matrix, blocks_remapping);
    }else if(srtype == SENDRECV_CONT){
        res = swing_coll_cont(recvbuf, recvbuf, recvcounts, displs, MPI_SUM, comm, size, rank, sendtype, recvtype, SWING_ALLGATHER, peers, my_blocks_matrix, blocks_remapping);
    }else{
        res = swing_coll(recvbuf, recvbuf, recvcounts, displs, MPI_SUM, comm, size, rank, sendtype, recvtype, SWING_ALLGATHER, peers, my_blocks_matrix);
    }
    return res;
}

// Code copied from MPICH repo (https://github.com/pmodels/mpich/tree/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce)
static int MPI_Allreduce_recdoub_l(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank, rem, newdst;
    MPI_Aint extent;
    void *tmp_buf;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);
    extent = dtsize;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    is_commutative = true;

    tmp_buf = malloc(count*extent);
    memcpy(recvbuf, sendbuf, count*extent);

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = pow(2, floor(log2(comm_size)));
    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPI_Send(recvbuf, count,
                                datatype, rank + 1, TAG_SWING_ALLREDUCE, comm);
            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPI_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
             * into tmp_buf */
            mpi_errno = MPI_Sendrecv(recvbuf, count, datatype,
                                      dst, TAG_SWING_ALLREDUCE, tmp_buf,
                                      count, datatype, dst,
                                      TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);        

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
                mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            } else {
                /* op is noncommutative and the order is not right */
                mpi_errno = MPI_Reduce_local(recvbuf, tmp_buf, count, datatype, op);
                /* copy result back into recvbuf */
                memcpy(recvbuf, tmp_buf, count*extent);
            }
            mask <<= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank - 1, TAG_SWING_ALLREDUCE, comm);
        else    /* even */
            mpi_errno = MPI_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    }
    return mpi_errno_ret;
}

// Code copied from MPICH repo (https://github.com/pmodels/mpich/tree/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce)
static int MPI_Allreduce_recdoub_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i, send_idx, recv_idx, last_idx;
    void *tmp_buf;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);
    MPI_Aint extent = dtsize;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    /* need to allocate temporary buffer to store incoming data */
    tmp_buf = malloc(count * dtsize);

    /* copy local data into recvbuf */
    memcpy(recvbuf, sendbuf, count*dtsize);

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = pow(2, floor(log2(comm_size)));

    rem = comm_size - pof2;
    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank + 1, TAG_SWING_ALLREDUCE, comm);

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPI_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */
    if (newrank != -1) {
        MPI_Aint *cnts, *disps;
        cnts = (MPI_Aint*) malloc(pof2 * sizeof(MPI_Aint));
        disps = (MPI_Aint*) malloc(pof2 * sizeof(MPI_Aint));
        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }            

            /* Send data from recvbuf. Recv into tmp_buf */
            mpi_errno = MPI_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, TAG_SWING_ALLREDUCE,
                                      (char *) tmp_buf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            mpi_errno = MPI_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPI_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, TAG_SWING_ALLREDUCE,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank - 1, TAG_SWING_ALLREDUCE, comm);
        else    /* even */
            mpi_errno = MPI_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    }
    return mpi_errno_ret;
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
                  datatype, recv_from, TAG_SWING_ALLREDUCE, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[send_chunk],
                MPI_FLOAT, send_to, TAG_SWING_ALLREDUCE, MPI_COMM_WORLD);

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
                     datatype, send_to, TAG_SWING_ALLREDUCE, segment_recv,
                     segment_sizes[recv_chunk], datatype, recv_from,
                     TAG_SWING_ALLREDUCE, MPI_COMM_WORLD, &recv_status);
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
        int size, rank;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        // Compute the peers
        uint** peers = (uint**) malloc(sizeof(uint*)*size);
        int num_steps = (int) ceil(log2(size));
        for(uint i = 0; i < size; i++){
            peers[i] = (uint*) malloc(sizeof(uint)*num_steps);
        }

        DPRINTF("[%d] Computing peers\n", rank);
        compute_peers(peers, 0, num_steps, 0, size); // TODO: For now we assume it is single-ported (and we pass port 0), extending should be trivial
        uint8_t* reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
        char** my_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);    
        for(size_t step = 0; step < num_steps; step++){    
            my_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
        }
        DPRINTF("[%d] Getting bitmaps\n", rank);
        getBitmapsMatrix(rank, size, my_blocks_matrix, reached_step, num_steps, peers, 0, NULL, -1);
        free(reached_step);
        int res = MPI_Reduce_scatter_swing(sendbuf, recvbuf, recvcounts, datatype, op, comm, peers, my_blocks_matrix, NULL);
        for(uint i = 0; i < size; i++){
            free(peers[i]);
        }
        free(peers);
        for(uint i = 0; i < num_steps; i++){
            free(my_blocks_matrix[i]);
        }
        free(my_blocks_matrix);
        return res;
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
        int size, rank;    
        MPI_Comm_size(comm, &size);    
        MPI_Comm_rank(comm, &rank);
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
        int size, rank;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        uint** peers = (uint**) malloc(sizeof(uint*)*size);
        int num_steps = (int) ceil(log2(size));
        for(uint i = 0; i < size; i++){
            peers[i] = (uint*) malloc(sizeof(uint)*num_steps);
        }

        DPRINTF("[%d] Computing peers\n", rank);
        compute_peers(peers, 0, num_steps, 0, size); // TODO: For now we assume it is single-ported (and we pass port 0), extending should be trivial        
        uint8_t* reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
        char** my_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);    
        for(size_t step = 0; step < num_steps; step++){    
            my_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
        }
        DPRINTF("[%d] Getting bitmaps\n", rank);
        getBitmapsMatrix(rank, size, my_blocks_matrix, reached_step, num_steps, peers, 0, NULL, -1);
        free(reached_step);
        int res = MPI_Allgatherv_swing(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, peers, my_blocks_matrix, NULL);
        for(uint i = 0; i < size; i++){
            free(peers[i]);
        }
        free(peers);
        for(uint i = 0; i < num_steps; i++){
            free(my_blocks_matrix[i]);
        }
        free(my_blocks_matrix);
        return res;
    }else{
        assert("Only Swing supported for Allgatherv." == 0);
    }
}

static std::string vectostr(const std::vector<int>& v){
    std::string s = "";
    for(auto i : v){
        s += std::to_string(i) + ",";
    }
    return s;
}

static void remap(const std::vector<int>& nodes, uint start_range, uint end_range, uint* blocks_remapping, char** zero_blocks_matrix, int step, int size, int rank, uint** peers, int simrank, int num_steps){
    if(nodes.size() == 2){
        blocks_remapping[nodes[0]] = start_range;
        blocks_remapping[nodes[1]] = end_range - 1;
#ifdef DEBUG
        if(rank == 0){
            printf("Base case, mapping %d->%d and %d->%d\n", nodes[0], start_range, nodes[1], end_range - 1);
        }
#endif
        assert(end_range == start_range + 2);
    }else{
        // Find two partitions of node that talk with each other. If I have n nodes, 
        // if I see what happens in next step, I have two disjoint sets of nodes.
        std::vector<int> left, right;
        for(auto n : nodes){
            if(zero_blocks_matrix[step][n] == 0){
                left.push_back(n);
            }else{
                right.push_back(n);
            }
        }
#ifdef DEBUG
        if(rank == 0){
            std::cout << "Remapping " << vectostr(left) << " (" << left.size() << " elements) on range " << start_range << " " << start_range + left.size() << std::endl;
        }              
#endif
        remap(left, start_range, start_range + left.size(), blocks_remapping, zero_blocks_matrix, step + 1, size, rank, peers, simrank, num_steps);

#ifdef DEBUG
        if(rank == 0){
            std::cout << "Remapping " << vectostr(right) << " (" << right.size() << " elements) on range " << end_range - right.size() << " " << end_range << std::endl;
        }              
#endif
        uint8_t* reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
        char** tmp_block_matrix = (char**) malloc(sizeof(char*)*num_steps);    
        for(size_t step = 0; step < num_steps; step++){
            tmp_block_matrix[step] = (char*) malloc(sizeof(char)*size);
        }
        getBitmapsMatrix(peers[simrank][step], size, tmp_block_matrix, reached_step, num_steps, peers, 0, NULL, -1);
        free(reached_step);        
        remap(right, end_range - right.size(), end_range, blocks_remapping, tmp_block_matrix, step + 1, size, rank, peers, peers[simrank][step], num_steps);
        for(size_t step = 0; step < num_steps; step++){    
            free(tmp_block_matrix[step]);
        }
        free(tmp_block_matrix);
    }
}

static int MPI_Allreduce_int(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    if(disable_allreduce || algo == ALGO_DEFAULT){
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_SWING){
        int size, dtsize, rank;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        MPI_Type_size(datatype, &dtsize);        
        // Compute total size of data
        size_t total_size_bytes = count*dtsize;

        // TODO: Fix (Cont not working for non-powers-of-two)
        if(srtype == SENDRECV_CONT && (size & (size - 1)) != 0){
            srtype = SENDRECV_BBBN;
        }

        // Compute the peers
        
        int latency_optimal = (total_size_bytes <= latency_optimal_threshold) || (count < size); // TODO Adjust for tori        
        if(latency_optimal){
            return MPI_Allreduce_lat_optimal_swing(sendbuf, recvbuf, count, datatype, op, comm); // Misleading, should call the function in a different way            
        }else{
            int res;
            int* recvcounts = (int*) malloc(sizeof(int)*size);
            int* displs = (int*) malloc(sizeof(int)*size);
            int num_steps = (int) ceil(log2(size));
            uint** peers; 
            char** my_blocks_matrix;
            uint* blocks_remapping = NULL;

            if(!cache || cached_p != size){
                // Delete previous cache if available (different p)
                if(cache && cached_peers){
                    for(size_t step = 0; step < num_steps; step++){    
                        free(cached_my_blocks_matrix[step]);
                    }
                    free(cached_my_blocks_matrix);
                    for(uint i = 0; i < size; i++){
                        free(cached_peers[i]);
                    }
                    free(cached_peers);
                    free(cached_blocks_remapping);
                }
                peers = (uint**) malloc(sizeof(uint*)*size);
                for(uint i = 0; i < size; i++){
                    peers[i] = (uint*) malloc(sizeof(uint)*num_steps);
                }           
                DPRINTF("[%d] Computing peers\n", rank);
                compute_peers(peers, 0, num_steps, 0, size); // TODO: For now we assume it is single-ported (and we pass port 0), extending should be trivial
                
                my_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);    
                for(size_t step = 0; step < num_steps; step++){    
                    my_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
                }
                DPRINTF("[%d] Getting bitmaps\n", rank);
                uint8_t* reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
                getBitmapsMatrix(rank, size, my_blocks_matrix, reached_step, num_steps, peers, 0, NULL, -1);
                free(reached_step);


                // Block remapping
                if(srtype == SENDRECV_CONT){
                    uint8_t* reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
                    char** zero_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);    
                    for(size_t step = 0; step < num_steps; step++){    // TODO: Cache it
                        zero_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
                    }
                    getBitmapsMatrix(0, size, zero_blocks_matrix, reached_step, num_steps, peers, 0, NULL, -1);
                    free(reached_step);
                    blocks_remapping = (uint*) malloc(sizeof(uint)*size);                
                    // Setting everything to 'size' to indicate it has not been set yet
                    for(size_t j = 0; j < size; j++){
                        blocks_remapping[j] = size;
                    }
                    
                    std::vector<int> nodes(size);
                    for(size_t i = 0; i < size; i++){
                        nodes[i] = i;
                    }
                    remap(nodes, 0, size, blocks_remapping, zero_blocks_matrix, 0, size, rank, peers, 0, num_steps);
    #ifdef DEBUG
                    if(rank == 0){
                        printf("Remapped: ");
                        for(size_t i = 0; i < size; i++){
                            printf("%d ", blocks_remapping[i]);
                        }
                        printf("\n");
                    }
    #endif          

                    for(size_t step = 0; step < num_steps; step++){    
                        free(zero_blocks_matrix[step]);
                    }
                    free(zero_blocks_matrix);
                }

                if(cache){
                    cached_peers = peers;
                    cached_p = size;
                    cached_my_blocks_matrix = my_blocks_matrix;
                    cached_blocks_remapping = blocks_remapping;
                }
            }else{
                peers = cached_peers;
                my_blocks_matrix = cached_my_blocks_matrix;
                blocks_remapping = cached_blocks_remapping;
            }

            size_t last = 0;
            size_t normcount = ceil(count / size);
            for(size_t i = 0; i < size; i++){
                recvcounts[i] = normcount;
                if(i == size - 1){
                    recvcounts[i] = count - ((count / size)*(size - 1));
                } 
                displs[i] = last;
                last += recvcounts[i];
            }
            int intermediate_rank = rank;
            if(srtype == SENDRECV_CONT){
                intermediate_rank = blocks_remapping[rank];
            }
            char* intermediate_buf = ((char*) recvbuf) + displs[intermediate_rank]*dtsize;
            res = MPI_Reduce_scatter_swing(sendbuf, intermediate_buf, recvcounts, datatype, op, comm, peers, my_blocks_matrix, blocks_remapping);
            if(res == MPI_SUCCESS){        
                res = MPI_Allgatherv_swing(intermediate_buf, recvcounts[intermediate_rank], datatype, recvbuf, recvcounts, displs, datatype, comm, peers, my_blocks_matrix, blocks_remapping);
            }
            free(recvcounts);
            free(displs);            

            if(!cache){
                for(uint i = 0; i < size; i++){
                    free(peers[i]);
                }
                free(peers);     
                for(size_t step = 0; step < num_steps; step++){    
                    free(my_blocks_matrix[step]);
                }
                free(my_blocks_matrix);
                if(srtype == SENDRECV_CONT){
                    free(blocks_remapping);
                }
            }
            return res;  
        }         
    }else if(algo == ALGO_RING){
        return MPI_Allreduce_ring(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_RECDOUB_B){ // Recdoub_b
        return MPI_Allreduce_recdoub_b(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_RECDOUB_L){ // Recdoub_l
        return MPI_Allreduce_recdoub_l(sendbuf, recvbuf, count, datatype, op, comm);
    }else{
        return 1;
    }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_allreduce || algo == ALGO_DEFAULT){
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(switch_point){
        int size, dtsize, rank, res;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        MPI_Type_size(datatype, &dtsize);    
        int remaining_count = count;
        int max_count, next_offset = 0;
        if(algo == ALGO_RING){
            if(((count*dtsize)/size) <= switch_point){
                max_count = count;
            }else{
                max_count = (switch_point*size)/dtsize;
            }
        }else{ // RECDOUB/SWING
            if(((count*dtsize)/2) <= switch_point){
                max_count = count;
            }else{
                max_count = (switch_point*2)/dtsize;
            }
        }
        do{
            int next_count = std::min(remaining_count, max_count);
            res = MPI_Allreduce_int(((char*)sendbuf) + next_offset*dtsize, ((char*) recvbuf) + next_offset*dtsize, next_count, datatype, op, comm);
            if(res != MPI_SUCCESS){
                return res;
            }
            next_offset += next_count;
            remaining_count -= next_count;
        }while(remaining_count > 0);            
        return MPI_SUCCESS;
    }else{
        return MPI_Allreduce_int(sendbuf, recvbuf, count, datatype, op, comm);
    }
}
// TODO: Don't use Swing for non-continugous non-native datatypes (tedious implementation)