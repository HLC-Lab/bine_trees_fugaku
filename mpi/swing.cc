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
    SWING_ALLGATHER
}CollType;

typedef enum{
    ALGO_SWING = 0,
    ALGO_RING,
    ALGO_RECDOUB
}Algo;

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allgather = 0, disable_allreduce = 0, 
                    dimensions_num = 1, latency_optimal_threshold = 1024, force_env_reload = 1, env_read = 0;
static Algo algo;
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
            if(strcmp(env_str, "SWING") == 0){
                algo = ALGO_SWING;
            }else if(strcmp(env_str, "RING") == 0){
                algo = ALGO_RING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                algo = ALGO_RECDOUB;
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

static int sendrecv(char* send_bitmap, char* recv_bitmap, 
                    int* array_of_blocklengths_s, int* array_of_displacements_s,
                    int* array_of_blocklengths_r, int* array_of_displacements_r,
                    int dest, int source, int sendtag, int recvtag, 
                    void* buf, void* rbuf, const int *blocks_sizes, 
                    const int *blocks_displs, MPI_Comm comm, int size, int rank,  
                    MPI_Datatype sendtype, MPI_Datatype recvtype){
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
    if(rbuf){
        res = MPI_Sendrecv(buf, 1, indexed_sendtype, dest, sendtag,
                           rbuf, 1, indexed_recvtype, source, recvtag,  
                           comm, MPI_STATUS_IGNORE);
    }else{
        // For allgather we can use buf instead of rbuf
        res = MPI_Sendrecv(buf, 1, indexed_sendtype, dest, sendtag,
                           buf, 1, indexed_recvtype, source, recvtag,  
                           comm, MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&indexed_sendtype);
    MPI_Type_free(&indexed_recvtype);
    return res;
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

    my_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);
    peer_blocks_matrix = (char**) malloc(sizeof(char*)*num_steps);
    reached_step = (uint8_t*) malloc(sizeof(uint8_t)*size);
    array_of_blocklengths_s = (int*) malloc(sizeof(int)*size);
    array_of_displacements_s = (int*) malloc(sizeof(int)*size);
    array_of_blocklengths_r = (int*) malloc(sizeof(int)*size);
    array_of_displacements_r = (int*) malloc(sizeof(int)*size);
    for(size_t step = 0; step < num_steps; step++){    
        my_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
        peer_blocks_matrix[step] = (char*) malloc(sizeof(char)*size);
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

    // Compute total size of data
    size_t total_size_bytes = 0;
    for(size_t i = 0; i < size; i++){
        total_size_bytes += dtsize*blocks_sizes[i]; // TODO: Check for custom datatypes
    }
    DPRINTF("[%d] swing_coll called on %d bytes\n", rank, total_size_bytes);
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
        // Find which blocks I must send and recv.
        if(total_size_bytes <= latency_optimal_threshold){
            // For latency optimal, I send/recv all the data
            for(size_t i = 0; i < size; i++){
                blocks_bitmap_s[i] = 1;
                blocks_bitmap_r[i] = 1;
            }
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
                uint reversed_step = int(num_steps - step - 1);
                peer = peers[rank][reversed_step];
                getBitmaps(peer, size, peer_blocks_matrix, reached_step, num_steps, peers);
                blocks_bitmap_s = peer_blocks_matrix[reversed_step];
                blocks_bitmap_r = my_blocks_matrix[reversed_step];
            }
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

        // Sendrecv
        int res = sendrecv(blocks_bitmap_s, blocks_bitmap_r, 
                           array_of_blocklengths_s, array_of_displacements_s,
                           array_of_blocklengths_r, array_of_displacements_r,
                           peer, peer, tag, tag, 
                           buf, rbuf, blocks_sizes, blocks_displs, 
                           comm, size, rank, sendtype, recvtype);
        if(res != MPI_SUCCESS){
            return res;
        }

#ifdef PERF_DEBUGGING
        end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
#endif

        // Aggregate    
        if(coll_type == SWING_REDUCE_SCATTER){
            for(size_t i = 0; i < size; i++){
                if(blocks_bitmap_r[i]){
                    void* rbuf_block = (void*) (((char*) rbuf) + dtsize*blocks_displs[i]);
                    void* buf_block = (void*) (((char*) buf) + dtsize*blocks_displs[i]);
                    DPRINTF("[%d] Step %d, aggregating %d elements, displ %d. Before: %f, rbuf block: %f\n", rank, step, blocks_sizes[i], blocks_displs[i], ((float*)buf_block)[0], ((float*)rbuf_block)[0]);
                    MPI_Reduce_local(rbuf_block, buf_block, blocks_sizes[i], sendtype, op);
                }
            }
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
    free(array_of_blocklengths_s);
    free(array_of_displacements_s);
    free(array_of_blocklengths_r);
    free(array_of_displacements_r);
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
    char* buf = (char*) malloc(buf_size*dtsize);    
    char* rbuf = (char*) malloc(buf_size*dtsize);
    memset(buf, 0, sizeof(char)*buf_size*dtsize);
    size_t my_displ_bytes = displs[rank]*dtsize;
    size_t my_count_bytes = recvcounts[rank]*dtsize;
#ifndef PERF_DEBUGGING
    memcpy(buf, sendbuf, buf_size*dtsize);
#endif
    res = swing_coll(buf, rbuf, recvcounts, displs, op, comm, size, rank, datatype, datatype, SWING_REDUCE_SCATTER);
    // Copy the block in recvbuf
#ifndef PERF_DEBUGGING
    memcpy(recvbuf, ((char*) buf) + my_displ_bytes, my_count_bytes);
#endif    
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
    char* buf = (char*) malloc(buf_size*dtsize);    
#ifndef PERF_DEBUGGING
    memcpy(((char*) buf) + displs[rank]*dtsize, sendbuf, recvcounts[rank]*dtsize);
#endif
    res = swing_coll(buf, NULL, recvcounts, displs, MPI_SUM, comm, size, rank, sendtype, recvtype, SWING_ALLGATHER);
#ifndef PERF_DEBUGGING
    memcpy(recvbuf, buf, buf_size*dtsize);
#endif
    free(buf);
    return res;
}

static int MPI_Allreduce_ring(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    return 0;
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_reducescatter){
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
    if(disable_allgather){
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
    if(disable_allgatherv){
        return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }else if(algo == ALGO_SWING){
        return MPI_Allgatherv_swing(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }else{
        assert("Only Swing supported for Allgatherv." == 0);
    }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_allreduce){
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

        char* intermediate_buf = (char*) recvbuf + displs[rank];
        if(disable_reducescatter){
            res = PMPI_Reduce_scatter(sendbuf, intermediate_buf, recvcounts, datatype, op, comm);
        }else{
            res = MPI_Reduce_scatter_swing(sendbuf, intermediate_buf, recvcounts, datatype, op, comm);
        }
        if(res == MPI_SUCCESS){        
            if(disable_allgatherv){
                res = PMPI_Allgatherv(intermediate_buf, recvcounts[rank], datatype, recvbuf, recvcounts, displs, datatype, comm);
            }else{
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
