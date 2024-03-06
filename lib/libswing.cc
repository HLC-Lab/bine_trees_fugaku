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
#include <inttypes.h>

#define MAX_SUPPORTED_DIMENSIONS 6 // We support up to 6D torus
#define MAX_SUPPORTED_PORTS (MAX_SUPPORTED_DIMENSIONS*2)
#define LIBSWING_MAX_STEPS 20 // With this we are ok up to 2^20 nodes, add other terms to the following arrays if needed.
static int rhos[LIBSWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};
static int smallest_negabinary[LIBSWING_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBSWING_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

#define TAG_SWING_REDUCESCATTER (0x7FFF - MAX_SUPPORTED_PORTS*1)
#define TAG_SWING_ALLGATHER     (0x7FFF - MAX_SUPPORTED_PORTS*2)
#define TAG_SWING_ALLREDUCE     (0x7FFF - MAX_SUPPORTED_PORTS*3)

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
    ALGO_SWING_L,
    ALGO_SWING_B,
    ALGO_SWING_B_CONT,
    ALGO_RING,
    ALGO_RECDOUB_L,
    ALGO_RECDOUB_B,
}Algo;

typedef struct{
    size_t offset;
    size_t count;
}ChunkInfo;

typedef struct{
    int rank;
    int size;
    int dtsize;
    int num_steps;
    size_t num_steps_per_dim[MAX_SUPPORTED_DIMENSIONS];
    uint num_ports;
    ChunkInfo** chunks; // One per chunk (each of size max_size). Each element has one offset/count per port.
    size_t num_chunks;
}SwingInfo;

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allgather = 0, disable_allreduce = 0, 
                    dimensions_num = 1, force_env_reload = 1, env_read = 0, contiguous_blocks = 0,
                    rdma = 0, max_size = 0; //2097152;
static Algo algo = ALGO_SWING_B;
static uint dimensions[MAX_SUPPORTED_DIMENSIONS];
static int multiport = 0; // If 1, assumes that the number of ports is equal to twice the number of dimensions

static inline void read_env(MPI_Comm comm){
    char* env_str = getenv("LIBSWING_FORCE_ENV_RELOAD");
    if(env_str){
        force_env_reload = 1;
    }else{
        force_env_reload = 0;
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

        env_str = getenv("LIBSWING_CONTIGUOUS_BLOCKS");
        if(env_str){
            contiguous_blocks = atoi(env_str);
        }
       
        env_str = getenv("LIBSWING_RDMA");
        if(env_str){
            rdma = atoi(env_str);
        }
        
        env_str = getenv("LIBSWING_MAX_SIZE");
        if(env_str){
            max_size = atoi(env_str);
        }

        env_str = getenv("LIBSWING_MULTIPORT");
        if(env_str){
            multiport = atoi(env_str);
        }

        env_str = getenv("LIBSWING_ALGO");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                algo = ALGO_DEFAULT;
            }else if(strcmp(env_str, "SWING_L") == 0){
                algo = ALGO_SWING_L;
            }else if(strcmp(env_str, "SWING_B") == 0){
                algo = ALGO_SWING_B;
            }else if(strcmp(env_str, "SWING_B_CONT") == 0){
                algo = ALGO_SWING_B_CONT;
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
            printf("LIBSWING_RDMA: %d\n", rdma);
            printf("LIBSWING_ALGO: %d\n", algo);
            printf("LIBSWING_MAX_SIZE: %d\n", max_size);
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


// Convert a rank id into a list of d-dimensional coordinates (adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c)
// Row-major order, i.e., row coordinates change the slowest (i.e., we first increase depth, than cols, then rows -- https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays) 
static inline void getCoordFromId(int id, uint* coord, int nnodes, uint* dimensions_virtual = NULL){
    if(dimensions_virtual == NULL){
        dimensions_virtual = dimensions;
    }

    for (uint i = 0; i < dimensions_num; i++) {
        nnodes = nnodes / dimensions_virtual[i];
        coord[i] = id / nnodes;
        id = id % nnodes;
    }
}

// Convert d-dimensional coordinates into a rank id (adapted from MPICH code -- https://github.com/pmodels/mpich/blob/94b1cd6f060cafbf68d6d83ea551a8bcc8fcecd4/src/mpi/topo/topo_impl.c)
// Dimensions are (rows, cols, depth)
static inline int getIdFromCoord(uint* coords, uint* dimensions, uint dimensions_num){
    int rank = 0;
    int multiplier = 1;
    uint coord;
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

static int inline is_odd(int x){
    return x & 1;
}

// Computes the mapping between rank and coordinates. We do this only
// once at the beginning to avoid doing this multiple times.
static inline void compute_rank_to_coord_mapping(uint size, uint* dimensions, uint* coordinates, int nnodes){
    for(uint i = 0; i < size; i++){
        getCoordFromId(i, &(coordinates[i*dimensions_num]), nnodes, dimensions);
    }
}

static inline void retrieve_coord_mapping(uint* coordinates, uint rank, uint* coord){
    memcpy(coord, &(coordinates[rank*dimensions_num]), sizeof(uint)*dimensions_num);
}

static inline void compute_peers(uint* peers, int port, int num_steps, uint rank, uint* coordinates, uint* dimensions_virtual = NULL){
    if(dimensions_virtual == NULL){
        dimensions_virtual = dimensions;
    }
    bool terminated_dimensions_bitmap[MAX_SUPPORTED_DIMENSIONS];
    int num_steps_per_dim[MAX_SUPPORTED_DIMENSIONS];
    uint8_t next_step_per_dim[MAX_SUPPORTED_DIMENSIONS];
    memset(next_step_per_dim, 0, sizeof(uint8_t)*MAX_SUPPORTED_DIMENSIONS);
    for(size_t i = 0; i < dimensions_num; i++){
        num_steps_per_dim[i] = ceil(log2(dimensions_virtual[i]));
    }
    // Compute default directions
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, rank, coord);
    for(size_t i = 0; i < dimensions_num; i++){
        terminated_dimensions_bitmap[i] = false;            
    }
    
    int target_dim, relative_step, distance, last_dim = port - 1;
    uint terminated_dimensions = 0, o = 0;
    
    // Generate peers
    for(size_t i = 0; i < (uint) num_steps; ){            
        if(dimensions_num > 1){
            retrieve_coord_mapping(coordinates, rank, coord); // Regenerate rank coord
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
            coord[target_dim] = mod((coord[target_dim] + distance), dimensions_virtual[target_dim]); // We need to use mod to avoid negative coordinates
            if(dimensions_num > 1){
                peers[i] = getIdFromCoord(coord, dimensions_virtual, dimensions_num);
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

// Code copied from MPICH repo (https://github.com/pmodels/mpich/tree/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce)
static int MPI_Allreduce_recdoub_l(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
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
        if (!is_odd(rank)) {    /* even */
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
        if (is_odd(rank))   /* odd */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank - 1, TAG_SWING_ALLREDUCE, comm);
        else    /* even */
            mpi_errno = MPI_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    }
    free(tmp_buf);
    return mpi_errno;
}

// Code copied from MPICH repo (https://github.com/pmodels/mpich/tree/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce)
static int MPI_Allreduce_recdoub_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
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
        if (!is_odd(rank)) {    /* even */
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
        if (is_odd(rank))   /* odd */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank - 1, TAG_SWING_ALLREDUCE, comm);
        else    /* even */
            mpi_errno = MPI_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    }
    free(tmp_buf);
    return mpi_errno;
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
    assert(segment_ends[size - 1] == (uint) count);

     // Copy your data to the output buffer to avoid modifying the input buffer.
    memcpy(recvbuf, sendbuf, count*dtsize);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    char* buffer;
    buffer = (char*) malloc(segment_sizes[0]*dtsize);

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

/*
static inline int get_peer(int step, int rank, int size){
    uint32_t negabinary_repunit = (1 << (step + 1)) - 1; // 000...000111...111 (least significant step+1 bits set to 1)          
    uint32_t delta = negabinary_to_binary(negabinary_repunit); // TODO: Replace with a lookup table?
    // Flip the sign for odd nodes
    if(is_odd(rank)){delta = -delta;} 
    // TODO: manage multiport
    return mod((rank + delta), size);
}
*/

// Sends the data from nodes outside of the power-of-two boundary to nodes within the boundary.
// This is done one dimension at a time.
// Returns the new rank.
static inline int shrink_non_power_of_two(const void *sendbuf, void *recvbuf, int count, 
                                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, 
                                          char* tmpbuf, SwingInfo* info, uint* dimensions_virtual, 
                                          uint* coordinates, int* idle,
                                          int* rank_virtual, int* all_p2_dimensions){    
    uint coord_peer[MAX_SUPPORTED_DIMENSIONS];
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, info->rank, coord);
    *all_p2_dimensions = 1;

    // We do the swapping between buffers to avoid additional
    // explicit memcopies.
    // Contains the data aggregated so far.
    void* aggregated_buf = (void*) sendbuf;
    // Contains the buffer into which the data must be received.
    void* recvbuf_real = recvbuf;
    // Contains the data to aggregate to recvbuf.
    void* aggregation_source = (void*) sendbuf;

    for(size_t i = 0; i < dimensions_num; i++){
        // This dimensions is not a power of two, shrink it
        if(!is_power_of_two(dimensions[i])){
            memcpy(coord_peer, coord, sizeof(uint)*dimensions_num);
            dimensions_virtual[i] = pow(2, info->num_steps_per_dim[i] - 1);
            int extra = dimensions[i] - dimensions_virtual[i];
            if(coord[i] >= dimensions_virtual[i]){            
                coord_peer[i] = coord[i] - extra;
                int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                DPRINTF("[%d] Sending to %d\n", info->rank, peer);
                int res = MPI_Send(aggregated_buf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm);                
                if(res != MPI_SUCCESS){return res;}
                *idle = 1;
                break;
            }else if(coord[i] + extra >= dimensions_virtual[i]){
                coord_peer[i] = coord[i] + extra;
                int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                DPRINTF("[%d] Receiving from %d\n", info->rank, peer);
                int res = MPI_Recv(recvbuf_real, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, NULL);                
                if(res != MPI_SUCCESS){return res;}
                MPI_Reduce_local(aggregation_source, recvbuf, count, datatype, op);
                aggregated_buf = recvbuf;
                recvbuf_real = tmpbuf;
                aggregation_source = tmpbuf;
            }else{
                // I am neither sending or receiving, thus I copy my sendbuf to recvbuf
                // so that later I can aggregate directly rbuf with recvbuf
                // I only need to do it once.
                if(*all_p2_dimensions){
                    memcpy(recvbuf, sendbuf, count*info->dtsize);
                    aggregated_buf = recvbuf;
                    recvbuf_real = tmpbuf;
                    aggregation_source = tmpbuf;
                }
            }
            *all_p2_dimensions = 0;
        }
    }
    *rank_virtual = getIdFromCoord(coord, dimensions_virtual, dimensions_num);
    return MPI_SUCCESS;
}

static inline int enlarge_non_power_of_two(void *recvbuf, int count, MPI_Datatype datatype, 
                                           MPI_Op op, MPI_Comm comm, SwingInfo* info, uint* coordinates){
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, info->rank, coord);
    uint coord_peer[MAX_SUPPORTED_DIMENSIONS];
    //for(size_t d = 0; d < dimensions_num; d++){
    for(int d = dimensions_num - 1; d >= 0; d--){
        // This dimensions was a non-power of two, enlarge it
        if(!is_power_of_two(dimensions[d])){
            memcpy(coord_peer, coord, sizeof(uint)*dimensions_num);
            int dim_virtual = pow(2, info->num_steps_per_dim[d] - 1);
            int extra = dimensions[d] - dim_virtual;
            if(coord[d] >= (uint) dim_virtual){                
                coord_peer[d] = coord[d] - extra;
                int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                DPRINTF("[%d] Receiving from %d\n", info->rank, peer);
                // I can overwrite the recvbuf and don't need to aggregate, since 
                // I was an extra node and did not participate to the actual allreduce
                int r = MPI_Recv(recvbuf, count, datatype, peer, TAG_SWING_ALLREDUCE, comm, NULL);
                if(r != MPI_SUCCESS){return r;}
            }else if(coord[d] + extra >= (uint) dim_virtual){
                coord_peer[d] = coord[d] + extra;
                int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                DPRINTF("[%d] Sending to %d\n", info->rank, peer);
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
// TODO: Do steps 1. and 3. also considering max_size?? (Probably not needed since for latency-optimal the buffer should never be that large)
// TODO: Exploit multiport for steps 1. and 3.?? (probably not worth for small messages)
static inline int MPI_Allreduce_lat_optimal_swing(const void *sendbuf, void *recvbuf, int count, 
                                                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info){
    // Reduce the number of steps since we shrink
    info->num_steps = 0;
    for(size_t d = 0; d < dimensions_num; d++){
        info->num_steps += floor(log2(dimensions[d]));
    }
    char* tmpbuf = (char*) malloc(count*info->dtsize); // Temporary buffer (to avoid overwriting sendbuf)
    // To avoid memcpying, the first recv+aggregation and the subsequent ones use different buffers
    // (see MPI_Allreduce_lat_optimal_swing_sendrecv). This variable keeps track of that.
    uint dimensions_virtual[MAX_SUPPORTED_DIMENSIONS];   

    // Compute real and virtual coordinates
    uint *coordinates, *coordinates_virtual;
    coordinates = (uint*) malloc(sizeof(uint)*info->size*dimensions_num);
    coordinates_virtual = (uint*) malloc(sizeof(uint)*info->size*dimensions_num);
    compute_rank_to_coord_mapping(info->size, dimensions, coordinates, info->size);
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, info->rank, coord);
    memcpy(dimensions_virtual, dimensions, sizeof(dimensions));
    
    int idle = 0;
    int rank_virtual = info->rank;
    int all_p2_dimensions = 1;
    int res;
    res = shrink_non_power_of_two(sendbuf, recvbuf, count, datatype, op, comm, tmpbuf, info, 
                                  dimensions_virtual, coordinates, &idle, &rank_virtual,
                                  &all_p2_dimensions);
    if(res != MPI_SUCCESS){return res;}
    int size_virtual = 1, num_steps_virtual = 0;
    for(size_t i = 0; i < dimensions_num; i++){
        size_virtual *= dimensions_virtual[i];
        num_steps_virtual += ceil(log2(dimensions_virtual[i]));
    }
    compute_rank_to_coord_mapping(info->size, dimensions_virtual, coordinates_virtual, size_virtual);
    DPRINTF("[%d] Virtual dimensions (%d, %d, %d)\n", info->rank, dimensions_virtual[0], dimensions_virtual[1], dimensions_virtual[2]);

    if(!idle){
        // If the topology has some dimension which was not a power of 2,
        // then we can receive in tmpbuf and aggregate into recvbuf.
        // Compute the number of steps on the shrunk topology

        // Computes the peer sequence on each port.
        DPRINTF("[%d] Computing peers\n", info->rank);  
        uint** peers_per_port = (uint**) malloc(sizeof(uint*)*info->num_ports);
        for(size_t p = 0; p < info->num_ports; p++){
            peers_per_port[p] = (uint*) malloc(sizeof(uint)*num_steps_virtual);
            compute_peers(peers_per_port[p], p, num_steps_virtual, rank_virtual, coordinates_virtual, dimensions_virtual); 
        }
        DPRINTF("[%d] Peers computed\n", info->rank);

        // Do the step-by-step communication on the shrunk topology.
        MPI_Request requests_s[MAX_SUPPORTED_PORTS];
        MPI_Request requests_r[MAX_SUPPORTED_PORTS];
        const void *sendbuf_real, *aggbuff_a;
        void *aggbuff_b, *recvbuf_real;
        for(size_t c = 0; c < info->num_chunks; c++){            
            for(size_t step = 0; step < (uint) num_steps_virtual; step++){     
                // Isend/Irecv requests
                memset(requests_s, 0, sizeof(requests_s));                
                memset(requests_r, 0, sizeof(requests_r));

                // Schedule all the send and recv
                for(size_t p = 0; p < info->num_ports; p++){
                    //DPRINTF("[%d] Chunk %d, step %d, port %d, offset %d count %d\n", info->rank, c, step, p, info->collectives[c][p].offset, info->collectives[c][p].count);
                    // Get the peer
                    int virtual_peer = peers_per_port[p][step]; 
                    uint coord_peer[MAX_SUPPORTED_DIMENSIONS];
                    retrieve_coord_mapping(coordinates_virtual, virtual_peer, coord_peer);
                    int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                    DPRINTF("[%d] Sending to %d\n", info->rank, peer);
                    // Get the buffers
                    // If all dimensions are powers of two, the ranks
                    // did not send, recv, and aggregate anything and thus
                    // the trick with the buffers swap has not been done.
                    // We need to do it here (once per port and chunk -- i.e.,
                    // always at the first step)
                    if(all_p2_dimensions && step == 0){
                        sendbuf_real = ((char*) sendbuf) + info->chunks[c][p].offset;
                        recvbuf_real = ((char*) recvbuf) + info->chunks[c][p].offset;
                    }else{
                        sendbuf_real = ((char*) recvbuf) + info->chunks[c][p].offset;
                        recvbuf_real = ((char*) tmpbuf) + info->chunks[c][p].offset;
                    }
                    DPRINTF("[%d] Count: %d\n", info->rank, info->chunks[c][p].count);
                    // Schedule the sends and recvs
                    res = MPI_Isend(sendbuf_real, info->chunks[c][p].count, datatype, peer, TAG_SWING_ALLREDUCE + p, comm, &(requests_s[p]));
                    if(res != MPI_SUCCESS){DPRINTF("[%d] Error on isend\n", info->rank); return res;}

                    res = MPI_Irecv(recvbuf_real, info->chunks[c][p].count, datatype, peer, TAG_SWING_ALLREDUCE + p, comm, &(requests_r[p]));
                    if(res != MPI_SUCCESS){DPRINTF("[%d] Error on irecv\n", info->rank); return res;}                       
                }  
                DPRINTF("[%d] Send/Recv issued, going to wait\n", info->rank);
                // Wait and aggregate
                for(size_t p = 0; p < info->num_ports; p++){
                    int terminated_port;
                    res = MPI_Waitany(info->num_ports, requests_r, &terminated_port, MPI_STATUS_IGNORE);
                    if(res != MPI_SUCCESS){DPRINTF("[%d] Error on waitany\n", info->rank); return res;}     
                    // Now wait also for the send. If we wait for all the sends only at the end,
                    // we might overwrite recvbuf while still being used by a send. // TODO
                    res = MPI_Wait(&(requests_s[terminated_port]), MPI_STATUS_IGNORE);
                    if(res != MPI_SUCCESS){DPRINTF("[%d] Error on wait\n", info->rank); return res;}     
                    DPRINTF("[%d] Irecv/Isend on port %d completed\n", info->rank, terminated_port);
                    
                    // See comment above about buffers trick
                    if(all_p2_dimensions && step == 0){
                        aggbuff_a = ((char*) sendbuf) + info->chunks[c][terminated_port].offset;
                    }else{
                        aggbuff_a = ((char*) tmpbuf) + info->chunks[c][terminated_port].offset;
                    }                    
                    aggbuff_b = ((char*) recvbuf) + info->chunks[c][terminated_port].offset;                        
                    
                    //DPRINTF("[%d] sendbuf %p recvbuf %p sendbuf_real %p recvbuf_real %p tmpbuf %p aggbuff_a %p aggbuff_b %p\n", info->rank, sendbuf, recvbuf, sendbuf_real, recvbuf_real, tmpbuf, aggbuff_a, aggbuff_b);

                    res = MPI_Reduce_local(aggbuff_a, aggbuff_b, info->chunks[c][terminated_port].count, datatype, op); 
                    if(res != MPI_SUCCESS){DPRINTF("[%d] Error on reduce_local\n", info->rank); return res;}
                }
                // Wait for all the send to finish
                DPRINTF("[%d] All sends and receive completed\n", info->rank);
            }
        }
        for(size_t p = 0; p < info->num_ports; p++){
            free(peers_per_port[p]);
        }
        free(peers_per_port);  
    }

    DPRINTF("[%d] Propagating data to extra nodes\n", info->rank);
    res = enlarge_non_power_of_two(recvbuf, count, datatype, op, comm, info, coordinates);
    if(res != MPI_SUCCESS){return res;}
    DPRINTF("[%d] Data propagated\n", info->rank);
    
    free(tmpbuf);
    free(coordinates);
    free(coordinates_virtual);
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

static inline int get_last_step(uint32_t block_distance){
    // The last step is the position of the most significant bit.
    return 32 - clz(block_distance) - 1;
}

static int swing_coll_sendrecv_bbb(void *buf, void* rbuf, size_t count, size_t chunk, size_t step, 
                              MPI_Request* requests_s, MPI_Request* requests_r, uint* num_requests_s, uint* num_requests_r, ChunkInfo* req_idx_to_block_idx,
                              MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                              CollType coll_type, SwingInfo* info, uint** peers_per_port, char*** bitmap_send, char*** bitmap_recv){
    int tag, res;
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(info->num_steps - step - 1);            
    *num_requests_s = 0;
    *num_requests_r = 0;
    memset(requests_s, 0, sizeof(MPI_Request)*info->size*MAX_SUPPORTED_PORTS);
    memset(requests_r, 0, sizeof(MPI_Request)*info->size*MAX_SUPPORTED_PORTS);
    for(size_t port = 0; port < info->num_ports; port++){
        ChunkInfo bi = info->chunks[chunk][port];
        uint partition_size = bi.count / info->size;
        uint remaining = bi.count % info->size;
        uint peer = peers_per_port[port][block_step];
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", info->rank, step, port, info->num_ports, peer);                

        if(coll_type == SWING_REDUCE_SCATTER){
            tag = TAG_SWING_REDUCESCATTER + port;
        }else{
            tag = TAG_SWING_ALLGATHER + port;
        }

        // Sendrecv + aggregate
        // Search for the blocks that must be sent.
        size_t count_so_far = 0;
        for(size_t i = 0; i < (uint) info->size; i++){
            int send_block, recv_block;
            if(coll_type == SWING_REDUCE_SCATTER){
                send_block = bitmap_send[port][block_step][i];
                recv_block = bitmap_recv[port][block_step][i];
            }else{                
                send_block = bitmap_recv[port][block_step][i];
                recv_block = bitmap_send[port][block_step][i];
            }
            
            size_t block_count = partition_size + (i < remaining ? 1 : 0);
            size_t block_within_port_offset = count_so_far*info->dtsize;
            count_so_far += block_count;

            // The actual offset is the offset of the data for this port, 
            // plus the offset of the block within that data
            size_t block_offset = bi.offset + block_within_port_offset; 

            //DPRINTF("[%d] Block %d (send %d recv %d)\n", rank, i, send_block, recv_block);
            if(send_block){              
                DPRINTF("[%d] Sending block %d to %d at step %d (coll %d)\n", info->rank, i, peer, step, coll_type);
                res = MPI_Isend(((char*) buf) + block_offset, block_count, sendtype, peer, tag, comm, &(requests_s[*num_requests_s]));
                if(res != MPI_SUCCESS){return res;}
                ++(*num_requests_s);
            }
            if(recv_block){
                DPRINTF("[%d] Receiving block %d from %d at step %d (coll %d)\n", info->rank, i, peer, step, coll_type);
                res = MPI_Irecv(((char*) rbuf) + block_offset, block_count, recvtype, peer, tag, comm, &(requests_r[*num_requests_r]));
                if(res != MPI_SUCCESS){return res;}
                req_idx_to_block_idx[*(num_requests_r)].offset = block_offset;
                req_idx_to_block_idx[*(num_requests_r)].count = block_count;
                ++(*num_requests_r);
            }
        }
    }
    DPRINTF("[%d] Issued %d send requests and %d receive requests\n", info->rank, *num_requests_s, *num_requests_r);
    return MPI_SUCCESS;
}

static int swing_coll_sendrecv_cont(void *buf, void* rbuf, size_t count, size_t chunk, size_t step, 
                              MPI_Request* requests_s, MPI_Request* requests_r, uint* num_requests_s, uint* num_requests_r, ChunkInfo* req_idx_to_block_idx,
                              MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                              CollType coll_type, SwingInfo* info, uint** peers_per_port, char*** bitmap_send, char*** bitmap_recv){
    int tag, res;
    size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(info->num_steps - step - 1); 
    *num_requests_s = 0;
    *num_requests_r = 0;           
    memset(requests_s, 0, sizeof(MPI_Request)*info->size*MAX_SUPPORTED_PORTS);
    memset(requests_r, 0, sizeof(MPI_Request)*info->size*MAX_SUPPORTED_PORTS);
    for(size_t port = 0; port < info->num_ports; port++){
        ChunkInfo bi = info->chunks[chunk][port];
        uint partition_size = bi.count / info->size;
        uint remaining = bi.count % info->size;
        uint peer = peers_per_port[port][block_step];
        DPRINTF("[%d] Starting step %d on port %d (out of %d) peer %d\n", info->rank, step, port, info->num_ports, peer);                

        if(coll_type == SWING_REDUCE_SCATTER){
            tag = TAG_SWING_REDUCESCATTER + port;
        }else{
            tag = TAG_SWING_ALLGATHER + port;
        }

        // Sendrecv + aggregate
        // Search for the blocks that must be sent.
        size_t count_so_far = 0;
        bool start_found_s = false, start_found_r = false;
        size_t offset_s, offset_r, count_s = 0, count_r = 0;
        for(size_t i = 0; i < (uint) info->size; i++){
            int send_block, recv_block;
            if(coll_type == SWING_REDUCE_SCATTER){
                send_block = bitmap_send[port][block_step][i];
                recv_block = bitmap_recv[port][block_step][i];
            }else{                
                send_block = bitmap_recv[port][block_step][i];
                recv_block = bitmap_send[port][block_step][i];
            }
            
            size_t block_count = partition_size + (i < remaining ? 1 : 0);
            size_t block_within_port_offset = count_so_far*info->dtsize;
            count_so_far += block_count;

            // The actual offset is the offset of the data for this port, 
            // plus the offset of the block within that data
            size_t block_offset = bi.offset + block_within_port_offset; 

            if(send_block){
                if(!start_found_s){
                    start_found_s = true;
                    offset_s = block_offset;
                }
                count_s += block_count;
            }
            if(start_found_s && (!send_block || i == info->size - 1)){ // The train of consecutive blocks is over
                DPRINTF("[%d] Sending offset %d count %d at step %d (coll %d)\n", info->rank, offset_s, count_s, step, coll_type);            
                res = MPI_Isend(((char*) buf) + offset_s, count_s, sendtype, peer, tag, comm, &(requests_s[*num_requests_s]));
                (*num_requests_s)++;
                if(res != MPI_SUCCESS){return res;}

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
            if(start_found_r && (!recv_block || i == info->size - 1)){ // The train of consecutive blocks is over
                DPRINTF("[%d] Receiving offset %d count %d at step %d (coll %d)\n", info->rank, offset_r, count_r, step, coll_type);
                req_idx_to_block_idx[*num_requests_r].offset = offset_r;
                req_idx_to_block_idx[*num_requests_r].count = count_r;
                res = MPI_Irecv(((char*) rbuf) + offset_r, count_r, recvtype, peer, tag, comm, &(requests_r[*num_requests_r]));
                (*num_requests_r)++;
                if(res != MPI_SUCCESS){return res;}

                // In some rare cases (e.g., for 10 nodes), I might have not one but two consecutive trains of blocks
                // Reset everything in case we need to send another train of blocks
                count_r = 0;
                offset_r = 0;
                start_found_r = false;
            }
        }
    }
    DPRINTF("[%d] Issued %d send requests and %d receive requests\n", info->rank, *num_requests_s, *num_requests_r);
    return MPI_SUCCESS;
}

static int swing_coll_sendrecv(void *buf, void* rbuf, size_t count, size_t chunk, size_t step, 
                              MPI_Request* requests_s, MPI_Request* requests_r, uint* num_requests_s, uint* num_requests_r, ChunkInfo* req_idx_to_block_idx,
                              MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                              CollType coll_type, SwingInfo* info, uint** peers_per_port, char*** bitmap_send, char*** bitmap_recv){

    if(algo == ALGO_SWING_B){
        return swing_coll_sendrecv_bbb(buf, rbuf, count, chunk, step, requests_s, requests_r, num_requests_s, num_requests_r, req_idx_to_block_idx, op, comm, sendtype, recvtype, coll_type, info, peers_per_port, bitmap_send, bitmap_recv);
    }else if(algo == ALGO_SWING_B_CONT){
        return swing_coll_sendrecv_cont(buf, rbuf, count, chunk, step, requests_s, requests_r, num_requests_s, num_requests_r, req_idx_to_block_idx, op, comm, sendtype, recvtype, coll_type, info, peers_per_port, bitmap_send, bitmap_recv);
    }else{
        assert("Unknown algo" == 0);
    }
}

static int swing_coll_wait(void *buf, void* rbuf, size_t count, size_t chunk, size_t step, MPI_Request* requests_s, MPI_Request* requests_r,
                            uint num_requests_s, uint num_requests_r,
                           ChunkInfo* req_idx_to_block_idx, MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type, SwingInfo* info){
    int res;
    // Wait for all the recvs to be over
    if(coll_type == SWING_REDUCE_SCATTER){
        int index;
        for(size_t i = 0; i < num_requests_r; i++){
            res = MPI_Waitany(num_requests_r, requests_r, &index, MPI_STATUS_IGNORE);
            if(res != MPI_SUCCESS){return res;}
            void* rbuf_block = (void*) (((char*) rbuf) + req_idx_to_block_idx[index].offset);
            void* buf_block = (void*) (((char*) buf) + req_idx_to_block_idx[index].offset);  
            DPRINTF("[%d] Aggregating from %p to %p (i %d index %d offset %d count %d)\n", info->rank, rbuf_block, buf_block, i, index, req_idx_to_block_idx[index].offset, req_idx_to_block_idx[index].count);
            MPI_Reduce_local(rbuf_block, buf_block, req_idx_to_block_idx[index].count, sendtype, op); 
        }
    }else{
        res = MPI_Waitall(num_requests_r, requests_r, MPI_STATUSES_IGNORE);
        if(res != MPI_SUCCESS){return res;}        
    }
    // Wait for all the sends to be over
    res = MPI_Waitall(num_requests_s, requests_s, MPI_STATUSES_IGNORE);
    return res;
}

static int swing_coll(void *buf, void* rbuf, size_t count, size_t chunk, size_t step, MPI_Request* requests_s, MPI_Request* requests_r, 
                      ChunkInfo* req_idx_to_block_idx, MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                      CollType coll_type, SwingInfo* info, uint** peers_per_port, char*** bitmap_send, char*** bitmap_recv){

    uint num_requests_s, num_requests_r;
    int res = swing_coll_sendrecv(buf, rbuf, count, chunk, step, requests_s, requests_r, &num_requests_s, &num_requests_r, req_idx_to_block_idx, op, comm, sendtype, recvtype, coll_type, info, peers_per_port, bitmap_send, bitmap_recv);
    if(res != MPI_SUCCESS){return res;}
    return swing_coll_wait(buf, rbuf, count, chunk, step, requests_s, requests_r, num_requests_s, num_requests_r, req_idx_to_block_idx, op, comm, sendtype, recvtype, coll_type, info);
}

static inline void get_blocks_bitmaps(int rank, int step, int max_steps, int size, char* bitmap_send, size_t port){
    size_t max_num_blocks = pow(2, (max_steps - step - 1));
    // Generate all binary strings with a 0 in position step+1 and a 1 in all the bits in position j (j <= step)
    for(size_t i = 0; i < max_num_blocks; i++){
        if(!(i & 0x1)){ // Only if it has a 0 as LSB (we generate in one shot bot the string with LSB=0 and that with LSB=1)         
            int nbin[2]; // At position 0, we have the numbers with 0 as LSB, at position 1, the numbers with 1 as LSB
            nbin[1] = (i << (step + 1)) | ((1 << (step + 1)) - 1); // LSB=1
            nbin[0] = ~nbin[1]          & ((1 <<  max_steps) - 1); // LSB=0

            // If mirrored collectives, the direction of the communication are inverted
            // It is thus enough to invert nbin_1 and nbin_0
            if(port >= dimensions_num){
                int tmp = nbin[1];
                nbin[1] = nbin[0];
                nbin[0] = tmp;
            }
            

            int distance[2] = {negabinary_to_binary(nbin[0]),  // At position 0, we have the numbers with 0 as LSB, 
                               negabinary_to_binary(nbin[1])}; // at position 1, the numbers with 1 as LSB.

            DPRINTF("[%d] distancee (%d %d) (nbin: %d %d)\n", rank, distance[0], distance[1], nbin[0], nbin[1]);

            char distance_valid[2] = {1, 1};
            for(size_t q = 0; q < 2; q++){
                // A rank never sends its data in reduce-scatter.
                if(mod(distance[q], size) == 0){
                    distance_valid[q] = 0;
                    continue;
                }

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
                    assert("This should never happen!" && 0);
                }

                for(size_t k = 0; k < 3; k++){
                    if(in_range(alternatives[k], max_steps)){ // First of all, check if the corresponding negabinary number can be represented with the given number of bits
                        int nbin_alt = binary_to_negabinary(alternatives[k]);
                        int first_step_alt = get_first_step(nbin_alt);
                        if(first_step_alt != step && first_step_alt > get_first_step(nbin[q])){
                            //DPRINTF("[%d] Invalid distance %d (nbin %d vs %d) for step %d\n", rank, alternatives[k], nbin_alt, nbin[q], step);
                            DPRINTF("[%d] Step %d I am skipping block at distance %d %d, I am going to send at step %d (nbin %d %d)\n", rank, step, distance[q], alternatives[k], get_first_step(nbin_alt), nbin[q], nbin_alt);
                            distance_valid[q] = 0;
                            break;
                        }
                    }
                }
            }

            if(is_odd(rank)){ // Odd rank
                // Sum r to all the strings with LSB=0                
                if(distance_valid[0]){
                    bitmap_send[mod(rank + distance[0], size)] = 1; 
                }
                
                // Subtract from r all the strings with LSB=1
                if(distance_valid[1]){
                    bitmap_send[mod(rank - distance[1], size)] = 1; 
                }
            }else{ // Even rank
                // Subtract form r all the strings with LSB=0
                if(distance_valid[0]){
                    bitmap_send[mod(rank - distance[0], size)] = 1;
                }

                // Sum r to all the strings with LSB=1
                if(distance_valid[1]){
                    bitmap_send[mod(rank + distance[1], size)] = 1;
                }
            }
        }
    }
}

static inline void get_blocks_bitmaps_multid(size_t* next_step_per_dim, size_t* current_d, size_t step,
                                             size_t port, uint* coordinates, uint* coord_peer, char** bitmap_send, 
                                             char** bitmap_recv, char* bitmap_send_merged, char* bitmap_recv_merged, 
                                             uint* coord_mine, SwingInfo* info){    
    // Compute the bitmap for each dimension
    for(size_t k = 0; k < dimensions_num; k++){
        size_t d = (k + current_d[port]) % dimensions_num;

        memset(bitmap_send[d], 0, sizeof(char)*dimensions[d]);
        memset(bitmap_recv[d], 0, sizeof(char)*dimensions[d]);

        // To deal with the case where I don't move in that dimension.
        // e.g. if I send on the row dimension to a node with ID 0001,
        // (i.e., I send on the first step on the row, and never on the column (00))
        if(k){
            bitmap_send[d][coord_mine[d]] = 1;
            bitmap_recv[d][coord_peer[d]] = 1;
        }
                        
        // We skip dimension d if we are done with that dimension.
        if(next_step_per_dim[d] >= info->num_steps_per_dim[d]){
            continue;
        }     
                    
        size_t last_step;
        if(k == 0){
            last_step = next_step_per_dim[d] + 1;
        }else{
            last_step = info->num_steps_per_dim[d];
        }

        //DPRINTF("[%d] step %d port %d target dim %d rel step %d last step %d\n", info->rank, step, p, d, rel_step, last_step);                
        for(size_t sk = next_step_per_dim[d]; sk < last_step; sk++){
            get_blocks_bitmaps(coord_mine[d], sk, info->num_steps_per_dim[d], dimensions[d], bitmap_send[d], port);
            get_blocks_bitmaps(coord_peer[d], sk, info->num_steps_per_dim[d], dimensions[d], bitmap_recv[d], port);
        }
    }

    // Combine the per-dimension bitmaps
    uint coord_block[MAX_SUPPORTED_DIMENSIONS];    
    for(size_t i = 0; i < (uint) info->size; i++){
        retrieve_coord_mapping(coordinates, i, coord_block);
        //DPRINTF("[%d] Step %d Peer (%d, %d) Block %d coord (%d,%d) bitmap send (%d,%d)\n", info->rank, step, coord_peer[0], coord_peer[1], i, coord_block[0], coord_block[1], bitmap_send[0][coord_block[0]], bitmap_send[1][coord_block[1]]);
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
    if(step < (size_t) info->num_steps - 1){
        size_t d = current_d[port];              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < info->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d[port] = (current_d[port] + 1) % dimensions_num;
            d = current_d[port];
        }while(next_step_per_dim[d] >= info->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
}

// Same as the one above, but we compute next_step_per_dim and current_d on the fly so that we do not need to do bookeping
static inline void get_blocks_bitmaps_multid(size_t step, size_t port, uint* coordinates, uint* coord_peer, 
                                             char* bitmap_send_merged, char* bitmap_recv_merged, 
                                             uint* coord_mine, SwingInfo* info){
    size_t next_step_per_dim[MAX_SUPPORTED_DIMENSIONS];
    size_t current_d[MAX_SUPPORTED_PORTS];
    char* bitmap_send[MAX_SUPPORTED_DIMENSIONS];
    char* bitmap_recv[MAX_SUPPORTED_DIMENSIONS];    
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
        if(next_step_per_dim[d] < info->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d[port] = (current_d[port] + 1) % dimensions_num;
            d = current_d[port];
        }while(next_step_per_dim[d] >= info->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
    //DPRINTF("[%d] Going to do step %d on dim %d rel %d\n", info->rank, step, current_d[port], next_step_per_dim[current_d[port]]);    
    get_blocks_bitmaps_multid(next_step_per_dim, current_d, step, port, coordinates, coord_peer, bitmap_send, bitmap_recv, bitmap_send_merged, bitmap_recv_merged, coord_mine, info);
    for(size_t d = 0; d < dimensions_num; d++){
        free(bitmap_send[d]);
        free(bitmap_recv[d]);   
    }
}

#ifdef DEBUG
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
#endif

static void get_peer(uint* coord_rank, size_t step, size_t port, SwingInfo* info, uint* coord_peer){
    size_t next_step_per_dim[MAX_SUPPORTED_DIMENSIONS];
    size_t current_d = port % dimensions_num;
    memcpy(coord_peer, coord_rank, sizeof(uint)*dimensions_num);
    memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);
    for(size_t i = 0; i < step; i++){
        // Move to the next dimension for the next step
        size_t d = current_d;              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < info->num_steps_per_dim[d]){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d = (current_d + 1) % dimensions_num;
            d = current_d;
        }while(next_step_per_dim[d] >= info->num_steps_per_dim[d]); // If we exhausted this dimension, move to the next one
    }
    size_t distance = rhos[next_step_per_dim[current_d]];
    // Flip the sign for odd nodes
    if(is_odd(coord_rank[current_d])){distance *= -1;}
    // Mirrored collectives
    if((uint) port >= dimensions_num){distance *= -1;}

    coord_peer[current_d] = mod(coord_peer[current_d] + distance, dimensions[current_d]);
}

static void remap(const std::vector<int>& nodes, uint start_range, uint end_range, uint* blocks_remapping,
                  int step, size_t port, uint* coord_rank, uint* coordinates, SwingInfo* info){
    if(nodes.size() < 2){ // Needed for cases with non-power of two sizes (e.g., 6x6)
        return;
    }else if(nodes.size() == 2){
        blocks_remapping[nodes[0]] = start_range;
        blocks_remapping[nodes[1]] = end_range - 1;
        assert(end_range == start_range + 2);
    }else{
        // Find two partitions of node that talk with each other. If I have n nodes, 
        // if I see what happens in next step, I have two disjoint sets of nodes.
        char* my_bitmap_send = (char*) malloc(sizeof(char)*info->size);        
        
        uint coord_peer[MAX_SUPPORTED_DIMENSIONS];   
        get_peer(coord_rank, step, port, info, coord_peer);
        get_blocks_bitmaps_multid(step, port, coordinates, coord_peer, my_bitmap_send, NULL, coord_rank, info);

        std::vector<int> left, right;
        left.reserve(info->size);
        right.reserve(info->size);
        for(auto n : nodes){
            if(my_bitmap_send[n] == 0){
                left.push_back(n);
            }else{
                right.push_back(n);
            }
        }

        DPRINTF("[%d] step %d NODESIZE %d %d\n", info->rank, step, left.size(), right.size());        

        remap(left , start_range             , start_range + left.size(), blocks_remapping, step + 1, port, coord_rank, coordinates, info);
        remap(right, end_range - right.size(), end_range                , blocks_remapping, step + 1, port, coord_peer, coordinates, info);
        
        free(my_bitmap_send);
    }
}

static inline void compute_bitmaps(SwingInfo* info, size_t step, uint* coordinates, uint** peers_per_port, char** bitmap_ready, char** bitmap_send, char** bitmap_recv,
                                   size_t next_step_per_dim[MAX_SUPPORTED_PORTS][MAX_SUPPORTED_DIMENSIONS], size_t* current_d, uint* coord_mine, uint** remapping_per_port, char* tmp_s, char* tmp_r,
                                   char*** bitmap_send_merged, char*** bitmap_recv_merged){
    for(size_t p = 0; p < info->num_ports; p++){                
        // Compute bitmaps of blocks to send and receive (we do not need to do this for allgather since bitmap_ready would always be 1)
        if(!bitmap_ready[p][step]){
            bitmap_send_merged[p][step] = (char*) malloc(sizeof(char)*info->size);
            bitmap_recv_merged[p][step] = (char*) malloc(sizeof(char)*info->size);
            
            uint coord_peer[MAX_SUPPORTED_DIMENSIONS];   
            retrieve_coord_mapping(coordinates, peers_per_port[p][step], coord_peer);

            get_blocks_bitmaps_multid(next_step_per_dim[p], current_d, step, p, coordinates, coord_peer, bitmap_send, bitmap_recv, bitmap_send_merged[p][step], bitmap_recv_merged[p][step], coord_mine, info);
            bitmap_ready[p][step] = 1;

            // Remapping
            if(algo == ALGO_SWING_B_CONT){
                memcpy(tmp_s, bitmap_send_merged[p][step], sizeof(char)*info->size);
                memcpy(tmp_r, bitmap_recv_merged[p][step], sizeof(char)*info->size);
                for(size_t i = 0; i < (uint) info->size; i++){
                    DPRINTF("[%d] Remapping %d to %d\n", info->rank, i, remapping_per_port[p][i]);
                    bitmap_send_merged[p][step][remapping_per_port[p][i]] = tmp_s[i];
                    bitmap_recv_merged[p][step][remapping_per_port[p][i]] = tmp_r[i];
                }
            }
#ifdef DEBUG
            print_bitmaps(info, step, bitmap_send_merged[p][step], bitmap_recv_merged[p][step]);
#endif
        }
    }
}


static inline int MPI_Allreduce_bw_optimal_swing(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info){    
    int res;
    uint* coordinates = (uint*) malloc(sizeof(uint)*info->size*dimensions_num);
    compute_rank_to_coord_mapping(info->size, dimensions, coordinates, info->size);
    DPRINTF("[%d] Computing peers\n", info->rank);
    uint** peers_per_port = (uint**) malloc(sizeof(uint*)*info->num_ports);
    for(size_t p = 0; p < info->num_ports; p++){
        peers_per_port[p] = (uint*) malloc(sizeof(uint)*info->num_steps);
        compute_peers(peers_per_port[p], p, info->num_steps, info->rank, coordinates);
    }
    DPRINTF("[%d] Peers computed\n", info->rank);
    
    size_t total_size_bytes = count*info->dtsize;
    char* rbuf = (char*) malloc(total_size_bytes);
    memcpy(recvbuf, sendbuf, total_size_bytes);    
    uint coord_mine[MAX_SUPPORTED_DIMENSIONS];    
    getCoordFromId(info->rank, coord_mine, info->size, dimensions);

    char* bitmap_send[MAX_SUPPORTED_DIMENSIONS];
    char* bitmap_recv[MAX_SUPPORTED_DIMENSIONS];
    char** bitmap_send_merged[MAX_SUPPORTED_PORTS];
    char** bitmap_recv_merged[MAX_SUPPORTED_PORTS];
    char* bitmap_ready[MAX_SUPPORTED_PORTS]; // For each port and step, if 1, the send/recv bitmaps have been already computed.
    size_t current_d[MAX_SUPPORTED_PORTS]; // For each port, what's the current dimension we are sending in.
    size_t next_step_per_dim[MAX_SUPPORTED_PORTS][MAX_SUPPORTED_DIMENSIONS]; // For each port and for each dimension, what's the next step to execute in that dimension.
    char *tmp_s = NULL, *tmp_r = NULL;
    
    for(size_t d = 0; d < dimensions_num; d++){
        bitmap_send[d] = (char*) malloc(sizeof(char)*dimensions[d]);
        bitmap_recv[d] = (char*) malloc(sizeof(char)*dimensions[d]);   
    }

    for(size_t p = 0; p < info->num_ports; p++){
        bitmap_ready[p] = (char*) malloc(sizeof(char)*info->num_steps);
        memset(bitmap_ready[p], 0, sizeof(char)*info->num_steps);        
        bitmap_send_merged[p] = (char**) malloc(sizeof(char*)*info->num_steps);
        bitmap_recv_merged[p] = (char**) malloc(sizeof(char*)*info->num_steps);                                
    }

    /********************/
    /** Send/Recv part **/
    /********************/
    uint num_steps = info->num_steps;
    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*info->size*MAX_SUPPORTED_PORTS);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*info->size*MAX_SUPPORTED_PORTS);
    ChunkInfo* req_idx_to_block_idx = (ChunkInfo*) malloc(sizeof(ChunkInfo)*info->size*MAX_SUPPORTED_PORTS);


    /*************/
    /* REMAPPING */
    /*************/
    uint* remapping_per_port[MAX_SUPPORTED_PORTS];
    if(algo == ALGO_SWING_B_CONT){
        DPRINTF("[%d] Remapping\n", info->rank);
        std::vector<int> nodes(info->size);
        uint coord_zero[MAX_SUPPORTED_DIMENSIONS];
        memset(coord_zero, 0, sizeof(uint)*dimensions_num);
        for(size_t n = 0; n < (size_t) info->size; n++){nodes[n] = n;}
        for(size_t p = 0; p < info->num_ports; p++){
            remapping_per_port[p] = (uint*) malloc(sizeof(uint)*info->size);              
            remap(nodes, 0, info->size, remapping_per_port[p], 0, p, coord_zero, coordinates, info);
        }
    }
    if(algo == ALGO_SWING_B_CONT){
        tmp_s = (char*) malloc(sizeof(char)*info->size);
        tmp_r = (char*) malloc(sizeof(char)*info->size);
    }
    
    /******************/
    /* Reduce-scatter */
    /******************/
    for(size_t c = 0; c < info->num_chunks; c++){
        // Reset info for the next series of steps        
        for(size_t p = 0; p < info->num_ports; p++){                
            memset(next_step_per_dim[p], 0, sizeof(size_t)*dimensions_num);
            current_d[p] = p % dimensions_num;
        }

        // Compute bitmaps needed to execute next step (step 0)
        compute_bitmaps(info, 0, coordinates, peers_per_port, bitmap_ready, bitmap_send, bitmap_recv, next_step_per_dim, current_d, coord_mine, remapping_per_port, tmp_s, tmp_r, bitmap_send_merged, bitmap_recv_merged);

        for(size_t step = 0; step < num_steps; step++){        
            // Run reduce-scatter
            uint num_requests_s, num_requests_r;
            res = swing_coll_sendrecv(recvbuf, rbuf, count, c, step, requests_s, requests_r, &num_requests_s, &num_requests_r, req_idx_to_block_idx,
                                      op, comm, datatype, datatype, 
                                      SWING_REDUCE_SCATTER, info, peers_per_port, bitmap_send_merged, bitmap_recv_merged);
            if(res != MPI_SUCCESS){return res;} 

            // Start overlap
            // While communicating, compute bitmaps needed to execute next step
            // We do not need to do the same for allgather since they have already been computed here
            if(step != num_steps - 1){
                compute_bitmaps(info, step + 1, coordinates, peers_per_port, bitmap_ready, bitmap_send, bitmap_recv, next_step_per_dim, current_d, coord_mine, remapping_per_port, tmp_s, tmp_r, bitmap_send_merged, bitmap_recv_merged);
            }
            // End overlap

            res = swing_coll_wait(recvbuf, rbuf, count, c, step, requests_s, requests_r, num_requests_s, num_requests_r, req_idx_to_block_idx, op, comm, datatype, datatype, SWING_REDUCE_SCATTER, info);
            if(res != MPI_SUCCESS){return res;} 
        }
    }
    
    /**************/
    /* All-gather */
    /**************/
    for(size_t c = 0; c < info->num_chunks; c++){         
        for(size_t step = 0; step < num_steps; step++){
            res = swing_coll(recvbuf, recvbuf, count, c, step, requests_s, requests_r, req_idx_to_block_idx,
                             op, comm, datatype, datatype, 
                             SWING_ALLGATHER, info, peers_per_port, bitmap_send_merged, bitmap_recv_merged);
            if(res != MPI_SUCCESS){return res;}
        }
    }

    /********/
    /* Free */
    /********/
    free(requests_s);
    free(requests_r);
    free(req_idx_to_block_idx);
    free(coordinates);
    free(rbuf);
    for(size_t p = 0; p < info->num_ports; p++){
        for(size_t s = 0; s < (uint) info->num_steps; s++){
            free(bitmap_send_merged[p][s]);
            free(bitmap_recv_merged[p][s]);
        }
        free(bitmap_send_merged[p]);
        free(bitmap_recv_merged[p]);
        free(peers_per_port[p]);
        if(algo == ALGO_SWING_B_CONT){
            free(remapping_per_port[p]); 
        }
    }
    free(peers_per_port);  
    for(size_t d = 0; d < dimensions_num; d++){
        free(bitmap_send[d]);
        free(bitmap_recv[d]);
    }
    if(tmp_s){free(tmp_s);}
    if(tmp_r){free(tmp_r);}
    return res;
}

// Gets the count and offset for each port.
static inline void get_count_and_offset_per_port(const char *sendbuf, char *recvbuf, int count, int dtsize, uint num_ports, ChunkInfo* ci){
    uint partition_size = count / num_ports;
    uint remaining = count % num_ports;
    uint count_so_far = 0;
    for(size_t i = 0; i < num_ports; i++){
        ci[i].count = partition_size + (i < remaining ? 1 : 0);
        ci[i].offset = count_so_far*dtsize;
        count_so_far += ci[i].count;
    }
}


int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_allreduce || algo == ALGO_DEFAULT){
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }else{
        SwingInfo info;
        MPI_Type_size(datatype, &info.dtsize);    
        MPI_Comm_size(comm, &info.size);
        MPI_Comm_rank(comm, &info.rank);
        
        size_t offset = 0;
        for(size_t i = 0; i < dimensions_num; i++){
            info.num_steps_per_dim[i] = (int) ceil(log2(dimensions[i]));
            offset += info.num_steps_per_dim[i];
        }
        // The number of steps is not ceil(log2(size)) but the sum of the number of steps for each dimension.
        // This is needed for those cases where dimensions are not powers of two. Consider for example a 
        // 10x10 torus. We would perform ceil(log2(10)) + ceil(log2(10)) = 4 + 4 = 8 steps, not ceil(log2(100)) = 7.
        info.num_steps = offset;

        if(info.num_steps > LIBSWING_MAX_STEPS){
            assert("Max steps limit must be increased and constants updated.");
        }

        // Split the data if too big (> max_size)
        size_t remaining_count = count;
        size_t max_count, next_offset = 0;
    
        if(!max_size || count*(size_t)info.dtsize <= max_size){
            max_count = count;
        }else{
            max_count = max_size / info.dtsize;
        }

        uint i = 0;
        info.num_ports = 1;
        if(multiport){
            info.num_ports = dimensions_num*2; // TODO: Always assumes num port is 2*num_dimension, refactor
        }
        DPRINTF("[%d] Going to run on %d ports\n", info.rank, info.num_ports);
        info.num_chunks = (size_t) ceil(count / max_count);
        info.chunks = (ChunkInfo**) malloc(sizeof(ChunkInfo*) * info.num_chunks);   
        for(size_t c = 0; c < info.num_chunks; c++){
            info.chunks[c] = (ChunkInfo*) malloc(sizeof(ChunkInfo) * info.num_ports);
        }
 
        do{
            int next_count = std::min(remaining_count, max_count);
            char* sendbuf_chunk = ((char*)sendbuf) + next_offset*info.dtsize;
            char* recvbuf_chunk = ((char*)recvbuf) + next_offset*info.dtsize;
            get_count_and_offset_per_port(sendbuf_chunk, recvbuf_chunk, next_count, info.dtsize, info.num_ports, info.chunks[i]);
            next_offset += next_count;
            remaining_count -= next_count;
            ++i;
        }while(remaining_count > 0);   

        assert(info.num_chunks == i);
#ifdef DEBUG
        for(size_t i = 0; i < info.num_chunks; i++){
            for(size_t j = 0; j < info.num_ports; j++){
                DPRINTF("[%d] Chunk %d Port %d Offset: %d Count: %d\n", info.rank, i, j, info.chunks[i][j].offset, info.chunks[i][j].count);
            }
        }
#endif
        // Call the algo
        if(algo == ALGO_SWING_L){ // Swing_l
            return MPI_Allreduce_lat_optimal_swing(sendbuf, recvbuf, count, datatype, op, comm, &info);
        }else if(algo == ALGO_SWING_B || algo == ALGO_SWING_B_CONT){ // Swing_b
            return MPI_Allreduce_bw_optimal_swing(sendbuf, recvbuf, count, datatype, op, comm, &info);
        }else if(algo == ALGO_RING){ // Ring
            // TODO: Implement multiported ring
            for(size_t i = 0; i < info.num_chunks; i++){
                return MPI_Allreduce_ring(((char*) sendbuf) + info.chunks[i][0].offset, ((char*) recvbuf) + info.chunks[i][0].offset, info.chunks[i][0].count, datatype, op, comm);
            }
        }else if(algo == ALGO_RECDOUB_B){ // Recdoub_b
            // TODO: Implement multiported recdoub b
            for(size_t i = 0; i < info.num_chunks; i++){
                return MPI_Allreduce_recdoub_b(((char*) sendbuf) + info.chunks[i][0].offset, ((char*) recvbuf) + info.chunks[i][0].offset, info.chunks[i][0].count, datatype, op, comm);
            }
        }else if(algo == ALGO_RECDOUB_L){ // Recdoub_l
            // TODO: Implement multiported recdoub l
            for(size_t i = 0; i < info.num_chunks; i++){
                return MPI_Allreduce_recdoub_l(((char*) sendbuf) + info.chunks[i][0].offset, ((char*) recvbuf) + info.chunks[i][0].offset, info.chunks[i][0].count, datatype, op, comm);
            }
        }else{
            return 1;
        }

        for(size_t c = 0; c < info.num_chunks; c++){
            free(info.chunks[c]);
        }
        free(info.chunks);
        return MPI_SUCCESS;
    }
}
// TODO: Don't use Swing for non-continugous non-native datatypes (tedious implementation)
