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
#define LIBSWING_MAX_STEPS 20 // With this we are ok up to 2^20 nodes, add other terms if needed.
static int rhos[LIBSWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};
static int smallest_negabinary[LIBSWING_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBSWING_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

#define TAG_SWING_REDUCESCATTER (0x7FFF - MAX_SUPPORTED_DIMENSIONS*2*1)
#define TAG_SWING_ALLGATHER     (0x7FFF - MAX_SUPPORTED_DIMENSIONS*2*2)
#define TAG_SWING_ALLREDUCE     (0x7FFF - MAX_SUPPORTED_DIMENSIONS*2*3)

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
    ALGO_RING,
    ALGO_RECDOUB_L,
    ALGO_RECDOUB_B,
}Algo;

typedef struct{
    int rank;
    int size;
    int dtsize;
    int num_steps;
    int num_steps_per_dim[MAX_SUPPORTED_DIMENSIONS];
    int offset_per_dim[MAX_SUPPORTED_DIMENSIONS];
}SwingInfo;

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allgather = 0, disable_allreduce = 0, 
                    dimensions_num = 1, latency_optimal_threshold = 1024, force_env_reload = 1, env_read = 0,
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

        env_str = getenv("LIBSWING_LATENCY_OPTIMAL_THRESHOLD");
        if(env_str){
            latency_optimal_threshold = atoi(env_str);
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
            printf("LIBSWING_LATENCY_OPTIMAL_THRESHOLD: %d\n", latency_optimal_threshold);
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

static inline void compute_peers(uint** peers, int port, int num_steps, uint rank, uint* coordinates, uint* dimensions_virtual = NULL){
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
    // TODO: Peers can be a 1D array
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
                peers[rank][i] = getIdFromCoord(coord, dimensions_virtual, dimensions_num);
            }else{
                peers[rank][i] = coord[0];
            }
        
            /*
            if(rank == 0){
                DPRINTF("eeeeee step %d relative step %d target_dim %d rho %d distance %d peer %d\n", i, relative_step, target_dim, rhos[relative_step], distance, peers[rank][i]);
            }
            */
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

static inline int MPI_Allreduce_lat_optimal_swing_sendrecv(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info, 
                                                           int skip_send, int skip_recv, char** rbuf, int peer, int step, int overwrite, uint port){    
    int res;
    const void *sendbuf_real, *aggbuff_a;
    void *aggbuff_b, *recvbuf_real;
    if(!*rbuf){
        sendbuf_real = sendbuf;
        recvbuf_real = recvbuf;
        aggbuff_a = sendbuf;
        aggbuff_b = recvbuf;
    }else{
        sendbuf_real = recvbuf;
        recvbuf_real = *rbuf;
        aggbuff_a = *rbuf;
        aggbuff_b = recvbuf;
    }

    int tag = TAG_SWING_ALLREDUCE + port;

    DPRINTF("[%d] Starting step %d, communicating with %d (skip_send=%d, skip_recv=%d)\n", info->rank, step, peer, skip_send, skip_recv);


    //printf("Rank %d sending %d bytes to %d on tag %d\n", info->rank, count*info->dtsize, peer, tag);
    //printf("Rank %d receiving %d bytes from %d on tag %d\n", info->rank, count*info->dtsize, peer, tag);

    MPI_Request requests[2];
    memset(requests, 0, sizeof(requests));
    if(!skip_send){
        res = MPI_Isend(sendbuf_real, count, datatype, peer, tag, comm, &(requests[0]));
        if(res != MPI_SUCCESS){return res;}
    }
    if(!skip_recv){
        res = MPI_Irecv(recvbuf_real, count, datatype, peer, tag, comm, &(requests[1]));
        if(res != MPI_SUCCESS){return res;}
    }
    // While data is transmitted in the first step, we allocate buffer for the next steps
    if(!*rbuf){
        *rbuf = (char*) malloc(count*info->dtsize);
    }    
    if(!skip_send){
        res = MPI_Wait(&(requests[0]), MPI_STATUS_IGNORE);
        if(res != MPI_SUCCESS){return res;}
    }
    if(!skip_recv){
        res = MPI_Wait(&(requests[1]), MPI_STATUS_IGNORE);
        if(res != MPI_SUCCESS){return res;}
    }
    
    if(res != MPI_SUCCESS){return res;}
    if(!skip_recv){
        if(overwrite){
            memcpy(aggbuff_b, aggbuff_a, count*info->dtsize);
        }else{
            MPI_Reduce_local(aggbuff_a, aggbuff_b, count, datatype, op);
        }
    }
    
    return MPI_SUCCESS;
}

// https://stackoverflow.com/questions/37637781/calculating-the-negabinary-representation-of-a-given-number-without-loops
static uint32_t binary_to_negabinary(int32_t bin) {
    if (bin > 0x55555555) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}

/*
static int32_t negabinary_to_binary(uint32_t neg) {
    //const int32_t even = 0x2AAAAAAA, odd = 0x55555555;
    //if ((neg & even) > (neg & odd)) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask ^ neg) - mask;
}
*/

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
static inline int shrink_non_power_of_two(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, char** rbuf, SwingInfo* info, uint* dimensions_virtual, uint* coordinates, int* idle, uint port){    
    uint coord_peer[MAX_SUPPORTED_DIMENSIONS];
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    int skip_send, skip_recv;
    int res;

    retrieve_coord_mapping(coordinates, info->rank, coord);
    memcpy(dimensions_virtual, dimensions, sizeof(dimensions));
    *idle = 0;
    int do_something = 0;
    for(size_t i = 0; i < dimensions_num; i++){
        // This dimensions is not a power of two, shrink it
        if(!is_power_of_two(dimensions[i])){
            memcpy(coord_peer, coord, sizeof(uint)*dimensions_num);
            dimensions_virtual[i] = pow(2, info->num_steps_per_dim[i] - 1);
            int extra = dimensions[i] - dimensions_virtual[i];
            if(coord[i] >= dimensions_virtual[i]){            
                coord_peer[i] = coord[i] - extra;
                skip_send = 0;
                skip_recv = 1;
                *idle = 1;
                do_something = 1;
            }else if(coord[i] + extra >= dimensions_virtual[i]){
                coord_peer[i] = coord[i] + extra;
                skip_send = 1;
                skip_recv = 0;
                do_something = 1;
            }else{
                do_something = 0;
            }
            if(do_something){
                int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                res = MPI_Allreduce_lat_optimal_swing_sendrecv(sendbuf, recvbuf, count, datatype, op, comm, info, skip_send, skip_recv, rbuf, peer, 998, 0, port);
                if(res != MPI_SUCCESS){
                    return res;
                }
                if(*idle){break;}
            }
        }
    }
    int rank_virtual = getIdFromCoord(coord, dimensions_virtual, dimensions_num);
    DPRINTF("[%d] Virtual rank %d coord (%d, %d, %d) dimensions_virtual (%d, %d, %d) idle %d\n", info->rank, rank_virtual, coord[0], coord[1], coord[2], dimensions_virtual[0], dimensions_virtual[1], dimensions_virtual[2], *idle);
    return rank_virtual;
}

static inline int enlarge_non_power_of_two(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, char** rbuf, SwingInfo* info, uint* coordinates, uint port){
    int skip_send, skip_recv;
    int res;
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, info->rank, coord);
    int do_something = 0;
    uint coord_peer[MAX_SUPPORTED_DIMENSIONS];
    //for(size_t d = 0; d < dimensions_num; d++){
    for(int d = dimensions_num - 1; d >= 0; d--){
        // This dimensions was a non-power of two, enlarge it
        if(!is_power_of_two(dimensions[d])){
            memcpy(coord_peer, coord, sizeof(uint)*dimensions_num);
            int dim_virtual = pow(2, info->num_steps_per_dim[d] - 1);
            int extra = dimensions[d] - dim_virtual;
            int overwrite = 0;
            if(coord[d] >= (uint) dim_virtual){                
                coord_peer[d] = coord[d] - extra;
                skip_send = 1;
                skip_recv = 0;
                overwrite = 1;
                do_something = 1;
            }else if(coord[d] + extra >= (uint) dim_virtual){
                coord_peer[d] = coord[d] + extra;
                skip_send = 0;
                skip_recv = 1;
                do_something = 1;
            }else{
                do_something = 0;
            }
            if(do_something){
                int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);
                res = MPI_Allreduce_lat_optimal_swing_sendrecv(sendbuf, recvbuf, count, datatype, op, comm, info, skip_send, skip_recv, rbuf, peer, 999, overwrite, port);
                if(res != MPI_SUCCESS){
                    return res;
                }
            }
        }
    }
    return MPI_SUCCESS;
}

static inline int MPI_Allreduce_lat_optimal_swing(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info, uint port){
    // Reduce the number of steps since we shrink
    info->num_steps = 0;
    for(size_t d = 0; d < dimensions_num; d++){
        info->num_steps += floor(log2(dimensions[d]));
    }
    int res;    
    char* rbuf = NULL; // Temporary buffer (to avoid overwriting sendbuf)
    int skip_send = 0, skip_recv = 0;
    uint** peers = (uint**) malloc(sizeof(uint*)*info->size);    
    uint dimensions_virtual[MAX_SUPPORTED_DIMENSIONS];   

    // Compute real and virtual coordinates
    uint *coordinates, *coordinates_virtual;
    coordinates = (uint*) malloc(sizeof(uint)*info->size*dimensions_num);
    coordinates_virtual = (uint*) malloc(sizeof(uint)*info->size*dimensions_num);
    compute_rank_to_coord_mapping(info->size, dimensions, coordinates, info->size);
    uint coord[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, info->rank, coord);
    
    int idle = 0;
    int rank_virtual = shrink_non_power_of_two(sendbuf, recvbuf, count, datatype, op, comm, &rbuf, info, dimensions_virtual, coordinates, &idle, port);
    int size_virtual = 1;
    for(size_t i = 0; i < dimensions_num; i++){
        size_virtual *= dimensions_virtual[i];
    }
    compute_rank_to_coord_mapping(info->size, dimensions_virtual, coordinates_virtual, size_virtual);
    DPRINTF("[%d] Virtual dimensions (%d, %d, %d)\n", info->rank, dimensions_virtual[0], dimensions_virtual[1], dimensions_virtual[2]);
    peers[rank_virtual] = (uint*) malloc(sizeof(uint)*info->num_steps);

    if(!idle){
        int num_steps_virtual = 0;
        for(size_t i = 0; i < dimensions_num; i++){
            num_steps_virtual += ceil(log2(dimensions_virtual[i]));
        }
        DPRINTF("[%d] Computing peers\n", info->rank);  
        compute_peers(peers, port, num_steps_virtual, rank_virtual, coordinates_virtual, dimensions_virtual); 
        DPRINTF("[%d] Peers computed\n", info->rank);
        for(size_t step = 0; step < (uint) num_steps_virtual; step++){     
            skip_send = 0;
            skip_recv = 0;
            int virtual_peer = peers[rank_virtual][step]; 
            uint coord_peer[MAX_SUPPORTED_DIMENSIONS];
            retrieve_coord_mapping(coordinates_virtual, virtual_peer, coord_peer);
            int peer = getIdFromCoord(coord_peer, dimensions, dimensions_num);        

            DPRINTF("[%d] Step %d Peer (virtual): %d (real): %d coord: (%d, %d, %d)\n", info->rank, step, virtual_peer, peer, coord_peer[0], coord_peer[1], coord_peer[2]);  
            
            res = MPI_Allreduce_lat_optimal_swing_sendrecv(sendbuf, recvbuf, count, datatype, op, comm, info, skip_send, skip_recv, &rbuf, peer, step, 0, port);
            if(res != MPI_SUCCESS){
                return res;
            }
        }    
    }

    enlarge_non_power_of_two(sendbuf, recvbuf, count, datatype, op, comm, &rbuf, info, coordinates, port);
    
    if(rbuf){
        free(rbuf);
    }
    free(peers[rank_virtual]);
    free(peers);
    free(coordinates);
    free(coordinates_virtual);
    return res;
}

static inline int check_last_n_bits_equal(uint32_t a, uint32_t b, uint32_t n){
    uint32_t mask = (1 << n) - 1;
    return (a & mask) == (b & mask);
}

static inline int get_block_distance(int rank, int block, uint port){
    if(((!is_odd(rank) && is_odd(block)) || (is_odd(rank) && is_odd(block)))){
        if(port < dimensions_num){
            return block - rank;                
        }else{
            // If port >= dimensions_num, this is a mirrored collective.
            // This means that the signs in Eq. 4 would be flipped, as well as the
            // conditions to determine how to compute the block distance (r-q or q-r)
            return rank - block;
        }
    }else{
        if(port < dimensions_num){
            return rank - block;
        }else{
            // Mirrored
            return block - rank;
        }
    }
}

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

// Returns the step in which rank will send the block.
static int get_step_to_reach(int rank, int block, int num_steps, int size, SwingInfo* info, uint port){
    int first_step_a = -1, first_step_b = -1, is_in_range_a = 0, is_in_range_b = 0;
    int block_distance_a, block_distance_b;
    uint32_t block_distance_neg_a = 0, block_distance_neg_b = 0;
    block_distance_a = get_block_distance(rank, block, port);
    if(block_distance_a < 0){
        block_distance_b = block_distance_a + size;
    }else{
        block_distance_b = block_distance_a - size;
    }

    is_in_range_a = in_range(block_distance_a, num_steps);
    is_in_range_b = in_range(block_distance_b, num_steps);

    if(is_in_range_a){
        block_distance_neg_a = binary_to_negabinary(block_distance_a);
        first_step_a = get_first_step(block_distance_neg_a);
    }

    if(is_in_range_b){
        block_distance_neg_b = binary_to_negabinary(block_distance_b);
        first_step_b = get_first_step(block_distance_neg_b);
    }
    if(rank == info->rank){
        DPRINTF("[%d] Block %d, distance (%d, %d), distance_neg (%d, %d), is_in_range (%d, %d), first_step (%d, %d)\n", info->rank,  block, block_distance_a, block_distance_b, block_distance_neg_a, block_distance_neg_b, is_in_range_a, is_in_range_b, first_step_a, first_step_b);
    }
    
    assert(!(first_step_a == -1 && first_step_b == -1));
    if(first_step_a == -1 && first_step_b != -1){ // I can reach it only in one way
        return first_step_b;
    }else if(first_step_a != -1 && first_step_b == -1){ // I can reach it only in one way
        return first_step_a;
    }else if(first_step_a == first_step_b){ // I can reach it with two different combinations of steps, but they both start at the first step
        return first_step_a;
    }else{ // I can reach it in two different ways, but they start at different steps. We choose the one that arrives later (i.e., in the last step)
        if(get_last_step(block_distance_neg_a) > get_last_step(block_distance_neg_b)){
            return first_step_a;
        }else{
            return first_step_b;
        }
    }
}


static uint get_step_to_reach_multid(uint* coord_mine, uint* coord_block, int num_steps, int size, SwingInfo* info, uint* coordinates, uint port){
#if MAX_SUPPORTED_DIMENSIONS == 1
        return get_step_to_reach(rank, block, info->num_steps, info->size, info);
#else       
        int min_step = info->num_steps + 1, min_d = MAX_SUPPORTED_DIMENSIONS;
        uint starting_dimension = port % dimensions_num; 
        // TODO: For now we assume we start from dimension 0 and go forward, fix when moving to multiport.
        // Fix by starting d = start_dimension and looping thorugh the rest
        for(size_t i = 0; i < dimensions_num; i++){
            size_t d = (i + starting_dimension) % dimensions_num;
            if(coord_block[d] != coord_mine[d]){
                //DPRINTF("[%d] Rank %d Block %d, coord_block[%d] %d, coord_mine[%d] %d\n", info->rank, rank, block, d, coord_block[d], d, coord_mine[d]);
                int st = get_step_to_reach(coord_mine[d], coord_block[d], info->num_steps_per_dim[d], dimensions[d], info, port);
                if(st < min_step){
                    min_step = st;
                    min_d = i;
                }
            }
        }

        // Convert from relative to absolute step
        uint actual_step = 0;
        for(size_t i = 0; i < dimensions_num; i++){
            uint d = (i + starting_dimension) % dimensions_num;
            if(i < (uint) min_d){
                actual_step += std::min(min_step, info->num_steps_per_dim[d] - 1) + 1;
            }else{
                actual_step += std::min(min_step - 1, info->num_steps_per_dim[d] - 1) + 1;
            }
        }


        /*
        if(info->rank == 0){
            DPRINTF("eeee going to reach block (%d,%d) at step %d on dim %d actual step %d\n", coord_block[0], coord_block[1], min_step, min_d, actual_step);
        }
        */

        /*
        if(rank == info->rank){
            DPRINTF("[%d] Block %d, min_step %d, min_d %d, actual_step %d\n", info->rank, block, min_step, min_d, actual_step);
        }
        */
        return actual_step;
#endif
}

static int swing_coll_new(void *buf, void* rbuf, const int *blocks_sizes, const int *blocks_displs, 
                           MPI_Op op, MPI_Comm comm, int size, int rank, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type, SwingInfo* info, uint** peers, uint port, uint32_t* step_to_send, uint32_t* step_to_recv){
    int tag, res;  
    uint num_steps = info->num_steps;

    if(coll_type == SWING_REDUCE_SCATTER){
        tag = TAG_SWING_REDUCESCATTER + port;
    }else{
        tag = TAG_SWING_ALLGATHER + port;
    }

    MPI_Win win = MPI_WIN_NULL;
    if(rdma){
        int count = 0;
        for(size_t i = 0; i < (uint) size; i++){
            count += blocks_sizes[i];
        }
        MPI_Win_create(buf, count*info->dtsize, info->dtsize, MPI_INFO_NULL, comm, &win);
        MPI_Win_fence(0, win);
    }

    MPI_Request* requests_s = (MPI_Request*) malloc(sizeof(MPI_Request)*size);
    MPI_Request* requests_r = (MPI_Request*) malloc(sizeof(MPI_Request)*size);
    int* req_idx_to_block_idx = (int*) malloc(sizeof(int)*size);
    
    // Iterate over steps
    for(size_t step = 0; step < num_steps; step++){        
        size_t block_step = (coll_type == SWING_REDUCE_SCATTER)?step:(num_steps - step - 1);            

        //int peer = get_peer(block_step, info->rank, info->size);
        int peer = peers[info->rank][block_step];
        DPRINTF("[%d] Starting step %d peer %d\n", rank, step, peer);

        // Sendrecv + aggregate
        uint num_requests_s = 0, num_requests_r = 0;
        for(size_t i = 0; i < (uint) size; i++){
            int send_block, recv_block;
            if(coll_type == SWING_REDUCE_SCATTER){
                send_block = step_to_send[i] & (0x1 << block_step);
                recv_block = step_to_recv[i] & (0x1 << block_step);
            }else{
                recv_block = step_to_send[i] & (0x1 << block_step);
                send_block = step_to_recv[i] & (0x1 << block_step);
            }

            //DPRINTF("[%d] Block %d (send %d recv %d)\n", rank, i, send_block, recv_block);
            if(send_block){              
                DPRINTF("[%d] Sending block %d to %d at step %d (coll %d)\n", rank, i, peer, step, coll_type);
                if(rdma){
                    if(coll_type == SWING_REDUCE_SCATTER){
                        res = MPI_Accumulate(((char*) buf) + blocks_displs[i]*info->dtsize, blocks_sizes[i], sendtype, peer, blocks_displs[i], blocks_sizes[i], recvtype, op, win);
                    }else{
                        res = MPI_Put(((char*) buf) + blocks_displs[i]*info->dtsize, blocks_sizes[i], sendtype, peer, blocks_displs[i], blocks_sizes[i], recvtype, win);
                    }
                }else{                    
                    res = MPI_Isend(((char*) buf) + blocks_displs[i]*info->dtsize, blocks_sizes[i], sendtype, peer, tag, comm, &(requests_s[num_requests_s]));
                    if(res != MPI_SUCCESS){return res;}
                    ++num_requests_s;
                }                
            }
            if(recv_block){
                DPRINTF("[%d] Receiving block %d from %d at step %d (coll %d)\n", rank, i, peer, step, coll_type);
                if(!rdma){                    
                    res = MPI_Irecv(((char*) rbuf) + blocks_displs[i]*info->dtsize, blocks_sizes[i], recvtype, peer, tag, comm, &(requests_r[num_requests_r]));
                    if(res != MPI_SUCCESS){return res;}
                    req_idx_to_block_idx[num_requests_r] = i;
                    ++num_requests_r;
                }
            }
        }

        // Overlap here
        
        // End overlap
        if(!rdma){
            if(coll_type == SWING_REDUCE_SCATTER){
                int index;
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
                    size_t displ_bytes = info->dtsize*blocks_displs[block_idx];
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


static inline int MPI_Allreduce_bw_optimal_swing(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info, uint port){    
    int res;
    int* recvcounts = (int*) malloc(sizeof(int)*info->size);
    int* displs = (int*) malloc(sizeof(int)*info->size);
    uint** peers = (uint**) malloc(sizeof(uint*)*info->size);    
    peers[info->rank] = (uint*) malloc(sizeof(uint)*info->num_steps);
    DPRINTF("[%d] Computing peers\n", info->rank);       
    uint* coordinates = (uint*) malloc(sizeof(uint)*info->size*dimensions_num);
    compute_rank_to_coord_mapping(info->size, dimensions, coordinates, info->size);
    compute_peers(peers, port, info->num_steps, info->rank, coordinates); 

    // Define blocks sizes and displacements
    size_t last = 0;
    size_t normcount = ceil(count / info->size);
    for(size_t i = 0; i < (uint) info->size; i++){
        recvcounts[i] = normcount;
        if(i == (uint) info->size - 1){
            recvcounts[i] = count - ((count / info->size)*(info->size - 1));
        } 
        displs[i] = last;
        last += recvcounts[i];
    }
    
    size_t total_size_bytes = last*info->dtsize;
    char* rbuf = (char*) malloc(total_size_bytes);
    memcpy(recvbuf, sendbuf, total_size_bytes);


    // For each block, compute the step in which it must be sent
    uint32_t* step_to_send; // Steps in which each block must be sent. Each element of the array is a 32-bit integer, where each bit represents a step. If 1 the block must be sent in that step
    step_to_send = (uint32_t*) malloc(sizeof(uint32_t)*info->size);
    uint32_t* step_to_recv; // Steps in which each block must be recvd. Each element of the array is a 32-bit integer, where each bit represents a step. If 1 the block must be recvd in that step
    step_to_recv = (uint32_t*) malloc(sizeof(uint32_t)*info->size);   
    memset(step_to_send, 0, sizeof(uint32_t)*info->size);
    memset(step_to_recv, 0, sizeof(uint32_t)*info->size); 
    uint coord_block[MAX_SUPPORTED_DIMENSIONS];
    uint coord_mine[MAX_SUPPORTED_DIMENSIONS];
    retrieve_coord_mapping(coordinates, info->rank, coord_mine);

    for(size_t i = 0; i < (uint) info->size; i++){
        // In reducescatter I never send my block
        // I precompute it so that get_step_to_reach_multid is called only 'size' times rather than 'size x num_steps' times.
        if(i != (uint) info->rank){            
            retrieve_coord_mapping(coordinates, i, coord_block);      
            step_to_send[i] |= (0x1 << get_step_to_reach_multid(coord_mine, coord_block, info->num_steps, info->size, info, coordinates, port));
        }
    }
    
    // TODO: Don't like this nested loop, find a way to simplify it...    
    for(size_t i = 0; i < (uint) info->size; i++){
        retrieve_coord_mapping(coordinates, i, coord_block);        
        for(size_t step = 0; step < (uint) info->num_steps; step++){
            uint peer = peers[info->rank][step];
            if(i != peer){
                retrieve_coord_mapping(coordinates, peer, coord_mine);
                if(get_step_to_reach_multid(coord_mine, coord_block, info->num_steps, info->size, info, coordinates, port) == step){
                    step_to_recv[i] |= (0x1 << step);
                }
            }
        }
    }

    res = swing_coll_new(recvbuf, rbuf   , recvcounts, displs, op, comm, info->size, info->rank, datatype, datatype, SWING_REDUCE_SCATTER, info, peers, port, step_to_send, step_to_recv);
    if(res != MPI_SUCCESS){return res;} 
    res = swing_coll_new(recvbuf, recvbuf, recvcounts, displs, op, comm, info->size, info->rank, datatype, datatype, SWING_ALLGATHER     , info, peers, port, step_to_send, step_to_recv);
    if(res != MPI_SUCCESS){return res;}

    free(coordinates);
    free(step_to_send);
    free(step_to_recv);
    free(recvcounts);
    free(displs);            
    free(rbuf);
    free(peers[info->rank]);
    free(peers);
    return res;  
}

static int MPI_Allreduce_int(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info, uint port){
    //printf("[%d] Running an allreduce on %d elements on port %d (starting pointer %p)\n", info->rank, count, port, sendbuf);
    if(algo == ALGO_SWING_L){ // Swing_l
        return MPI_Allreduce_lat_optimal_swing(sendbuf, recvbuf, count, datatype, op, comm, info, port);
    }else if(algo == ALGO_SWING_B){ // Swing_b
        return MPI_Allreduce_bw_optimal_swing(sendbuf, recvbuf, count, datatype, op, comm, info, port);
    }else if(algo == ALGO_RING){ // Ring
        return MPI_Allreduce_ring(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_RECDOUB_B){ // Recdoub_b
        return MPI_Allreduce_recdoub_b(sendbuf, recvbuf, count, datatype, op, comm);
    }else if(algo == ALGO_RECDOUB_L){ // Recdoub_l
        return MPI_Allreduce_recdoub_l(sendbuf, recvbuf, count, datatype, op, comm);
    }else{
        return 1;
    }
}

static int MPI_Allreduce_split_max_size(const void *sendbuf, void *recvbuf, uint count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, SwingInfo* info, uint port){        
    int res;            
    int remaining_count = count;
    int max_count, next_offset = 0;

    if(!max_size || count*info->dtsize <= max_size){
        max_count = count;
    }else{
        max_count = max_size / info->dtsize;
    }
    do{
        int next_count = std::min(remaining_count, max_count);
        res = MPI_Allreduce_int(((char*)sendbuf) + next_offset*info->dtsize, ((char*) recvbuf) + next_offset*info->dtsize, next_count, datatype, op, comm, info, port);
        if(res != MPI_SUCCESS){
            return res;
        }
        next_offset += next_count;
        remaining_count -= next_count;
    }while(remaining_count > 0);   
    return MPI_SUCCESS;         
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
            info.offset_per_dim[i] = offset;
            offset += info.num_steps_per_dim[i];
        }
        // The number of steps is not ceil(log2(size)) but the sum of the number of steps for each dimension.
        // This is needed for those cases where dimensions are not powers of two. Consider for example a 
        // 10x10 torus. We would perform ceil(log2(10)) + ceil(log2(10)) = 4 + 4 = 8 steps, not ceil(log2(100)) = 7.
        info.num_steps = offset;

        if(info.num_steps > LIBSWING_MAX_STEPS){
            assert("Max steps limit must be increased and constants updated.");
        }
        
        if(!multiport || (algo != ALGO_SWING_B && algo != ALGO_SWING_L)){
            return MPI_Allreduce_split_max_size(sendbuf, recvbuf, count, datatype, op, comm, &info, 0);
        }else{
            // First split the data across the ports.
            // If still to big (> max_size), further split it in chunks of max_size
            uint num_ports = dimensions_num*2;
            uint offsets[MAX_SUPPORTED_DIMENSIONS*2];
            uint counts_per_port[MAX_SUPPORTED_DIMENSIONS*2];
            uint partition_size = count / num_ports;
            uint remaining = count % num_ports;
            uint count_so_far = 0;
            for(size_t i = 0; i < num_ports; i++){
                counts_per_port[i] = partition_size + (i < remaining ? 1 : 0);
                offsets[i] = count_so_far;
                count_so_far += counts_per_port[i];
            }

            int res[MAX_SUPPORTED_DIMENSIONS*2];
            // Go in parallel over all the ports
            #pragma omp parallel for num_threads(num_ports)
            for(size_t i = 0; i < num_ports; i++){
                res[i] = MPI_Allreduce_split_max_size(((char*) sendbuf) + offsets[i]*info.dtsize, ((char*) recvbuf) + offsets[i]*info.dtsize, counts_per_port[i], datatype, op, comm, &info, i);
            }
            for(size_t i = 0; i < num_ports; i++){
                if(res[i] != MPI_SUCCESS){
                    return res[i];
                }
            }
            return MPI_SUCCESS;
        }
    }
}
// TODO: Don't use Swing for non-continugous non-native datatypes (tedious implementation)
