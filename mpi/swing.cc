#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define MAX_SUPPORTED_DIMENSIONS 8 // We support up to 8D torus

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allreduce = 0, debug = 0, dimensions_num = 1;
static uint dimensions[MAX_SUPPORTED_DIMENSIONS];

void read_env(MPI_Comm comm){
    char* env_str = getenv("LIBSWING_DISABLE_REDUCESCATTER");
    if(env_str){
        disable_reducescatter = atoi(env_str);
    }

    env_str = getenv("LIBSWING_DISABLE_ALLGATHERV");
    if(env_str){
        disable_allgatherv = atoi(env_str);
    }

    env_str = getenv("LIBSWING_DISABLE_ALLREDUCE");
    if(env_str){
        disable_allreduce = atoi(env_str);
    }

    env_str = getenv("LIBSWING_DEBUG");
    if(env_str){
        debug = atoi(env_str);
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

    if(debug){
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0){
            printf("Libswing called. Environment:\n");
            printf("------------------------------------\n");
            printf("LIBSWING_DISABLE_REDUCESCATTER: %d\n", disable_reducescatter);
            printf("LIBSWING_DISABLE_ALLGATHERV: %d\n", disable_allgatherv);
            printf("LIBSWING_DISABLE_ALLREDUCE: %d\n", disable_allreduce);
            printf("LIBSWING_DEBUG: %d\n", debug);
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
    }
}

static int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

// Convert a rank id into a list of d-dimensional coordinates
static void getCoordFromId(int id, int* coord){
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
static int getIdFromCoord(int* coord, uint* dimensions, uint dimensions_num){
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

static void compute_peers(uint** peers, int size, int rank, int port){
    int coord[MAX_SUPPORTED_DIMENSIONS];
    bool terminated_dimensions_bitmap[MAX_SUPPORTED_DIMENSIONS];
    int next_directions[MAX_SUPPORTED_DIMENSIONS];
    for(uint rank = 0; rank < size; rank++){
        // Compute default directions
        getCoordFromId(rank, coord);
        for(size_t i = 0; i < dimensions_num; i++){
            next_directions[i] = -1^((coord[i] % 2) + (port % 2));
            terminated_dimensions_bitmap[i] = false;            
        }
        
        int target_dim, relative_step, distance;
        uint terminated_dimensions = 0, o = 0;
        
        // Generate peers
        for(size_t i = 0; i < ceil(log2(size)); ){
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

int MPI_Reduce_scatter_swing(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int res, size, rank, dtsize;    
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(datatype, &dtsize);
    uint** peers = (uint**) malloc(sizeof(uint*)*size);
    for(uint rank = 0; rank < size; rank++){
        peers[rank] = (uint*) malloc(sizeof(uint)*ceil(log2(size)));
    }
    compute_peers(peers, size, rank, 0); // TODO: For now we assume it is single-ported (and we pass port 0), extending should be trivial

    // Create a temporary buffer (to avoid overwriting sendbuf)
    size_t buf_size = 0;
    int* displs = (int*) malloc(sizeof(int)*size);
    for(size_t i = 0; i < size; i++){
        displs[i] = buf_size;
        buf_size += recvcounts[i]; // TODO: If custom datatypes, manage appropriately        
    }
    char* buf = (char*) malloc(dtsize*buf_size);
    memcpy(buf, sendbuf, buf_size);


    // Copy the block in recvbuf
    memcpy(recvbuf, buf + displs[rank], recvcounts[rank]);
    free(buf);
    return 0;
}


int MPI_Allgatherv_swing(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, 
                         const int *displs, MPI_Datatype recvtype, MPI_Comm comm){
    return 0;
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_reducescatter){
        return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    }else{
        return MPI_Reduce_scatter_swing(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    }
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, 
                 MPI_Datatype recvtype, MPI_Comm comm){
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
}

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, 
                   const int *displs, MPI_Datatype recvtype, MPI_Comm comm){
    read_env(comm);
    if(disable_allgatherv){
        return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }else{
        return MPI_Allgatherv_swing(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
    }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_allreduce){
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }else{
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
                recvcounts[i] = count - ((count / size)*(size - 1)); // TODO: Can be unbalanced, there are better ways to partition
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
    }
}

// TODO: Don't use Swing for non-continugous non-native datatypes (tedious implementation)