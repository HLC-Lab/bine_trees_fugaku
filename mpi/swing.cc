#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static unsigned int disable_reducescatter = 0, disable_allgatherv = 0, disable_allreduce = 0, debug = 0;

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
            printf("------------------------------------\n");
        }
    }
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_reducescatter){
        return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
    }else{
        return 0;
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
        return 0;
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
            recvcounts[i] = count / size;
            if(i == size - 1){
                recvcounts[i] = count - ((count / size)*(size - 1));
            } 
            displs[i] = last;
            last += recvcounts[i];
        }
        char* intermediate_buf = (char*) recvbuf + displs[rank];
        res = MPI_Reduce_scatter(sendbuf, intermediate_buf, recvcounts, datatype, op, comm);
        if(res == MPI_SUCCESS){        
            res = MPI_Allgatherv(intermediate_buf, recvcounts[rank], datatype, recvbuf, recvcounts, displs, datatype, comm);
        }
        free(recvcounts);
        free(displs);
        return res;
    }
}