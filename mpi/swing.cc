#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static unsigned int disable_reducescatter, disable_allgather, disable_allreduce, debug, env_read = 0;

void read_env(MPI_Comm comm){
    if(!env_read){
        char* env_str = getenv("LIBSWING_DISABLE_REDUCESCATTER");
        if(!env_str){
            disable_reducescatter = 0;
        }else{
            disable_reducescatter = atoi(env_str);
        }

        env_str = getenv("LIBSWING_DISABLE_ALLGATHER");
        if(!env_str){
            disable_allgather = 0;
        }else{
            disable_allgather = atoi(env_str);
        }

        env_str = getenv("LIBSWING_DISABLE_ALLREDUCE");
        if(!env_str){
            disable_allreduce = 0;
        }else{
            disable_allreduce = atoi(env_str);
        }

        env_str = getenv("LIBSWING_DEBUG");
        if(!env_str){
            debug = 0;
        }else{
            debug = atoi(env_str);
        }

        if(debug){
            int rank;
            MPI_Comm_rank(comm, &rank);
            if(rank == 0){
                printf("Libswing called. Environment:\n");
                printf("------------------------------------\n");
                printf("LIBSWING_DISABLE_REDUCESCATTER: %d\n", disable_reducescatter);
                printf("LIBSWING_DISABLE_ALLGATHER: %d\n", disable_allgather);
                printf("LIBSWING_DISABLE_ALLREDUCE: %d\n", disable_allreduce);
                printf("LIBSWING_DEBUG: %d\n", debug);
                printf("------------------------------------\n");
            }
        }
        env_read = 1;
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

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
    read_env(comm);
    if(disable_allgather){
        return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }else{
        return 0;
    }
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    if(disable_allreduce){
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }else{
        int r;
        int size;
        MPI_Comm_size(comm, &size);
        int* recvcounts = (int*) malloc(sizeof(int)*size);
        for(size_t i = 0; i < size; i++){
            recvcounts[i] = count / size;
            if(i == size - 1){
                //assert("might create problems with allgather, should use allgatherv to manage last block" == NULL);
                recvcounts[i] = count - ((count / size)*(size - 1));
            } 
        }
        r = MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
        free(recvcounts);
        if(r != MPI_SUCCESS){        
            return r;
        }
        return MPI_Allgather(sendbuf, count / size, datatype, recvbuf, count / size, datatype, comm);
    }
}