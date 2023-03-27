#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    int r, rank;
    int count = 16;
    float* sendbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf_v = (float*) malloc(sizeof(float)*count);

    // Fill sendbuf with random data
    for(size_t i = 0; i < count; i++){
        sendbuf[i] = 1; //rand() % 1024;
    }

    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Run first swing allreduce
    setenv("LIBSWING_DISABLE_REDUCESCATTER", "0", 1);
    setenv("LIBSWING_DISABLE_ALLGATHERV", "1", 1);
    setenv("LIBSWING_DISABLE_ALLREDUCE", "0", 1);
    r = MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if(r != MPI_SUCCESS){
        fprintf(stderr, "Allreduce failed with error %d\n", r);
        return r;
    }

    // Then run the original MPI allreduce
    setenv("LIBSWING_DISABLE_REDUCESCATTER", "1", 1);
    setenv("LIBSWING_DISABLE_ALLGATHERV", "1", 1);
    setenv("LIBSWING_DISABLE_ALLREDUCE", "1", 1);
    r = MPI_Allreduce(sendbuf, recvbuf_v, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if(r != MPI_SUCCESS){
        fprintf(stderr, "Allreduce failed with error %d\n", r);
        return r;
    }

    // Fini
    MPI_Finalize();    

    // Then validate
    bool valid = true;
    for(size_t i = 0; i < count; i++){
        if(recvbuf[i] != recvbuf_v[i]){
            fprintf(stderr, "[%d] Validation failed at index %d (%f but should be %f)\n", rank, i, recvbuf[i], recvbuf_v[i]);
            valid = false;
        }
    }

    if(valid){
        printf("Validation succeeded.\n");
    }
    return 0;
}