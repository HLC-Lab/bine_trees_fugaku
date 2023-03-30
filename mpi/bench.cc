#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

// Usage: ./bench msgsize(elems) iterations
int main(int argc, char** argv){
    int warmup = 10;    
    int count = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    double* samples = (double*) malloc(sizeof(double)*iterations);
    double* samples_all;
    long i, r;
    MPI_Init(&argc, &argv);
    int rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char* buffer = (char*) malloc(sizeof(char)*comm_size);
    float* sendbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf = (float*) malloc(sizeof(float)*count);
    if(rank == 0){
        samples_all = (double*) malloc(sizeof(double)*comm_size*iterations);
    }
    for(i = -10; i < iterations; i++){
        //usleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if(i >= 0){
            samples[i] = ((MPI_Wtime() - start_time)*1000000.0);
        }
    }
    MPI_Gather(samples, iterations, MPI_DOUBLE, samples_all, iterations, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("#MessageSize ");
        for(r = 0; r < comm_size; r++){
            printf("Rank%ldTime(us) ", r);
        }
        printf("\n");
        double avg_iteration = 0.0;
        for(i = 0; i < iterations; i++){
            printf("%d ", count);
            double avg_ranks = 0.0;
            for(r = 0; r < comm_size; r++){
               printf("%f ", samples_all[r*iterations + i]);
               avg_ranks += samples_all[r*iterations + i];
            }
            avg_iterations += avg_ranks / comm_size;
            printf("\n");
        }
        avg_iteration /= iterations;
        printf("Average runtime: %f\n", avg_iterations);
    }
    MPI_Finalize();
    free(sendbuf);
    free(recvbuf);
    free(buffer);
    free(samples);
    return 0;
}