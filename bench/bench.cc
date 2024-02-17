#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

void voidop(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
    return;
}


// Usage: ./bench type msgsize(elems) iterations
int main(int argc, char** argv){
    int warmup = 10;    
    char* type = argv[1];
    int count = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    double* samples = (double*) malloc(sizeof(double)*iterations);
    double* samples_all;
    long i, r;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if(provided < MPI_THREAD_MULTIPLE){
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Op MPI_VOIDOP;
    MPI_Op_create(voidop, 1, &MPI_VOIDOP);
    MPI_Datatype dt;
    MPI_Op op = MPI_SUM;
    if(strcmp(type, "CHAR") == 0){
        dt = MPI_CHAR;
        count *= 4;
    }else if(strcmp(type, "FLOAT") == 0){
        dt = MPI_FLOAT;
    }else if(strcmp(type, "INT") == 0){
        dt = MPI_INT;
    }else if(strcmp(type, "VOID") == 0){
        dt = MPI_FLOAT;
        op = MPI_VOIDOP;
    }else{
        fprintf(stderr, "Unknown type %s\n", type);
        return 1;
    }
    int dtsize;
    MPI_Type_size(dt, &dtsize);
    char* sendbuf = (char*) malloc(dtsize*count);
    char* recvbuf = (char*) malloc(dtsize*count);
    if(rank == 0){
        samples_all = (double*) malloc(sizeof(double)*comm_size*iterations);
    }
    for(i = -warmup; i < iterations; i++){
        //usleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        MPI_Allreduce(sendbuf, recvbuf, count, dt, op, MPI_COMM_WORLD);

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
            double max_ranks = 0.0;
            for(r = 0; r < comm_size; r++){
                double sample = samples_all[r*iterations + i];
               printf("%f ", sample);
               if(sample > max_ranks){
                max_ranks = sample;
               }
            }
            avg_iteration += max_ranks;
            printf("\n");
        }
        avg_iteration /= iterations;
        printf("Average runtime: %f\n", avg_iteration);
    }
    MPI_Finalize();
    free(sendbuf);
    free(recvbuf);
    free(samples);
    return 0;
}
