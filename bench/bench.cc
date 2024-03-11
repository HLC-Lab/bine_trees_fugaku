#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

void voidop(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
    return;
}

typedef enum{
    RUN_TYPE_VALIDATION = 0,
    RUN_TYPE_BENCHMARK = 1
}RunType;

int run_collective(RunType rt, const char* collective, const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype dt, MPI_Op op, size_t size){
    int r = MPI_SUCCESS;
    if(!strcmp(collective, "MPI_Allreduce")){
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Allreduce(sendbuf, recvbuf, count, dt, op, MPI_COMM_WORLD);
        }else{
            r = MPI_Allreduce(sendbuf, recvbuf, count, dt, op, MPI_COMM_WORLD);
        }
    }else if(!strcmp(collective, "MPI_Reduce_scatter")){
        int* recvcounts = (int*) malloc(sizeof(int)*size);
        size_t partition_size = count / size;
        size_t remaining = count % size;                
        for(size_t i = 0; i < size; i++){
            size_t count_i = partition_size + (i < remaining ? 1 : 0);
            recvcounts[i] = count_i;
        }
        if(rt == RUN_TYPE_VALIDATION){
            r = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, dt, op, MPI_COMM_WORLD);
        }else{
            r = MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, dt, op, MPI_COMM_WORLD);
        }
        free(recvcounts);
    }
    return r;
}


// Usage: ./bench collective type msgsize(elems) iterations
int main(int argc, char** argv){
    int warmup = 10;    
    char* collective = argv[1];
    char* type = argv[2];
    int count = atoi(argv[3]);
    int iterations = atoi(argv[4]);
    double* samples = (double*) malloc(sizeof(double)*iterations);
    double* samples_all = NULL;
    long i;
    int r;
    MPI_Init(&argc, &argv);
    int rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Op MPI_VOIDOP;
    MPI_Op_create(voidop, 1, &MPI_VOIDOP);
    MPI_Datatype dt;
    MPI_Op op = MPI_SUM;
    if(strcmp(type, "CHAR") == 0){
        dt = MPI_CHAR;
    }else if(strcmp(type, "BYTE") == 0){
        dt = MPI_BYTE;
    }else if(strcmp(type, "FLOAT") == 0){
        dt = MPI_FLOAT;
    }else if(strcmp(type, "INT") == 0){
        dt = MPI_INT;
    }else if(strcmp(type, "INT32") == 0){
        dt = MPI_INT32_T;
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
    char* recvbuf_validation = (char*) malloc(dtsize*count); // To check correctness of results
    if(rank == 0){
        samples_all = (double*) malloc(sizeof(double)*comm_size*iterations);
    }

    // Initialize sendbuf with random values
    srand(time(NULL));
    for(i = 0; i < count; i++){
        sendbuf[i] = (char) rand() % 1024;
    }
    r = run_collective(RUN_TYPE_VALIDATION, collective, sendbuf, recvbuf_validation, count, dt, op, comm_size);
    if(r != MPI_SUCCESS){
        fprintf(stderr, "Rank %d: Validation failed with error %d\n", rank, r);
        return 1;
    }

    for(i = -warmup; i < iterations; i++){
        //usleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        // Run the collective
        r = run_collective(RUN_TYPE_BENCHMARK, collective, sendbuf, recvbuf, count, dt, op, comm_size);
        if(r != MPI_SUCCESS){
            fprintf(stderr, "Rank %d: Benchmark failed with error %d\n", rank, r);
            return 1;
        }

        if(i >= 0){
            samples[i] = ((MPI_Wtime() - start_time)*1000000.0);
        }
    }

    // Check correctness of results
    for(i = 0; i < dtsize*count; i++){
        if(recvbuf[i] != recvbuf_validation[i]){
            fprintf(stderr, "Rank %d: Validation failed at index %ld: %d != %d\n", rank, i, recvbuf[i], recvbuf_validation[i]);
            return 1;
        }
    }

    MPI_Gather(samples, iterations, MPI_DOUBLE, samples_all, iterations, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("#MessageSize ");
        for(size_t r = 0; r < (size_t) comm_size; r++){
            printf("Rank%ldTime(us) ", r);
        }
        printf("\n");
        double avg_iteration = 0.0;
        for(i = 0; i < iterations; i++){
            printf("%d ", count);
            double max_ranks = 0.0;
            for(size_t r = 0; r < (size_t) comm_size; r++){
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
