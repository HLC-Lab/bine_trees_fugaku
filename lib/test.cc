#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <string.h>

// Only one argument is required: the name of the collective (camelcase as in MPI)
int main(int argc, char** argv){
    int r, rank;
#ifdef PROFILE
    int count = 65536;
    int iterations = 100000;
#else
    int count = 131072;
    int iterations = 4;
#endif
    float* sendbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf_v = (float*) malloc(sizeof(float)*count);
    const char* collective = argv[1];

    // Fill sendbuf with random data
    for(int i = 0; i < count; i++){
        sendbuf[i] = rand() % 1024;
    }

    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#ifndef PROFILE
    // Run the original collective
    if(!strcmp(collective, "MPI_Allreduce")){
        r = PMPI_Allreduce(sendbuf, recvbuf_v, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }else if(!strcmp(collective, "MPI_Reduce_scatter")){
        int* recvcounts = (int*) malloc(sizeof(int)*size);
        size_t partition_size = count / size;
        size_t remaining = count % size;                
        for(size_t i = 0; i < size; i++){
            size_t count_i = partition_size + (i < remaining ? 1 : 0);
            recvcounts[i] = count_i;
        }
        r = PMPI_Reduce_scatter(sendbuf, recvbuf_v, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        free(recvcounts);
    }
    if(r != MPI_SUCCESS){
        fprintf(stderr, "Collective failed with error %d\n", r);
        return r;
    }
#endif

    size_t num_algos = 0;
    const char** algos;

    if(!strcmp(collective, "MPI_Allreduce")){    
#ifdef PROFILE
        num_algos = 1;
        algos = (const char**) malloc(num_algos*sizeof(char*));
        algos[0] = "SWING_B_CONT";
#else
        num_algos = 4;
        algos = (const char**) malloc(num_algos*sizeof(char*));
        algos[0] = "SWING_L";
        algos[1] = "SWING_B_CONT";
        algos[2] = "SWING_B";
        algos[3] = "SWING_B_COALESCE";
#endif
    }else if(!strcmp(collective, "MPI_Reduce_scatter")){
        num_algos = 1;
        algos = (const char**) malloc(num_algos*sizeof(char*));
        algos[0] = "SWING_B";
    }

    for(size_t algo = 0; algo < num_algos; algo++){
        std::cout << "Running " << algos[algo] << std::endl;
        memset(recvbuf, 0, sizeof(float)*count);
        // Run first swing allreduce
        setenv("LIBSWING_FORCE_ENV_RELOAD", "1", 1);
        setenv("LIBSWING_ALGO", algos[algo], 1);
        for(int i = 0; i < iterations; i++){
            if(!strcmp(collective, "MPI_Allreduce")){
                r = MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }else if(!strcmp(collective, "MPI_Reduce_scatter")){
                int* recvcounts = (int*) malloc(sizeof(int)*size);
                size_t partition_size = count / size;
                size_t remaining = count % size;                
                for(size_t i = 0; i < size; i++){
                    size_t count_i = partition_size + (i < remaining ? 1 : 0);
                    recvcounts[i] = count_i;
                }
                r = MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                free(recvcounts);
            }
        }
        if(r != MPI_SUCCESS){
            fprintf(stderr, "Collective failed with error %d\n", r);
            return r;
        }

#ifndef PROFILE
        // Then validate
        bool valid = true;
        for(int i = 0; i < count; i++){
            if(recvbuf[i] != recvbuf_v[i]){
                fprintf(stderr, "[%d][%s] Validation failed at index %d (%f but should be %f)\n", rank, algos[algo], (int) i, recvbuf[i], recvbuf_v[i]);
                valid = false;
                return 1;
            }
        }

        if(valid){
            printf("%s Validation succeeded.\n", algos[algo]);
        }
#endif
    }
    // Fini
    MPI_Finalize();    
    return 0;
}
