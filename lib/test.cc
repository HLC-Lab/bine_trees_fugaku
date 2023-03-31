#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

int main(int argc, char** argv){
    int r, rank;
#ifdef PROFILE
    int count = 65536;
    int warmup = 0;
    int iterations = 100000;
#else
    int count = 65536;
    int warmup = 5;
    int iterations = 1;
#endif
    float* sendbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf = (float*) malloc(sizeof(float)*count);
    float* recvbuf_v = (float*) malloc(sizeof(float)*count);

    // Fill sendbuf with random data
    for(size_t i = 0; i < count; i++){
        sendbuf[i] = rand() % 1024;
    }

    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
#ifndef PROFILE
    // Run the original MPI allreduce
    setenv("LIBSWING_FORCE_ENV_RELOAD", "1", 1);
    setenv("LIBSWING_ALGO", "DEFAULT", 1);
    setenv("LIBSWING_DISABLE_REDUCESCATTER", "1", 1);
    setenv("LIBSWING_DISABLE_ALLGATHERV", "1", 1);
    setenv("LIBSWING_DISABLE_ALLREDUCE", "1", 1);
    for(int i = warmup; i >= 0; i--){
        r = MPI_Allreduce(sendbuf, recvbuf_v, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    unsetenv("LIBSWING_FORCE_ENV_RELOAD");

    start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < iterations; i++){
        r = MPI_Allreduce(sendbuf, recvbuf_v, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    end = std::chrono::high_resolution_clock::now();
    if(r != MPI_SUCCESS){
        fprintf(stderr, "Allreduce failed with error %d\n", r);
        return r;
    }else{
        std::cout << "Default terminated in: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / iterations << "ns\n";
    }
#endif

#ifdef PROFILE
    const char* algos[1] = {"SWING"};
#else
    const char* algos[2] = {"SWING", "RING"};
#endif
    for(size_t algo = 0; algo < sizeof(algos)/sizeof(char*); algo++){
        // Run first swing allreduce
        setenv("LIBSWING_FORCE_ENV_RELOAD", "1", 1);
        setenv("LIBSWING_ALGO", algos[algo], 1);
        setenv("LIBSWING_DISABLE_REDUCESCATTER", "0", 1);
        setenv("LIBSWING_DISABLE_ALLGATHERV", "0", 1);
        setenv("LIBSWING_DISABLE_ALLREDUCE", "0", 1);
        for(int i = warmup; i >= 0; i--){
            r = MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        unsetenv("LIBSWING_FORCE_ENV_RELOAD");
        start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < iterations; i++){
            r = MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        end = std::chrono::high_resolution_clock::now();
        if(r != MPI_SUCCESS){
            fprintf(stderr, "Allreduce failed with error %d\n", r);
            return r;
        }else{
            std::cout << algos[algo] << " terminated in: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / iterations << "ns\n";
        }

#ifndef PROFILE
        // Then validate
        bool valid = true;
        for(size_t i = 0; i < count; i++){
            if(recvbuf[i] != recvbuf_v[i]){
                fprintf(stderr, "[%d] Validation failed at index %d (%f but should be %f)\n", rank, (int) i, recvbuf[i], recvbuf_v[i]);
                valid = false;
                return 1;
            }
        }

        if(valid){
            printf("Validation succeeded.\n");
        }
#endif
    }
    // Fini
    MPI_Finalize();    
    return 0;
}