// Computes reduce_local vs. my reduce_local performance
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <sys/time.h>
#include <cstring>
#include <mpi.h>

void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    if(datatype == MPI_INT32_T){
        const int32_t *in = (const int32_t *)inbuf;
        int32_t *inout = (int32_t *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            //fprintf(stderr, "Unknown reduction op\n");
            //exit(EXIT_FAILURE);
            // We assume this is the custom VOID operator
        }
    }else if(datatype == MPI_INT){
        const int *in = (const int *)inbuf;
        int *inout = (int *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_CHAR){
        const char *in = (const char *)inbuf;
        char *inout = (char *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_FLOAT){
        const float *in = (const float *)inbuf;
        float *inout = (float *)inoutbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                inout[i] += in[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else{
        fprintf(stderr, "Unknown reduction datatype\n");
        exit(EXIT_FAILURE);
    }
}

// Use C buffers rather than vectors
int main(int argc, char** argv){
    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }
    int size = atoi(argv[1]);
    MPI_Init(NULL, NULL);
    // Allocate two vectors
    std::vector<int> v1(size);
    std::vector<int> v2(size);
    // Now compare the performance of reduce_local vs MPI_Reduce_local
    // Initialize the vectors
    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 0);

    double time_mine = 0, time_mpi = 0;

    // Now call reduce_local
    struct timeval start, end;

    size_t iters = 10;
    for(size_t i = 0; i < iters; i++){
      // gettimeofday(&start, NULL);
      // reduce_local(v1.data(), v2.data(), size, MPI_INT, MPI_SUM);
      // gettimeofday(&end, NULL);
      // time_mine += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        

        // Now call MPI_Reduce_local
        gettimeofday(&start, NULL);
        MPI_Reduce_local(v1.data(), v2.data(), size, MPI_INT, MPI_SUM);
        gettimeofday(&end, NULL);
        time_mpi += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
    }  
    std::cout << "Time taken by my reduce_local: " << time_mine / iters << " us" << std::endl;
    std::cout << "Time taken by MPI_Reduce_local: " << time_mpi / iters << " us" << std::endl;
    
    MPI_Finalize();
    return 0;
}
