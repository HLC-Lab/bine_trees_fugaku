#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cinttypes>
#include "../lib/libbine_coll.h"
#include "../lib/libbine_common.h"
#include "../lib/fugaku/bine_utofu.h"
#include <mpi-ext.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char* env_str;
    env_str = getenv("LIBBINE_DIMENSIONS");
    uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    uint dimensions_num = 0;
    if(env_str){
        char* copy = (char*) malloc(sizeof(char)*(strlen(env_str) + 1));
        strcpy(copy, env_str);
        const char *delim = "x";
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
    }
    assert(dimensions_num <= LIBBINE_MAX_SUPPORTED_DIMENSIONS);

    int dimensions_num_fj;
    int dimensions_fj[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    FJMPI_Topology_get_dimension(&dimensions_num_fj);
    FJMPI_Topology_get_shape(&(dimensions_fj[0]), &(dimensions_fj[1]), &(dimensions_fj[2]));

    printf("dimensions_num (me): %d (fj): %d \n", dimensions_num, dimensions_num_fj);
    for(int i = 0; i < dimensions_num; ++i){
        printf("dimensions[%d] (me): %d (fj): %d \n", i, dimensions[i], dimensions_fj[i]);
    }

    BineCoordConverter* scc = new BineCoordConverter(dimensions, dimensions_num);
    int coord[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    int coord_fj[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    scc->getCoordFromId(rank, coord);

    FJMPI_Topology_get_coords(MPI_COMM_WORLD, rank, FJMPI_LOGICAL, dimensions_num_fj, coord_fj);
    printf("rank: %d, coord (me): ", rank);
    for(int i = 0; i < dimensions_num; ++i){
        printf("%d ", coord[i]);
    }
    printf(" (fj): ");
    for(int i = 0; i < dimensions_num_fj; ++i){
        printf("%d ", coord_fj[i]);
    }
    printf("\n");


    MPI_Finalize();
    return 0;
}
