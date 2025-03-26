#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cinttypes>
#include "../lib/libswing_coll.h"
#include "../lib/libswing_common.h"

int main(int argc, char** argv){
    if(argc != 3){
        fprintf(stderr, "Usage: %s <num_ranks> <iter>\n", argv[0]);
        return 1;
    }
    int num_ranks = atoi(argv[1]);
    int iter = atoi(argv[2]);

    SwingCoordConverter* scc = new SwingCoordConverter((uint*) &num_ranks, 1);
    int accumul_parent = 0, accumul_reached = 0, accumul_remap = 0, accumul_remap_max = 0, accumul_peers = 0;
    uint64_t elapsed_total = 0, elapsed_total_compute = 0;
    uint* peers = (uint*) malloc(num_ranks * sizeof(uint));

    for(size_t i = 0; i < iter; i++){
        // Get current timestamp
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        swing_tree_t t = get_tree(i % num_ranks, 0, SWING_ALGO_FAMILY_SWING, SWING_DISTANCE_DECREASING, scc);
        clock_gettime(CLOCK_MONOTONIC, &end);
        uint64_t elapsed = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        elapsed_total += elapsed;      

        clock_gettime(CLOCK_MONOTONIC, &start);
        compute_peers((i % num_ranks), 0, SWING_ALGO_FAMILY_SWING, scc, peers);
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec);
        elapsed_total_compute += elapsed;      

        accumul_parent += t.parent[i % num_ranks];
        accumul_reached += t.reached_at_step[i % num_ranks];
        accumul_remap += t.remapped_ranks[i % num_ranks];
        accumul_remap_max += t.remapped_ranks_max[i % num_ranks];
        accumul_peers += peers[i % num_ranks];
        destroy_tree(&t);
    }
    free(peers);
    printf("Elapsed time (get_tree): %" PRIu64 " ns\n", elapsed_total / iter);
    printf("Elapsed time (compute_peers): %" PRIu64 " ns\n", elapsed_total_compute / iter);
    
    printf("Average parent: %f\n", (float) accumul_parent / iter);
    printf("Average reached: %f\n", (float) accumul_reached / iter);
    printf("Average remap: %f\n", (float) accumul_remap / iter);
    printf("Average remap_max: %f\n", (float) accumul_remap_max / iter);
    printf("Average peers: %f\n", (float) accumul_peers / iter);
    return 0;
}
