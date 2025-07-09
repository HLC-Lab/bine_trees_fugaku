#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <unistd.h>
#include "libbine_common.h"

int ceil_log2(unsigned long long x);
int mod(int a, int b);
int is_power_of_two(int x);

void compute_peers(uint rank, int port, bine_algo_family_t algo, BineCoordConverter* scc, uint* peers);

void get_peer_c(int* coord_rank, size_t step, uint port, bine_step_info_t* step_info, bine_algo_family_t algo, uint dimensions_num, uint* dimensions, int* coord_peer);
bine_step_info_t* compute_step_info(uint port, BineCoordConverter* scc, uint dimensions_num, uint* dimensions);

int get_mirroring_port(int num_ports, uint dimensions_num);

/**
 * @brief Get a Bine binomial tree.
 * @param root (IN) The root of the tree
 * @param port (IN) The port on which we are working on
 * @param algo (IN) The algorithm to use (BINE or RECDOUB)
 * @param dist_type (IN) The type of distance between nodes in the tree (increasing or decreasing)
 * @param scc (IN) The BineCoordConverter object
 * @return A bine_tree_t object
 */
bine_tree_t get_tree(uint root, uint port, bine_algo_family_t algo, bine_distance_type_t dist_type, BineCoordConverter* scc);
bine_tree_t* get_tree_fast(uint root, uint port, bine_algo_family_t algo, bine_distance_type_t dist_type, BineCoordConverter* scc);
void destroy_tree(bine_tree_t* tree);

void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op);
void reduce_local(const void* inbuf_a, const void* inbuf_b, void* outbuf, int count, MPI_Datatype datatype, MPI_Op op);
