#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <unistd.h>
#include "libswing_common.h"

int ceil_log2(unsigned long long x);
int mod(int a, int b);
int is_power_of_two(int x);

void compute_peers(uint rank, int port, swing_algo_family_t algo, SwingCoordConverter* scc, uint* peers);

void get_peer_c(int* coord_rank, size_t step, uint port, swing_step_info_t* step_info, swing_algo_family_t algo, uint dimensions_num, uint* dimensions, int* coord_peer);
swing_step_info_t* compute_step_info(uint port, SwingCoordConverter* scc, uint dimensions_num, uint* dimensions);

int get_mirroring_port(int num_ports, uint dimensions_num);

/**
 * @brief Get a Swing binomial tree.
 * @param root (IN) The root of the tree
 * @param port (IN) The port on which we are working on
 * @param algo (IN) The algorithm to use (SWING or RECDOUB)
 * @param dist_type (IN) The type of distance between nodes in the tree (increasing or decreasing)
 * @param scc (IN) The SwingCoordConverter object
 * @return A swing_tree_t object
 */
swing_tree_t get_tree(uint root, uint port, swing_algo_family_t algo, swing_distance_type_t dist_type, SwingCoordConverter* scc);
swing_tree_t* get_tree_fast(uint root, uint port, swing_algo_family_t algo, swing_distance_type_t dist_type, SwingCoordConverter* scc);
void destroy_tree(swing_tree_t* tree);

void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op);
void reduce_local(const void* inbuf_a, const void* inbuf_b, void* outbuf, int count, MPI_Datatype datatype, MPI_Op op);
