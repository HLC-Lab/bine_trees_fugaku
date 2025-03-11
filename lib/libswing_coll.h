#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <unistd.h>

typedef struct {
    uint* parent; // For each node in the tree, its parent.
    uint* reached_at_step; // For each node in the tree, the step at which it is reached.
    uint* remapped_ranks; // The remapped rank so that each subtree contains contiguous remapped ranks    
    uint* remapped_ranks_max; // remapped_ranks_max[i] is the maximum remapped rank in the subtree rooted at i
    // We do not need to store the min because it is the remapped rank itself (the node is the last in the subtree to be numbered)
    //uint* remapped_ranks_min; // remapped_ranks_min[i] is the minimum remapped rank in the subtree rooted at i
} swing_tree_t;

int ceil_log2(unsigned long long x);
int mod(int a, int b);
int is_power_of_two(int x);

void compute_peers(uint rank, int port, Algo algo, SwingCoordConverter* scc, uint* peers);

void get_peer_c(int* coord_rank, size_t step, int* coord_peer, uint port, uint dimensions_num, uint* dimensions, Algo algo);

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
swing_tree_t get_tree(uint root, uint port, Algo algo, swing_distance_type_t dist_type, SwingCoordConverter* scc);
void destroy_tree(swing_tree_t* tree);

void reduce_local(const void* inbuf, void* inoutbuf, int count, MPI_Datatype datatype, MPI_Op op);
void reduce_local(const void* inbuf_a, const void* inbuf_b, void* outbuf, int count, MPI_Datatype datatype, MPI_Op op);
