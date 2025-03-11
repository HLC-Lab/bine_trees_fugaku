#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <unistd.h>

#include "libswing_common.h"
#include "libswing_coll.h"
#include <climits>
#ifdef FUGAKU
#include "fugaku/swing_utofu.h"
#endif

static int is_mirroring_port(int port, uint dimensions_num){
    if(dimensions_num == 3){
        return port >= dimensions_num;
    }else if(dimensions_num == 2){
        if(port == 0 || port == 1){
            return 0;
        }else if(port == 2 || port == 3){
            return 1;
        }else if(port == 4 || port == 5){
            // TODO: On 2D torus we might have some unbalance (i.e., 4 ports for plain collectives and 2 for mirrored) The data we sent on plain collectives is 2x higher than what we send on mirrored. We should unbalance the 6 partitions of the vector accordingly.
            return 0;
        }
    }else if(dimensions_num == 1){
        return port % 2;
    }
    return 0;
}

static int get_distance_sign(size_t rank, size_t port, size_t dimensions_num){
    int multiplier = 1;
    if(is_odd(rank)){ // Invert sign if odd rank
        multiplier *= -1;
    }
    if(is_mirroring_port(port, dimensions_num)){ // Invert sign if mirrored collective
        multiplier *= -1;     
    }
    return multiplier;
}

int get_mirroring_port(int num_ports, uint dimensions_num){
    int p = -1;
    for(size_t p = 0; p < num_ports; p++){
        if(is_mirroring_port(p, dimensions_num)){
            return p;
        }
    }
    return p;
}

// TODO: This and some of those before to macros
int is_power_of_two(int x){
    return (x != 0) && ((x & (x - 1)) == 0);
}

static void get_peer_swing(int* coord_rank, size_t step, int* coord_peer, uint port, uint dimensions_num, uint* dimensions){
    size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    size_t current_d = port % dimensions_num;
    memcpy(coord_peer, coord_rank, sizeof(uint)*dimensions_num);
    memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);
    for(size_t i = 0; i < step; i++){
        // Move to the next dimension for the next step
        size_t d = current_d;              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < ceil_log2(dimensions[d])){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d = (current_d + 1) % dimensions_num;
            d = current_d;
        }while(next_step_per_dim[d] >= ceil_log2(dimensions[d])); // If we exhausted this dimension, move to the next one
    }
    int distance = rhos[next_step_per_dim[current_d]];
    distance *= get_distance_sign(coord_rank[current_d], port, dimensions_num);
    coord_peer[current_d] = mod(coord_peer[current_d] + distance, dimensions[current_d]);
}

static void get_peer_recdoub(int* coord_rank, size_t step, int* coord_peer, uint port, uint dimensions_num, uint* dimensions){
    size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    size_t current_d = port % dimensions_num;
    memcpy(coord_peer, coord_rank, sizeof(uint)*dimensions_num);
    memset(next_step_per_dim, 0, sizeof(size_t)*dimensions_num);
    for(size_t i = 0; i < step; i++){
        // Move to the next dimension for the next step
        size_t d = current_d;              
        // Increase the next step, unless we are done with this dimension
        if(next_step_per_dim[d] < ceil_log2(dimensions[d])){ 
            next_step_per_dim[d] += 1;
        }
        
        // Select next dimension
        do{ 
            current_d = (current_d + 1) % dimensions_num;
            d = current_d;
        }while(next_step_per_dim[d] >= ceil_log2(dimensions[d])); // If we exhausted this dimension, move to the next one
    }
    int distance = (coord_peer[current_d] ^ (1 << (next_step_per_dim[current_d]))) - coord_peer[current_d];
    if(is_mirroring_port(port, dimensions_num)){ // Invert sign if mirrored collective
        distance *= -1;     
    }
    coord_peer[current_d] = mod(coord_peer[current_d] + distance, dimensions[current_d]);
}


int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}


int ceil_log2(unsigned long long x){
    static const unsigned long long t[6] = {
      0xFFFFFFFF00000000ull,
      0x00000000FFFF0000ull,
      0x000000000000FF00ull,
      0x00000000000000F0ull,
      0x000000000000000Cull,
      0x0000000000000002ull
    };
  
    int y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 32;
    int i;
  
    for (i = 0; i < 6; i++) {
      int k = (((x & t[i]) == 0) ? 0 : j);
      y += k;
      x >>= k;
      j >>= 1;
    }
  
    return y;
}

/**
 * Computes a Swing binomial tree. 
 * @param coord_root (IN) The root of the tree
 * @param step (IN) The current step
 * @param port (IN) The port on which we are working on
 * @param algo (IN) The algorithm to use (SWING or RECDOUB)
 * @param dist_type (IN) The type of distance between nodes in the tree (increasing or decreasing)
 * @param scc (IN) The SwingCoordConverter object
 * @param reached_at_step (OUT) An array of size scc->size to store the step at which a node is reached
 * @param parent (OUT) An array of size scc->size to store the parent of a node
 */
static void build_tree(int* coord_root, size_t step, uint port, Algo algo, swing_distance_type_t dist_type, SwingCoordConverter* scc, uint32_t* reached_at_step, uint32_t* parent){
    for(size_t i = step; i < scc->num_steps; i++){
        int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        int real_step;
        if(dist_type == SWING_DISTANCE_DECREASING){
            real_step = scc->num_steps - 1 - i;
        }else{
            real_step = i;
        }
        get_peer_c(coord_root, real_step, peer_rank, port, scc->dimensions_num, scc->dimensions, algo);
        
        uint32_t rank = scc->getIdFromCoord(peer_rank);
        if(parent[rank] == UINT32_MAX || i < reached_at_step[rank]){
            parent[rank] = scc->getIdFromCoord(coord_root);
            reached_at_step[rank] = i;
        }
        build_tree(peer_rank, i + 1, port, algo, dist_type, scc, reached_at_step, parent);
    }
}

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
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
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

void reduce_local(const void* inbuf_a, const void* inbuf_b, void* outbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    if(datatype == MPI_INT32_T){
        const int32_t *in_a = (const int32_t *)inbuf_a;
        const int32_t *in_b = (const int32_t *)inbuf_b;
        int32_t *out = (int32_t *)outbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                out[i] = in_a[i] + in_b[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_INT){
        const int *in_a = (const int *)inbuf_a;
        const int *in_b = (const int *)inbuf_b;
        int *out = (int *)outbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                out[i] = in_a[i] + in_b[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_CHAR){
        const char *in_a = (const char *)inbuf_a;
        const char *in_b = (const char *)inbuf_b;
        char *out = (char *)outbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                out[i] = in_a[i] + in_b[i];
            }
        }else{
            fprintf(stderr, "Unknown reduction op\n");
            exit(EXIT_FAILURE);
        }
    }else if(datatype == MPI_FLOAT){
        const float *in_a = (const float *)inbuf_a;
        const float *in_b = (const float *)inbuf_b;
        float *out = (float *)outbuf;
        if(op == MPI_SUM){
//#pragma omp parallel for // Should be automatically parallelized by the compiler
            for (int i = 0; i < count; i++) {
                out[i] = in_a[i] + in_b[i];
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


/**
 * Remaps the ranks of a Swing binomial tree so to have subtrees with contiguous ranks.
 * @param coord_root (IN) The root of the tree
 * @param step (IN) The current step
 * @param port (IN) The port on which we are working on
 * @param algo (IN) The algorithm to use (SWING or RECDOUB)
 * @param dist_type (IN) The type of distance between nodes in the tree (increasing or decreasing)
 * @param scc (IN) The SwingCoordConverter object
 * @param next_rank (IN) The next rank to assign
 * @param parent (IN) An array of size scc->size to store the parent of a node
 * @param remapped_ranks (OUT) An array of size scc->size to store the remapped ranks
 * @param remapped_ranks_max (OUT) An array of size scc->size to store the maximum remapped rank in the subtree rooted at i
 */
static void remap_ranks(int* coord_root, size_t step, uint port, Algo algo, swing_distance_type_t dist_type, SwingCoordConverter* scc, uint* next_rank, const uint* parent, uint* remapped_ranks, uint* remapped_ranks_max){
    remapped_ranks_max[scc->getIdFromCoord(coord_root)] = *next_rank;
    for(size_t i = step; i < scc->num_steps; i++){
        int peer_rank[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        int real_step;
        if(dist_type == SWING_DISTANCE_DECREASING){
            real_step = scc->num_steps - 1 - i;
        }else{
            real_step = i;
        }
        get_peer_c(coord_root, real_step, peer_rank, port, scc->dimensions_num, scc->dimensions, algo);
        // I need to check if I am actually the parent of that peer.
        // When we have a number of nodes that is not a power of 2, we may have peers which are reached by
        // more than one node, so we must do this check.
        if(parent[scc->getIdFromCoord(peer_rank)] == scc->getIdFromCoord(coord_root)){
            remap_ranks(peer_rank, i + 1, port, algo, dist_type, scc, next_rank, parent, remapped_ranks, remapped_ranks_max);
        }
    }
    remapped_ranks[scc->getIdFromCoord(coord_root)] = (*next_rank);
    DPRINTF("Remapped rank %d to %d\n", scc->getIdFromCoord(coord_root), *next_rank);
    (*next_rank)--;
}

swing_tree_t get_tree(uint root, uint port, Algo algo, swing_distance_type_t dist_type, SwingCoordConverter* scc){
    int coord_root[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    scc->getCoordFromId(root, coord_root);
    swing_tree_t tree;
    uint* buffer = (uint*) malloc(sizeof(uint)*scc->size*4); // Do one single malloc rather than 4
    tree.parent = buffer;
    tree.reached_at_step = buffer + scc->size;
    tree.remapped_ranks = buffer + scc->size*2;
    tree.remapped_ranks_max = buffer + scc->size*3;
    for(size_t i = 0; i < scc->size; i++){
        tree.parent[i] = UINT32_MAX;
        tree.reached_at_step[i] = scc->num_steps;
    }
    
    // Compute the basic tree informations (parent and reached_at_step)
    build_tree(coord_root, 0, port, algo, dist_type, scc, tree.reached_at_step, tree.parent);
    tree.parent[root] = UINT32_MAX;
    tree.reached_at_step[root] = 0; // To avoid sending the step for myself at a wrong value

    // Now that we have a loopless tree, do a DFS to compute the remapped rank
    uint next_rank = scc->size - 1;
    remap_ranks(coord_root, 0, port, algo, dist_type, scc, &(next_rank), tree.parent, tree.remapped_ranks, tree.remapped_ranks_max);
    assert(next_rank == UINT32_MAX);
    return tree;
}

void destroy_tree(swing_tree_t* tree){
    free(tree->parent);
}

void get_peer_c(int* coord_rank, size_t step, int* coord_peer, uint port, uint dimensions_num, uint* dimensions, Algo algo){
    if(algo == ALGO_RECDOUB_L_UTOFU || algo == ALGO_RECDOUB_B_UTOFU){
        get_peer_recdoub(coord_rank, step, coord_peer, port, dimensions_num, dimensions);
    }else{
        get_peer_swing(coord_rank, step, coord_peer, port, dimensions_num, dimensions);
    }
}

void compute_peers(uint rank, int port, Algo algo, SwingCoordConverter* scc, uint* peers){
    bool terminated_dimensions_bitmap[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    uint8_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    memset(next_step_per_dim, 0, sizeof(uint8_t)*LIBSWING_MAX_SUPPORTED_DIMENSIONS);
    
    // Compute default directions
    int coord[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
    scc->retrieve_coord_mapping(rank, coord);
    for(size_t i = 0; i < scc->dimensions_num; i++){
        terminated_dimensions_bitmap[i] = false;            
    }
    
    int target_dim, relative_step, distance, last_dim = port - 1;
    uint terminated_dimensions = 0, o = 0;
    
    // Generate peers
    for(size_t i = 0; i < (uint) scc->num_steps; ){            
        if(scc->dimensions_num > 1){
            scc->retrieve_coord_mapping(rank, coord); // Regenerate rank coord
            o = 0;
            do{
                target_dim = (last_dim + 1 + o) % (scc->dimensions_num);            
                o++;
            }while(terminated_dimensions_bitmap[target_dim]);
            relative_step = next_step_per_dim[target_dim];
            ++next_step_per_dim[target_dim];
            last_dim = target_dim;
        }else{
            target_dim = 0;
            relative_step = i;
            coord[0] = rank;
        }

        if(algo == ALGO_RECDOUB_B_UTOFU || algo == ALGO_RECDOUB_L_UTOFU || algo == ALGO_RECDOUB_B || algo == ALGO_RECDOUB_L){
            distance = (coord[target_dim] ^ (1 << relative_step)) - coord[target_dim];
        }else{
            distance = rhos[relative_step];
            // Flip the sign for odd nodes
            if(is_odd(coord[target_dim])){distance *= -1;}
        }
        
        // Mirrored collectives
        if(is_mirroring_port(port, scc->dimensions_num)){distance *= -1;}

        if(relative_step < scc->num_steps_per_dim[target_dim]){
            coord[target_dim] = mod((coord[target_dim] + distance), scc->dimensions[target_dim]); // We need to use mod to avoid negative coordinates
            if(scc->dimensions_num > 1){
                peers[i] = scc->getIdFromCoord(coord);
            }else{
                peers[i] = coord[0];
            }
    
            i += 1;
        }else{
            terminated_dimensions_bitmap[target_dim] = true;
            terminated_dimensions++;                
        }        
    }        
}
