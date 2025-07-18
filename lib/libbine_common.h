#ifndef LIBBINE_COMMON_H
#define LIBBINE_COMMON_H

#include <chrono>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#ifdef FUGAKU
#include <utofu.h>
#endif

typedef struct bine_utofu_comm_d bine_utofu_comm_descriptor;

#define LIBBINE_MAX_SUPPORTED_DIMENSIONS 3 // We support up to 3D torus
#define LIBBINE_MAX_SUPPORTED_PORTS (LIBBINE_MAX_SUPPORTED_DIMENSIONS*2)
#define LIBBINE_MAX_STEPS 40 // With this we are ok up to 2^20 nodes, add other terms to the following arrays if needed.
#define LIBBINE_MAX_COLLECTIVE_SEQUENCE 2
#define LIBBINE_TMPBUF_ALIGNMENT 256 // uTofu STag alignment

#define CACHE_LINE_SIZE 256

static int rhos[LIBBINE_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};
static int smallest_negabinary[LIBBINE_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBBINE_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

#define TAG_BINE_REDUCESCATTER (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*1)
#define TAG_BINE_ALLGATHER     (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*2)
#define TAG_BINE_ALLREDUCE     (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*3)
#define TAG_BINE_BCAST         (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*4)
#define TAG_BINE_ALLTOALL      (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*5)
#define TAG_BINE_SCATTER       (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*6)
#define TAG_BINE_GATHER        (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*7)
#define TAG_BINE_REDUCE        (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*8)

typedef struct {
    uint* parent; // For each node in the tree, its parent.
    uint* reached_at_step; // For each node in the tree, the step at which it is reached.
    uint* remapped_ranks; // The remapped rank so that each subtree contains contiguous remapped ranks    
    uint* remapped_ranks_max; // remapped_ranks_max[i] is the maximum remapped rank in the subtree rooted at i
    uint* subtree_roots; // subtree_roots[i] is the rank of the root of the subtree to which i belongs
    // We do not need to store the min because it is the remapped rank itself (the node is the last in the subtree to be numbered)
    //uint* remapped_ranks_min; // remapped_ranks_min[i] is the minimum remapped rank in the subtree rooted at i
} bine_tree_t;

typedef struct{
    uint d; // In which dimension is this global step performed
    uint step_in_d; // What's the relative step in this specific dimension
} bine_step_info_t;

typedef enum{
    BINE_REDUCE_SCATTER = 0,
    BINE_ALLGATHER,
    BINE_ALLREDUCE
}CollType;

/**
 * @brief Enum to specify whether a binomial tree is built with distance between
 * nodes increasing or decreasing at each step.
 */
typedef enum {
    BINE_DISTANCE_INCREASING = 0,
    BINE_DISTANCE_DECREASING = 1
} bine_distance_type_t;

typedef struct{
    size_t offset;
    size_t count;
}BlockInfo;

typedef enum {
    // Default
    BINE_ALGO_FAMILY_DEFAULT = 0,
    // Bine
    BINE_ALGO_FAMILY_BINE,
    // Recdoub
    BINE_ALGO_FAMILY_RECDOUB,
    // Bruck
    BINE_ALGO_FAMILY_BRUCK,
    // Ring/Bucket
    BINE_ALGO_FAMILY_RING,
} bine_algo_family_t;

typedef enum {
    BINE_ALGO_LAYER_MPI = 0,
    BINE_ALGO_LAYER_UTOFU,
} bine_algo_layer_t;

typedef enum {
    BINE_ALLREDUCE_ALGO_L = 0,
    BINE_ALLREDUCE_ALGO_B,
    BINE_ALLREDUCE_ALGO_REDUCE_BCAST,
    BINE_ALLREDUCE_ALGO_B_CONT,
    BINE_ALLREDUCE_ALGO_B_COALESCE,
} bine_allreduce_algo_t;

typedef enum {
    BINE_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE = 0, // Permutation at the end
    BINE_ALLGATHER_ALGO_VEC_DOUBLING_CONT_SEND, // Sendrecv at the beginning
    BINE_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS, // Block-by-block
    BINE_ALLGATHER_ALGO_GATHER_BCAST, // Gather + bcast
} bine_allgather_algo_t;

typedef enum {
    BINE_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE = 0, // Permutation at the beginning
    BINE_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_SEND, // Sendrecv at the end
    BINE_REDUCE_SCATTER_ALGO_VEC_HALVING_BLOCKS, // Block-by-block
    BINE_REDUCE_SCATTER_ALGO_REDUCE_SCATTER, // Reduce + scatter
} bine_reduce_scatter_algo_t;

typedef enum {
    BINE_BCAST_ALGO_BINOMIAL_TREE = 0, // Binomial tree
    BINE_BCAST_ALGO_BINOMIAL_TREE_TMPBUF, // Binomial tree with tmpbuf (avoids sending the rmtaddr of the recvbuf but needs a final memcpy)
    BINE_BCAST_ALGO_SCATTER_ALLGATHER, // Scatter + allgather
} bine_bcast_algo_t;

typedef enum {
    BINE_ALLTOALL_ALGO_LOG = 0,
} bine_alltoall_algo_t;

typedef enum {
    BINE_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE = 0, // Permutation at the beginning
    BINE_SCATTER_ALGO_BINOMIAL_TREE_CONT_SEND, // Sendrecv at the end
    BINE_SCATTER_ALGO_BINOMIAL_TREE_BLOCKS, // Block-by-block
} bine_scatter_algo_t;

typedef enum {
    BINE_GATHER_ALGO_BINOMIAL_TREE_CONT_PERMUTE = 0, // Permutation at the end
    BINE_GATHER_ALGO_BINOMIAL_TREE_CONT_SEND, // Sendrecv at the beginning
    BINE_GATHER_ALGO_BINOMIAL_TREE_BLOCKS, // Block-by-block
} bine_gather_algo_t;

typedef enum {
    BINE_REDUCE_ALGO_BINOMIAL_TREE = 0, // Binomial tree
    BINE_REDUCE_ALGO_REDUCE_SCATTER_GATHER, // Reduce-scatter + gather
} bine_reduce_algo_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_allreduce_algo_t algo;
    bine_distance_type_t distance_type;
} bine_allreduce_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_allgather_algo_t algo;
    bine_distance_type_t distance_type;
} bine_allgather_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_reduce_scatter_algo_t algo;
    bine_distance_type_t distance_type;
} bine_reduce_scatter_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_bcast_algo_t algo;
    bine_distance_type_t distance_type;
} bine_bcast_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_alltoall_algo_t algo;
    bine_distance_type_t distance_type;
} bine_alltoall_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_scatter_algo_t algo;
    bine_distance_type_t distance_type;
} bine_scatter_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_gather_algo_t algo;
    bine_distance_type_t distance_type;
} bine_gather_config_t;

typedef struct {
    bine_algo_family_t algo_family;
    bine_algo_layer_t algo_layer;
    bine_reduce_algo_t algo;
    bine_distance_type_t distance_type;
} bine_reduce_config_t;

typedef struct bine_comm_info_key {
    uint root;
    uint port;
    bine_algo_family_t algo;
    bine_distance_type_t dist_type;
    MPI_Comm comm;  

    bool operator==(const bine_comm_info_key &other) const
    { return (root == other.root &&
              port == other.port &&
              algo == other.algo &&
              dist_type == other.dist_type &&
              comm == other.comm);
    }
} bine_comm_info_key_t;

template <>
struct std::hash<bine_comm_info_key_t>
{
  std::size_t operator()(const bine_comm_info_key_t& k) const
  {
    using std::size_t;
    using std::hash;
    using std::string;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:

    //return (hash<uint>()(k.root) ^ 
    //        hash<uint>()(k.port) ^
    //        hash<uint>()(k.algo) ^
    //        hash<uint>()(k.dist_type) ^
    //        hash<void*>()((void*) k.comm));
    return ((k.root) ^ 
            (k.port) ^
            (k.algo) ^
            (k.dist_type) ^
            ((uint64_t) k.comm));    
  }
};

typedef struct {
    bine_tree_t tree;
} bine_comm_info_t;

extern std::unordered_map<bine_comm_info_key_t, bine_comm_info_t> comm_info;

typedef struct {
    uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
    uint dimensions_num;
    uint num_ports;
    uint segment_size;
    size_t prealloc_size;
    char* prealloc_buf;
    char utofu_add_ag;
    char use_threads;
    bine_allreduce_config_t allreduce_config;
    bine_allgather_config_t allgather_config;
    bine_reduce_scatter_config_t reduce_scatter_config;
    bine_bcast_config_t bcast_config;
    bine_alltoall_config_t alltoall_config;
    bine_scatter_config_t scatter_config;
    bine_gather_config_t gather_config;
    bine_reduce_config_t reduce_config;    
} bine_env_t;


//#define PERF_DEBUGGING 
//#define ACTIVE_WAIT

//#define DEBUG
//#define PROFILE

#ifdef DEBUG
#define DPRINTF(...) printf(__VA_ARGS__); fflush(stdout)
#else
#define DPRINTF(...) 
#endif

#define PROFILE_TIMER_TYPE steady_clock

class Timer {
public:
    Timer(std::string fname, std::string name);
    Timer(std::string name);
    ~Timer();
    void stop();
    void reset(std::string name);
private:
    std::chrono::time_point<std::chrono::PROFILE_TIMER_TYPE> _start_time_point;
    std::chrono::time_point<std::chrono::PROFILE_TIMER_TYPE> _end_time_point;
    std::string _name;
    bool _timer_stopped;
    std::stringstream _ss;
    std::string _fname;
};

int is_odd(int x);

class BineCoordConverter {
    public:
        uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_num; 
        int* coordinates;
        uint size;
        uint num_steps_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint num_steps;
    
        BineCoordConverter(uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num);
        
        ~BineCoordConverter();

        // Convert a rank id into a list of d-dimensional coordinates
        // Row-major order, i.e., row coordinates change the slowest 
        // (i.e., we first increase depth, than cols, then rows -- https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays) 
        // @param id (IN): the rank id
        // @param coord (OUT): the array where the coordinates are stored
        void getCoordFromId(int id, int* coord);

        // Convert d-dimensional coordinates into a rank id).
        // Dimensions are (rows, cols, depth).
        // @param coords (IN): the array with the coordinates
        // @return the rank id
        int getIdFromCoord(int* coords);

        // Gets the real or virtual (for non-p2) coordinates associated to a rank.
        // @param rank (IN): the rank
        // @param coord (OUT): the array where the coordinates are stored
        void retrieve_coord_mapping(uint rank, int* coord);
};


typedef struct{
    size_t send_offset; // In bytes
    size_t send_count; // In number of elements
    size_t recv_offset; // In bytes
    size_t recv_count; // In number of elements
}ChunkParams;

class BineBitmapCalculator {
    private:
        volatile char padding1[CACHE_LINE_SIZE];
        uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_num; 
        uint port;
        BlockInfo** blocks_info;
        uint size;
        size_t num_steps_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint num_steps;
        bine_step_info_t* step_info;

        // I use either bitmaps or chunk params
        // depending on whether I can send contiguous blocks or not
        char** bitmap_send_merged;
        char** bitmap_recv_merged;
        ChunkParams* chunk_params_per_step; // For each step, the chunk params 

        uint* peers; // Peers per step, computed on the original, non-shrinked topology.
        int rank;
        uint32_t remapped_rank;
        size_t min_block_s, min_block_r, max_block_s, max_block_r;
        BineCoordConverter scc;
        bool remap_blocks;
        int coord_mine[LIBBINE_MAX_SUPPORTED_DIMENSIONS];    
        uint32_t* block_step;
        bine_algo_family_t algo;
        size_t next_step; 
        volatile char padding2[CACHE_LINE_SIZE];

        // Computes an array of valid distances (considering a plain collective on an even node),
        // for a collective working on a given dimension and executing a given step.
        // The result is stored in the reference_valid_distances array.
        // @param d (IN): the dimension
        // @param step (IN): the step
        void compute_valid_distances(uint d, int step);

        // Computes the step at which each block must be sent.
        // @param coord_rank (IN): the coordinates of the rank
        // @param starting_step (IN): the starting step. 
        // @param step (IN): the step. 
        // @param num_steps (IN): the number of steps
        // @param block_step (OUT): the array of steps at which each block must be sent
        void compute_block_step(int* coord_rank, size_t starting_step, size_t step, size_t num_steps, uint32_t* block_step);

        // Computes the bitmaps for the next step (assuming reduce_scatter)
        void compute_next_bitmaps();

    public:
        // Constructor
        // @param rank (IN): the rank
        // @param dimensions (IN): the dimensions of the torus
        // @param dimensions_num (IN): the number of dimensions
        // @param port (IN): the port the collective starts from
        // @param blocks_info (IN): the blocks info
        // @param remap_blocks (IN): if true, the blocks are remapped to be contiguous
        BineBitmapCalculator(uint rank, uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, uint port, BlockInfo** blocks_info, bool remap_blocks, bine_algo_family_t algo);

        // Destructor
        ~BineBitmapCalculator();

        // Computes the peer of this node, considering it started for a given port and it is at a specific step.
        // @param port (IN): the port
        // @param step (IN): the step
        // @param coll_type (IN): the collective type
        // @return the peer
        uint get_peer(uint step, CollType coll_type);

        // Computes the bitmaps to be used at a given step.
        // You can avoid calling this explicitly, as it is called by block_must_be_sent and block_must_be_recvd.
        // You can call it explicitly in case you want to separate the bitmap computation from the check of the blocks 
        // (e.g., for overlapping bitmap computation with communication).
        // @param step (IN): the step
        // @param coll_type (IN): the collective type
        void compute_bitmaps(uint step, CollType coll_type);

        // Chekcs if a specific block must be sent, based on the port this collective started from and the current step.
        // @param step (IN): the step
        // @param coll_type (IN): the collective type
        // @param block_id (IN): the block id
        // @return true if the block must be sent, false otherwise
        bool block_must_be_sent(uint step, CollType coll_type, uint block_id);

        // Chekcs if a specific block must be received, based on the port this collective started from and the current step.
        // @param step (IN): the step
        // @param coll_type (IN): the collective type
        // @param block_id (IN): the block id
        // @return true if the block must be received, false otherwise  
        bool block_must_be_recvd(uint step, CollType coll_type, uint block_id);

        // Gets the chunk params for a given step.
        // @param step (IN): the step
        // @param coll_type (IN): the collective type
        // @param chunk_params (OUT): the chunk params
        void get_chunk_params(uint step, CollType coll_type, ChunkParams* chunk_params);

        uint* get_peers(){return peers;}
};

class BineCommon {
    private:
        bine_env_t env;
        uint size;
        int rank;        
        bool all_p2_dimensions; // True if all the dimensions are power of 2
        size_t num_steps_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        size_t num_steps;
        uint dimensions_virtual[LIBBINE_MAX_SUPPORTED_DIMENSIONS]; // Used when we shrink torus with non-power of 2 size // TODO: Rename as dimensions_lower_p2
        uint* virtual_peers[LIBBINE_MAX_SUPPORTED_PORTS]; // For latency optimal, one per port
        size_t num_steps_virtual;
        BineCoordConverter* scc_real;
        BineCoordConverter* scc_virtual;
        BineBitmapCalculator *sbc[LIBBINE_MAX_SUPPORTED_PORTS]; 
#ifdef FUGAKU
        bine_utofu_comm_descriptor* utofu_descriptor;
        utofu_vcq_id_t* vcq_ids[LIBBINE_MAX_SUPPORTED_PORTS];
        utofu_stadd_t lcl_temp_stadd[LIBBINE_MAX_SUPPORTED_PORTS];
        utofu_stadd_t* temp_buffers[LIBBINE_MAX_SUPPORTED_PORTS]; // temp_buffers[p][r] contains the address of the buffer for the rank r on port p
#endif

        // Sends the data from nodes outside of the power-of-two boundary to nodes within the boundary.
        // This is done one dimension at a time.
        // @param sendbuf (INOUT): the allreduce sendbuf
        // @param recvbuf (IN): the allreduce recvbuf
        // @param tempbuf (IN): the allreduce tempbuf
        // @param count (IN): the number of elements in the sendbuf
        // @param datatype (IN): the datatype of the elements in the sendbuf
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator        
        // @param idle (OUT): if 1, the rank is one of the "extra" ranks and is going to be idle
        // @param rank_virtual (OUT): the virtual rank
        // @param first_copy_done (OUT): if 1, the copy from sendbuf to recvbuf has been already done
        // @return MPI_SUCCESS or an error code
        int shrink_non_power_of_two(const void *sendbuf, void* recvbuf, void* tempbuf, int count, 
                                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, 
                                    int* idle, int* rank_virtual,
                                    int* first_copy_done);  

        // Enlarges the data from nodes within the power-of-two boundary to nodes outside the boundary.
        // This is done one dimension at a time.
        // @param recvbuf (INOUT): the allreduce recvbuf
        // @param count (IN): the number of elements in the recvbuf
        // @param datatype (IN): the datatype of the elements in the recvbuf
        // @param comm (IN): the communicator
        // @return MPI_SUCCESS or an error code
        int enlarge_non_power_of_two(void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm);


        // Bandwidth-optimal Bine collective
        // @param buf (IN): the sendbuf
        // @param rbuf (OUT): the recvbuf
        // @param blocks_info (IN): the blocks info
        // @param step (IN): the step
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param sendtype (IN): the send datatype
        // @param recvtype (IN): the recv datatype
        // @param coll_type (IN): the collective type
        // @param bitmap_send (IN): the bitmap of the send
        // @param bitmap_recv (IN): the bitmap of the recv
        int bine_coll_step_b(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                             
                              MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                              CollType coll_type);

                                      // Bandwidth-optimal Bine collective with contiguous blocks
        // @param buf (IN): the sendbuf
        // @param rbuf (OUT): the recvbuf
        // @param blocks_info (IN): the blocks info
        // @param step (IN): the step
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param sendtype (IN): the send datatype
        // @param recvtype (IN): the recv datatype
        // @param coll_type (IN): the collective type
        // @param bitmap_send (IN): the bitmap of the send
        // @param bitmap_recv (IN): the bitmap of the recv
        int bine_coll_step_coalesce(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
            MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
            CollType coll_type);

        // Bandwidth-optimal Bine collective with contiguous blocks
        // @param buf (IN): the sendbuf
        // @param rbuf (OUT): the recvbuf
        // @param blocks_info (IN): the blocks info
        // @param step (IN): the step
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param sendtype (IN): the send datatype
        // @param recvtype (IN): the recv datatype
        // @param coll_type (IN): the collective type
        // @param bitmap_send (IN): the bitmap of the send
        // @param bitmap_recv (IN): the bitmap of the recv
        int bine_coll_step_cont(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
                                 MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                 CollType coll_type);


        // Wrapper for step_b/step_cont
        int bine_coll_step(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
                           MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type);

        // Bandwidth-optimal Bine collective with UTOFU
        // @param port (IN): the port
        // @param utofu_descriptor (IN): the UTOFU descriptor
        // @param sbuf (IN): the user_sbuf
        // @param buf (IN): the buf
        // @param rbuf (OUT): the rbuf
        // @param rbuf_size (IN): the size of the rbuf
        // @param blocks_info (IN): the blocks info
        // @param step (IN): the step
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param sendtype (IN): the send datatype
        // @param recvtype (IN): the recv datatype
        // @param coll_type (IN): the collective type
        int bine_coll_step_utofu(size_t port, bine_utofu_comm_descriptor* utofu_descriptor, const void* sbuf, void *buf, void* rbuf, size_t rbuf_size, const BlockInfo *const *const, size_t step, 
                                  MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                  CollType coll_type, bool is_first_coll);

        
        int bine_coll_l_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int bine_coll_l_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int bine_coll_l_utofu_omp(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int bine_coll_l_utofu_noomp(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
    public:
        // Constructor
        // @param comm (IN): the communicator
        // @param env (IN): the environment
        BineCommon(MPI_Comm comm, bine_env_t env);

        // Destructor
        ~BineCommon();

        uint get_num_ports(){return env.num_ports;}
        uint get_size(){return size;}
        uint get_rank(){return rank;}

        // Runs a latency-optimal allreduce
        // @param sendbuf (IN): the allreduce sendbuf
        // @param recvbuf (OUT): the allreduce recvbuf
        // @param count (IN): the number of elements in the sendbuf
        // @param datatype (IN): the datatype of the elements in the sendbuf
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @return MPI_SUCCESS or an error code
        int bine_coll_l(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        
        // TODO: Document
        int bine_coll_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type);
        int bine_coll_b_cont_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type);   


        /*******************************/
        /************ ALLREDUCE ************/
        /*******************************/
        void ringRedScatAG(char* data, int count, int nProc, int rank, int recvfrom, int sendto, int redscat, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, char* buffer, int port, size_t data_offset, size_t buffer_offset, size_t real_size);
        int bucket_allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);       

        /*******************************/
        /************ BCAST ************/
        /*******************************/                
        int bine_bcast_l(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_l_omp(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_l_noomp(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_l_tmpbuf(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_l_tmpbuf_omp(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_l_tmpbuf_noomp(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_b(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_l_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_b_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_scatter_allgather(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
        int bine_bcast_scatter_allgather_mpi(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

        /**********************************/
        /************ ALLTOALL ************/
        /**********************************/        
        int bruck_alltoall(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm);
        int bine_alltoall_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm);
        int bine_alltoall_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm);

        /*********************************/
        /************ SCATTER ************/
        /*********************************/                
        int bine_scatter_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_scatter_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm);

        /********************************/
        /************ GATHER ************/
        /********************************/        
        int bine_gather_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_gather_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, BlockInfo** blocks_info, MPI_Comm comm);
        
        /********************************/
        /************ REDUCE ************/
        /********************************/        
        int bine_reduce_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
        int bine_reduce_utofu_omp(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
        int bine_reduce_utofu_noomp(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
        int bine_reduce_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
        int bine_reduce_redscat_gather_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, BlockInfo** blocks_info);
        int bine_reduce_redscat_gather_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
        

        /***********************************/
        /************ ALLGATHER ************/
        /***********************************/    
        /****** CONT_SEND ******/
        int bine_allgather_send_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
        int bine_allgather_send_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

        /****** CONT_PERMUTE ******/
        int bine_allgather_utofu_contiguous(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_allgather_utofu_contiguous_threads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_allgather_utofu_contiguous_nothreads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_allgather_mpi_contiguous(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);

        /****** BLOCKS ******/
        int bine_allgather_blocks_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_allgather_blocks_utofu_threads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_allgather_blocks_utofu_nothreads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_allgather_blocks_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

        /****** BUCKET ******/
        int bucket_allgather(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

           
        /****************************************/
        /************ REDUCE-SCATTER ************/
        /****************************************/
        /****** CONT_PERMUTE ******/
        int bine_reduce_scatter_utofu_contiguous(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_reduce_scatter_mpi_contiguous(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm);

        /****** BLOCKS ******/
        int bine_reduce_scatter_utofu_blocks(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_reduce_scatter_mpi_blocks(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm);
        
        int bine_reduce_scatter_utofu(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm);    
        int bine_reduce_scatter_mpi(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm);
        int bine_reduce_scatter_mpi_new(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

        /***** BUCKET */
        int bucket_reduce_scatter(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

        // TODO: Add allreduce_l as reduce+bcast
        // TODO: add bcast as scatter + allgather
        // TODO: add reduce-scatter as reduce + scatter

};


#endif // LIBBINE_COMMON_H
