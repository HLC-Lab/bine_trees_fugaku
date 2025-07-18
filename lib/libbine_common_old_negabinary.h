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

typedef struct bine_utofu_comm_d bine_utofu_comm_descriptor;

#define LIBBINE_MAX_SUPPORTED_DIMENSIONS 3 // We support up to 3D torus
#define LIBBINE_MAX_SUPPORTED_PORTS (LIBBINE_MAX_SUPPORTED_DIMENSIONS*2)
#define LIBBINE_MAX_STEPS 20 // With this we are ok up to 2^20 nodes, add other terms to the following arrays if needed.
#define LIBBINE_MAX_COLLECTIVE_SEQUENCE 2

#define CACHE_LINE_SIZE 256

static int rhos[LIBBINE_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};
static int smallest_negabinary[LIBBINE_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBBINE_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

#define TAG_BINE_REDUCESCATTER (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*1)
#define TAG_BINE_ALLGATHER     (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*2)
#define TAG_BINE_ALLREDUCE     (0x7FFF - LIBBINE_MAX_SUPPORTED_PORTS*3)

typedef enum{
    BINE_REDUCE_SCATTER = 0,
    BINE_ALLGATHER,
    BINE_ALLREDUCE
}CollType;

typedef enum{
    ALGO_DEFAULT = 0,
    ALGO_BINE_L,
    ALGO_BINE_B,
    ALGO_BINE_B_COALESCE,
    ALGO_BINE_B_CONT,
    ALGO_BINE_B_UTOFU,
    ALGO_RING,
    ALGO_RECDOUB_L,
    ALGO_RECDOUB_B,
}Algo;

typedef struct{
    size_t offset;
    size_t count;
}BlockInfo;


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
    private:
        uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_virtual[LIBBINE_MAX_SUPPORTED_DIMENSIONS]; // Used when we shrink torus with non-power of 2 size // TODO: Rename as dimensions_lower_p2
        uint dimensions_num; 
        int* coordinates;
        int* coordinates_virtual;
        uint size;
    public:
        BineCoordConverter(uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num);
        
        ~BineCoordConverter();

        // Convert a rank id into a list of d-dimensional coordinates
        // Row-major order, i.e., row coordinates change the slowest 
        // (i.e., we first increase depth, than cols, then rows -- https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays) 
        // @param id (IN): the rank id
        // @param virt (IN): if true, the virtual coordinates are returned, otherwise the real ones
        // @param coord (OUT): the array where the coordinates are stored
        void getCoordFromId(int id, bool virt, int* coord);

        // Convert d-dimensional coordinates into a rank id).
        // Dimensions are (rows, cols, depth).
        // @param coords (IN): the array with the coordinates
        // @param virt (IN): if true, the virtual coordinates are considered, otherwise the real ones
        // @return the rank id
        int getIdFromCoord(int* coords, bool virt);

        // Gets the real or virtual (for non-p2) coordinates associated to a rank.
        // @param rank (IN): the rank
        // @param virt (IN): if true, the virtual coordinates are stored, otherwise the real ones
        // @param coord (OUT): the array where the coordinates are stored
        void retrieve_coord_mapping(uint rank, bool virt, int* coord);
};


typedef struct{
    size_t send_offset;
    size_t send_count;
    size_t recv_offset;
    size_t recv_count;
}ChunkParams;

class BineBitmapCalculator {
    private:
        volatile char padding1[CACHE_LINE_SIZE];
        uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_num; 
        uint port;
        uint size;
        size_t num_steps_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint num_steps;
        char** bitmap_send_merged;
        char** bitmap_recv_merged;

        uint* peers; // Peers per step, computed on the original, non-shrinked topology.
        int** reference_valid_distances[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint* num_valid_distances[LIBBINE_MAX_SUPPORTED_DIMENSIONS];    
        int rank;
        uint32_t remapped_rank;
        size_t min_block_s, min_block_r, max_block_s, max_block_r;
        BineCoordConverter scc;
        bool remap_blocks;
        int coord_mine[LIBBINE_MAX_SUPPORTED_DIMENSIONS];    
        uint32_t* block_step;

        ChunkParams chunk_params[LIBBINE_MAX_STEPS];
        bool valid_chunk_params[LIBBINE_MAX_STEPS];
        
        size_t next_step; 
        size_t current_d; // What's the current dimension we are sending in.
        size_t next_step_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS]; // For each port and for each dimension, what's the next step to execute in that dimension.
        volatile char padding2[CACHE_LINE_SIZE];

        // Computes an array of valid distances (considering a plain collective on an even node),
        // for a collective working on a given dimension and executing a given step.
        // The result is stored in the reference_valid_distances array.
        // @param d (IN): the dimension
        // @param step (IN): the step
        void compute_valid_distances(uint d, int step);

        // Gets the sign to be used for the distance calculation (-1 or +1), based on the rank and the port.
        // @param rank (IN): the rank
        // @param port (IN): the port
        // @return -1 or +1
        int get_distance_sign(size_t rank, size_t port);

        // Computes the peer of an arbitrary rank for an arbitrary step and port.
        // @param coord_rank (IN): the coordinates of the rank
        // @param step (IN): the step
        // @param coord_peer (OUT): the coordinates of the peer
        void get_peer(int* coord_rank, size_t step, int* coord_peer);

        // Finds the remapped rank of a given rank.
        // @param coord_rank (IN): It is always rank 0
        // @param step (IN): the step. It must be 0.
        // @param num_steps (IN): the number of steps
        // @param target_rank (IN): the rank to find
        // @param remap_rank (OUT): the remapped rank
        // @param found (OUT): if true, the rank was found
        void dfs(int* coord_rank, size_t step, size_t num_steps, int* target_rank, uint32_t* remap_rank, bool* found); 

        // Computes the step at which each block must be sent.
        // @param coord_rank (IN): the coordinates of the rank
        // @param starting_step (IN): the starting step. 
        // @param step (IN): the step. 
        // @param num_steps (IN): the number of steps
        // @param block_step (OUT): the array of steps at which each block must be sent
        void compute_block_step(int* coord_rank, size_t starting_step, size_t step, size_t num_steps, uint32_t* block_step);

        // Computes the bitmaps for the next step (assuming reduce_scatter)
        void compute_next_bitmaps();

        // Magic support functions (TODO: Document)
        void get_blocks_bitmaps_multid(size_t step, int* coord_peer, 
                                       char* bitmap_send_merged, char* bitmap_recv_merged, 
                                       int* coord_mine);
        void get_blocks_bitmaps_multid(size_t step, int* coord_peer, 
                                       char** bitmap_send, char** bitmap_recv, 
                                       char* bitmap_send_merged, char* bitmap_recv_merged, 
                                       int* coord_mine, size_t* next_step_per_dim = NULL, size_t* current_d = NULL);
    public:
        // Constructor
        // @param rank (IN): the rank
        // @param dimensions (IN): the dimensions of the torus
        // @param dimensions_num (IN): the number of dimensions
        // @param port (IN): the port the collective starts from
        // @param remap_blocks (IN): if true, the blocks are remapped to be contiguous
        BineBitmapCalculator(uint rank, uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, uint port, bool remap_blocks);

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

        ChunkParams get_chunk_params(uint step, CollType coll_type, const BlockInfo *const *const blocks_info);
};

class BineCommon {
    private:
        uint size;
        int rank;
        uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_virtual[LIBBINE_MAX_SUPPORTED_DIMENSIONS]; // Used when we shrink torus with non-power of 2 size // TODO: Rename as dimensions_lower_p2
        uint dimensions_num;
        Algo algo; 
        uint num_ports; 
        uint segment_size;
        bool all_p2_dimensions; // True if all the dimensions are power of 2
        size_t num_steps_per_dim[LIBBINE_MAX_SUPPORTED_DIMENSIONS];
        size_t num_steps;
        BineCoordConverter scc;
        BineBitmapCalculator *sbc[LIBBINE_MAX_SUPPORTED_PORTS]; 
        uint* virtual_peers[LIBBINE_MAX_SUPPORTED_PORTS]; // For latency optimal, one per port
        size_t size_virtual;
        size_t num_steps_virtual;
        size_t prealloc_size;
        char* prealloc_buf;

        // Sends the data from nodes outside of the power-of-two boundary to nodes within the boundary.
        // This is done one dimension at a time.
        // @param recvbuf (INOUT): the allreduce recvbuf
        // @param count (IN): the number of elements in the sendbuf
        // @param datatype (IN): the datatype of the elements in the sendbuf
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param tmpbuf (IN): a temporary buffer, as large as the sendbuf/recvbuf
        // @param idle (OUT): if 1, the rank is one of the "extra" ranks and is going to be idle
        // @param rank_virtual (OUT): the virtual rank
        // @return MPI_SUCCESS or an error code
        int shrink_non_power_of_two(void *recvbuf, int count, 
                                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, 
                                    char* tmpbuf, int* idle, int* rank_virtual);  

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

    public:
        // Constructor
        // @param comm (IN): the communicator
        // @param dimensions (IN): the dimensions of the torus
        // @param dimensions_num (IN): the number of dimensions
        // @param algo (IN): the algorithm to use
        // @param num_ports (IN): the number of ports
        // @param segment_size (IN): in allreduce and reducescatter, each send is segmented in blocks of at most this size 
        // @param prealloc_size (IN): the size of the preallocated buffer
        // @param prealloc_buf (IN): the preallocated buffer
        BineCommon(MPI_Comm comm, uint dimensions[LIBBINE_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, Algo algo, uint num_ports, uint segment_size, size_t prealloc_size, char* prealloc_buf);

        // Destructor
        ~BineCommon();

        uint get_num_ports(){return num_ports;}
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
        int MPI_Allreduce_lat_optimal(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        
        // TODO: Document
        int bine_coll_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type);
};


#endif // LIBBINE_COMMON_H
