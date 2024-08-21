#ifndef LIBSWING_COMMON_H
#define LIBSWING_COMMON_H

#include <chrono>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <mpi.h>

typedef struct swing_utofu_comm_d swing_utofu_comm_descriptor;

#define LIBSWING_MAX_SUPPORTED_DIMENSIONS 3 // We support up to 3D torus
#define LIBSWING_MAX_SUPPORTED_PORTS (LIBSWING_MAX_SUPPORTED_DIMENSIONS*2)
#define LIBSWING_MAX_STEPS 20 // With this we are ok up to 2^20 nodes, add other terms to the following arrays if needed.
#define LIBSWING_MAX_COLLECTIVE_SEQUENCE 2

static int rhos[LIBSWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};
static int smallest_negabinary[LIBSWING_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42, -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[LIBSWING_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85, 341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

#define TAG_SWING_REDUCESCATTER (0x7FFF - LIBSWING_MAX_SUPPORTED_PORTS*1)
#define TAG_SWING_ALLGATHER     (0x7FFF - LIBSWING_MAX_SUPPORTED_PORTS*2)
#define TAG_SWING_ALLREDUCE     (0x7FFF - LIBSWING_MAX_SUPPORTED_PORTS*3)


typedef enum{
    SWING_NULL = 0, // No collective, useful to force some bookeping data computation
    SWING_REDUCE_SCATTER,
    SWING_ALLGATHER,
    SWING_ALLREDUCE
}CollType;

typedef enum{
    ALGO_DEFAULT = 0,
    ALGO_SWING_L,
    ALGO_SWING_B,
    ALGO_SWING_B_COALESCE,
    ALGO_SWING_B_CONT,
    ALGO_SWING_B_UTOFU,
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

#define DEBUG
//#define PROFILE

#ifdef DEBUG
#define DPRINTF(...) printf(__VA_ARGS__); fflush(stdout)
#else
#define DPRINTF(...) 
#endif

class Timer {
public:
    Timer();
    ~Timer();
    void stop(std::string name);
    void reset(std::string name);
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
};

int is_odd(int x);

class SwingCoordConverter {
    private:
        uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_virtual[LIBSWING_MAX_SUPPORTED_DIMENSIONS]; // Used when we shrink torus with non-power of 2 size // TODO: Rename as dimensions_lower_p2
        uint dimensions_num; 
        int* coordinates;
        int* coordinates_virtual;
        uint size;
    public:
        SwingCoordConverter(uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num);
        
        ~SwingCoordConverter();

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

class SwingBitmapCalculator {
    private:
        uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_num; 
        uint num_ports;
        uint size;
        size_t num_steps_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        uint num_steps;
        char** bitmap_send_merged[LIBSWING_MAX_SUPPORTED_PORTS];
        char** bitmap_recv_merged[LIBSWING_MAX_SUPPORTED_PORTS];

        uint** peers_per_port; // Peers per port and step, computed on the original, non-shrinked topology.
        int** reference_valid_distances[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        uint* num_valid_distances[LIBSWING_MAX_SUPPORTED_DIMENSIONS];    
        uint* remapping_per_port[LIBSWING_MAX_SUPPORTED_PORTS];
        SwingCoordConverter scc;
        bool remap_blocks;
        int coord_mine[LIBSWING_MAX_SUPPORTED_DIMENSIONS];    
        
        size_t step; 
        size_t current_d[LIBSWING_MAX_SUPPORTED_PORTS]; // For each port, what's the current dimension we are sending in.
        size_t next_step_per_dim[LIBSWING_MAX_SUPPORTED_PORTS][LIBSWING_MAX_SUPPORTED_DIMENSIONS]; // For each port and for each dimension, what's the next step to execute in that dimension.

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
        // @param port (IN): the port
        // @param coord_peer (OUT): the coordinates of the peer
        void get_peer(int* coord_rank, size_t step, size_t port, int* coord_peer);

        // Magic support functions (TODO: Document)
        void get_blocks_bitmaps_multid(size_t step, size_t port, int* coord_peer, 
                                       char* bitmap_send_merged, char* bitmap_recv_merged, 
                                       int* coord_mine);
        void get_blocks_bitmaps_multid(size_t* next_step_per_dim, size_t* current_d, size_t step,
                                       size_t port, int* coord_peer, char** bitmap_send, 
                                       char** bitmap_recv, char* bitmap_send_merged, char* bitmap_recv_merged, 
                                       int* coord_mine);
        void remap(const std::vector<int>& nodes, uint start_range, uint end_range, uint* blocks_remapping,
                   int step, size_t port, int* coord_rank, char* bitmap_tmp);        
    public:
        SwingBitmapCalculator(uint rank, uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, uint num_ports, bool remap_blocks);

        ~SwingBitmapCalculator();

        uint get_peer(uint port, uint step, CollType coll_type){
            size_t block_step = (coll_type == SWING_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);            
            return (this->peers_per_port)[port][block_step];
        }

        void compute_next_bitmaps();

        bool block_must_be_sent(uint port, uint step, CollType coll_type, uint block_id){
            size_t block_step = (coll_type == SWING_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);
            if(coll_type == SWING_REDUCE_SCATTER){
                return bitmap_send_merged[port][block_step][block_id];
            }else{                
                return bitmap_recv_merged[port][block_step][block_id];
            }
        }

        bool block_must_be_recvd(uint port, uint step, CollType coll_type, uint block_id){
            size_t block_step = (coll_type == SWING_REDUCE_SCATTER) ? step : (this->num_steps - step - 1);            
            if(coll_type == SWING_REDUCE_SCATTER){
                return bitmap_recv_merged[port][block_step][block_id];
            }else{                
                return bitmap_send_merged[port][block_step][block_id];
            }
        }
};

class SwingCommon {
    private:
        uint size;
        int rank;
        uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        uint dimensions_virtual[LIBSWING_MAX_SUPPORTED_DIMENSIONS]; // Used when we shrink torus with non-power of 2 size // TODO: Rename as dimensions_lower_p2
        uint dimensions_num;
        Algo algo; 
        uint num_ports; 
        uint max_size;
        bool all_p2_dimensions; // True if all the dimensions are power of 2
        size_t num_steps_per_dim[LIBSWING_MAX_SUPPORTED_DIMENSIONS];
        size_t num_steps;
        SwingCoordConverter scc;

        // Sends the data from nodes outside of the power-of-two boundary to nodes within the boundary.
        // This is done one dimension at a time.
        // @param sendbuf (IN): the allreduce sendbuf
        // @param recvbuf (OUT): the allreduce recvbuf
        // @param count (IN): the number of elements in the sendbuf
        // @param datatype (IN): the datatype of the elements in the sendbuf
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param tmpbuf (IN): a temporary buffer, as large as the sendbuf/recvbuf
        // @param idle (OUT): if 1, the rank is one of the "extra" ranks and is going to be idle
        // @param rank_virtual (OUT): the virtual rank
        // @return MPI_SUCCESS or an error code
        int shrink_non_power_of_two(const void *sendbuf, void *recvbuf, int count, 
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


        // Bandwidth-optimal Swing collective
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
        int swing_coll_step_b(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                             
                              MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                              CollType coll_type, SwingBitmapCalculator& sbc);

        // Bandwidth-optimal Swing collective with contiguous blocks
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
        int swing_coll_step_cont(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
                                 MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                 CollType coll_type, SwingBitmapCalculator& sbc);

        // Wrapper for step_b/step_cont
        int swing_coll_step(void *buf, void* rbuf, BlockInfo** blocks_info, size_t step,                                 
                           MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                           CollType coll_type, SwingBitmapCalculator& sbc);

        // Bandwidth-optimal Swing collective with UTOFU
        // @param port (IN): the port
        // @param utofu_descriptor (IN): the UTOFU descriptor
        // @param buf (IN): the buf
        // @param rbuf (OUT): the rbuf
        // @param blocks_info (IN): the blocks info
        // @param step (IN): the step
        // @param op (IN): the operation to perform
        // @param comm (IN): the communicator
        // @param sendtype (IN): the send datatype
        // @param recvtype (IN): the recv datatype
        // @param coll_type (IN): the collective type
        // @param bitmap_send (IN): the bitmap of the send
        // @param bitmap_recv (IN): the bitmap of the recv
        int swing_coll_step_utofu(size_t port, swing_utofu_comm_descriptor* utofu_descriptor, void *buf, void* rbuf, BlockInfo** blocks_info, size_t step, 
                                  MPI_Op op, MPI_Comm comm, MPI_Datatype sendtype, MPI_Datatype recvtype,  
                                  CollType coll_type, SwingBitmapCalculator& sbc);

    public:
        // Constructor
        // @param comm (IN): the communicator
        // @param dimensions (IN): the dimensions of the torus
        // @param dimensions_num (IN): the number of dimensions
        // @param algo (IN): the algorithm to use
        // @param num_ports (IN): the number of ports
        // @param max_size (IN): the maximum transfer size
        SwingCommon(MPI_Comm comm, uint dimensions[LIBSWING_MAX_SUPPORTED_DIMENSIONS], uint dimensions_num, Algo algo, uint num_ports, uint max_size);

        // Destructor
        ~SwingCommon();

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
        int swing_coll_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, BlockInfo** blocks_info, CollType coll_type);
};


#endif // LIBSWING_COMMON_H
