#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <utofu.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../libswing_common.h"

// TODO: Grab with utofu_caps
#define MAX_PUTGET_SIZE 16777215
#define UTOFU_NUM_RESERVED_STAGS 256
#define UTOFU_STAG_ADDR_ALIGNMENT 256
#define MAX_EDATA 255 // 8 bits

// TODO: Add cache injection?
#define SWING_UTOFU_POST_FLAGS (UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE)

#define UTOFU_THREAD_SAFE 0

#if UTOFU_THREAD_SAFE
#define SWING_UTOFU_VCQ_FLAGS (UTOFU_VCQ_FLAG_THREAD_SAFE) // (UTOFU_VCQ_FLAG_EXCLUSIVE) // Allows different threads to work on different VCQs simultaneously
#else
#define SWING_UTOFU_VCQ_FLAGS 0
#endif

typedef struct{
    utofu_vcq_id_t vcq_id;
    utofu_stadd_t recv_stadd;
    utofu_stadd_t temp_stadd;
}swing_utofu_remote_info;

struct swing_utofu_comm_d{
    uint num_ports;
    SwingBitmapCalculator* sbc;
    utofu_vcq_hdl_t vcq_hdl[LIBSWING_MAX_SUPPORTED_PORTS]; // One handle per port
    utofu_vcq_id_t lcl_vcq_id[LIBSWING_MAX_SUPPORTED_PORTS]; // One local VCQ per port
    utofu_stadd_t lcl_send_stadd[LIBSWING_MAX_SUPPORTED_PORTS], lcl_recv_stadd[LIBSWING_MAX_SUPPORTED_PORTS], lcl_temp_stadd[LIBSWING_MAX_SUPPORTED_PORTS];
    std::unordered_map<uint, swing_utofu_remote_info>* rmt_info[LIBSWING_MAX_SUPPORTED_PORTS]; // For each port we have a map mapping the peer to the addresses
    std::unordered_set<utofu_stadd_t>* completed_recv[LIBSWING_MAX_SUPPORTED_PORTS]; // For each port we have a set of completed but unacknowledged recvs
    size_t acked_send[LIBSWING_MAX_SUPPORTED_PORTS];
    uint64_t* sbuffer;
    MPI_Request reqs[LIBSWING_MAX_STEPS];
};

// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup(const void* send_buffer, size_t length_s, 
                                               void* recv_buffer, size_t length_r, 
                                               void* tmp_buffer, size_t length_t,
                                               uint num_ports, uint num_steps, SwingBitmapCalculator* sbc);
void swing_utofu_setup_wait(swing_utofu_comm_descriptor* desc, uint num_steps);

// teardown communication
void swing_utofu_teardown(swing_utofu_comm_descriptor* desc);
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint port, size_t peer,
                       utofu_stadd_t lcl_addr, size_t length,
                       utofu_stadd_t rmt_addr);

// Sends and recv are waited in the same order they are posted
void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, uint port, char expected_count);
void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, uint port, utofu_stadd_t end_addr);
