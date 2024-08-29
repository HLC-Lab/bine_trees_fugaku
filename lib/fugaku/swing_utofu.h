#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <utofu.h>
#include <vector>
#include <unordered_map>
#include "../libswing_common.h"

// TODO: Grab with utofu_caps
#define MAX_PUTGET_SIZE 16777215
#define UTOFU_NUM_RESERVED_STAGS 256
#define UTOFU_STAG_ADDR_ALIGNMENT 256

#define MAX_NUM_CHUNKS 16
#define MAX_EDATA 255 // 8 bits

typedef struct{
    utofu_vcq_id_t vcq_id;
    utofu_stadd_t send_stadd;
    utofu_stadd_t recv_stadd;
}swing_utofu_remote_info;

struct swing_utofu_comm_d{
    uint num_ports;
    SwingBitmapCalculator* sbc;
    uint64_t next_edata[LIBSWING_MAX_SUPPORTED_PORTS]; // One per port/VCQ
    uint64_t expected_edata_s[LIBSWING_MAX_SUPPORTED_PORTS]; // One per port/VCQ
    uint64_t expected_edata_r[LIBSWING_MAX_SUPPORTED_PORTS]; // One per port/VCQ
    utofu_vcq_hdl_t vcq_hdl[LIBSWING_MAX_SUPPORTED_PORTS]; // One handle per port
    utofu_vcq_id_t lcl_vcq_id[LIBSWING_MAX_SUPPORTED_PORTS]; // One local VCQ per port
    utofu_stadd_t lcl_send_stadd[LIBSWING_MAX_SUPPORTED_PORTS], lcl_recv_stadd[LIBSWING_MAX_SUPPORTED_PORTS]; // One local STADD per port
    std::unordered_map<uint, swing_utofu_remote_info>* rmt_info[LIBSWING_MAX_SUPPORTED_PORTS]; // For each port we have a map mapping the peer to the addresses
    uint64_t* sbuffer;
    MPI_Request reqs[LIBSWING_MAX_STEPS];
    
    char completed_send[LIBSWING_MAX_SUPPORTED_PORTS][MAX_EDATA];
    char completed_recv[LIBSWING_MAX_SUPPORTED_PORTS][MAX_EDATA];    
};

// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup(void* send_buffer, size_t length_s, void* recv_buffer, size_t length_r, 
                                               uint num_ports, uint num_steps, SwingBitmapCalculator* sbc);
void swing_utofu_setup_wait(swing_utofu_comm_descriptor* desc, uint num_steps);

// teardown communication
void swing_utofu_teardown(swing_utofu_comm_descriptor* desc);
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint port, uint peer, size_t offset_s, size_t offset_r, size_t length, char is_allgather);
// Sends and recv are waited in the same order they are posted
void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, uint port, char expected_count);
void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, uint port);
