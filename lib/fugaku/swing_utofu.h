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
#define SWING_UTOFU_POST_FLAGS (UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE)
#define SWING_UTOFU_VCQ_FLAGS 0

typedef struct{
    utofu_vcq_id_t vcq_id;
    utofu_stadd_t recv_stadd;
    utofu_stadd_t temp_stadd;
}swing_utofu_remote_info;

typedef struct{
    utofu_vcq_hdl_t vcq_hdl;
    utofu_vcq_id_t lcl_vcq_id;
    utofu_stadd_t lcl_send_stadd;
    utofu_stadd_t lcl_recv_stadd;
    utofu_stadd_t lcl_temp_stadd;
    std::unordered_map<uint, swing_utofu_remote_info>* rmt_info; // mapping the peer to the addresses
    char completed_recv[LIBSWING_MAX_STEPS]; // set of completed recvs
    volatile char padding[CACHE_LINE_SIZE];
}swing_utofu_port_info;

struct swing_utofu_comm_d{
    uint num_ports;
    uint* peers;
    swing_utofu_port_info port_info[LIBSWING_MAX_SUPPORTED_PORTS];
    uint64_t* sbuffer;
    MPI_Request reqs[LIBSWING_MAX_STEPS];
};


void swing_utofu_setup(swing_utofu_comm_descriptor* desc, uint num_ports);
void swing_utofu_reg_buf(swing_utofu_comm_descriptor* desc,
                         const void* send_buffer, size_t length_s, 
                         void* recv_buffer, size_t length_r, 
                         void* temp_buffer, size_t length_t,
                         uint num_ports, uint num_steps, uint* peers);
void swing_utofu_reg_buf_wait(swing_utofu_comm_descriptor* desc, uint num_steps);
void swing_utofu_dereg_buffers(swing_utofu_comm_descriptor* desc);
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint port, size_t peer,
    utofu_stadd_t lcl_addr, size_t length, 
    utofu_stadd_t rmt_addr, uint64_t edata);
void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, uint port, char expected_count);
void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, uint port, size_t expected_step, size_t expected_segment);
