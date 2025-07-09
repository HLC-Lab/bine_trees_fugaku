#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <utofu.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../libbine_common.h"

// TODO: Grab with utofu_caps
#define MAX_PUTGET_SIZE 16777215
#define MAX_PIGGYBACK_SIZE 32
#define UTOFU_NUM_RESERVED_STAGS 256
#define UTOFU_STAG_ADDR_ALIGNMENT 256
#define MAX_EDATA 255 // 8 bits

// TODO: Add cache injection?
#define BINE_UTOFU_POST_FLAGS (UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_STRONG_ORDER)
#define BINE_UTOFU_VCQ_FLAGS 0

typedef struct{
    utofu_vcq_hdl_t vcq_hdl;
    utofu_stadd_t lcl_send_stadd;
    utofu_stadd_t lcl_recv_stadd;
    utofu_stadd_t lcl_temp_stadd;
    utofu_stadd_t* rmt_recv_stadd; 
    utofu_stadd_t* rmt_temp_stadd; 
    std::unordered_map<void*, utofu_stadd_t>* registration_cache; // Cache for the registration of the buffers
    size_t completed_recv[LIBBINE_MAX_STEPS]; // set of completed recvs
    size_t completed_send; // set of completed sends
    volatile char padding[CACHE_LINE_SIZE];
}bine_utofu_port_info;

struct bine_utofu_comm_d{
    uint num_ports;
    uint size;
    uint* peers;
    bine_utofu_port_info port_info[LIBBINE_MAX_SUPPORTED_PORTS];
};


bine_utofu_comm_descriptor* bine_utofu_setup(utofu_vcq_id_t* vcq_ids, uint num_ports, uint size);
void bine_utofu_teardown(bine_utofu_comm_descriptor* desc, uint num_ports);
void bine_utofu_reg_buf(bine_utofu_comm_descriptor* desc,
                         const void* send_buffer, size_t length_s, 
                         void* recv_buffer, size_t length_r, 
                         void* temp_buffer, size_t length_t,
                         uint num_ports);
void bine_utofu_dereg_buf(bine_utofu_comm_descriptor* desc, void* buffer, int port);                         
void bine_utofu_exchange_buf_info(bine_utofu_comm_descriptor* desc, uint num_steps, uint* peers);
void bine_utofu_exchange_buf_info_allgather(bine_utofu_comm_descriptor* desc, uint num_steps);
size_t bine_utofu_isend(bine_utofu_comm_descriptor* desc, utofu_vcq_id_t* vcq_id, uint port, size_t peer,
                         utofu_stadd_t lcl_addr, size_t length, 
                         utofu_stadd_t rmt_addr, uint64_t edata);
size_t bine_utofu_isend_piggyback(bine_utofu_comm_descriptor* desc, utofu_vcq_id_t* vcq_id, uint port, size_t peer,
                            void* lcl_data, size_t length, 
                         utofu_stadd_t rmt_addr, uint64_t edata);                         
size_t bine_utofu_isend_delayed(bine_utofu_comm_descriptor* desc, utofu_vcq_id_t* vcq_id, uint port, size_t peer,
                                utofu_stadd_t lcl_addr, size_t length, 
                                utofu_stadd_t rmt_addr, uint64_t edata);
void bine_utofu_wait_sends(bine_utofu_comm_descriptor* desc, uint port, size_t expected_count);
void bine_utofu_wait_recv(bine_utofu_comm_descriptor* desc, uint port, size_t expected_step, size_t expected_segment);
