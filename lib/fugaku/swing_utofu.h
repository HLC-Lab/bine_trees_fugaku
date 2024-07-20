#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <utofu.h>

#define MAX_PUTGET_SIZE 16777215
#define MAX_NUM_CHUNKS 16

typedef struct{
    utofu_vcq_hdl_t vcq_hdl;
    utofu_vcq_id_t lcl_vcq_id, rmt_vcq_id;
    utofu_stadd_t lcl_send_stadd, lcl_recv_stadd, rmt_recv_stadd;
    char completed_send_local[MAX_NUM_CHUNKS];
    char completed_send[MAX_NUM_CHUNKS];
    char completed_recv[MAX_NUM_CHUNKS];
}swing_utofu_comm_descriptor;

// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup_communication(utofu_tni_id_t tni_id, void* send_buffer, size_t length_s, void* recv_buffer, size_t length_r);
void swing_utofu_exchange_addr(swing_utofu_comm_descriptor* desc, uint peer);

// teardown communication
void swing_utofu_destroy_communication(swing_utofu_comm_descriptor* desc);

// Even if I registered the entire buffer, I might want to send only part of it
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, size_t step, size_t chunk, size_t offset, size_t length);
void swing_utofu_wait_tcq(swing_utofu_comm_descriptor* desc); // sbuff can be modify after this
void swing_utofu_wait_rmq(swing_utofu_comm_descriptor* desc, size_t expected_step); // local data successfully wrote in remote memory, or remote data successfully wrote in local memory. Corresponds to completion of send (or recv)
void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, size_t expected_step, char expected_count);
void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, size_t expected_step, char expected_chunk);

