#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <utofu.h>

typedef struct{
    utofu_vcq_hdl_t vcq_hdl;
    utofu_vcq_id_t lcl_vcq_id, rmt_vcq_id;
    utofu_stadd_t lcl_send_stadd, lcl_recv_stadd, rmt_recv_stadd;
    size_t length_s;
    char send_complete, recv_complete;
}swing_utofu_comm_descriptor;

// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup_communication(utofu_tni_id_t tni_id, uint peer, void* send_buffer, size_t length_s, void* recv_buffer, size_t length_r);

// teardown communication
void swing_utofu_destroy_communication(swing_utofu_comm_descriptor* desc);

void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint64_t edata);
void swing_utofu_wait(swing_utofu_comm_descriptor* desc, uint64_t edata);
void swing_utofu_waitsend(swing_utofu_comm_descriptor* desc);
void swing_utofu_waitrecv(swing_utofu_comm_descriptor* desc);
