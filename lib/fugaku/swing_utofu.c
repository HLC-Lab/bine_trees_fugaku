#include "swing_utofu.h"

// TODO: Add cache injection?
#define SWING_UTOFU_POST_FLAGS (UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE)
#define MAX_PUTGET_SIZE 16777215

// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup_communication(utofu_tni_id_t tni_id, uint peer, void* send_buffer, size_t length_s, void* recv_buffer, size_t length_r){
    if(length_s > MAX_PUTGET_SIZE || length_r > MAX_PUTGET_SIZE){
        fprintf(stderr, "Put maximum length exceeded (%ld,%ld) vs. %ld.\n", length_s, length_r, MAX_PUTGET_SIZE);
        exit(-1);
    }
    swing_utofu_comm_descriptor* desc = (swing_utofu_comm_descriptor*) malloc(sizeof(swing_utofu_comm_descriptor));    
    desc->length_s = length_s;
    desc->send_complete = 0;
    desc->recv_complete = 0;
    // TODO: We can do just once at the beginning. We can communicate alltoall all the base addresses of the vectors, and then just add offsets at each step.
    assert(sizeof(utofu_stadd_t) == sizeof(uint64_t));  // Since we send both as 2 64-bit values
    assert(sizeof(utofu_vcq_id_t) == sizeof(uint64_t)); // Since we send both as 2 64-bit values
    // query the capabilities of one-sided communication of the TNI
    // create a VCQ and get its VCQ ID
    assert(utofu_create_vcq(tni_id, 0, &(desc->vcq_hdl)) == UTOFU_SUCCESS);
    assert(utofu_query_vcq_id(desc->vcq_hdl, &(desc->lcl_vcq_id)) == UTOFU_SUCCESS);
    // register memory regions and get their STADDs
    assert(utofu_reg_mem(desc->vcq_hdl, send_buffer, length_s, 0, &(desc->lcl_send_stadd)) == UTOFU_SUCCESS);
    assert(utofu_reg_mem(desc->vcq_hdl, recv_buffer, length_r, 0, &(desc->lcl_recv_stadd)) == UTOFU_SUCCESS);

    // notify peer processes of the VCQ ID and the STADD
    uint64_t tmp_s_buffer[2] = {desc->lcl_vcq_id, desc->lcl_recv_stadd};
    uint64_t tmp_r_buffer[2];
    assert(MPI_Sendrecv(tmp_s_buffer, 2, MPI_UINT64_T, peer, 0,
                        tmp_r_buffer, 2, MPI_UINT64_T, peer, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS);
    desc->rmt_vcq_id = tmp_r_buffer[0];
    desc->rmt_recv_stadd = tmp_r_buffer[1];
    /*
    // Logically equivalent to the following, we just did one sendrecv rather than two.
    MPI_Sendrecv(&(desc->lcl_vcq_id), 1, MPI_UINT64_T, peer, 0,
                 &(desc->rmt_vcq_id), 1, MPI_UINT64_T, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(desc->lcl_recv_stadd), 1, MPI_UINT64_T, peer, 0,
                 &(desc->rmt_recv_stadd), 1, MPI_UINT64_T, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    */
    // embed the default communication path coordinates into the received VCQ ID.
    assert(utofu_set_vcq_id_path(&(desc->rmt_vcq_id), NULL) == UTOFU_SUCCESS);
    MPI_Barrier(MPI_COMM_WORLD);
    return desc;
}

// teardown communication
void swing_utofu_destroy_communication(swing_utofu_comm_descriptor* desc){
    utofu_dereg_mem(desc->vcq_hdl, desc->lcl_send_stadd, 0);
    utofu_dereg_mem(desc->vcq_hdl, desc->lcl_recv_stadd, 0);
    utofu_free_vcq(desc->vcq_hdl);
    free(desc);
}

// send data and confirm its completion
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint64_t edata){
    uintptr_t cbvalue = 0; // for tcq polling; the value is not used
    // instruct the TNI to perform a Put communication
    utofu_put(desc->vcq_hdl, desc->rmt_vcq_id, 
              desc->lcl_send_stadd, desc->rmt_recv_stadd, desc->length_s,
              edata, SWING_UTOFU_POST_FLAGS, (void *)cbvalue);
}

void swing_utofu_wait(swing_utofu_comm_descriptor* desc, uint64_t edata){
    int rc;    
    // confirm the TCQ notification
    if (SWING_UTOFU_POST_FLAGS & UTOFU_ONESIDED_FLAG_TCQ_NOTICE) {
        void *cbdata;
        do {
            rc = utofu_poll_tcq(desc->vcq_hdl, 0, &cbdata);
        } while (rc == UTOFU_ERR_NOT_FOUND);
        assert(rc == UTOFU_SUCCESS);
    }
    // At this point I could modify the sbuff
    
    // confirm the local MRQ notification
    struct utofu_mrq_notice notice;
    rc = UTOFU_ERR_NOT_FOUND;
    assert(!desc->send_complete && !desc->recv_complete);
    while (rc == UTOFU_ERR_NOT_FOUND) {
        rc = utofu_poll_mrq(desc->vcq_hdl, 0, &notice);
        if(rc == UTOFU_SUCCESS){
            if(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT){ // Remote put (recv) completed
                desc->recv_complete = 1;
                assert(notice.edata == edata);
                rc = UTOFU_ERR_NOT_FOUND; // So we keep polling
            }else if(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT){ // Local put (send) completed
                desc->send_complete = 1;
                assert(notice.edata == edata);
                rc = UTOFU_ERR_NOT_FOUND;  // So we keep polling
            }
        }else if(rc != UTOFU_ERR_NOT_FOUND){
            fprintf(stderr, "Unknown return status %d.\n", rc);
            exit(-1);
        }
        if(desc->recv_complete && desc->send_complete){
            return;
        }
    };
}

void swing_utofu_waitsend(swing_utofu_comm_descriptor* desc){
    if(desc->send_complete){return;}
    int rc;    
    // confirm the TCQ notification
    if (SWING_UTOFU_POST_FLAGS & UTOFU_ONESIDED_FLAG_TCQ_NOTICE) {
        void *cbdata;
        do {
            rc = utofu_poll_tcq(desc->vcq_hdl, 0, &cbdata);
        } while (rc == UTOFU_ERR_NOT_FOUND);
        assert(rc == UTOFU_SUCCESS);
        //assert((uintptr_t)cbdata == cbvalue);
    }
    // confirm the local MRQ notification
    if (SWING_UTOFU_POST_FLAGS & UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE) {
rewait_s:
        struct utofu_mrq_notice notice;
        do {
            rc = utofu_poll_mrq(desc->vcq_hdl, 0, &notice);
        } while (rc == UTOFU_ERR_NOT_FOUND);
        assert(rc == UTOFU_SUCCESS);
        if(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT){ // Remote put (recv) completed
            desc->recv_complete = 1;
            goto rewait_s;
        }else{
            assert(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT);
        }
        //assert(notice.edata == edata);
    }
}

// confirm receiving data
void swing_utofu_waitrecv(swing_utofu_comm_descriptor* desc){
    if(desc->recv_complete){return;}
    //uint64_t edata = 0; // Unused
    int rc;
    // confirm the remote MRQ notification or the memory update
    if (SWING_UTOFU_POST_FLAGS & UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE) {
rewait_r:
        struct utofu_mrq_notice notice;
        do {
            rc = utofu_poll_mrq(desc->vcq_hdl, 0, &notice);
        } while (rc == UTOFU_ERR_NOT_FOUND);
        assert(rc == UTOFU_SUCCESS);
        if(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT){ // Local put (send) completed
            desc->send_complete = 1;
            goto rewait_r;
        }else{
            assert(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT);
        }
        //assert(notice.edata == edata);
        //assert(*recv_buffer == expected_value);
    } else {
        //while (*(desc->recv_buffer) != expected_value);
    }
}
