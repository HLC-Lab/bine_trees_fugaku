#include "swing_utofu.h"

// TODO: Add cache injection?
#define SWING_UTOFU_POST_FLAGS (UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE)

// edata can be at most 8 bits. 
// We use the first 4 bits to encode the step we are, and the remaining 4 bits to encode the id of the chunk
// we are transmitting (large data must be chunked).
// - Regarding the step, this means we support up to 2^4 (16) steps which, in turn, means we support up to
// 2^16 nodes. In practice, we should be fine even with a larger number of nodes, since the probability to receive
// the same edata twice from two different steps should be negligible.abort
// - Regarding the chunk id, it means we can support up to 2^4 chunks. The maximum transmission size is around
// 16MiB. Moreover, we send at most half of the vector. Thus, we can reduce vectors of at most 16MiB*2^4*2 = 511MiB
// In principle we could modify the bit partitioning to support larger networks or larger vectors. Moreover,
// we could still split larger vectors into separate allreduce.
// I do not need to include the port in the edata since each port is manage on a different VCQ already.
uint64_t build_edata(size_t step, size_t chunk){
    return step << 4 | chunk;
}

void parse_edata(uint64_t edata, size_t* step, size_t* chunk){
    *step = (edata >> 4) & 0xF;
    *chunk = edata & 0xF;
}

// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup_communication(utofu_tni_id_t tni_id, uint peer, void* send_buffer, size_t length_s, void* recv_buffer, size_t length_r){
    swing_utofu_comm_descriptor* desc = (swing_utofu_comm_descriptor*) malloc(sizeof(swing_utofu_comm_descriptor));    
    memset(desc->completed_send_local, 0, sizeof(char)*MAX_NUM_CHUNKS);
    memset(desc->completed_send, 0, sizeof(char)*MAX_NUM_CHUNKS);
    memset(desc->completed_recv, 0, sizeof(char)*MAX_NUM_CHUNKS);
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
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, size_t step, size_t chunk, size_t offset, size_t length){    
    if(length > MAX_PUTGET_SIZE){
        fprintf(stderr, "Put maximum length exceeded %ld vs. %ld.\n", length, MAX_PUTGET_SIZE);
        exit(-1);
    }
    uintptr_t cbvalue = 0; // for tcq polling; the value is not used
    uint64_t edata = build_edata(step, chunk);
    // instruct the TNI to perform a Put communication
    utofu_put(desc->vcq_hdl, desc->rmt_vcq_id, 
              desc->lcl_send_stadd + offset, desc->rmt_recv_stadd + offset, length,
              edata, SWING_UTOFU_POST_FLAGS, (void *)cbvalue);
}

void swing_utofu_wait_tcq(swing_utofu_comm_descriptor* desc){
    int rc;    
    // confirm the TCQ notification
    void *cbdata;
    do {
        rc = utofu_poll_tcq(desc->vcq_hdl, 0, &cbdata);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    assert(rc == UTOFU_SUCCESS);
    //assert((uintptr_t)cbdata == cbvalue);
}


void swing_utofu_wait_rmq(swing_utofu_comm_descriptor* desc, size_t expected_step){
    size_t step, chunk;    
    int rc;    
    // confirm the local MRQ notification
    struct utofu_mrq_notice notice;
    do {
        rc = utofu_poll_mrq(desc->vcq_hdl, 0, &notice);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    assert(rc == UTOFU_SUCCESS);
    parse_edata(notice.edata, &step, &chunk);
    if(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT){ // Remote put (recv) completed
        desc->completed_recv[chunk] = 1;
    }else if(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT){
        desc->completed_send[chunk] = 1;
    }else{
        fprintf(stderr, "Unknown notice type.\n");
        exit(-1);
    }
    assert(step == expected_step);
}

void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, size_t expected_step, char expected_count){
    for(size_t i = 0; i < expected_count; i++){
        swing_utofu_wait_tcq(desc);
    }
    for(size_t i = 0; i < expected_count; i++){
        while(!desc->completed_send[i]){
            swing_utofu_wait_rmq(desc, expected_step);
        }
    }
}

void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, size_t expected_step, char expected_chunk){
    while(!desc->completed_recv[expected_chunk]){
        swing_utofu_wait_rmq(desc, expected_step);
    }
}
