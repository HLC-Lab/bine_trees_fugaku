#include "swing_utofu.h"

// TODO: Add cache injection?
#define SWING_UTOFU_POST_FLAGS (UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE)
#define SWING_UTOFU_VCQ_FLAGS 0 // (UTOFU_VCQ_FLAG_EXCLUSIVE) // Allows different threads to work on different VCQs simultaneously

static uint64_t* pack_local_info(swing_utofu_comm_descriptor* desc){
    uint64_t* buffer = (uint64_t*) malloc(3*sizeof(uint64_t)*desc->num_ports);
    for(size_t i = 0; i < desc->num_ports; i++){
        buffer[3*i] = desc->lcl_vcq_id[i];
        buffer[3*i+1] = desc->lcl_send_stadd[i];
        buffer[3*i+2] = desc->lcl_recv_stadd[i];
    }
    return buffer;
}

static void unpack_remote_info(swing_utofu_comm_descriptor* desc, uint64_t* buffer, uint peer){
    for(size_t i = 0; i < desc->num_ports; i++){
        assert(desc->rmt_info[i]->count(peer) == 0);
        // Add an empty entry for the peer
        swing_utofu_remote_info rmt;
        rmt.vcq_id     = buffer[3*i];
        rmt.send_stadd = buffer[3*i+1];
        rmt.recv_stadd = buffer[3*i+2];
        desc->rmt_info[i]->insert({peer, rmt});
        //std::unordered_map<uint, swing_utofu_remote_info>& m = (desc->rmt_info[i]);
        ////m[peer] = rmt;
        //m.insert({peer, rmt});

        // embed the default communication path coordinates into the received VCQ ID.
        assert(utofu_set_vcq_id_path(&(rmt.vcq_id), NULL) == UTOFU_SUCCESS);
    }
}


// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup(void* send_buffer, size_t length_s, void* recv_buffer, size_t length_r, 
                                               uint num_ports, uint num_steps, uint** peers_per_port){
    // Safety checks
    assert(sizeof(utofu_stadd_t) == sizeof(uint64_t));  // Since we send both as 2 64-bit values
    assert(sizeof(utofu_vcq_id_t) == sizeof(uint64_t)); // Since we send both as 2 64-bit values
    
    swing_utofu_comm_descriptor* desc = (swing_utofu_comm_descriptor*) malloc(sizeof(swing_utofu_comm_descriptor));    
    desc->num_ports = num_ports;
    desc->peers_per_port = peers_per_port;
    memset(desc->completed_send, 0, sizeof(desc->completed_send));
    memset(desc->completed_recv, 0, sizeof(desc->completed_recv));

    // Create all the VCQs (one per port) and register the buffers (once per port)
    for(size_t p = 0; p < num_ports; p++){
        desc->next_edata[p] = 0;
        desc->expected_edata_s[p] = 0;
        desc->expected_edata_r[p] = 0;
        utofu_tni_id_t tni_id = p;
        // query the capabilities of one-sided communication of the TNI
        // create a VCQ and get its VCQ ID
        double t0 = MPI_Wtime();
        assert(utofu_create_vcq(tni_id, SWING_UTOFU_VCQ_FLAGS, &(desc->vcq_hdl[p])) == UTOFU_SUCCESS);
        assert(utofu_query_vcq_id(desc->vcq_hdl[p], &(desc->lcl_vcq_id[p])) == UTOFU_SUCCESS);
        // register memory regions and get their STADDs
        assert(utofu_reg_mem(desc->vcq_hdl[p], send_buffer, length_s, 0, &(desc->lcl_send_stadd[p])) == UTOFU_SUCCESS);
        assert(utofu_reg_mem(desc->vcq_hdl[p], recv_buffer, length_r, 0, &(desc->lcl_recv_stadd[p])) == UTOFU_SUCCESS);
        desc->rmt_info[p] = new std::unordered_map<uint, swing_utofu_remote_info>();
    }


    // This must be done by a single thread!
    uint64_t* sbuffer = pack_local_info(desc);
    // Send the local info for my the ports, to all the peers
    for(size_t step = 0; step < num_steps; step++){
        uint peer = peers_per_port[0][step];
        MPI_Send(sbuffer, 3*num_ports, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD);
    }
    // Receive the remote info for all the ports, from all the peers
    uint64_t* rbuffer = (uint64_t*) malloc(3*sizeof(uint64_t)*num_ports);
    for(size_t step = 0; step < num_steps; step++){
        uint peer = peers_per_port[0][step];
        MPI_Recv(rbuffer, 3*num_ports, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        unpack_remote_info(desc, rbuffer, peer);
    }

    free(sbuffer);
    free(rbuffer);
    return desc;
}

// teardown communication
void swing_utofu_teardown(swing_utofu_comm_descriptor* desc){
    for(size_t p = 0; p < desc->num_ports; p++){        
        utofu_dereg_mem(desc->vcq_hdl[p], desc->lcl_send_stadd[p], 0);
        utofu_dereg_mem(desc->vcq_hdl[p], desc->lcl_recv_stadd[p], 0);
        utofu_free_vcq(desc->vcq_hdl[p]);
        delete desc->rmt_info[p];
    }    
    free(desc);
}

// send data and confirm its completion
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint port, size_t step, size_t chunk, size_t offset_s, size_t offset_r, size_t length, char is_allgather){    
    if(length > MAX_PUTGET_SIZE){
        fprintf(stderr, "Put maximum length exceeded %ld vs. %ld.\n", length, MAX_PUTGET_SIZE);
        exit(-1);
    }
    uintptr_t cbvalue = 0; // for tcq polling; the value is not used
    uint64_t edata = desc->next_edata[port];
    desc->next_edata[port] = (desc->next_edata[port] + 1) % MAX_EDATA;
    uint peer = desc->peers_per_port[port][step];
    swing_utofu_remote_info rmt = (*(desc->rmt_info[port]))[peer];
    utofu_stadd_t rmt_stadd = is_allgather ? rmt.send_stadd : rmt.recv_stadd;

    // instruct the TNI to perform a Put communication
#pragma omp critical // To remove this we should put SWING_UTOFU_VCQ_FLAGS to UTOFU_VCQ_FLAG_EXCLUSIVE. However, this adds crazy overhead when creating/destroying the VCQs
    {
    utofu_put(desc->vcq_hdl[port], rmt.vcq_id, 
              desc->lcl_send_stadd[port] + offset_s, rmt_stadd + offset_r, length,
              edata, SWING_UTOFU_POST_FLAGS, (void *)cbvalue);
    }
}

static void swing_utofu_wait_tcq(swing_utofu_comm_descriptor* desc, uint port){
    int rc;    
    // confirm the TCQ notification
    void *cbdata;
    do {
        rc = utofu_poll_tcq(desc->vcq_hdl[port], 0, &cbdata);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    assert(rc == UTOFU_SUCCESS);
    //assert((uintptr_t)cbdata == cbvalue);
}

static void swing_utofu_wait_rmq(swing_utofu_comm_descriptor* desc, uint port){
    int rc;    
    struct utofu_mrq_notice notice;
    do {
        rc = utofu_poll_mrq(desc->vcq_hdl[port], 0, &notice);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    assert(rc == UTOFU_SUCCESS);
    if(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT){ // Remote put (recv) completed
        desc->completed_recv[port][notice.edata] = 1;
    }else if(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT){ // Local put (send) completed
        desc->completed_send[port][notice.edata] = 1;
    }else{
        fprintf(stderr, "Unknown notice type.\n");
        exit(-1);
    }    
    //assert(step == expected_step);
}

void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, uint port, char expected_count){    
    for(size_t i = 0; i < expected_count; i++){
        uint64_t expected_edata = desc->expected_edata_s[port];
        swing_utofu_wait_tcq(desc, port);
        while(!desc->completed_send[port][expected_edata]){
            swing_utofu_wait_rmq(desc, port);
        }
        desc->completed_send[port][expected_edata] = 0;
        desc->expected_edata_s[port] = (desc->expected_edata_s[port] + 1) % MAX_EDATA;
    }    
}

void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, uint port){
    uint64_t expected_edata = desc->expected_edata_r[port];
    while(!desc->completed_recv[port][expected_edata]){
        swing_utofu_wait_rmq(desc, port);
    }
    desc->completed_recv[port][expected_edata] = 0;
    desc->expected_edata_r[port] = (desc->expected_edata_r[port] + 1) % MAX_EDATA;
}
