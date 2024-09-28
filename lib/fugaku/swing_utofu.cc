#include "swing_utofu.h"

static void pack_local_info(swing_utofu_comm_descriptor* desc){
    for(size_t i = 0; i < desc->num_ports; i++){
        desc->sbuffer[3*i] = desc->port_info[i].lcl_vcq_id;
        desc->sbuffer[3*i+1] = desc->port_info[i].lcl_recv_stadd;
        desc->sbuffer[3*i+2] = desc->port_info[i].lcl_temp_stadd;
    }
}

static void unpack_remote_info(swing_utofu_comm_descriptor* desc, uint64_t* buffer, uint peer){
    for(size_t i = 0; i < desc->num_ports; i++){
        assert(desc->port_info[i].rmt_info->count(peer) == 0);
        // Add an empty entry for the peer
        swing_utofu_remote_info rmt;
        rmt.vcq_id     = buffer[3*i];
        rmt.recv_stadd = buffer[3*i+1];
        rmt.temp_stadd = buffer[3*i+2];
        desc->port_info[i].rmt_info->insert({peer, rmt});
        //std::unordered_map<uint, swing_utofu_remote_info>& m = (desc->port_info[i].rmt_info);
        ////m[peer] = rmt;
        //m.insert({peer, rmt});

        // embed the default communication path coordinates into the received VCQ ID.
        assert(utofu_set_vcq_id_path(&(rmt.vcq_id), NULL) == UTOFU_SUCCESS);
    }
}


// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup(const void* send_buffer, size_t length_s, 
                                               void* recv_buffer, size_t length_r, 
                                               void* temp_buffer, size_t length_t,
                                               uint num_ports, uint num_steps, SwingBitmapCalculator* sbc){
    // Safety checks
    assert(sizeof(utofu_stadd_t) == sizeof(uint64_t));  // Since we send both as 2 64-bit values
    assert(sizeof(utofu_vcq_id_t) == sizeof(uint64_t)); // Since we send both as 2 64-bit values
    
    swing_utofu_comm_descriptor* desc = (swing_utofu_comm_descriptor*) malloc(sizeof(swing_utofu_comm_descriptor));    
    desc->num_ports = num_ports;
    desc->sbc = sbc;    

    // Create all the VCQs (one per port) and register the buffers (once per port)
    for(size_t p = 0; p < num_ports; p++){
        desc->port_info[p].acked = 0;
        utofu_tni_id_t tni_id = p;
        // query the capabilities of one-sided communication of the TNI
        // create a VCQ and get its VCQ ID
        assert(utofu_create_vcq(tni_id, SWING_UTOFU_VCQ_FLAGS, &(desc->port_info[p].vcq_hdl)) == UTOFU_SUCCESS);
        assert(utofu_query_vcq_id(desc->port_info[p].vcq_hdl, &(desc->port_info[p].lcl_vcq_id)) == UTOFU_SUCCESS);
        
        // register memory regions and get their STADDs
        assert(utofu_reg_mem(desc->port_info[p].vcq_hdl, (void*) send_buffer, length_s, 0, &(desc->port_info[p].lcl_send_stadd)) == UTOFU_SUCCESS);
        assert(utofu_reg_mem(desc->port_info[p].vcq_hdl, recv_buffer, length_r, 0, &(desc->port_info[p].lcl_recv_stadd)) == UTOFU_SUCCESS);
        assert(utofu_reg_mem(desc->port_info[p].vcq_hdl, temp_buffer, length_t, 0, &(desc->port_info[p].lcl_temp_stadd)) == UTOFU_SUCCESS);
        desc->port_info[p].rmt_info = new std::unordered_map<uint, swing_utofu_remote_info>();

        desc->port_info[p].completed_recv = new std::unordered_set<utofu_stadd_t>();
    }

    
    desc->sbuffer = (uint64_t*) malloc(3*sizeof(uint64_t)*desc->num_ports);
    pack_local_info(desc);
    // Send the local info for my the ports, to all the peers
    // TODO: Replace the individual sends with a collective so that this is done with HW tofu barrier which would be faster?
    for(size_t step = 0; step < num_steps; step++){
        uint peer = desc->sbc->get_peer(step, SWING_REDUCE_SCATTER);	
        MPI_Isend(desc->sbuffer, 3*num_ports, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD, &(desc->reqs[step]));
    }
    return desc;
}

void swing_utofu_setup_wait(swing_utofu_comm_descriptor* desc, uint num_steps){ // TODO: Do it synchronously with sendrecv ?
    // Receive the remote info for all the ports, from all the peers
    uint64_t* rbuffer = (uint64_t*) malloc(3*sizeof(uint64_t)*desc->num_ports);
    for(size_t step = 0; step < num_steps; step++){
        uint peer = desc->sbc->get_peer(step, SWING_REDUCE_SCATTER);
        MPI_Recv(rbuffer, 3*desc->num_ports, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        unpack_remote_info(desc, rbuffer, peer);
    }
    MPI_Waitall(num_steps, desc->reqs, MPI_STATUSES_IGNORE);

    free(desc->sbuffer);
    free(rbuffer);
}

// teardown communication
void swing_utofu_teardown(swing_utofu_comm_descriptor* desc){
    for(size_t p = 0; p < desc->num_ports; p++){        
        utofu_dereg_mem(desc->port_info[p].vcq_hdl, desc->port_info[p].lcl_send_stadd, 0);
        utofu_dereg_mem(desc->port_info[p].vcq_hdl, desc->port_info[p].lcl_recv_stadd, 0);
        utofu_dereg_mem(desc->port_info[p].vcq_hdl, desc->port_info[p].lcl_temp_stadd, 0);
        utofu_free_vcq(desc->port_info[p].vcq_hdl);
        delete desc->port_info[p].rmt_info;
        delete desc->port_info[p].completed_recv;
    }    
    free(desc);
}

// send data and confirm its completion
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, uint port, size_t peer,
                       utofu_stadd_t lcl_addr, size_t length, 
                       utofu_stadd_t rmt_addr){    
    if(length > MAX_PUTGET_SIZE){
        fprintf(stderr, "Put maximum length exceeded %ld vs. %ld.\n", length, MAX_PUTGET_SIZE);
        exit(-1);
    }
    uintptr_t cbvalue = 0; // for tcq polling; the value is not used
    uint64_t edata = 0; // not used
    utofu_vcq_id_t vcq_id = (*(desc->port_info[port].rmt_info))[peer].vcq_id;
    // instruct the TNI to perform a Put communication
    {
    utofu_put(desc->port_info[port].vcq_hdl, vcq_id, 
              lcl_addr, rmt_addr, length,
              edata, SWING_UTOFU_POST_FLAGS, (void *)cbvalue);
    }
}

static void swing_utofu_wait_tcq(swing_utofu_comm_descriptor* desc, uint port){
    int rc;    
    // confirm the TCQ notification
    void *cbdata;
    do {
        rc = utofu_poll_tcq(desc->port_info[port].vcq_hdl, 0, &cbdata);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    assert(rc == UTOFU_SUCCESS);
    //assert((uintptr_t)cbdata == cbvalue);
}

static void swing_utofu_wait_rmq(swing_utofu_comm_descriptor* desc, uint port){
    int rc;    
    struct utofu_mrq_notice notice;
    do {
        rc = utofu_poll_mrq(desc->port_info[port].vcq_hdl, 0, &notice);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    assert(rc == UTOFU_SUCCESS);
    if(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT){ // Remote put (recv) completed      
        DPRINTF("Recv completed at [X, %ld]\n", notice.rmt_stadd);
        desc->port_info[port].completed_recv->insert(notice.rmt_stadd);
    }else if(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT){ // Local put (send) completed
        ++desc->port_info[port].acked; // We do not need to store the address
    }else{
        fprintf(stderr, "Unknown notice type.\n");
        exit(-1);
    }    
    //assert(step == expected_step);
}

void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, uint port, char expected_count){    
    // For the sends it is enough to wait for the completion of expected_count sends, since we never issue
    // the sends to the next peer if the sends to the previous peer are not completed.
    // i.e., we do not need to match the exact send addresses but just count how many of those completed
    for(size_t i = 0; i < expected_count; i++){
        swing_utofu_wait_tcq(desc, port);        
    }    
    while(desc->port_info[port].acked < expected_count){
        swing_utofu_wait_rmq(desc, port);
    }
    desc->port_info[port].acked = 0;
}

/*
std::ostream& operator<<(std::ostream& os, std::unordered_set<utofu_stadd_t> const& s)
{
    os << "{";
    for (auto i : s)
        os << i << ' ';
    return os << "}\n";
}
*/

void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, uint port, utofu_stadd_t end_addr){
    DPRINTF("Wait for %ld\n", end_addr);
    while(!desc->port_info[port].completed_recv->count(end_addr)){
        swing_utofu_wait_rmq(desc, port);
    }
    desc->port_info[port].completed_recv->erase(end_addr);
}
