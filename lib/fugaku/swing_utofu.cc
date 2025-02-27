#include "swing_utofu.h"


// setup send/recv communication
swing_utofu_comm_descriptor* swing_utofu_setup(utofu_vcq_id_t* vcq_ids, uint num_ports, uint size){
    // Safety checks
    assert(sizeof(utofu_stadd_t) == sizeof(uint64_t));  // Since we send both as 2 64-bit values
    assert(sizeof(utofu_vcq_id_t) == sizeof(uint64_t)); // Since we send both as 2 64-bit values
    
    swing_utofu_comm_descriptor* desc = (swing_utofu_comm_descriptor*) malloc(sizeof(swing_utofu_comm_descriptor));
    memset(desc, 0, sizeof(swing_utofu_comm_descriptor));
    desc->num_ports = num_ports;

    utofu_tni_id_t* tni_ids;
    size_t num_tnis;
    int rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS || num_tnis == 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return NULL;
    }
    assert(num_tnis >= num_ports);


    // TODO: Actually everything could be done in parallel as long as different CQs are managed by different threads
    // this should always be true since each thread works on a different TNI.
    
    // Create all the VCQs (one per port) and register the buffers (once per port)
    for(size_t p = 0; p < num_ports; p++){
        utofu_tni_id_t tni_id = tni_ids[p];
        // query the capabilities of one-sided communication of the TNI
        // create a VCQ and get its VCQ ID
        assert(utofu_create_vcq(tni_id, SWING_UTOFU_VCQ_FLAGS, &(desc->port_info[p].vcq_hdl)) == UTOFU_SUCCESS);
        assert(utofu_query_vcq_id(desc->port_info[p].vcq_hdl, &(vcq_ids[p])) == UTOFU_SUCCESS);
        desc->port_info[p].rmt_recv_stadd = (utofu_stadd_t*) malloc(sizeof(utofu_stadd_t)*size);
        desc->port_info[p].rmt_temp_stadd = (utofu_stadd_t*) malloc(sizeof(utofu_stadd_t)*size);
        desc->port_info[p].registration_cache = new std::unordered_map<void*, utofu_stadd_t>();
    }
    free(tni_ids);    
    return desc;
}

void swing_utofu_teardown(swing_utofu_comm_descriptor* desc, uint num_ports){
    for(size_t p = 0; p < num_ports; p++){
        // Deregister all STADD in cache
        for(auto it = desc->port_info[p].registration_cache->begin(); it != desc->port_info[p].registration_cache->end(); it++){
            utofu_dereg_mem(desc->port_info[p].vcq_hdl, it->second, 0);
        }
        utofu_free_vcq(desc->port_info[p].vcq_hdl);        
        free(desc->port_info[p].rmt_recv_stadd);
        free(desc->port_info[p].rmt_temp_stadd);
        delete desc->port_info[p].registration_cache;
    }
    free(desc);
}

void swing_utofu_reg_buf(swing_utofu_comm_descriptor* desc,
                         const void* send_buffer, size_t length_s, 
                         void* recv_buffer, size_t length_r, 
                         void* temp_buffer, size_t length_t,
                         uint num_ports){
    for(size_t p = 0; p < num_ports; p++){        
        // register memory regions and get their STADDs

        // If send_buffer has already been registered, use the cached STADD
        // Otherwise, register it and cache the STADD
        // Use iterators/find instead of count to avoid multiple lookups
        auto it = desc->port_info[p].registration_cache->find((void*) send_buffer);
        if(it != desc->port_info[p].registration_cache->end()){
            desc->port_info[p].lcl_send_stadd = it->second;
        }else{
            assert(utofu_reg_mem(desc->port_info[p].vcq_hdl, (void*) send_buffer, length_s, 0, &(desc->port_info[p].lcl_send_stadd)) == UTOFU_SUCCESS);
            desc->port_info[p].registration_cache->insert({(void*) send_buffer, desc->port_info[p].lcl_send_stadd});
        }
        // TODO: What if a buffer is freed and then malloced with a different size?
        // We should probably cache the size as well ...
        it = desc->port_info[p].registration_cache->find(recv_buffer);
        if(it != desc->port_info[p].registration_cache->end()){
            desc->port_info[p].lcl_recv_stadd = it->second;
        }else{
            assert(utofu_reg_mem(desc->port_info[p].vcq_hdl, recv_buffer, length_r, 0, &(desc->port_info[p].lcl_recv_stadd)) == UTOFU_SUCCESS);
            desc->port_info[p].registration_cache->insert({recv_buffer, desc->port_info[p].lcl_recv_stadd});
        }

        if(length_t){
            it = desc->port_info[p].registration_cache->find(temp_buffer);
            if(it != desc->port_info[p].registration_cache->end()){
                desc->port_info[p].lcl_temp_stadd = it->second;
            }else{
                assert(utofu_reg_mem(desc->port_info[p].vcq_hdl, temp_buffer, length_t, 0, &(desc->port_info[p].lcl_temp_stadd)) == UTOFU_SUCCESS);
                desc->port_info[p].registration_cache->insert({temp_buffer, desc->port_info[p].lcl_temp_stadd});
            }
        }
        memset(desc->port_info[p].completed_recv, 0, sizeof(desc->port_info[p].completed_recv));
    }
}

void swing_utofu_exchange_buf_info(swing_utofu_comm_descriptor* desc, uint num_steps, uint* peers){
    uint64_t* sbuffer = (uint64_t*) malloc(2*sizeof(uint64_t)*desc->num_ports);
    MPI_Request reqs[LIBSWING_MAX_STEPS];
    for(size_t i = 0; i < desc->num_ports; i++){
        sbuffer[2*i] = desc->port_info[i].lcl_recv_stadd;
        sbuffer[2*i+1] = desc->port_info[i].lcl_temp_stadd;
    }
    // Send the local info for my the ports, to all the peers
    for(size_t step = 0; step < num_steps; step++){
        uint peer = peers[step];	
        MPI_Isend(sbuffer, 2*desc->num_ports, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD, &(reqs[step]));
    }

    // Receive the remote info for all the ports, from all the peers
    uint64_t* rbuffer = (uint64_t*) malloc(2*sizeof(uint64_t)*desc->num_ports);
    for(size_t step = 0; step < num_steps; step++){
        uint peer = peers[step];
        MPI_Recv(rbuffer, 2*desc->num_ports, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(size_t i = 0; i < desc->num_ports; i++){
            // Add an empty entry for the peer
            desc->port_info[i].rmt_recv_stadd[peer] = rbuffer[2*i];
            desc->port_info[i].rmt_temp_stadd[peer] = rbuffer[2*i+1];
        }
    }
    MPI_Waitall(num_steps, reqs, MPI_STATUSES_IGNORE);

    free(sbuffer);
    free(rbuffer);
}

// send data and confirm its completion
void swing_utofu_isend(swing_utofu_comm_descriptor* desc, utofu_vcq_id_t* vcq_id, 
                       uint port, size_t peer,
                       utofu_stadd_t lcl_addr, size_t length, 
                       utofu_stadd_t rmt_addr, uint64_t edata){    
    if(length > MAX_PUTGET_SIZE){
        fprintf(stderr, "Put maximum length exceeded %ld vs. %ld.\n", length, MAX_PUTGET_SIZE);
        exit(-1);
    }
    uintptr_t cbvalue = 0; // for tcq polling; the value is not used
    // instruct the TNI to perform a Put communication
    {
    // embed the default communication path coordinates into the received VCQ ID.
        assert(utofu_set_vcq_id_path(vcq_id, NULL) == UTOFU_SUCCESS);

        utofu_put(desc->port_info[port].vcq_hdl, *vcq_id, 
                  lcl_addr, rmt_addr, length,
                  edata, SWING_UTOFU_POST_FLAGS, (void *)cbvalue);
    }
}

void swing_utofu_wait_sends(swing_utofu_comm_descriptor* desc, uint port, char expected_count){    
    // For the sends it is enough to wait for the completion of expected_count sends, since we never issue
    // the sends to the next peer if the sends to the previous peer are not completed.
    // i.e., we do not need to match the exact send addresses but just count how many of those completed
    for(size_t i = 0; i < expected_count; i++){
        int rc;    
        // confirm the TCQ notification
        void *cbdata;
        do {
            rc = utofu_poll_tcq(desc->port_info[port].vcq_hdl, 0, &cbdata);
        } while (rc == UTOFU_ERR_NOT_FOUND);
        assert(rc == UTOFU_SUCCESS);
    }    
}

// uTofu guarantees that the data sent from a given source to a given destination is received in order, as long as it uses the same VCQ 
// and the same path. The same applies for remote RMQ notifications.
// However, the order is not guaranteed across different sources. For example, rank 0 might receive some data from rank p-1 (from which it
// receives from at step 1) before all the data from rank 1 (from which it receives from at step 0) has been received. 
// Thus, we need to use the EDATA to keep track of the order of the segments received from a given source.
// The EDATA is a 64-bit value that is sent with the PUT operation and is received with the MRQ notification.
// However, only 8 bits are usable. Thus, we have 256 possible values. We use the EDATA to mark the step at which that data was sent.
// Everytime we receive a segment with a given EDATA, we increment the count of segments for that EDATA/step. Since segments
// of the same step are received from the same source, they are received in order, so if we want to know if a specific segment
// has been received, we can just check how many segments were received for that step.
//
// TODO Check EDATA range
// Max putget size: 16777215
// Max edata size: 1
// Num reserved stags: 256
// STag address alignment: 256
// Cache line size: 256
//
// completed_recv[i] corresponds to the number of segments received for step i.
// i.e., if completed_recv[2] == 3, then 3 segments have been received at step 2.
// This works under the assumption that the segments from a given source are received in order (which holds for utofu).
void swing_utofu_wait_recv(swing_utofu_comm_descriptor* desc, uint port, size_t expected_step, size_t expected_segment){
    // If it was already received, return
    if(desc->port_info[port].completed_recv[expected_step] > expected_segment){
        return;
    }
    int rc;    
    struct utofu_mrq_notice notice;
    while(1){
        rc = utofu_poll_mrq(desc->port_info[port].vcq_hdl, 0, &notice);
        if(rc == UTOFU_SUCCESS){
            if(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT){
                DPRINTF("Recv completed at [X, %ld]\n", notice.rmt_stadd);
                desc->port_info[port].completed_recv[notice.edata] += 1;
                if(desc->port_info[port].completed_recv[expected_step] > expected_segment){
                    return;
                }
            }else if(notice.notice_type != UTOFU_MRQ_TYPE_LCL_PUT){
                fprintf(stderr, "Unknown notice type.\n");
                exit(-1);
            }
        }else{
            assert(rc == UTOFU_ERR_NOT_FOUND);
        }
    }   
}
