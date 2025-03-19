#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <unistd.h>

#include "libswing_common.h"
#include "libswing_coll.h"
#include <climits>
#ifdef FUGAKU
#include "fugaku/swing_utofu.h"
#endif

#define SWING_REDUCE_NOSYNC_THRESHOLD 1024 // TODO Read from env. Env should be passed to SwingCommon as a struct with all the variables.

int SwingCommon::swing_reduce_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.reduce_config.algo == SWING_REDUCE_ALGO_BINOMIAL_TREE);
#endif
#ifdef FUGAKU
    assert(count >= env.num_ports);
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_mpi (init)");
    Timer timer("swing_reduce_utofu (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size;
    if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
        tmpbuf_size = count*dtsize;
    }else{
        tmpbuf_size = count*dtsize*this->num_steps; // In this way each rank writes in a different part of tmpbuf, avoid the need to synchronize
    }
    
    timer.reset("= swing_reduce_utofu (utofu buf reg)"); 

    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, count*dtsize, recvbuf, count*dtsize, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_reduce_utofu (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.reduce_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.reduce_config.algo_family, this->scc_real, peers[mp]);
                swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, count*dtsize, recvbuf, count*dtsize, NULL, 0, env.num_ports); 
        tmpbuf = env.prealloc_buf;
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_temp_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }
    DPRINTF("tmpbuf allocated\n");    

    size_t max_count;
    if(env.segment_size){
        max_count = floor(env.segment_size / dtsize);
    }else{
        max_count = floor(MAX_PUTGET_SIZE / dtsize);
    }

    int res = MPI_SUCCESS; 

    uint partition_size = count / env.num_ports;
    uint remaining = count % env.num_ports;  
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the count and the offset of the piece of buffer that is aggregated on this port
        size_t count_port = partition_size + (port < remaining ? 1 : 0);
        size_t offset_port = 0;
        for(size_t j = 0; j < port; j++){
            offset_port += partition_size + (j < remaining ? 1 : 0);
        }
        offset_port *= dtsize;

        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_utofu (computing trees)");
        // If I construct a tree with increasing distance, then I can use it is a gather tree with decreasing distance (i.e., the last peer will be the first).
        // and vice-versa. Thus, I always need to use the opposite distance when building the tree.
        // e.g., for 4 nodes I have an increasing distance tree as follows:
        //       0
        //      / \
        //     3   1
        //          \
        //           2
        // Thus, if I want to gather the data, I should gather first from 3 and then from 1.
        // I.e., to construct a 'reduce/gather' we should construct a  'broadcast/scatter' tree with opposite distances.
        swing_tree_t tree = get_tree(root, port, env.reduce_config.algo_family, env.reduce_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);        

        // I do a bunch of receives (unless I am a leaf), and then I send the data to the parent
        // To understand at which step I must send the data, I need to check at which step I am 
        // reached by the root.
        // If this is step s, then I start sending at (num_steps - 1 - s)
        // To check if I am a leaf, I can just check if I am reached by the root in the last step
        int sending_step;
        if(root == this->rank){
            sending_step = this->num_steps;
        }else{
            sending_step = this->num_steps - 1 - tree.reached_at_step[this->rank];
        }

        DPRINTF("[%d] Sending step: %d\n", this->rank, sending_step);
        timer.reset("= swing_reduce_utofu (waiting recv)");
        char copied = 0;
        for(size_t step = 0; step < (uint) this->num_steps; step++){      
            size_t offset_port_tmpbuf;
            if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
                offset_port_tmpbuf = offset_port;                
            }else{
                offset_port_tmpbuf = offset_port*this->num_steps + step*count_port*dtsize;
            }  
            if(step < sending_step){
                // Receive from peer
                uint peer;
                if(env.reduce_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][this->num_steps - step - 1];                             
                }else{  
                    peer = peers[port][step];                      
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    DPRINTF("[%d] Receiving from %d at step %d\n", this->rank, peer, step);

                    if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
                        // Do a 0-byte put to notify I am ready to recv
                        swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, utofu_descriptor->port_info[port].lcl_send_stadd + offset_port, 0, utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + offset_port, step);                    
                    }

                    // Now start pipelining recv and reduce
                    size_t remaining = count_port;
                    size_t bytes_to_recv = 0;
                    size_t next_recv = 0;
                    char set_copied = 0;
                    size_t offset_segment = 0;

                    while(remaining){
                        size_t count = remaining < max_count ? remaining : max_count;
                        bytes_to_recv = count*dtsize;

                        swing_utofu_wait_recv(utofu_descriptor, port, step, next_recv);
                        if(!copied){
                            reduce_local(((char*) sendbuf) + offset_port + offset_segment, tmpbuf + offset_port_tmpbuf + offset_segment, ((char*) recvbuf) + offset_port + offset_segment, count, datatype, op);
                            set_copied = 1;
                        }else{
                            reduce_local(tmpbuf + offset_port_tmpbuf + offset_segment, ((char*) recvbuf) + offset_port + offset_segment, count, datatype, op);
                        }

                        offset_segment += bytes_to_recv;
                        remaining -= count;
                        ++next_recv;
                    }
                    
                    if(set_copied){
                        copied = 1;
                    }

                    // Now wait for the 0-byte put notify completion
                    if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
                        swing_utofu_wait_sends(utofu_descriptor, port, 1);
                    }
                }
            }else if(step == sending_step){
                // Send to parent
                uint peer = tree.parent[this->rank];            
                DPRINTF("[%d] Sending to %d at step %d\n", this->rank, peer, step);
                utofu_stadd_t lcl_addr, rmt_addr;
                if(!copied){
                    lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd + offset_port;                    
                    copied = 1;
                }else{
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd + offset_port;                    
                }
                rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + offset_port_tmpbuf;

                if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
                    // Do a 0-byte recv to check if the peer is ready to recv
                    swing_utofu_wait_recv(utofu_descriptor, port, step, 0);
                }

                // Now start pipelining recv and reduce
                size_t issued_sends = 0;
                size_t remaining = count_port;
                size_t bytes_to_send = 0;
                size_t offset_segment = 0;
                
                while(remaining){
                    size_t count = remaining < max_count ? remaining : max_count;
                    bytes_to_send = count*dtsize;
                    issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr + offset_segment, count*dtsize, rmt_addr + offset_segment, step);
                    offset_segment += bytes_to_send;
                    remaining -= count;
                }
                            
                swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_reduce_utofu (waiting all sends)");
        }
        
        free(peers[port]);
        destroy_tree(&tree);
        if(free_tmpbuf){
            swing_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, port);
        }        
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_utofu (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return -1;
#endif    
}
    
int SwingCommon::swing_reduce_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_config.algo == SWING_REDUCE_ALGO_BINOMIAL_TREE);
#endif
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_mpi (init)");
    Timer timer("swing_reduce_mpi (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = count*dtsize*this->num_steps; // A bit overkill, needed for uTofu to avoid overwriting the same buffer in different steps // TODO: Fix
    
    timer.reset("= swing_reduce_mpi (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }

    int res = MPI_SUCCESS; 

    uint partition_size = count / env.num_ports;
    uint remaining = count % env.num_ports;  

    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the count and the offset of the piece of buffer that is aggregated on this port
        size_t count_port = partition_size + (port < remaining ? 1 : 0);
        size_t offset_port = 0, offset_port_tmpbuf;
        for(size_t j = 0; j < port; j++){
            offset_port += partition_size + (j < remaining ? 1 : 0);
        }
        offset_port *= dtsize;
        offset_port_tmpbuf = offset_port * this->num_steps; // TODO: Ugly related to the overkill thing above for tmpbuf

        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_mpi (computing trees)");
        // If I construct a tree with increasing distance, then I can use it is a gather tree with decreasing distance (i.e., the last peer will be the first).
        // and vice-versa. Thus, I always need to use the opposite distance when building the tree.
        // e.g., for 4 nodes I have an increasing distance tree as follows:
        //       0
        //      / \
        //     3   1
        //          \
        //           2
        // Thus, if I want to gather the data, I should gather first from 3 and then from 1.
        // I.e., to construct a 'reduce/gather' we should construct a  'broadcast/scatter' tree with opposite distances.
        swing_tree_t tree = get_tree(root, port, env.reduce_config.algo_family, env.reduce_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);        

        // I do a bunch of receives (unless I am a leaf), and then I send the data to the parent
        // To understand at which step I must send the data, I need to check at which step I am 
        // reached by the root.
        // If this is step s, then I start sending at (num_steps - 1 - s)
        // To check if I am a leaf, I can just check if I am reached by the root in the last step
        int sending_step;
        if(root == this->rank){
            sending_step = this->num_steps;
        }else{
            sending_step = this->num_steps - 1 - tree.reached_at_step[this->rank];
        }

        DPRINTF("[%d] Sending step: %d\n", this->rank, sending_step);
        timer.reset("= swing_reduce_mpi (waiting recv)");
        char copied = 0;
        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            if(step < sending_step){
                // Receive from peer
                uint peer;
                if(env.reduce_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][this->num_steps - step - 1];         
                }else{  
                    peer = peers[port][step];                      
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    DPRINTF("[%d] Receiving from %d at step %d\n", this->rank, peer, step);
                    MPI_Recv(tmpbuf + offset_port_tmpbuf + step*count_port*dtsize, count_port, datatype, peer, TAG_SWING_REDUCE, comm, MPI_STATUS_IGNORE);                    
                    if(!copied){
                        // This instead of MPI_Reduce_local to avoid doing a memcpy form sendbuf to recvbuf at the beginning.
                        reduce_local(((char*) sendbuf) + offset_port, tmpbuf + offset_port_tmpbuf + step*count_port*dtsize, ((char*) recvbuf) + offset_port, count_port, datatype, op);
                        copied = 1;
                    }else{
                        MPI_Reduce_local(tmpbuf + offset_port_tmpbuf + step*count_port*dtsize, ((char*) recvbuf) + offset_port, count_port, datatype, op);
                    }
                }
            }else if(step == sending_step){
                // Send to parent
                uint peer = tree.parent[this->rank];            
                DPRINTF("[%d] Sending to %d at step %d\n", this->rank, peer, step);
                if(!copied){
                    MPI_Send(((char*) sendbuf) + offset_port, count_port, datatype, peer, TAG_SWING_REDUCE, comm);
                    copied = 1;
                }else{
                    MPI_Send(((char*) recvbuf) + offset_port, count_port, datatype, peer, TAG_SWING_REDUCE, comm);
                }
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_reduce_mpi (waiting all sends)");
        }
        
        free(peers[port]);
        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_mpi (writing profile data to file)");
    return res;
}


int SwingCommon::swing_reduce_redscat_gather_utofu(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.reduce_config.algo == SWING_REDUCE_ALGO_REDUCE_SCATTER_GATHER);
#endif
#ifdef FUGAKU
    assert(this->size > 2); // To work for two nodes we need to fix the tmpbuf/recvbuf in reduce_local below
    assert(env.reduce_config.distance_type == SWING_DISTANCE_INCREASING); // For now, we only support decreasing distance
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_redscat_gather_utofu (init)");
    Timer timer("swing_reduce_redscat_gather_utofu (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    
    // How big should the buffer be? I should consider the largest count possible
    // In case count not divisible by num ports, the first port will for sure have the largest blocks.
    // Thus, it is enough to check the count of the first block of the first port to know the largest block.
    // TODO: This work for reduce_block, generalize it to reduce
    size_t tmpbuf_size = count * dtsize * 2; // I need two buffers since the non-root ranks do not have the recvbuf
    
    timer.reset("= swing_reduce_redscat_gather_utofu (utofu buf reg)"); 
    // Also the root sends from tmpbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        assert(posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size) == 0);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, count*dtsize, recvbuf, count*dtsize, tmpbuf, tmpbuf_size, env.num_ports); 
    }else{
        tmpbuf = env.prealloc_buf;
        swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, count*dtsize, recvbuf, count*dtsize, NULL, 0, env.num_ports);         
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }
    timer.reset("= swing_reduce_redscat_gather_utofu (utofu buf exch)");           
    if(env.utofu_add_ag){
        swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
    }else{
        // TODO: Probably need to do this for all the ports for torus with different dimensions size
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, 0, env.reduce_config.algo_family, this->scc_real, peers[0]);
        swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
        
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
        if(mp != -1 && mp != 0){
            peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, mp, env.reduce_config.algo_family, this->scc_real, peers[mp]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
        }
    }

    char* tmpbuf2 = tmpbuf + count*dtsize;
    size_t offset_tmpbuf2 = count*dtsize;

    int res = MPI_SUCCESS; 
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_redscat_gather_utofu (computing trees)");
        swing_tree_t tree = get_tree(root, port, env.reduce_config.algo_family, env.reduce_config.distance_type, this->scc_real);

        size_t count_per_rank_per_port = count / (env.num_ports * this->size);
        size_t offset_port = count_per_rank_per_port * this->size * port * dtsize;
        size_t min_block_s, max_block_s;
        size_t min_block_r, max_block_r;
        min_block_r = min_block_s = 0;
        max_block_r = max_block_s = size;

        /******************/
        /* Reduce-Scatter */
        /******************/
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            uint peer;
            if(env.reduce_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{
                peer = peers[port][step];
            }

            min_block_s = min_block_r;
            max_block_s = max_block_r;
            size_t middle = (min_block_r + max_block_r + 1) / 2; // == min + (max - min) / 2
            if(tree.remapped_ranks[rank] < middle){
                min_block_s = middle;
                max_block_r = middle;
            }else{
                max_block_s = middle;
                min_block_r = middle;
            }
            size_t count_to_send = (max_block_s - min_block_s) * count_per_rank_per_port;
            size_t count_to_recv = (max_block_r - min_block_r) * count_per_rank_per_port;
            // Last block of last port is larger.
            if(max_block_s == this->size - 1 && port == env.num_ports - 1){
                count_to_send += (count % (env.num_ports * this->size));
            }
            if(max_block_r == this->size - 1 && port == env.num_ports - 1){
                count_to_recv += (count % (env.num_ports * this->size));
            }

            size_t offset_s = offset_port + min_block_s*count_per_rank_per_port*dtsize;
            size_t offset_r = offset_port + min_block_r*count_per_rank_per_port*dtsize;

            utofu_stadd_t lcl_addr, rmt_addr;
            if(step == 0){
                lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd + offset_s;
            }else{
                lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd + offset_s;
            }
            rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + offset_tmpbuf2 + offset_s; 

            size_t issued_sends = swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_send*dtsize, rmt_addr, step); 
            swing_utofu_wait_recv(utofu_descriptor, port, step, issued_sends - 1);
            swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
            
            if(step == 0){
                reduce_local((char*) sendbuf + offset_r, tmpbuf2 + offset_r, tmpbuf + offset_r, count_to_recv, datatype, op);
            }else{
                // Rank 0 in the last step receives directly in recvbuf since that
                // data is finalize and will not move anymore.
                if(step == this->num_steps - 1 && rank == root){
                    reduce_local(tmpbuf + offset_r, tmpbuf2 + offset_r, (char*) recvbuf + offset_r, count_to_recv, datatype, op);
                }else{
                    reduce_local(tmpbuf2 + offset_r, tmpbuf + offset_r, count_to_recv, datatype, op);
                }                
            }
        }    

        /**********/
        /* Gather */
        /**********/
        int sending_step;
        if(root == this->rank){
            sending_step = this->num_steps;
        }else{
            sending_step = this->num_steps - 1 - tree.reached_at_step[this->rank];
        }
        DPRINTF("[%d] Sending step: %d\n", this->rank, sending_step);
        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            if(step < sending_step){
                // Receive from peer
                uint peer;                
                // We need to do the opposite -- see the comment in gather code with the tree drawing
                if(env.reduce_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][step];                      
                }else{                      
                    peer = peers[port][this->num_steps - step - 1];                             
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    size_t min_block_r = tree.remapped_ranks[peer];
                    size_t max_block_r = tree.remapped_ranks_max[peer];            
                    size_t num_blocks = (max_block_r - min_block_r) + 1;                     
                    size_t count_to_recv = num_blocks * count_per_rank_per_port;
                    if(max_block_r == this->size - 1 && port == env.num_ports - 1){
                        count_to_recv += (count % (env.num_ports * this->size));
                    }
                    size_t segments_max_put_size = ceil((float) (count_to_recv*dtsize) / ((float) MAX_PUTGET_SIZE));
                    // + this->num_steps since we did already num_steps in the reduce-scatter phase
                    swing_utofu_wait_recv(utofu_descriptor, port, step + this->num_steps, segments_max_put_size - 1);
                }
            }else if(step == sending_step){
                // Send to parent
                uint peer = tree.parent[this->rank];            
                size_t min_block_s = tree.remapped_ranks[this->rank];
                size_t max_block_s = tree.remapped_ranks_max[this->rank];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                size_t offset_s = offset_port + min_block_s*count_per_rank_per_port*dtsize;
                size_t count_to_send = num_blocks * count_per_rank_per_port;
                if(max_block_s == this->size - 1 && port == env.num_ports - 1){
                    count_to_send += (count % (env.num_ports * this->size));
                }    
                
                utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd + offset_s;
                utofu_stadd_t rmt_addr;
                if(peer == root){
                    rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + offset_s;
                }else{
                    rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + offset_s;
                }
                DPRINTF("[%d] Sending [%d, %d] to %d at step %d\n", this->rank, min_block_s, max_block_s, peer, step);
                // + this->num_steps since we did already num_steps in the reduce-scatter phase
                size_t issued_sends = swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_send*dtsize, rmt_addr, step + this->num_steps); 
                swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);                
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_reduce_redscat_gather_utofu (waiting all sends)");
        }
        if(free_tmpbuf){
            swing_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, port);
        }
        free(peers[port]);
        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_redscat_gather_utofu (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif
}

int SwingCommon::swing_reduce_redscat_gather_mpi(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_config.algo == SWING_REDUCE_ALGO_REDUCE_SCATTER_GATHER);
#endif
    assert(this->size > 2); // To work for two nodes we need to fix the tmpbuf/recvbuf in reduce_local below
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_redscat_gather_mpi (init)");
    Timer timer("swing_reduce_redscat_gather_mpi (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    
    // How big should the buffer be? I should consider the largest count possible
    // In case count not divisible by num ports, the first port will for sure have the largest blocks.
    // Thus, it is enough to check the count of the first block of the first port to know the largest block.
    // TODO: This work for reduce_block, generalize it to reduce
    size_t tmpbuf_size = count * dtsize * 2; // I need two buffers since the non-root ranks do not have the recvbuf
    
    timer.reset("= swing_reduce_redscat_gather_mpi (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }
    char* tmpbuf2 = tmpbuf + count*dtsize;

    int res = MPI_SUCCESS; 
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_redscat_gather_mpi (computing trees)");
        swing_tree_t tree = get_tree(root, port, env.reduce_config.algo_family, env.reduce_config.distance_type, this->scc_real);

        size_t count_per_rank_per_port = count / (env.num_ports * this->size);
        size_t offset_port = count_per_rank_per_port * this->size * port * dtsize;
        size_t min_block_s, max_block_s;
        size_t min_block_r, max_block_r;
        min_block_r = min_block_s = 0;
        max_block_r = max_block_s = size;

        /******************/
        /* Reduce-Scatter */
        /******************/
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            uint peer;
            if(env.reduce_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{
                peer = peers[port][step];
            }

            min_block_s = min_block_r;
            max_block_s = max_block_r;
            size_t middle = (min_block_r + max_block_r + 1) / 2; // == min + (max - min) / 2
            if(tree.remapped_ranks[rank] < middle){
                min_block_s = middle;
                max_block_r = middle;
            }else{
                max_block_s = middle;
                min_block_r = middle;
            }
            size_t count_to_send = (max_block_s - min_block_s) * count_per_rank_per_port;
            size_t count_to_recv = (max_block_r - min_block_r) * count_per_rank_per_port;
            // Last block of last port is larger.
            if(max_block_s == this->size - 1 && port == env.num_ports - 1){
                count_to_send += (count % (env.num_ports * this->size));
            }
            if(max_block_r == this->size - 1 && port == env.num_ports - 1){
                count_to_recv += (count % (env.num_ports * this->size));
            }

            char *sbuf;
            if(step == 0){
                sbuf = (char*) sendbuf;
            }else{
                sbuf = tmpbuf;
            }

            size_t offset_s = offset_port + min_block_s*count_per_rank_per_port*dtsize;
            size_t offset_r = offset_port + min_block_r*count_per_rank_per_port*dtsize;

            MPI_Sendrecv(sbuf    + offset_s, count_to_send, datatype, peer, TAG_SWING_REDUCE, 
                         tmpbuf2 + offset_r, count_to_recv, datatype, peer, TAG_SWING_REDUCE, comm, MPI_STATUS_IGNORE);
            
            if(step == 0){
                reduce_local(sbuf + offset_r, tmpbuf2 + offset_r, tmpbuf + offset_r, count_to_recv, datatype, op);
            }else{
                // Rank 0 in the last step receives directly in recvbuf since that
                // data is finalize and will not move anymore.
                if(step == this->num_steps - 1 && rank == root){
                    reduce_local(tmpbuf + offset_r, tmpbuf2 + offset_r, (char*) recvbuf + offset_r, count_to_recv, datatype, op);
                }else{
                    MPI_Reduce_local(tmpbuf2 + offset_r, tmpbuf + offset_r, count_to_recv, datatype, op);
                }                
            }
        }    

        /**********/
        /* Gather */
        /**********/
        int sending_step;
        if(root == this->rank){
            sending_step = this->num_steps;
        }else{
            sending_step = this->num_steps - 1 - tree.reached_at_step[this->rank];
        }
        DPRINTF("[%d] Sending step: %d\n", this->rank, sending_step);
        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            if(step < sending_step){
                // Receive from peer
                uint peer;                
                // We need to do the opposite -- see the comment in gather code with the tree drawing
                if(env.reduce_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][step];                      
                }else{                      
                    peer = peers[port][this->num_steps - step - 1];                             
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    size_t min_block_r = tree.remapped_ranks[peer];
                    size_t max_block_r = tree.remapped_ranks_max[peer];            
                    size_t num_blocks = (max_block_r - min_block_r) + 1;                     
                    size_t count_to_recv = num_blocks * count_per_rank_per_port;
                    if(max_block_r == this->size - 1 && port == env.num_ports - 1){
                        count_to_recv += (count % (env.num_ports * this->size));
                    }
                    size_t offset_r = offset_port + min_block_r*count_per_rank_per_port*dtsize;
                    DPRINTF("[%d] Receiving [%d, %d] from %d at step %d at offset %d (count=%d)\n", this->rank, min_block_r, max_block_r, peer, step, offset_r, count_to_recv);
                    if(this->rank == root){
                        MPI_Recv((char*) recvbuf + offset_r, count_to_recv, datatype, peer, TAG_SWING_REDUCE, comm, MPI_STATUS_IGNORE);
                    }else{
                        MPI_Recv(tmpbuf + offset_r, count_to_recv, datatype, peer, TAG_SWING_REDUCE, comm, MPI_STATUS_IGNORE);
                    }
                }
            }else if(step == sending_step){
                // Send to parent
                uint peer = tree.parent[this->rank];            
                size_t min_block_s = tree.remapped_ranks[this->rank];
                size_t max_block_s = tree.remapped_ranks_max[this->rank];            
                size_t num_blocks = (max_block_s - min_block_s) + 1; 
                size_t offset_s = offset_port + min_block_s*count_per_rank_per_port*dtsize;
                size_t count_to_send = num_blocks * count_per_rank_per_port;
                if(max_block_s == this->size - 1 && port == env.num_ports - 1){
                    count_to_send += (count % (env.num_ports * this->size));
                }                
                DPRINTF("[%d] Sending [%d, %d] to %d at step %d\n", this->rank, min_block_s, max_block_s, peer, step);
                MPI_Send(tmpbuf + offset_s, count_to_send, datatype, peer, TAG_SWING_REDUCE, comm);
            }
            // Wait all the sends for this segment before moving to the next one
            timer.reset("= swing_reduce_redscat_gather_mpi (waiting all sends)");
        }
        free(peers[port]);
        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_redscat_gather_mpi (writing profile data to file)");
    return res;
}
