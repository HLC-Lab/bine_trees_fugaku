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
        swing_tree_t tree = get_tree(root, port, env.reduce_config.algo_family, env.reduce_config.distance_type, this->scc_real);

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
                    peer = peers[port][step];                               
                }else{  
                    peer = peers[port][this->num_steps - step - 1];
                }

                if(tree.parent[peer] == this->rank){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                    DPRINTF("[%d] Receiving from %d at step %d\n", this->rank, peer, step);

                    if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
                        // Do a 0-byte put to notify I am ready to recv
                        swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, utofu_descriptor->port_info[port].lcl_send_stadd + offset_port, 0, utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + offset_port, step);                    
                    }

                    size_t segments_max_put_size = ceil(count_port*dtsize / ((float) MAX_PUTGET_SIZE));
                    swing_utofu_wait_recv(utofu_descriptor, port, step, segments_max_put_size - 1);

                    if(!copied){
                        reduce_local(((char*) sendbuf) + offset_port, tmpbuf + offset_port_tmpbuf, ((char*) recvbuf) + offset_port, count_port, datatype, op);
                        copied = 1;
                    }else{
                        reduce_local(tmpbuf + offset_port_tmpbuf, ((char*) recvbuf) + offset_port, count_port, datatype, op);
                    }
                    
                    if(count*dtsize > SWING_REDUCE_NOSYNC_THRESHOLD){
                        swing_utofu_wait_sends(utofu_descriptor, port, 1);
                    }
                }
            }else if(step == sending_step){
                // Send to parent
                size_t issued_sends = 0;
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

                issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_port*dtsize, rmt_addr, step);
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
        swing_tree_t tree = get_tree(root, port, env.reduce_config.algo_family, env.reduce_config.distance_type, this->scc_real);

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
                    peer = peers[port][step];                               
                }else{  
                    peer = peers[port][this->num_steps - step - 1];
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
