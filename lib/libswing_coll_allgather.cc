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

int SwingCommon::swing_allgather_utofu_contiguous_threads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE);
#endif
#ifdef FUGAKU
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_utofu_contiguous (init)");
    Timer timer("swing_allgather_utofu_contiguous (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= swing_allgather_utofu_contiguous (utofu buf reg)"); 

    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_allgather_utofu_contiguous (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.allgather_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.allgather_config.algo_family, this->scc_real, peers[mp]);
                swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        // Still done to reset everything
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, NULL, 0, env.num_ports); 
        tmpbuf = env.prealloc_buf;
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_temp_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }

    int res = MPI_SUCCESS; 
    
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_allgather_utofu_contiguous (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type, this->scc_real);

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * port;
        memcpy(tmpbuf + tmpbuf_offset_port, ((char*) sendbuf) + blocks_info[port][0].offset, blocks_info[port][0].count*dtsize);

        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            uint peer;
            if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];                               
            }else{  
                peer = peers[port][step];
            }

            // Always send from the beginning of the buffer
            // and receive in the remaining part.
            timer.reset("= swing_allgather_utofu_contiguous (sendrecv)");
            size_t num_blocks = pow(2, step);
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;

            utofu_stadd_t lcl_addr, rmt_addr;
            lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd + tmpbuf_offset_port;
            rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + tmpbuf_offset_port + count_to_sendrecv*dtsize;
            
            DPRINTF("[%d] Sending/receiving %d bytes from %d\n", this->rank, count_to_sendrecv*dtsize, peer);
            
            size_t issued_sends = swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step);
            swing_utofu_wait_recv(utofu_descriptor, port, step, issued_sends - 1);
            swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
        }

        // For each port we need to permute back the data in the correct position
        timer.reset("= swing_allgather_utofu_contiguous (permute)");    
        swing_tree_t perm_tree = get_tree(rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving block %d to %d\n", this->rank, i, perm_tree.remapped_ranks[i]);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            size_t pos_in_tmpbuf_port = perm_tree.remapped_ranks[i]*blocks_info[port][0].count*dtsize;
            size_t pos_in_recvbuf = blocks_info[port][i].offset;
            
            memcpy(((char*) recvbuf) + pos_in_recvbuf, (char*) tmpbuf + tmpbuf_offset_port + pos_in_tmpbuf_port, blocks_info[port][i].count*dtsize);
        }    
        destroy_tree(&perm_tree);   
        
        free(peers[port]);
        destroy_tree(&tree);
        if(free_tmpbuf){
            swing_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, port);
        }        
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_allgather_utofu_contiguous (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif   
}


int SwingCommon::swing_allgather_utofu_contiguous_nothreads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE);
#endif
#ifdef FUGAKU
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_utofu_contiguous (init)");
    Timer timer("swing_allgather_utofu_contiguous (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    timer.reset("= swing_allgather_utofu_contiguous (utofu buf reg)"); 

    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_allgather_utofu_contiguous (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.allgather_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.allgather_config.algo_family, this->scc_real, peers[mp]);
                swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        // Still done to reset everything
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, NULL, 0, env.num_ports); 
        tmpbuf = env.prealloc_buf;
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_temp_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }

    int res = MPI_SUCCESS; 
    size_t tmpbuf_offset_port[LIBSWING_MAX_SUPPORTED_PORTS];

    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
        }        

        tmpbuf_offset_port[port] = (tmpbuf_size / env.num_ports) * port;
        memcpy(tmpbuf + tmpbuf_offset_port[port], ((char*) sendbuf) + blocks_info[port][0].offset, blocks_info[port][0].count*dtsize);              
    }
    size_t issued_sends[LIBSWING_MAX_SUPPORTED_PORTS];

    for(size_t step = 0; step < (uint) this->num_steps; step++){
        for(size_t port = 0; port < env.num_ports; port++){
            uint peer;
            if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];                               
            }else{  
                peer = peers[port][step];
            }
            // Always send from the beginning of the buffer
            // and receive in the remaining part.
            timer.reset("= swing_allgather_utofu_contiguous (sendrecv)");
            size_t num_blocks = pow(2, step);
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;

            utofu_stadd_t lcl_addr, rmt_addr;
            lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd + tmpbuf_offset_port[port];
            rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + tmpbuf_offset_port[port] + count_to_sendrecv*dtsize;
            
            DPRINTF("[%d] Sending/receiving %d bytes from %d\n", this->rank, count_to_sendrecv*dtsize, peer);
            
            issued_sends[port] = swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step);
        }
        for(size_t port = 0; port < env.num_ports; port++){
            swing_utofu_wait_recv(utofu_descriptor, port, step, issued_sends[port] - 1);
            swing_utofu_wait_sends(utofu_descriptor, port, issued_sends[port]);
        }        
    }

    for(size_t port = 0; port < env.num_ports; port++){        
        // For each port we need to permute back the data in the correct position
        timer.reset("= swing_allgather_utofu_contiguous (permute)");    
        swing_tree_t perm_tree = get_tree(rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving block %d to %d\n", this->rank, i, perm_tree.remapped_ranks[i]);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 4 6 8 9 12 13, and port 1 on 2 3 6 7 10 11 14 15
            size_t pos_in_tmpbuf_port = perm_tree.remapped_ranks[i]*blocks_info[port][0].count*dtsize;
            size_t pos_in_recvbuf = blocks_info[port][i].offset;
            
            memcpy(((char*) recvbuf) + pos_in_recvbuf, (char*) tmpbuf + tmpbuf_offset_port[port] + pos_in_tmpbuf_port, blocks_info[port][i].count*dtsize);
        }    
        destroy_tree(&perm_tree);   
        free(peers[port]);
        if(free_tmpbuf){
            swing_utofu_dereg_buf(this->utofu_descriptor, tmpbuf, port);
        }                
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_allgather_utofu_contiguous (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif   
}

int SwingCommon::swing_allgather_utofu_contiguous(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
    if(env.use_threads){
        return swing_allgather_utofu_contiguous_threads(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }else{
        return swing_allgather_utofu_contiguous_nothreads(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }
}

int SwingCommon::swing_allgather_mpi_contiguous(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE);
#endif
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_mpi_contiguous (init)");
    Timer timer("swing_allgather_mpi_contiguous (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We always need a tempbuf since non root ranks might not specify a recvbuf
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    size_t tmpbuf_size = ceil((float) sendcount / env.num_ports)*env.num_ports*dtsize*this->size;
    
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }

    int res = MPI_SUCCESS; 

    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_allgather_mpi_contiguous (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type, this->scc_real);

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * port;
        memcpy(tmpbuf + tmpbuf_offset_port, ((char*) sendbuf) + blocks_info[port][0].offset, blocks_info[port][0].count*dtsize);              

        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            uint peer;
            if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];                               
            }else{  
                peer = peers[port][step];
            }

            // Always send from the beginning of the buffer
            // and receive in the remaining part.
            timer.reset("= swing_allgather_mpi_contiguous (sendrecv)");
            size_t num_blocks = pow(2, step);
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;
            MPI_Sendrecv(tmpbuf + tmpbuf_offset_port                           , count_to_sendrecv, sendtype, peer, TAG_SWING_ALLGATHER, 
                         tmpbuf + tmpbuf_offset_port + count_to_sendrecv*dtsize, count_to_sendrecv, sendtype, peer, TAG_SWING_ALLGATHER, 
                         comm, MPI_STATUS_IGNORE);                                   
        }

        // For each port we need to permute back the data in the correct position
        timer.reset("= swing_allgather_mpi_contiguous (permute)");    
        swing_tree_t perm_tree = get_tree(rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving block %d to %d\n", this->rank, i, perm_tree.remapped_ranks[i]);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            size_t pos_in_tmpbuf_port = perm_tree.remapped_ranks[i]*blocks_info[port][0].count*dtsize;
            size_t pos_in_recvbuf = blocks_info[port][i].offset;
            
            memcpy(((char*) recvbuf) + pos_in_recvbuf, (char*) tmpbuf + tmpbuf_offset_port + pos_in_tmpbuf_port, blocks_info[port][i].count*dtsize);
        }    
        destroy_tree(&perm_tree);   
        
        free(peers[port]);
        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_allgather_mpi_contiguous (writing profile data to file)");
    return res;
}

int SwingCommon::swing_allgather_send_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_SEND);
#endif
#ifdef FUGAKU
    assert(env.num_ports == 1); // TODO: Implement the case where env.num_ports > 1. It would require memcpys, so probably does not make sense ... (just use cont_permute ...)
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_utofu_cont_send (init)");
    Timer timer("swing_allgather_utofu_cont_send (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We don't need a tmpbuf since we do not need to reorder
    // We can do everything with recvbuf
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    size_t port = 0;
    // Compute the peers of this port if I did not do it yet
    if(peers[port] == NULL){
        peers[port] = (uint*) malloc(sizeof(uint)*(this->num_steps)); 
        compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
    }        
    timer.reset("= swing_allgather_utofu_cont_send (computing trees)");
    // We need to invert everything for the usual reason of the allgather etc...
    // The remapping must be the same for all the ranks so all trees must be built for the same rank (0 in this case)
    uint root = 0;
    swing_tree_t tree = get_tree(root, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);
    uint peer_send;
    for(size_t i = 0; i < this->size; i++){
        if(tree.remapped_ranks[i] == rank){
            peer_send = i;
            break;
        }
    }            

    int res = MPI_SUCCESS; 
    DPRINTF("[%d] Permuting send. Sending to %d and receiving from %d\n", rank, peer_send, tree.remapped_ranks[rank]); 
    timer.reset("= swing_allgather_utofu_contiguous (utofu buf reg+exch)");   
    swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, sendcount*dtsize, recvbuf, ((size_t) recvcount)*dtsize*this->size, NULL, 0, env.num_ports); 
    // + 1 because of the peer I talk to at the beginning to permute
    swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[port]);                 
    // I should receive the block I am remapped to from the rank I am remapped to,
    // and send my block to the rank it is remapped as my rank (used the inverse remapping) 
    // i.e., send to inverse_remapping[rank] and received from remapped_ranks[rank]
    // Rank 0 does not need to do anything because it is 0 also in the remapped tree
    if(rank != root){
        // Send my addresses to tree.remapped_ranks[rank]
        // and receive the addresses of peer_send
        uint64_t sbuffer[2] = {this->utofu_descriptor->port_info[port].lcl_recv_stadd, this->utofu_descriptor->port_info[port].lcl_temp_stadd};
        uint64_t rbuffer[2];
        MPI_Request req;
        MPI_Isend(sbuffer, 2, MPI_UINT64_T, tree.remapped_ranks[rank], 0, MPI_COMM_WORLD, &req);
        MPI_Recv(rbuffer, 2, MPI_UINT64_T, peer_send, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        this->utofu_descriptor->port_info[port].rmt_recv_stadd[peer_send] = rbuffer[0];
        this->utofu_descriptor->port_info[port].rmt_temp_stadd[peer_send] = rbuffer[1];
        assert(tree.remapped_ranks[peer_send] == rank);
        utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd;
        utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer_send] + tree.remapped_ranks[peer_send]*sendcount*dtsize;
        size_t issued_sends = swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer_send]), port, peer_send, lcl_addr, sendcount*dtsize, rmt_addr, 0);
        swing_utofu_wait_recv(utofu_descriptor, port, 0, issued_sends - 1);
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
    }else{
        memcpy(recvbuf, sendbuf, sendcount*dtsize);
    }
    
    DPRINTF("[%d] Permutation done\n", rank); 
    
    size_t num_blocks = 1;
    size_t min_block_resident = tree.remapped_ranks[this->rank];
    size_t min_block_r;        
    for(size_t step = 0; step < (uint) this->num_steps; step++){        
        uint peer;            
        if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - step - 1];                         
        }else{  
            peer = peers[port][step];
        }

        size_t count_to_sendrecv = num_blocks*sendcount;

        // The data I am going to receive contains the block
        // with id equal to the remapped rank of my peer,
        // and is aligned to a power of 2^step
        // Thus, I need to do proper masking to get the block id
        // i.e., I need to set to 0 the least significant step bits
        min_block_r = tree.remapped_ranks[peer] & ~((1 << step) - 1);

        utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + min_block_resident*sendcount*dtsize;
        utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + min_block_resident*sendcount*dtsize;;

        DPRINTF("[%d] Sending/receiving %d bytes to %d offset %d\n", this->rank, count_to_sendrecv*dtsize, peer, min_block_resident);

        // step + 1 because I did already a transmission before starting with the actual steps
        size_t issued_sends = swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step + 1);
        swing_utofu_wait_recv(utofu_descriptor, port, step + 1, issued_sends - 1);
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
                        
        min_block_resident = std::min(min_block_resident, min_block_r);
        num_blocks *= 2;
    }         
    free(peers[port]);
    destroy_tree(&tree);
    timer.reset("= swing_allgather_utofu_cont_send (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif
}

int SwingCommon::swing_allgather_send_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_SEND);
#endif
    assert(env.num_ports == 1);
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_mpi_cont_send (init)");
    Timer timer("swing_allgather_mpi_cont_send (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We don't need a tmpbuf since we do not need to reorder
    // We can do everything with recvbuf
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    int res = MPI_SUCCESS; 

    size_t port = 0;

    // Compute the peers of this port if I did not do it yet
    if(peers[port] == NULL){
        peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
    }        
    timer.reset("= swing_allgather_mpi_cont_send (computing trees)");
    // We need to invert everything for the usual reason of the allgather etc...
    // The remapping must be the same for all the ranks so all trees must be built for the same rank (0 in this case)
    swing_tree_t tree = get_tree(0, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);
    size_t remapped_rank = tree.remapped_ranks[rank];
    uint* inverse_remapping = (uint*) malloc(sizeof(uint)*this->size);
    for(size_t i = 0; i < this->size; i++){
        inverse_remapping[tree.remapped_ranks[i]] = i;
    }
    
    // I should receive the block I am remapped to from the rank I am remapped to,
    // and send my block to the rank it is remapped as my rank (used the inverse remapping) 
    // i.e., send to inverse_remapping[rank] and received from remapped_ranks[rank]
    DPRINTF("[%d] Remap: Sending to %d and receiving from %d\n", rank, inverse_remapping[rank], remapped_rank);
    MPI_Sendrecv((char*) sendbuf                                 , sendcount, sendtype, inverse_remapping[rank], TAG_SWING_ALLGATHER, 
                 (char*) recvbuf + remapped_rank*sendcount*dtsize, sendcount, sendtype, remapped_rank          , TAG_SWING_ALLGATHER, 
                 comm, MPI_STATUS_IGNORE);
    free(inverse_remapping);
    
    size_t recvbuf_offset_port = ((recvcount*dtsize) / env.num_ports) * port;        
    size_t num_blocks = 1;
    size_t min_block_resident = tree.remapped_ranks[this->rank];
    size_t min_block_r;        
    for(size_t step = 0; step < (uint) this->num_steps; step++){        
        uint peer;            
        if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - step - 1];                         
        }else{  
            peer = peers[port][step];
        }

        size_t count_to_sendrecv = num_blocks*sendcount;
        // The data I am going to receive contains the block
        // with id equal to the remapped rank of my peer,
        // and is aligned to a power of 2^step
        // Thus, I need to do proper masking to get the block id
        // i.e., I need to set to 0 the least significant step bits
        min_block_r = tree.remapped_ranks[peer] & ~((1 << step) - 1);


        DPRINTF("[%d] Sending to %d and receiving from %d\n", rank, peer, peer);
        MPI_Sendrecv((char*) recvbuf + recvbuf_offset_port + min_block_resident*sendcount*dtsize, count_to_sendrecv, sendtype, peer, TAG_SWING_ALLGATHER, 
                     (char*) recvbuf + recvbuf_offset_port + min_block_r*sendcount*dtsize       , count_to_sendrecv, sendtype, peer, TAG_SWING_ALLGATHER, 
                        comm, MPI_STATUS_IGNORE);     
                        
        min_block_resident = std::min(min_block_resident, min_block_r);
        num_blocks *= 2;
    }         
    free(peers[port]);
    destroy_tree(&tree);

    timer.reset("= swing_allgather_mpi_cont_send (writing profile data to file)");
    return res;
}


#if 1
int SwingCommon::swing_allgather_blocks_utofu_nothreads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS);
#endif
#ifdef FUGAKU
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    Timer timer("swing_allgather_blocks_utofu (init)");
    timer.reset("= swing_allgather_blocks_utofu (serial fraction)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We don't need a tmpbuf since we do not need to reorder
    // We can do everything with recvbuf
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, sendcount*dtsize, recvbuf, recvcount*this->size*dtsize, NULL, 0, env.num_ports);     
    if(env.utofu_add_ag){
        swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
    }else{
        // TODO: Probably need to do this for all the ports for torus with different dimensions size
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, 0, env.allgather_config.algo_family, this->scc_real, peers[0]);
        swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
        
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
        if(mp != -1 && mp != 0){
            peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, mp, env.allgather_config.algo_family, this->scc_real, peers[mp]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
        }
    }

    //timer.reset("= swing_allgather_blocks_utofu (memcpy)");           
    memcpy((char*) recvbuf + this->rank*sendcount*dtsize, sendbuf, sendcount*dtsize);

    swing_tree_t trees[LIBSWING_MAX_SUPPORTED_PORTS];
    // resident_blocks[i] == 0 => block i is not resident
    // resident_blocks[i] == 1 => we must wait for block i to be received
    // resident_blocks[i] == 2 => block i has been received
    char* resident_blocks[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t segments_max_put_size[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t issued_sends[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t issued_recvs[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t expected_recv[LIBSWING_MAX_SUPPORTED_PORTS];

    // Do the first step
    for(size_t port = 0; port < env.num_ports; port++){
        //Timer timer("swing_allgather_blocks_utofu (init)");
        // Compute the peers of this port if I did not do it yet
        timer.reset("= swing_allgather_blocks_utofu (computing peers)");
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_allgather_blocks_utofu (computing trees)");
        trees[port] = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);


        resident_blocks[port] = (char*) calloc(this->size, sizeof(char)); 
        
        timer.reset("= swing_allgather_blocks_utofu (first send)");
        issued_sends[port] = 0;
        issued_recvs[port] = 0;

        // Do send for the first step, and compute what to receive
        uint peer;            
        if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - 0 - 1];                         
        }else{  
            peer = peers[port][0];
        }

        utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][this->rank].offset;
        utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][this->rank].offset;
        issued_sends[port] += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][this->rank].count*dtsize, rmt_addr, 0);
        resident_blocks[port][this->rank] = 2;
        resident_blocks[port][peer] = 1;

        segments_max_put_size[port] = ceil(blocks_info[port][peer].count*dtsize / ((float) MAX_PUTGET_SIZE));
        issued_recvs[port] = segments_max_put_size[port];
        expected_recv[port] = segments_max_put_size[port];
    }

    // For each step, I wait for the blocks that were supposed to be received in the previous step
    // and I send each of those blocks, as well as the other blocks that were already resident.
    for(size_t step = 1; step < (uint) this->num_steps; step++){            
        memset(issued_recvs, 0, sizeof(size_t)*env.num_ports);

        for(size_t block = 0; block < this->size; block++){
            for(size_t port = 0; port < env.num_ports; port++){
                uint peer;            
                if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][this->num_steps - step - 1];                         
                }else{  
                    peer = peers[port][step];
                }

                if(resident_blocks[port][block] == 1){
                    timer.reset("= swing_allgather_blocks_utofu (waiting for recv)");
                    // We must wait for this block (from previous step)                    
                    swing_utofu_wait_recv(utofu_descriptor, port, step - 1, expected_recv[port] - 1);
                    expected_recv[port] += segments_max_put_size[port];
                    resident_blocks[port][block] = 2;
                }

                if(resident_blocks[port][block] == 2){
                    timer.reset("= swing_allgather_blocks_utofu (sending)");
                    // Block is here, send it
                    utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][block].offset;
                    utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][block].offset;
                    issued_sends[port] += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);
                }    

                if(trees[port].subtree_roots[block] == peer){
                    issued_recvs[port] += segments_max_put_size[port];
                    resident_blocks[port][block] = 1; // Wait for this block at the next step
                }                    
            }    
        }
            
        // How much will the block be segmented?
        for(size_t port = 0; port < env.num_ports; port++){
            expected_recv[port] = segments_max_put_size[port];            
        }
    }

    for(size_t port = 0; port < env.num_ports; port++){
        swing_utofu_wait_recv(utofu_descriptor, port, this->num_steps - 1, issued_recvs[port] - 1);
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends[port]);
        free(peers[port]);
        free(resident_blocks[port]);
        destroy_tree(&trees[port]);
    }

    return MPI_SUCCESS;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif
}

#else

int SwingCommon::swing_allgather_blocks_utofu_nothreads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS);
#endif
#ifdef FUGAKU
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    Timer timer("swing_allgather_blocks_utofu (init)");
    timer.reset("= swing_allgather_blocks_utofu (serial fraction)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We don't need a tmpbuf since we do not need to reorder
    // We can do everything with recvbuf
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, sendcount*dtsize, recvbuf, recvcount*this->size*dtsize, NULL, 0, env.num_ports);     
    if(env.utofu_add_ag){
        swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
    }else{
        // TODO: Probably need to do this for all the ports for torus with different dimensions size
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, 0, env.allgather_config.algo_family, this->scc_real, peers[0]);
        swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
        
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
        if(mp != -1 && mp != 0){
            peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, mp, env.allgather_config.algo_family, this->scc_real, peers[mp]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
        }
    }

    //timer.reset("= swing_allgather_blocks_utofu (memcpy)");           
    memcpy((char*) recvbuf + this->rank*sendcount*dtsize, sendbuf, sendcount*dtsize);

    swing_tree_t trees[LIBSWING_MAX_SUPPORTED_PORTS];
    // resident_blocks[i] == 0 => block i is not resident
    // resident_blocks[i] == 1 => we must wait for block i to be received
    // resident_blocks[i] == 2 => block i has been received
    char* resident_blocks[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t segments_max_put_size[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t issued_sends[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t issued_recvs[LIBSWING_MAX_SUPPORTED_PORTS];
    size_t expected_recv[LIBSWING_MAX_SUPPORTED_PORTS];

    // Do the first step
    for(size_t port = 0; port < env.num_ports; port++){
        //Timer timer("swing_allgather_blocks_utofu (init)");
        // Compute the peers of this port if I did not do it yet
        timer.reset("= swing_allgather_blocks_utofu (computing peers)");
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_allgather_blocks_utofu (computing trees)");
        trees[port] = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);


        resident_blocks[port] = (char*) calloc(this->size, sizeof(char)); 
        
        timer.reset("= swing_allgather_blocks_utofu (first send)");
        issued_sends[port] = 0;
        issued_recvs[port] = 0;

        // Do send for the first step, and compute what to receive
        uint peer;            
        if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - 0 - 1];                         
        }else{  
            peer = peers[port][0];
        }

        utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][this->rank].offset;
        utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][this->rank].offset;
        issued_sends[port] += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][this->rank].count*dtsize, rmt_addr, 0);
        resident_blocks[port][this->rank] = 2;
        resident_blocks[port][peer] = 1;

        segments_max_put_size[port] = ceil(blocks_info[port][peer].count*dtsize / ((float) MAX_PUTGET_SIZE));
        issued_recvs[port] = segments_max_put_size[port];
        expected_recv[port] = segments_max_put_size[port];
    }

    // For each step, I wait for the blocks that were supposed to be received in the previous step
    // and I send each of those blocks, as well as the other blocks that were already resident.
    for(size_t step = 1; step < (uint) this->num_steps; step++){            
        for(size_t block = 0; block < this->size; block++){
            for(size_t port = 0; port < env.num_ports; port++){
                uint peer;            
                if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][this->num_steps - step - 1];                         
                }else{  
                    peer = peers[port][step];
                }

                if(resident_blocks[port][block] == 2){
                    timer.reset("= swing_allgather_blocks_utofu (sending)");
                    // Block is here, send it
                    utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][block].offset;
                    utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][block].offset;
                    issued_sends[port] += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);
                }    
            }    
        }
        
        memset(issued_recvs, 0, sizeof(size_t)*env.num_ports);

        for(size_t block = 0; block < this->size; block++){
            for(size_t port = 0; port < env.num_ports; port++){
                uint peer;            
                if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                    peer = peers[port][this->num_steps - step - 1];                         
                }else{  
                    peer = peers[port][step];
                }

                if(resident_blocks[port][block] == 1){
                    timer.reset("= swing_allgather_blocks_utofu (waiting for recv)");
                    // We must wait for this block (from previous step)                    
                    swing_utofu_wait_recv(utofu_descriptor, port, step - 1, expected_recv[port] - 1);
                    expected_recv[port] += segments_max_put_size[port];
                    resident_blocks[port][block] = 2;

                    timer.reset("= swing_allgather_blocks_utofu (sending)");
                    // Block is here, send it
                    utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][block].offset;
                    utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][block].offset;
                    issued_sends[port] += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);                    
                }            
                
                if(trees[port].subtree_roots[block] == peer){
                    issued_recvs[port] += segments_max_put_size[port];
                    resident_blocks[port][block] = 1; // Wait for this block at the next step
                }                    
            }    
        }        
        // How much will the block be segmented?
        for(size_t port = 0; port < env.num_ports; port++){
            expected_recv[port] = segments_max_put_size[port];            
        }
    }

    for(size_t port = 0; port < env.num_ports; port++){
        swing_utofu_wait_recv(utofu_descriptor, port, this->num_steps - 1, issued_recvs[port] - 1);
        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends[port]);
        free(peers[port]);
        free(resident_blocks[port]);
        destroy_tree(&trees[port]);
    }

    return MPI_SUCCESS;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif
}
#endif

int SwingCommon::swing_allgather_blocks_utofu_threads(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS);
#endif
#ifdef FUGAKU
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    Timer timer("swing_allgather_blocks_utofu (init)");
    timer.reset("= swing_allgather_blocks_utofu (serial fraction)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We don't need a tmpbuf since we do not need to reorder
    // We can do everything with recvbuf
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    swing_utofu_reg_buf(this->utofu_descriptor, sendbuf, sendcount*dtsize, recvbuf, recvcount*this->size*dtsize, NULL, 0, env.num_ports);     
    if(env.utofu_add_ag){
        swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
    }else{
        // TODO: Probably need to do this for all the ports for torus with different dimensions size
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, 0, env.allgather_config.algo_family, this->scc_real, peers[0]);
        swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
        
        // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
        int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
        if(mp != -1 && mp != 0){
            peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, mp, env.allgather_config.algo_family, this->scc_real, peers[mp]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
        }
    }

    //timer.reset("= swing_allgather_blocks_utofu (memcpy)");           
    //memcpy((char*) recvbuf + this->rank*sendcount*dtsize, sendbuf, sendcount*dtsize);

    int res = MPI_SUCCESS; 
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        //Timer timer("swing_allgather_blocks_utofu (init)");
        // Compute the peers of this port if I did not do it yet
        timer.reset("= swing_allgather_blocks_utofu (computing peers)");
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_allgather_blocks_utofu (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);

        // resident_blocks[i] == 0 => block i is not resident
        // resident_blocks[i] == 1 => we must wait for block i to be received
        // resident_blocks[i] == 2 => block i has been received
        char* resident_blocks = (char*) calloc(this->size, sizeof(char)); 
        
        timer.reset("= swing_allgather_blocks_utofu (first send)");
        size_t issued_sends = 0, issued_recvs = 0;
        // Do send for the first step, and compute what to receive
        uint peer;            
        if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - 0 - 1];                         
        }else{  
            peer = peers[port][0];
        }
        
        utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_send_stadd       + blocks_info[port][0].offset;
        utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][this->rank].offset;
        issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][this->rank].count*dtsize, rmt_addr, 0);
        resident_blocks[this->rank] = 2;
        resident_blocks[peer] = 1;

#pragma omp critical
{
        memcpy((char*) recvbuf + blocks_info[port][this->rank].offset, (char*) sendbuf +  blocks_info[port][0].offset, blocks_info[port][this->rank].count*dtsize);
}

        size_t bytes_to_recv = blocks_info[port][peer].count*dtsize;
        size_t segments_max_put_size = ceil(bytes_to_recv / ((float) MAX_PUTGET_SIZE));
        issued_recvs = segments_max_put_size;
        size_t expected_recv = segments_max_put_size;

        // For each step, I wait for the blocks that were supposed to be received in the previous step
        // and I send each of those blocks, as well as the other blocks that were already resident.
        for(size_t step = 1; step < (uint) this->num_steps; step++){        
            issued_recvs = 0;
            timer.reset("= swing_allgather_blocks_utofu (step " + std::to_string(step) + ")");
            uint peer;            
            if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];                         
            }else{  
                peer = peers[port][step];
            }
            
            #if 0
            for(size_t block = 0; block < this->size; block++){
                if(resident_blocks[block] == 1){
                    timer.reset("= swing_allgather_blocks_utofu (waiting for recv)");
                    // We must wait for this block (from previous step)                    
                    swing_utofu_wait_recv(utofu_descriptor, port, step - 1, expected_recv - 1);
                    // What's the ID of the next recv?
                    bytes_to_recv = blocks_info[port][peer].count*dtsize;
                    segments_max_put_size = ceil(bytes_to_recv / ((float) MAX_PUTGET_SIZE));                    
                    expected_recv += segments_max_put_size;
                    resident_blocks[block] = 2;
                }
                
                if(resident_blocks[block] == 2){
                    timer.reset("= swing_allgather_blocks_utofu (sending)");
                    // Block is here, send it
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][block].offset;
                    rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][block].offset;
                    issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);
                }
            }
            #else

            char push_every = 1; //this->size;
            uint posted_sends = 0;

            for(size_t block = 0; block < this->size; block++){
                if(resident_blocks[block] == 2){
                    timer.reset("= swing_allgather_blocks_utofu (sending)");
                    // Block is here, send it
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][block].offset;
                    rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][block].offset;
                    ++posted_sends;

                    if(posted_sends % push_every == 0){
                        issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);
                    }else{
                        issued_sends += swing_utofu_isend_delayed(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);
                    }
                }                
            }
            // Just to push all the posted sends (0-byte put) (unless I didn't do it already for the last send)
            if(posted_sends % push_every){
                issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, 0, rmt_addr, step);
            }

            for(size_t block = 0; block < this->size; block++){
                if(resident_blocks[block] == 1){
                    timer.reset("= swing_allgather_blocks_utofu (waiting for recv)");
                    // We must wait for this block (from previous step)                    
                    swing_utofu_wait_recv(utofu_descriptor, port, step - 1, expected_recv - 1);
                    expected_recv += segments_max_put_size;
                    resident_blocks[block] = 2;

                    timer.reset("= swing_allgather_blocks_utofu (sending)");
                    // Block is here, send it
                    lcl_addr = utofu_descriptor->port_info[port].lcl_recv_stadd       + blocks_info[port][block].offset;
                    rmt_addr = utofu_descriptor->port_info[port].rmt_recv_stadd[peer] + blocks_info[port][block].offset;
                    issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, blocks_info[port][block].count*dtsize, rmt_addr, step);                    
                }            
                
                if(tree.subtree_roots[block] == peer){
                    issued_recvs += segments_max_put_size;
                    resident_blocks[block] = 1; // Wait for this block at the next step
                }                
            }

            #endif            
            // How much will the block be segmented?
            expected_recv = segments_max_put_size;            
        }    

        timer.reset("= swing_allgather_blocks_utofu (last recvs waiting)");
        // Now wait for all the recvs that were issued in the last step.
        swing_utofu_wait_recv(utofu_descriptor, port, this->num_steps - 1, issued_recvs - 1);
        DPRINTF("[%d] All receives waited\n", this->rank);

        swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);
        DPRINTF("[%d] All sends waited\n", this->rank);
        free(peers[port]);
        destroy_tree(&tree);
        free(resident_blocks);
        timer.reset("= swing_allgather_blocks_utofu (writing profile data to file)");
    }    
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif
}

int SwingCommon::swing_allgather_blocks_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
    if(env.use_threads){
        return swing_allgather_blocks_utofu_threads(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }else{
        return swing_allgather_blocks_utofu_nothreads(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }
}


#if 1

static uint32_t btonb(int32_t bin) {
    if (bin > 0x55555555) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask + bin) ^ mask;
}

static int32_t nbtob(uint32_t neg) {
    //const int32_t even = 0x2AAAAAAA, odd = 0x55555555;
    //if ((neg & even) > (neg & odd)) throw std::overflow_error("value out of range");
    const uint32_t mask = 0xAAAAAAAA;
    return (mask ^ neg) - mask;
}

static inline uint32_t reverse(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}


static inline int in_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}


static inline uint32_t nb_to_nu(uint32_t nb, uint32_t size){
    return reverse(nb ^ (nb >> 1)) >> 32 - ceil_log2(size);
}

static inline uint32_t get_nu(uint32_t rank, uint32_t size){
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
    size_t num_bits = ceil_log2(size);
    if(rank % 2){
        if(in_range(rank, num_bits)){
            nba = btonb(rank);
        }
        if(in_range(rank - size, num_bits)){
            nbb = btonb(rank - size);
        }
    }else{
        if(in_range(-rank, num_bits)){
            nba = btonb(-rank);
        }
        if(in_range(-rank + size, num_bits)){
            nbb = btonb(-rank + size);
        }
    }
    assert(nba != UINT32_MAX || nbb != UINT32_MAX);

    if(nba == UINT32_MAX && nbb != UINT32_MAX){
        return nb_to_nu(nbb, size);
    }else if(nba != UINT32_MAX && nbb == UINT32_MAX){
        return nb_to_nu(nba, size);
    }else{ // Check MSB
        int nu_a = nb_to_nu(nba, size);
        int nu_b = nb_to_nu(nbb, size);
        if(nu_a < nu_b){
            return nu_a;
        }else{
            return nu_b;
        }
    }
}


int SwingCommon::swing_allgather_blocks_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS);
#endif    
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    int size, rank, dtsize;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Type_size(recvtype, &dtsize);
    memcpy((char*) recvbuf + sendcount*rank*dtsize, sendbuf, sendcount*dtsize);  
    
    int mask = 0x1;
    int inverse_mask = 0x1 << (int) (ceil(log2(size)) - 1);
    int block_first_mask = ~(inverse_mask - 1);
    int step = 0;
    while(inverse_mask > 0){
      int partner;
      if(rank % 2 == 0){
          partner = mod(rank + nbtob((inverse_mask << 1) - 1), size); 
      }else{
          partner = mod(rank - nbtob((inverse_mask << 1) - 1), size); 
      }   
    
      // We start from 1 because 0 never sends block 0
      for(size_t block = 1; block < size; block++){
          // Get the position of the highest set bit using clz
          // That gives us the first at which block departs from 0
          int k = 31 - __builtin_clz(get_nu(block, size));
          //int k = __builtin_ctz(get_nu(block, size));
          // Check if this must be sent (recvd in allgather)
          if(k == step || block == 0){
              // 0 would send this block
              size_t block_to_send, block_to_recv;
              // I invert what to send and what to receive wrt reduce-scatter
              if(rank % 2 == 0){
                  // I am even, thus I need to shift by rank position to the right
                  block_to_recv = mod(block + rank, size);
                  // What to receive? What my partner is sending
                  // Since I am even, my partner is odd, thus I need to mirror it and then shift
                  block_to_send = mod(partner - block, size);
              }else{
                  // I am odd, thus I need to mirror it
                  block_to_recv = mod(rank - block, size);
                  // What to receive? What my partner is sending
                  // Since I am odd, my partner is even, thus I need to mirror it and then shift   
                  block_to_send = mod(block + partner, size);
              }

              int partner_send = (block_to_send != partner) ? partner : MPI_PROC_NULL;
              int partner_recv = (block_to_recv != rank)    ? partner : MPI_PROC_NULL;

              MPI_Sendrecv((char*) recvbuf + block_to_send*sendcount*dtsize, sendcount, sendtype, partner_send, 0,
                           (char*) recvbuf + block_to_recv*recvcount*dtsize, recvcount, recvtype, partner_recv, 0,
                           comm, MPI_STATUS_IGNORE);
          }        
      }
    
      mask <<= 1;
      inverse_mask >>= 1;
      block_first_mask >>= 1;
      step++;
    }
    return MPI_SUCCESS;
}

#endif

#if 0
int SwingCommon::swing_allgather_blocks_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.allgather_config.algo_family == SWING_ALGO_FAMILY_SWING || env.allgather_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.allgather_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS);
#endif    
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_blocks_mpi (init)");
    Timer timer("swing_allgather_blocks_mpi (init)");
    int dtsize;
    MPI_Type_size(sendtype, &dtsize);    

    // We don't need a tmpbuf since we do not need to reorder
    // We can do everything with recvbuf
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);

    int res = MPI_SUCCESS; 

    size_t port = 0;

    // Compute the peers of this port if I did not do it yet
    if(peers[port] == NULL){
        peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
        compute_peers(this->rank, port, env.allgather_config.algo_family, this->scc_real, peers[port]);
    }        
    timer.reset("= swing_allgather_blocks_mpi (computing trees)");
    swing_tree_t tree = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);

    memcpy((char*) recvbuf + this->rank*sendcount*dtsize, sendbuf, sendcount*dtsize);
    MPI_Request* reqs_send = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size);
    MPI_Request* reqs_recv = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size);

    char* resident_blocks = (char*) calloc(this->size, sizeof(char));
    resident_blocks[this->rank] = 1;

    size_t issued_sends = 0, issued_recvs = 0;
    for(size_t step = 0; step < (uint) this->num_steps; step++){        
        uint peer;            
        if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
            peer = peers[port][this->num_steps - step - 1];                         
        }else{  
            peer = peers[port][step];
        }

        issued_recvs = 0;
        swing_tree_t tree_peer = get_tree(peer, port, env.allgather_config.algo_family, env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING, this->scc_real);
        for(size_t block = 0; block < this->size; block++){
            char send_block = 0, recv_block = 0;
            // Check first if I actually need to send something to that peer
            // (for non-p2 cases)
            if(tree_peer.parent[rank] == peer){
                if(tree_peer.subtree_roots[block] == rank){
                    send_block = 1;
                }            
            }
            // Check first if I actually need to recv something from that peer
            // (for non-p2 cases)
            if(tree.parent[peer] == rank){
                if(tree.subtree_roots[block] == peer){
                    recv_block = 1;
                }
            }

            // TODO: This does not work for some non p2 cases. You should use compute_block_step from libswing_common.cc rather than get_tree
            // The issue is how we disconnect the nodes, which should be done based on the starting step rather than arrival step.
            if(send_block && resident_blocks[block]){ // Sometimes (e.g., on a 6x6 torus) it might turn out that I need to both send and recv a block at the same time in the last step.
                //if(resident_blocks[block] == 0){
                //    printf("[%d] Error: block %d is not resident\n", rank, block);
                //    //exit(1);    
                //}
                ////assert(resident_blocks[block] == 1);
                DPRINTF("[%d] Step %d Sending block %d to %d\n", this->rank, step, block, peer);
                MPI_Isend((char*) recvbuf + block*sendcount*dtsize, sendcount, sendtype, peer, TAG_SWING_ALLGATHER, comm, &reqs_send[issued_sends]);
                ++issued_sends;
            }

            if(recv_block){
                DPRINTF("[%d] Step %d Receiving block %d from %d\n", this->rank, step, block, peer);
                MPI_Irecv((char*) recvbuf + block*recvcount*dtsize, recvcount, recvtype, peer, TAG_SWING_ALLGATHER, comm, &reqs_recv[issued_recvs]);
                ++issued_recvs;
                resident_blocks[block] = 1;
            }

        }
        destroy_tree(&tree_peer);
        MPI_Waitall(issued_recvs, reqs_recv, MPI_STATUSES_IGNORE);
        DPRINTF("[%d] All receives waited\n", this->rank);
    }             
    MPI_Waitall(issued_sends, reqs_send, MPI_STATUSES_IGNORE);
    DPRINTF("[%d] All sends waited\n", this->rank);
    free(peers[port]);
    destroy_tree(&tree);
    free(reqs_send);
    free(reqs_recv);
    free(resident_blocks);

    timer.reset("= swing_allgather_blocks_mpi (writing profile data to file)");
    return res;
}
#endif
