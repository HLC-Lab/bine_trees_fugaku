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


int SwingCommon::swing_reduce_scatter_utofu_blocks(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    return MPI_ERR_OTHER;
}

int SwingCommon::swing_reduce_scatter_utofu_contiguous(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_UTOFU);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
#ifdef FUGAKU
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_scatter_utofu_contiguous (init)");
    Timer timer("swing_reduce_scatter_utofu_contiguous (init)");
    int dtsize;
    MPI_Type_size(datatype, &dtsize);    

    // We always need a tempbuf since recvbuf is only large enough to accomodate the final block
    char* tmpbuf;
    bool free_tmpbuf = false;
    uint* peers[LIBSWING_MAX_SUPPORTED_PORTS];
    memset(peers, 0, sizeof(uint*)*env.num_ports);
    
    // sendbuf is read-only, and recvbuf has space only for one (final) block.
    // Thus, I need two larger temporary buffers: one for receiving the new data, and the other one to aggregate the data in (and to send the data from).
    // To avoid registering two different pointers, I will allocate a single buffer and then I will use two pointers to point to the two different parts of the buffer.
    // I call these two buffers tmpbuf_send and tmpbuf_recv.
    // How big should they be? I should consider the largest count possible
    // In case count not divisible by num ports, the first port will for sure have the largest blocks.
    // Thus, it is enough to check the count of the first block of the first port to know the largest block.
    // TODO: This work for reduce_scatter_block, generalize it to reduce_scatter
    size_t tmpbuf_send_size = blocks_info[0][0].count*env.num_ports*dtsize*this->size;
    // For tmpbuf_recv_size I have two options:
    // 1. I can either make it as large as the tmpbuf_send. In this case, at each step each rank can write in a different part of the buffer.
    // 2. I can make it half of that, but this would require explicit synchronization between rank before doing the PUTs
    //
    // I chose the first option. // TODO: Switch between the two to save some memory
    size_t tmpbuf_recv_size = tmpbuf_send_size;
    
    size_t tmpbuf_size = tmpbuf_send_size + tmpbuf_recv_size;    
    timer.reset("= swing_reduce_scatter_utofu_contiguous (utofu buf reg)"); 
    // Also the root sends from tmpbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        assert(posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size) == 0);
        free_tmpbuf = true;
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, tmpbuf, tmpbuf_size, env.num_ports); 
        timer.reset("= swing_reduce_scatter_utofu_contiguous (utofu buf exch)");           
        if(env.utofu_add_ag){
            swing_utofu_exchange_buf_info_allgather(this->utofu_descriptor, this->num_steps);
        }else{
            // TODO: Probably need to do this for all the ports for torus with different dimensions size
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            peers[0] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, 0, env.reduce_scatter_config.algo_family, this->scc_real, peers[0]);
            swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[0]); 
            
            // We need to exchange buffer info both for a normal port and for a mirrored one (peers are different)
            int mp = get_mirroring_port(env.num_ports, env.dimensions_num);
            if(mp != -1 && mp != 0){
                peers[mp] = (uint*) malloc(sizeof(uint)*this->num_steps);
                compute_peers(this->rank, mp, env.reduce_scatter_config.algo_family, this->scc_real, peers[mp]);
                swing_utofu_exchange_buf_info(this->utofu_descriptor, num_steps, peers[mp]); 
            }
        }            
    }else{
        // Everything to 0/NULL just to initialize the internal status.
        swing_utofu_reg_buf(this->utofu_descriptor, NULL, 0, NULL, 0, NULL, 0, env.num_ports); 
        tmpbuf = env.prealloc_buf;
        // Store the rmt_temp_stadd of all the other ranks
        for(size_t i = 0; i < env.num_ports; i++){
            this->utofu_descriptor->port_info[i].rmt_temp_stadd = temp_buffers[i];
            this->utofu_descriptor->port_info[i].lcl_temp_stadd = lcl_temp_stadd[i];
        }
    }
    char* tmpbuf_send = tmpbuf;
    char* tmpbuf_recv = tmpbuf + tmpbuf_send_size;

    int res = MPI_SUCCESS; 
#pragma omp parallel for num_threads(env.num_ports) schedule(static, 1) collapse(1)
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_scatter_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_scatter_utofu_contiguous (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.reduce_scatter_config.algo_family, env.reduce_scatter_config.distance_type, this->scc_real);

        size_t offset_port = tmpbuf_send_size / env.num_ports * port;
        size_t offset_port_recv = tmpbuf_recv_size / env.num_ports * port;

        DPRINTF("Offset_port_recv %d: %d\n", port, offset_port_recv);
        DPRINTF("Offset_port %d: %d\n", port, offset_port);

        char* tmpbuf_send_port = tmpbuf_send + offset_port;
        char* tmpbuf_recv_port = tmpbuf_recv + offset_port_recv;

        timer.reset("= swing_reduce_scatter_utofu_contiguous (permute)");    
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving %d bytes from %d to %d\n", this->rank, blocks_info[port][i].count*dtsize, blocks_info[port][i].offset, offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            // For a given port, all the sub-blocks have the same size, so we can just consider the count of any block (e.g., 0)
            memcpy(tmpbuf_send_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize, (char*) sendbuf + blocks_info[port][i].offset, blocks_info[port][i].count*dtsize);
        }        

        // Now perform all the subsequent steps            
        size_t offset_step_recv = 0; // This is used so that each step writes at a different location in tmpbuf_recv
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            uint peer;
            if(env.reduce_scatter_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{
                peer = peers[port][step];
            }

            // Always send from the end of the buffer
            // and receive in the beginning of the buffer
            timer.reset("= swing_reduce_scatter_utofu_contiguous (sendrecv)");            
            size_t num_blocks = this->size / (pow(2, step + 1));                        
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;
            
            utofu_stadd_t lcl_addr = utofu_descriptor->port_info[port].lcl_temp_stadd                          + offset_port      + count_to_sendrecv*dtsize;
            utofu_stadd_t rmt_addr = utofu_descriptor->port_info[port].rmt_temp_stadd[peer] + tmpbuf_send_size + offset_port_recv + offset_step_recv; 

            DPRINTF("Port %d sending from %d to %d\n", port, lcl_addr - utofu_descriptor->port_info[port].lcl_temp_stadd, rmt_addr - utofu_descriptor->port_info[port].rmt_temp_stadd[peer]);
            DPRINTF("tmpbuf[0] (port %d) at step %d before send: %d \n", port, step, ((char*) tmpbuf_send_port)[0]);

            size_t issued_sends = 0;
            issued_sends += swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step); 
            swing_utofu_wait_recv(utofu_descriptor, port, step, issued_sends - 1);
            swing_utofu_wait_sends(utofu_descriptor, port, issued_sends);

            DPRINTF("tmpbuf_recv[0] (port %d) at step %d after send: %d \n", port, step, ((char*) tmpbuf_recv_port)[0]);

            if(step == this->num_steps - 1){
                // To avoid doing a memcpy at the end
                reduce_local(tmpbuf_recv_port + offset_step_recv, tmpbuf_send_port, (char*) recvbuf + blocks_info[port][0].offset, count_to_sendrecv, datatype, op);
                DPRINTF("tmpbuf_send[0] (port %d) at step %d after aggr:  %d \n", port, step, ((char*) recvbuf + blocks_info[port][0].offset)[0]);
            }else{
                reduce_local(tmpbuf_recv_port + offset_step_recv, tmpbuf_send_port, count_to_sendrecv, datatype, op);
                DPRINTF("tmpbuf_send[0] (port %d) at step %d after aggr:  %d \n", port, step, ((char*) tmpbuf_send_port)[0]);
            }

            offset_step_recv += count_to_sendrecv*dtsize;
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
    timer.reset("= swing_reduce_scatter_utofu_contiguous (writing profile data to file)");
    return res;
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif 
}

int SwingCommon::swing_reduce_scatter_utofu(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    if(is_power_of_two(this->size)){
        return swing_reduce_scatter_utofu_contiguous(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }else{
        return swing_reduce_scatter_utofu_blocks(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }
}


int SwingCommon::swing_reduce_scatter_mpi_contiguous(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef VALIDATE
    printf("func_called: %s\n", __func__);
    assert(env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_SWING || env.reduce_scatter_config.algo_family == SWING_ALGO_FAMILY_RECDOUB);
    assert(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_MPI);
    assert(env.reduce_scatter_config.algo == SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE); 
#endif    
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_reduce_scatter_mpi_contiguous (init)");
    Timer timer("swing_reduce_scatter_mpi_contiguous (init)");
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
    // TODO: This work for reduce_scatter_block, generalize it to reduce_scatter
    size_t tmpbuf_size = blocks_info[0][0].count * dtsize * env.num_ports * this->size;
    
    timer.reset("= swing_reduce_scatter_mpi_contiguous (utofu buf reg)"); 

    // Also the root sends from tmbuf because it needs to permute the sendbuf
    if(tmpbuf_size > env.prealloc_size){
        posix_memalign((void**) &tmpbuf, LIBSWING_TMPBUF_ALIGNMENT, tmpbuf_size);
        free_tmpbuf = true;           
    }else{
        tmpbuf = env.prealloc_buf;
    }

    char* tmpbuf_send = tmpbuf;

    int res = MPI_SUCCESS; 
    for(size_t port = 0; port < env.num_ports; port++){
        // Compute the peers of this port if I did not do it yet
        if(peers[port] == NULL){
            peers[port] = (uint*) malloc(sizeof(uint)*this->num_steps);
            compute_peers(this->rank, port, env.reduce_scatter_config.algo_family, this->scc_real, peers[port]);
        }        
        timer.reset("= swing_reduce_scatter_mpi_contiguous (computing trees)");
        swing_tree_t tree = get_tree(this->rank, port, env.reduce_scatter_config.algo_family, env.reduce_scatter_config.distance_type, this->scc_real);

        size_t offset_port = tmpbuf_size / env.num_ports * port;
        DPRINTF("Offset_port %d: %d\n", port, offset_port);
        char* tmpbuf_send_port = tmpbuf_send + offset_port;

        timer.reset("= swing_reduce_scatter_mpi_contiguous (permute)");    
        for(size_t i = 0; i < size; i++){      
            DPRINTF("[%d] Moving %d bytes from %d to %d\n", this->rank, blocks_info[port][i].count*dtsize, blocks_info[port][i].offset, offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize);          
            // We need to pay attention here. Each port will work on non-contiguous sub-blocks of the buffer. So we need to copy the data in a contiguous buffer.
            // E.g., with two ports and 4 ranks
            // | 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 |
            // If we have 2 ports, each block is split in two parts, thus port 0 will work on 0 1 2 3 8 9 10 11, and port 1 on 4 5 6 7 12 13 14 15
            // For a given port, all the sub-blocks have the same size, so we can just consider the count of any block (e.g., 0)
            memcpy(tmpbuf + offset_port + blocks_info[port][0].count*tree.remapped_ranks[i]*dtsize, (char*) sendbuf + blocks_info[port][i].offset, blocks_info[port][i].count*dtsize);
        }        

        // Now perform all the subsequent steps            
        for(size_t step = 0; step < (uint) this->num_steps; step++){
            uint peer;
            if(env.reduce_scatter_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];
            }else{
                peer = peers[port][step];
            }

            // Always send from the end of the buffer
            // and receive in the beginning of the buffer
            timer.reset("= swing_reduce_scatter_mpi_contiguous (sendrecv)");
            size_t num_blocks = this->size / (pow(2, step + 1));                        
            size_t count_to_sendrecv = num_blocks*blocks_info[port][0].count;

            MPI_Sendrecv_replace(tmpbuf_send_port + count_to_sendrecv*dtsize, count_to_sendrecv, datatype, 
                                 peer, TAG_SWING_REDUCESCATTER, 
                                 peer, TAG_SWING_REDUCESCATTER, comm, MPI_STATUS_IGNORE); 

            if(step == this->num_steps - 1){
                // To avoid doing a memcpy at the end
                reduce_local(tmpbuf_send_port + count_to_sendrecv*dtsize, tmpbuf_send_port, (char*) recvbuf + offset_port, count_to_sendrecv, datatype, op);
            }else{
                MPI_Reduce_local(tmpbuf_send_port + count_to_sendrecv*dtsize, tmpbuf_send_port, count_to_sendrecv, datatype, op);
            }
        }        
        free(peers[port]);

        destroy_tree(&tree);
    }

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_reduce_scatter_mpi_contiguous (writing profile data to file)");
    return res;
}

int SwingCommon::swing_reduce_scatter_mpi_blocks(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    return MPI_ERR_OTHER;
}

int SwingCommon::swing_reduce_scatter_mpi(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op, BlockInfo** blocks_info, MPI_Comm comm){
    if(is_power_of_two(this->size)){
        return swing_reduce_scatter_mpi_contiguous(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }else{
        return swing_reduce_scatter_mpi_blocks(sendbuf, recvbuf, datatype, op, blocks_info, comm);
    }
}
