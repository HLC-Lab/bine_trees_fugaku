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


int SwingCommon::swing_allgather_utofu_blocks(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
    assert("Not yet implemented.\n");
    return MPI_ERR_OTHER;
}

int SwingCommon::swing_allgather_utofu_contiguous(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
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
            
            swing_utofu_isend(utofu_descriptor, &(this->vcq_ids[port][peer]), port, peer, lcl_addr, count_to_sendrecv*dtsize, rmt_addr, step);
            swing_utofu_wait_recv(utofu_descriptor, port, step, 0);
            swing_utofu_wait_sends(utofu_descriptor, port, 1);
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

int SwingCommon::swing_allgather_utofu(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
#ifdef FUGAKU
    if(is_power_of_two(this->size)){
        return swing_allgather_utofu_contiguous(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }else{
        return swing_allgather_utofu_blocks(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }
#else
    assert("uTofu not supported");
    return MPI_ERR_OTHER;
#endif   
}

int SwingCommon::swing_allgather_mpi_contiguous(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
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

int SwingCommon::swing_allgather_mpi_blocks(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
    assert(sendcount == recvcount); // TODO: Implement the case where sendcount != recvcount
    assert(sendtype == recvtype); // TODO: Implement the case where sendtype != recvtype
    //Timer timer("profile_" + std::to_string(count) + "_" + std::to_string(env.num_ports) + "/master.profile", "= swing_allgather_mpi_blocks (init)");
    Timer timer("swing_allgather_mpi_blocks (init)");
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
        timer.reset("= swing_allgather_mpi_blocks (computing trees)");
        swing_distance_type_t inverse_distance = env.allgather_config.distance_type == SWING_DISTANCE_DECREASING ? SWING_DISTANCE_INCREASING : SWING_DISTANCE_DECREASING;
        swing_tree_t tree    = get_tree(this->rank, port, env.allgather_config.algo_family, env.allgather_config.distance_type, this->scc_real);
        swing_tree_t tree_rs = get_tree(this->rank, port, env.allgather_config.algo_family, inverse_distance           , this->scc_real); // Since we are doing an allgather, we consider the tree of the corresponding reduce-scatter

        size_t tmpbuf_offset_port = (tmpbuf_size / env.num_ports) * port;
        memcpy(tmpbuf + tmpbuf_offset_port + blocks_info[port][rank].offset, ((char*) sendbuf) + blocks_info[port][0].offset, blocks_info[port][0].count*dtsize);              

        MPI_Request* reqs_send = (MPI_Request*) malloc(sizeof(MPI_Request)*this->size);
        for(size_t step = 0; step < (uint) this->num_steps; step++){        
            uint peer;
            if(env.allgather_config.distance_type == SWING_DISTANCE_DECREASING){
                peer = peers[port][this->num_steps - step - 1];                               
            }else{  
                peer = peers[port][step];
            }

            size_t issued_sends = 0;
            
            swing_tree_t tree_peer_rs = get_tree(peer, port, env.allgather_config.algo_family, inverse_distance           , this->scc_real);
            swing_tree_t tree_peer    = get_tree(peer, port, env.allgather_config.algo_family, env.allgather_config.distance_type, this->scc_real);
            if(tree_peer_rs.parent[rank] == peer || 1){ // Needed to avoid trees which are actually graphs in non-p2 cases.
                // Loop over all the blocks to determine what to send
                // At step s, I send what I received in the reduce-scatter at step num_steps - 1 - s
                // (i.e., what the peer sent me at step num_steps - 1 - s)                
                for(size_t block = 0; block < this->size; block++){
                    size_t remapped_block = tree_peer_rs.remapped_ranks[block];
                    if(remapped_block >= tree_peer_rs.remapped_ranks[rank] && remapped_block <= tree_peer_rs.remapped_ranks_max[rank]){
                        DPRINTF("[%d] Sending block %d to %d at step %d\n", this->rank, block, peer, step);
                        MPI_Isend(tmpbuf + tmpbuf_offset_port + blocks_info[port][block].offset, blocks_info[port][block].count, sendtype, peer, TAG_SWING_ALLGATHER, comm, &reqs_send[issued_sends]);
                        issued_sends++;
                    }                
                }                
            }

            if(tree_rs.parent[peer] == rank || 1){
                // Loop over all the blocks to determine what to recv
                // At step s, I recv what I sent in the reduce-scatter at step num_steps - 1 - s            
                for(size_t block = 0; block < this->size; block++){
                    size_t remapped_block = tree_rs.remapped_ranks[block];
                    if(remapped_block >= tree_rs.remapped_ranks[peer] && remapped_block <= tree_rs.remapped_ranks_max[peer]){
                        DPRINTF("[%d] Receiving block %d from %d at step %d\n", this->rank, block, peer, step);
                        MPI_Recv(tmpbuf + tmpbuf_offset_port + blocks_info[port][block].offset, blocks_info[port][block].count, sendtype, peer, TAG_SWING_ALLGATHER, comm, MPI_STATUS_IGNORE);
                    }                
                }
            }

            MPI_Waitall(issued_sends, reqs_send, MPI_STATUSES_IGNORE);
            destroy_tree(&tree_peer);
            destroy_tree(&tree_peer_rs);
        }
        //memcpy(recvbuf + tmpbuf_offset_port, tmpbuf + tmpbuf_offset_port, blocks_info[port][0].count*dtsize);

        free(reqs_send);        
        free(peers[port]);
        destroy_tree(&tree);
        destroy_tree(&tree_rs);
    }
    memcpy(recvbuf, tmpbuf, sendcount*this->size*dtsize);

    if(free_tmpbuf){
        free(tmpbuf);
    }    
    timer.reset("= swing_allgather_mpi_blocks (writing profile data to file)");
    return res;
}

int SwingCommon::swing_allgather_mpi(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, BlockInfo** blocks_info, MPI_Comm comm){
    if(is_power_of_two(this->size)){
        return swing_allgather_mpi_contiguous(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }else{
        return swing_allgather_mpi_blocks(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
    }
}
