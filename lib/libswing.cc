#include <mpi.h>
#ifdef FUGAKU
#include <mpi-ext.h>
#include "fugaku/swing_utofu.h"
#endif
#include <algorithm>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include <unistd.h>
#include <omp.h>
#include "libswing_common.h"

static int force_env_reload = 1, env_read = 0;
static SwingCommon* swing_common = NULL;
static swing_env_t env;

std::unordered_map<swing_comm_info_key_t, swing_comm_info_t> comm_info;

static void init_env(swing_env_t* env, MPI_Comm comm){
    MPI_Comm_size(comm, (int*) &(env->dimensions[0]));
    env->dimensions_num = 1;
    env->num_ports = 1;
    env->segment_size = 0;
    env->prealloc_size = 0;
    env->prealloc_buf = NULL;
    env->utofu_add_ag = 0;
    env->use_threads = 1;

    env->allreduce_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->allgather_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->reduce_scatter_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->bcast_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->alltoall_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->scatter_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->gather_config.algo_family = SWING_ALGO_FAMILY_SWING;
    env->reduce_config.algo_family = SWING_ALGO_FAMILY_SWING;

    env->allreduce_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->allgather_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->reduce_scatter_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->bcast_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->alltoall_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->scatter_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->gather_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
    env->reduce_config.algo_layer = SWING_ALGO_LAYER_UTOFU;

    env->allreduce_config.algo = SWING_ALLREDUCE_ALGO_B;
    env->allgather_config.algo = SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE;
    env->reduce_scatter_config.algo = SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE;
    env->bcast_config.algo = SWING_BCAST_ALGO_BINOMIAL_TREE;
    env->alltoall_config.algo = SWING_ALLTOALL_ALGO_LOG;
    env->scatter_config.algo = SWING_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE;
    env->gather_config.algo = SWING_GATHER_ALGO_BINOMIAL_TREE_CONT_PERMUTE;
    env->reduce_config.algo = SWING_REDUCE_ALGO_BINOMIAL_TREE;

    env->allreduce_config.distance_type = SWING_DISTANCE_INCREASING;
    env->allgather_config.distance_type = SWING_DISTANCE_DECREASING;
    env->reduce_scatter_config.distance_type = SWING_DISTANCE_INCREASING;
    env->bcast_config.distance_type = SWING_DISTANCE_DECREASING;    
    env->alltoall_config.distance_type = SWING_DISTANCE_INCREASING;
    env->scatter_config.distance_type = SWING_DISTANCE_INCREASING;
    env->gather_config.distance_type = SWING_DISTANCE_DECREASING;
    env->reduce_config.distance_type = SWING_DISTANCE_INCREASING;
}

static inline void read_env(MPI_Comm comm){
    char* env_str = getenv("LIBSWING_FORCE_ENV_RELOAD");
    if(env_str){
        force_env_reload = 1;
    }else{
        force_env_reload = 0;
    }

    if(!env_read || force_env_reload){
        env_read = 1;

        init_env(&env, comm);

        env_str = getenv("LIBSWING_SEGMENT_SIZE");
        if(env_str){
            env.segment_size = atoi(env_str);
        }

        env_str = getenv("LIBSWING_NUM_PORTS");
        if(env_str){
            env.num_ports = atoi(env_str);
        }
        assert(env.num_ports <= LIBSWING_MAX_SUPPORTED_PORTS);

        env_str = getenv("LIBSWING_PREALLOC_SIZE");
        if(env_str){
            env.prealloc_size = atol(env_str);
        }

        env_str = getenv("LIBSWING_UTOFU_ADD_AG");
        if(env_str){
            env.utofu_add_ag = atoi(env_str);
        }

        env_str = getenv("LIBSWING_USE_THREADS");
        if(env_str){
            env.use_threads = atoi(env_str);
        }        

        env_str = getenv("LIBSWING_DIMENSIONS");
        if(env_str){
            char* copy = (char*) malloc(sizeof(char)*(strlen(env_str) + 1));
            strcpy(copy, env_str);
            const char *delim = "x";
            char* rest = NULL;
            char *ptr = strtok_r(copy, delim, &rest);
            uint i = 0;
            while(ptr != NULL){
                env.dimensions[i] = atoi(ptr);
                ptr = strtok_r(NULL, delim, &rest);
                ++i;
            } 
            free(copy);
            env.dimensions_num = i;       
        }
        assert(env.dimensions_num <= LIBSWING_MAX_SUPPORTED_DIMENSIONS);

        /****************************/
        /* Algo family              */
        /****************************/
        env_str = getenv("LIBSWING_ALLREDUCE_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.allreduce_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.allreduce_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.allreduce_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.allreduce_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.allreduce_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_ALLREDUCE_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_ALLGATHER_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.allgather_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.allgather_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.allgather_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.allgather_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.allgather_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_ALLGATHER_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_REDUCE_SCATTER_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.reduce_scatter_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.reduce_scatter_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.reduce_scatter_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.reduce_scatter_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.reduce_scatter_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_REDUCE_SCATTER_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_BCAST_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.bcast_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.bcast_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.bcast_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.bcast_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.bcast_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_BCAST_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_ALLTOALL_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.alltoall_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.alltoall_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.alltoall_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.alltoall_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.alltoall_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_ALLTOALL_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_SCATTER_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.scatter_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.scatter_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.scatter_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.scatter_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.scatter_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_SCATTER_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_GATHER_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.gather_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.gather_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.gather_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.gather_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.gather_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_GATHER_ALGO_FAMILY" && 0);
            }
        }

        env_str = getenv("LIBSWING_REDUCE_ALGO_FAMILY");
        if(env_str){
            if(strcmp(env_str, "DEFAULT") == 0){
                env.reduce_config.algo_family = SWING_ALGO_FAMILY_DEFAULT;
            }else if(strcmp(env_str, "SWING") == 0){
                env.reduce_config.algo_family = SWING_ALGO_FAMILY_SWING;
            }else if(strcmp(env_str, "RECDOUB") == 0){
                env.reduce_config.algo_family = SWING_ALGO_FAMILY_RECDOUB;
            }else if(strcmp(env_str, "BRUCK") == 0){
                env.reduce_config.algo_family = SWING_ALGO_FAMILY_BRUCK;
            }else if(strcmp(env_str, "RING") == 0){
                env.reduce_config.algo_family = SWING_ALGO_FAMILY_RING;
            }else{
                assert("Invalid value for LIBSWING_REDUCE_ALGO_FAMILY" && 0);
            }
        }

        /****************************/
        /* Algo layer               */
        /****************************/
        env_str = getenv("LIBSWING_ALLREDUCE_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.allreduce_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.allreduce_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_ALLREDUCE_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_ALLGATHER_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.allgather_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.allgather_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_ALLGATHER_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_REDUCE_SCATTER_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.reduce_scatter_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.reduce_scatter_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_REDUCE_SCATTER_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_BCAST_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.bcast_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.bcast_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_BCAST_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_ALLTOALL_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.alltoall_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.alltoall_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_ALLTOALL_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_SCATTER_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.scatter_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.scatter_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_SCATTER_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_GATHER_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.gather_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.gather_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_GATHER_ALGO_LAYER" && 0);
            }
        }

        env_str = getenv("LIBSWING_REDUCE_ALGO_LAYER");
        if(env_str){
            if(strcmp(env_str, "UTOFU") == 0){
                env.reduce_config.algo_layer = SWING_ALGO_LAYER_UTOFU;
            }else if(strcmp(env_str, "MPI") == 0){
                env.reduce_config.algo_layer = SWING_ALGO_LAYER_MPI;
            }else{
                assert("Invalid value for LIBSWING_REDUCE_ALGO_LAYER" && 0);
            }
        }

        /****************************/
        /* Algo                     */
        /****************************/
        env_str = getenv("LIBSWING_ALLREDUCE_ALGO");
        if(env_str){
            if(strcmp(env_str, "L") == 0){
                env.allreduce_config.algo = SWING_ALLREDUCE_ALGO_L;
            }else if(strcmp(env_str, "B") == 0){
                env.allreduce_config.algo = SWING_ALLREDUCE_ALGO_B;
            }else if(strcmp(env_str, "REDUCE_BCAST") == 0){
                env.allreduce_config.algo = SWING_ALLREDUCE_ALGO_REDUCE_BCAST;
            }else if(strcmp(env_str, "B_CONT") == 0){
                env.allreduce_config.algo = SWING_ALLREDUCE_ALGO_B_CONT;
            }else if(strcmp(env_str, "B_COALESCE") == 0){
                env.allreduce_config.algo = SWING_ALLREDUCE_ALGO_B_COALESCE;
            }else{
                assert("Invalid value for LIBSWING_ALLREDUCE_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_ALLGATHER_ALGO");
        if(env_str){
            if(strcmp(env_str, "VEC_DOUBLING_CONT_PERMUTE") == 0){
                env.allgather_config.algo = SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE;
            }else if(strcmp(env_str, "VEC_DOUBLING_CONT_SEND") == 0){
                env.allgather_config.algo = SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_SEND;
            }else if(strcmp(env_str, "VEC_DOUBLING_BLOCKS") == 0){
                env.allgather_config.algo = SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS;
            }else if(strcmp(env_str, "GATHER_BCAST") == 0){
                env.allgather_config.algo = SWING_ALLGATHER_ALGO_GATHER_BCAST;
            }else{
                assert("Invalid value for LIBSWING_ALLGATHER_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_REDUCE_SCATTER_ALGO");
        if(env_str){
            if(strcmp(env_str, "VEC_HALVING_CONT_PERMUTE") == 0){
                env.reduce_scatter_config.algo = SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE;
            }else if(strcmp(env_str, "VEC_HALVING_CONT_SEND") == 0){
                env.reduce_scatter_config.algo = SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_SEND;
            }else if(strcmp(env_str, "VEC_HALVING_BLOCKS") == 0){
                env.reduce_scatter_config.algo = SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_BLOCKS;
            }else if(strcmp(env_str, "REDUCE_SCATTER") == 0){
                env.reduce_scatter_config.algo = SWING_REDUCE_SCATTER_ALGO_REDUCE_SCATTER;
            }else{
                assert("Invalid value for LIBSWING_REDUCE_SCATTER_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_BCAST_ALGO");
        if(env_str){
            if(strcmp(env_str, "BINOMIAL_TREE") == 0){
                env.bcast_config.algo = SWING_BCAST_ALGO_BINOMIAL_TREE;
            }else if(strcmp(env_str, "BINOMIAL_TREE_TMPBUF") == 0){
                env.bcast_config.algo = SWING_BCAST_ALGO_BINOMIAL_TREE_TMPBUF;
            }else if(strcmp(env_str, "SCATTER_ALLGATHER") == 0){
                env.bcast_config.algo = SWING_BCAST_ALGO_SCATTER_ALLGATHER;
            }else{
                assert("Invalid value for LIBSWING_BCAST_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_ALLTOALL_ALGO");
        if(env_str){
            if(strcmp(env_str, "LOG") == 0){
                env.alltoall_config.algo = SWING_ALLTOALL_ALGO_LOG;
            }else{
                assert("Invalid value for LIBSWING_ALLTOALL_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_SCATTER_ALGO");
        if(env_str){
            if(strcmp(env_str, "BINOMIAL_TREE_CONT_PERMUTE") == 0){
                env.scatter_config.algo = SWING_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE;
            }else if(strcmp(env_str, "BINOMIAL_TREE_CONT_SEND") == 0){
                env.scatter_config.algo = SWING_SCATTER_ALGO_BINOMIAL_TREE_CONT_SEND;
            }else{
                assert("Invalid value for LIBSWING_SCATTER_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_GATHER_ALGO");
        if(env_str){
            if(strcmp(env_str, "BINOMIAL_TREE_CONT_PERMUTE") == 0){
                env.gather_config.algo = SWING_GATHER_ALGO_BINOMIAL_TREE_CONT_PERMUTE;
            }else if(strcmp(env_str, "BINOMIAL_TREE_CONT_SEND") == 0){
                env.gather_config.algo = SWING_GATHER_ALGO_BINOMIAL_TREE_CONT_SEND;
            }else{
                assert("Invalid value for LIBSWING_GATHER_ALGO" && 0);
            }
        }

        env_str = getenv("LIBSWING_REDUCE_ALGO");
        if(env_str){
            if(strcmp(env_str, "BINOMIAL_TREE") == 0){
                env.reduce_config.algo = SWING_REDUCE_ALGO_BINOMIAL_TREE;
            }else if(strcmp(env_str, "REDUCE_SCATTER_GATHER") == 0){
                env.reduce_config.algo = SWING_REDUCE_ALGO_REDUCE_SCATTER_GATHER;
            }else{
                assert("Invalid value for LIBSWING_REDUCE_ALGO" && 0);
            }
        }

        /****************************/
        /* Distance                 */
        /****************************/
        env_str = getenv("LIBSWING_ALLREDUCE_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.allreduce_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.allreduce_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        env_str = getenv("LIBSWING_ALLGATHER_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.allgather_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.allgather_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        env_str = getenv("LIBSWING_REDUCE_SCATTER_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.reduce_scatter_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.reduce_scatter_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        env_str = getenv("LIBSWING_BCAST_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.bcast_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.bcast_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }
        
        env_str = getenv("LIBSWING_ALLTOALL_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.alltoall_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.alltoall_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        env_str = getenv("LIBSWING_SCATTER_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.scatter_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.scatter_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        env_str = getenv("LIBSWING_GATHER_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.gather_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.gather_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        env_str = getenv("LIBSWING_REDUCE_DISTANCE");
        if(env_str){
            if(strcmp(env_str, "INCREASING") == 0){
                env.reduce_config.distance_type = SWING_DISTANCE_INCREASING;
            }else{
                env.reduce_config.distance_type = SWING_DISTANCE_DECREASING;
            }
        }

        if(env.prealloc_size){
            posix_memalign((void**) &env.prealloc_buf, LIBSWING_TMPBUF_ALIGNMENT, env.prealloc_size);
        }

        swing_common = new SwingCommon(comm, env);

#if 0
        // TODO: Dump env
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0){
            printf("Libswing called. Environment:\n");
            printf("------------------------------------\n");
            //printf("LIBSWING_DISABLE_REDUCESCATTER: %d\n", disable_reducescatter);
            //printf("LIBSWING_DISABLE_ALLGATHERV: %d\n", disable_allgatherv);
            //printf("LIBSWING_DISABLE_ALLGATHER: %d\n", disable_allgather);
            //printf("LIBSWING_DISABLE_ALLREDUCE: %d\n", disable_allreduce);
            //printf("LIBSWING_RDMA: %d\n", rdma);
            printf("LIBSWING_ALGO: %d\n", algo);
            printf("LIBSWING_SEGMENT_SIZE: %d\n", segment_size);
            printf("LIBSWING_PREALLOC_SIZE: %d\n", prealloc_size);
            printf("LIBSWING_DIMENSIONS: ");
            for(size_t i = 0; i < dimensions_num; i++){
                printf("%d", dimensions[i]);
                if(i < dimensions_num - 1){
                    printf(",");
                }
            }
            printf("\n");
            printf("------------------------------------\n");
        }
#endif
    }
}

#if 0
// Code copied from MPICH repo (https://github.com/pmodels/mpich/tree/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce)
static int MPI_Allreduce_recdoub_l(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank, rem, newdst;
    MPI_Aint extent;
    void *tmp_buf;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);
    extent = dtsize;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    is_commutative = true;
    tmp_buf = malloc(count*extent);
    memcpy(recvbuf, sendbuf, count*extent);

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = pow(2, floor(log2(comm_size)));
    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (!is_odd(rank)) {    /* even */
            mpi_errno = MPI_Send(recvbuf, count,
                                datatype, rank + 1, TAG_SWING_ALLREDUCE, comm);
            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPI_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
             * into tmp_buf */
            mpi_errno = MPI_Sendrecv(recvbuf, count, datatype,
                                      dst, TAG_SWING_ALLREDUCE, tmp_buf,
                                      count, datatype, dst,
                                      TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);        

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
                mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            } else {
                /* op is noncommutative and the order is not right */
                mpi_errno = MPI_Reduce_local(recvbuf, tmp_buf, count, datatype, op);
                /* copy result back into recvbuf */
                memcpy(recvbuf, tmp_buf, count*extent);
            }
            mask <<= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (is_odd(rank))   /* odd */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank - 1, TAG_SWING_ALLREDUCE, comm);
        else    /* even */
            mpi_errno = MPI_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    }
    free(tmp_buf);
    return mpi_errno;
}

// Code copied from MPICH repo (https://github.com/pmodels/mpich/tree/bb7f0a9f61dbee66c67073f9c68fa28b6f443e0a/src/mpi/coll/allreduce)
static int MPI_Allreduce_recdoub_b(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i, send_idx, recv_idx, last_idx;
    void *tmp_buf;
    int dtsize;
    MPI_Type_size(datatype, &dtsize);
    MPI_Aint extent = dtsize;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    /* need to allocate temporary buffer to store incoming data */
    tmp_buf = malloc(count * dtsize);
    /* copy local data into recvbuf */
    memcpy(recvbuf, sendbuf, count*dtsize);

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = pow(2, floor(log2(comm_size)));

    rem = comm_size - pof2;
    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */
    if (rank < 2 * rem) {
        if (!is_odd(rank)) {    /* even */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank + 1, TAG_SWING_ALLREDUCE, comm);

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPI_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */
    if (newrank != -1) {
        MPI_Aint *cnts, *disps;
        cnts = (MPI_Aint*) malloc(pof2 * sizeof(MPI_Aint));
        disps = (MPI_Aint*) malloc(pof2 * sizeof(MPI_Aint));
        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }            

            /* Send data from recvbuf. Recv into tmp_buf */
            mpi_errno = MPI_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, TAG_SWING_ALLREDUCE,
                                      (char *) tmp_buf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            mpi_errno = MPI_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPI_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, TAG_SWING_ALLREDUCE,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (is_odd(rank))   /* odd */
            mpi_errno = MPI_Send(recvbuf, count,
                                  datatype, rank - 1, TAG_SWING_ALLREDUCE, comm);
        else    /* even */
            mpi_errno = MPI_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  TAG_SWING_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    }
    free(tmp_buf);
    return mpi_errno;
}
#endif

static int MPI_Allreduce_ring(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int rank;
    int r = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(r != MPI_SUCCESS)
        return r;
    int size;
    r = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(r != MPI_SUCCESS)
        return r;
    int dtsize;
    r = MPI_Type_size(datatype, &dtsize);
    if(r != MPI_SUCCESS)
        return r;

    const size_t segment_size = count / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = count % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[size - 1] == (uint) count);

     // Copy your data to the output buffer to avoid modifying the input buffer.
    memcpy(recvbuf, sendbuf, count*dtsize);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    char* buffer;
    buffer = (char*) malloc(segment_sizes[0]*dtsize);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    MPI_Status recv_status;
    MPI_Request recv_req;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).
    for (int i = 0; i < size - 1; i++) {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        char* segment_send = &(((char*)recvbuf)[dtsize*segment_ends[send_chunk] - dtsize*segment_sizes[send_chunk]]);

        MPI_Irecv(buffer, segment_sizes[recv_chunk],
                  datatype, recv_from, TAG_SWING_ALLREDUCE, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[send_chunk],
                datatype, send_to, TAG_SWING_ALLREDUCE, MPI_COMM_WORLD);

        char *segment_update = &(((char*)recvbuf)[dtsize*segment_ends[recv_chunk] - dtsize*segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);
        MPI_Reduce_local(buffer, segment_update, segment_sizes[recv_chunk], datatype, op);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + 1 + size) % size;
        int recv_chunk = (rank - i + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        char* segment_send = &(((char*)recvbuf)[dtsize*segment_ends[send_chunk] - dtsize*segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        char* segment_recv = &(((char*)recvbuf)[dtsize*segment_ends[recv_chunk] - dtsize*segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                     datatype, send_to, TAG_SWING_ALLREDUCE, segment_recv,
                     segment_sizes[recv_chunk], datatype, recv_from,
                     TAG_SWING_ALLREDUCE, MPI_COMM_WORLD, &recv_status);
    }

    // Free temporary memory.
    free(buffer);    
    return 0;
}

/**
 * @param offset In bytes!
*/
static inline void get_count_and_offset_per_block(size_t offset, size_t count, BlockInfo* blocks_info, SwingCommon* swing_common, uint dtsize){
    // Compute the count and offset of each block(for each port)
    uint partition_size = count / swing_common->get_size();
    uint remaining = count % swing_common->get_size();
    size_t count_so_far = 0;
    for(size_t i = 0; i < (size_t) swing_common->get_size(); i++){
        size_t count_block = partition_size + (i < remaining ? 1 : 0);
        size_t offset_block = offset + count_so_far*dtsize;
        count_so_far += count_block;
        blocks_info[i].count = count_block;
        blocks_info[i].offset = offset_block;
    }
}

// Gets the count and offset for each port.
static inline void get_count_and_offset_per_port(size_t count, BlockInfo** ci, SwingCommon* swing_common, uint dtsize){
    uint partition_size = count / swing_common->get_num_ports();
    uint remaining = count % swing_common->get_num_ports();
    uint count_so_far = 0;
    for(size_t i = 0; i < swing_common->get_num_ports(); i++){
        size_t count_port = partition_size + (i < remaining ? 1 : 0);
        size_t offset_port = count_so_far*dtsize;
        count_so_far += count_port;
        get_count_and_offset_per_block(offset_port, count_port, ci[i], swing_common, dtsize);
    }
}

/**
 * For collectives that send/recv blocks rather than the entire buffer, computes the
 * offset and count of each block (for each port).
 */
static inline BlockInfo** get_blocks_info(size_t count, SwingCommon* swing_common, uint dtsize){    
    // Allocate blocks_info
    BlockInfo** blocks_info = (BlockInfo**) malloc(sizeof(BlockInfo*)*swing_common->get_num_ports());
    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
        blocks_info[p] = (BlockInfo*) malloc(sizeof(BlockInfo)*swing_common->get_size());        
    }
    get_count_and_offset_per_port(count, blocks_info, swing_common, dtsize);
    return blocks_info;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    read_env(comm);
    switch(env.allreduce_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:
            switch(env.allreduce_config.algo){
                case SWING_ALLREDUCE_ALGO_L:{
                    int dtsize;
                    MPI_Type_size(datatype, &dtsize);
                    BlockInfo** blocks_info = get_blocks_info(count, swing_common, dtsize);
                    int res = swing_common->swing_coll_l(sendbuf, recvbuf, count, datatype, op, comm);
                    // Free blocks_info
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        free(blocks_info[p]);
                    }
                    free(blocks_info);
                    return res;
                }
                case SWING_ALLREDUCE_ALGO_B:
                case SWING_ALLREDUCE_ALGO_B_CONT:
                case SWING_ALLREDUCE_ALGO_B_COALESCE:{
                    assert(env.allreduce_config.distance_type == SWING_DISTANCE_INCREASING); // See notes. Doing with decreasing with lead to 0 and 1 having different trees and would not work.
                    assert(count >= swing_common->get_num_ports()*swing_common->get_size());
                    int dtsize;
                    MPI_Type_size(datatype, &dtsize);
                    BlockInfo** blocks_info = get_blocks_info(count, swing_common, dtsize);
                    int res;
                    if(env.allreduce_config.algo_layer == SWING_ALGO_LAYER_MPI){
                        res = swing_common->swing_coll_b(sendbuf, recvbuf, count, datatype, op, comm, blocks_info, SWING_ALLREDUCE);            
                    }else{
                        res = swing_common->swing_coll_b_cont_utofu(sendbuf, recvbuf, count, datatype, op, comm, blocks_info, SWING_ALLREDUCE);
                    }
                    // Free blocks_info
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        free(blocks_info[p]);
                    }
                    free(blocks_info);
                    return res;
                }
                default:
                    assert("Invalid value for LIBSWING_ALLREDUCE_ALGO" && 0);
            }
        case SWING_ALGO_FAMILY_RING:
            return swing_common->bucket_allreduce((char*) sendbuf, (char*) recvbuf, count, datatype, op, comm);
        default:
            assert("Invalid value for LIBSWING_ALLREDUCE_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    // TODO: Actually there are assumption of it being a reduce_scatter_block, so we need to fix that
    read_env(comm);
    switch(env.reduce_scatter_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
        case SWING_ALGO_FAMILY_RING:
            return swing_common->bucket_reduce_scatter(sendbuf, recvbuf, recvcounts[0]*swing_common->get_size(), datatype, op, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:{
            size_t count = 0;
            for(size_t i = 0; i < (size_t) swing_common->get_size(); i++){
                count += recvcounts[i];
            }
            assert(count >= swing_common->get_num_ports()*swing_common->get_size());

            switch(env.reduce_scatter_config.algo){
                case SWING_REDUCE_SCATTER_ALGO_VEC_HALVING_CONT_PERMUTE:{
                    // For reduce-scatter we do not do chunking (would complicate things too much). 
                    // We first split the data by block, and then by port (i.e., we split each block in num_ports parts). 
                    // This is the opposite of what we do in allreduce.
                    // Allocate blocks_info
                    BlockInfo** blocks_info = (BlockInfo**) malloc(sizeof(BlockInfo*)*swing_common->get_num_ports());
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        blocks_info[p] = (BlockInfo*) malloc(sizeof(BlockInfo)*swing_common->get_size());
                    }
                    size_t count_so_far = 0;
                    //size_t my_offset = 0;
                    int dtsize;
                    MPI_Type_size(datatype, &dtsize);
                    for(size_t i = 0; i < (size_t) swing_common->get_size(); i++){
                        DPRINTF("[%d] recvcnt %d: %d\n", swing_common->get_rank(), i, recvcounts[i]);
                        size_t partition_size = recvcounts[i] / swing_common->get_num_ports();
                        size_t remaining = recvcounts[i] % swing_common->get_num_ports();                
                        size_t block_offset = count_so_far*dtsize;
                        size_t block_count_so_far = 0;
                        for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                            size_t count_port = partition_size + (p < remaining ? 1 : 0);
                            size_t offset_port = block_offset + block_count_so_far*dtsize;
                            block_count_so_far += count_port;
                            blocks_info[p][i].count = count_port;
                            blocks_info[p][i].offset = offset_port;
                            DPRINTF("[%d] Port %d Offset %d Count %d\n", swing_common->get_rank(), p, offset_port, count_port);
                        }

                        //if(i == (size_t) swing_common->get_rank()){
                        //    my_offset = block_offset;
                        //}

                        count_so_far += recvcounts[i];
                    }
                    assert(count == count_so_far);
                    // Call the actual collective
                    
                    // OLD way of doing it
                    //char* tmpbuf = (char*) malloc(count*dtsize);
                    //int res = swing_common->swing_coll_b(sendbuf, tmpbuf, count, datatype, op, comm, blocks_info, SWING_REDUCE_SCATTER);            
                    //DPRINTF("[%d] Copying %d bytes from offset %d into recvbuf\n", swing_common->get_rank(), recvcounts[swing_common->get_rank()]*dtsize, my_offset);
                    //memcpy(recvbuf, tmpbuf + my_offset, recvcounts[swing_common->get_rank()]*dtsize);
                    //free(tmpbuf);
                    int res;
                    if(env.reduce_scatter_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        res = swing_common->swing_reduce_scatter_utofu(sendbuf, recvbuf, datatype, op, blocks_info, comm);
                    }else{
                        res = swing_common->swing_reduce_scatter_mpi_contiguous(sendbuf, recvbuf, datatype, op, blocks_info, comm);
		
                        //res = swing_common->swing_reduce_scatter_mpi(sendbuf, recvbuf, datatype, op, blocks_info, comm);
                    }        
        
                    // Free blocks_info
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        free(blocks_info[p]);
                    }
                    free(blocks_info);
                    return res;
                }                    
                default:
                    assert("Invalid value for LIBSWING_REDUCE_SCATTER_ALGO" && 0);
            }
            break;
        }
        default:
            assert("Invalid value for LIBSWING_REDUCE_SCATTER_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

#ifdef FUGAKU
int MPI_Init(int *argc, char ***argv){
    int r = PMPI_Init(argc, argv);
    int dtsize;
    MPI_Type_size(MPI_INT, &dtsize);
    assert(dtsize == sizeof(int)); // This is needed in the reduce_local function since we assume MPI_INT is int

    MPI_Type_size(MPI_CHAR, &dtsize);
    assert(dtsize == sizeof(char)); // This is needed in the reduce_local function since we assume MPI_INT is int
    return r;
}

int MPI_Finalize(void){
    return PMPI_Finalize();
}
#endif

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
    read_env(comm);
    assert(sendtype == recvtype); // Right now not supported if datatypes are different
    assert(sendcount == recvcount); // Right now not supported if counts are different
    switch(env.allgather_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        case SWING_ALGO_FAMILY_RING:
            return swing_common->bucket_allgather((char*) sendbuf, (char*) recvbuf, sendcount*swing_common->get_size(), sendtype, MPI_SUM, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:{
            switch(env.allgather_config.algo){
                case SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS:
                case SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE:{
                    // We first split the data by block, and then by port (i.e., we split each block in num_ports parts). 
                    // This is the opposite of what we do in allreduce.
                    // Allocate blocks_info
                    BlockInfo** blocks_info = (BlockInfo**) malloc(sizeof(BlockInfo*)*swing_common->get_num_ports());
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        blocks_info[p] = (BlockInfo*) malloc(sizeof(BlockInfo)*swing_common->get_size());
                    }
                    size_t count_so_far = 0;
                    int dtsize;
                    MPI_Type_size(sendtype, &dtsize);
                    for(size_t i = 0; i < (size_t) swing_common->get_size(); i++){
                        size_t partition_size = recvcount / swing_common->get_num_ports();
                        size_t remaining = recvcount % swing_common->get_num_ports();                
                        size_t block_offset = count_so_far*dtsize;
                        size_t block_count_so_far = 0;
                        for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                            size_t count_port = partition_size + (p < remaining ? 1 : 0);
                            size_t offset_port = block_offset + block_count_so_far*dtsize;
                            block_count_so_far += count_port;
                            blocks_info[p][i].count = count_port;
                            blocks_info[p][i].offset = offset_port;
                            DPRINTF("[%d] Port %d Offset %d Count %d\n", swing_common->get_rank(), p, offset_port, count_port);
                        }
                        count_so_far += recvcount;
                    }


                    size_t count = swing_common->get_size()*recvcount*dtsize;
                    int res;
                    //res = swing_common->swing_coll_b(sendbuf, recvbuf, count, sendtype, op, comm, blocks_info, SWING_ALLGATHER);    
                    if(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_BLOCKS){
                        if(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                            res = swing_common->swing_allgather_blocks_utofu(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
                        }else{
                            res = swing_common->swing_allgather_blocks_mpi(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
                        }  
                    }else if(env.allgather_config.algo == SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_PERMUTE){
                        if(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                            res = swing_common->swing_allgather_utofu_contiguous(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
                        }else{
                            res = swing_common->swing_allgather_mpi_contiguous(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, blocks_info, comm);
                        }        
                    }

                    // Free blocks_info
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        free(blocks_info[p]);
                    }
                    free(blocks_info);        
                    return res;
                }
                case SWING_ALLGATHER_ALGO_VEC_DOUBLING_CONT_SEND:{
                    int res;
                    if(env.allgather_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        res = swing_common->swing_allgather_send_utofu(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
                    }else{
                        res = swing_common->swing_allgather_send_mpi(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
                    }  
                    return res;      
                }
                default:
                    assert("Invalid value for LIBSWING_ALLGATHER_ALGO" && 0);
            }
        }
        default:
            assert("Invalid value for LIBSWING_ALLGATHER_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

int MPI_Bcast_f(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm ){
    read_env(comm);
    switch(env.bcast_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Bcast(buffer, count, datatype, root, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:
            switch(env.bcast_config.algo){
                case SWING_BCAST_ALGO_BINOMIAL_TREE:{
                    if(env.bcast_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        return swing_common->swing_bcast_l(buffer, count, datatype, root, comm);
                    }else{
                        return swing_common->swing_bcast_l_mpi(buffer, count, datatype, root, comm);
                    }
                }
                case SWING_BCAST_ALGO_BINOMIAL_TREE_TMPBUF:{
                    if(env.bcast_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        return swing_common->swing_bcast_l_tmpbuf(buffer, count, datatype, root, comm);
                    }else{
                        assert("Invalid value for LIBSWING_BCAST_ALGO_LAYER" && 0);
                    }
                }
                case SWING_BCAST_ALGO_SCATTER_ALLGATHER:{
                    assert(env.bcast_config.distance_type == SWING_DISTANCE_INCREASING); // See notes
                    if(env.bcast_config.algo_layer == SWING_ALGO_LAYER_UTOFU){                        
                        return swing_common->swing_bcast_scatter_allgather(buffer, count, datatype, root, comm);
                    }else{
                        return swing_common->swing_bcast_scatter_allgather_mpi(buffer, count, datatype, root, comm);
                    }
                }
                default:
                    assert("Invalid value for LIBSWING_BCAST_ALGO" && 0);
            }
        default:
            assert("Invalid value for LIBSWING_BCAST_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){
    read_env(comm);
    switch(env.alltoall_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        case SWING_ALGO_FAMILY_SWING:{
            if(env.alltoall_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                return swing_common->swing_alltoall_utofu(sendbuf, recvbuf, sendcount, sendtype, comm);        
            }else{
                return swing_common->swing_alltoall_mpi(sendbuf, recvbuf, sendcount, sendtype, comm);        
            }
        }
        case SWING_ALGO_FAMILY_BRUCK:
            return swing_common->bruck_alltoall(sendbuf, recvbuf, sendcount, sendtype, comm);    
        default:
            assert("Invalid value for LIBSWING_ALLTOALL_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm){    
    read_env(comm);
    switch(env.scatter_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:{
            switch(env.scatter_config.algo){
                case SWING_SCATTER_ALGO_BINOMIAL_TREE_CONT_PERMUTE:{
                    // We first split the data by block, and then by port (i.e., we split each block in num_ports parts). 
                    // This is the opposite of what we do in allreduce.
                    // Allocate blocks_info
                    DPRINTF("Creating block infos.\n");
                    assert(recvcount >= swing_common->get_num_ports());
                    BlockInfo** blocks_info = (BlockInfo**) malloc(sizeof(BlockInfo*)*swing_common->get_num_ports());
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        blocks_info[p] = (BlockInfo*) malloc(sizeof(BlockInfo)*swing_common->get_size());
                    }
                    size_t count_so_far = 0;
                    int dtsize;
                    MPI_Type_size(sendtype, &dtsize);
                    size_t partition_size = recvcount / swing_common->get_num_ports();
                    size_t remaining = recvcount % swing_common->get_num_ports();     
                    
                    for(size_t i = 0; i < (size_t) swing_common->get_size(); i++){           
                        size_t block_offset = count_so_far*dtsize;
                        size_t block_count_so_far = 0;
                        for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                            size_t count_port = partition_size + (p < remaining ? 1 : 0);
                            size_t offset_port = block_offset + block_count_so_far*dtsize;
                            block_count_so_far += count_port;
                            blocks_info[p][i].count = count_port;
                            blocks_info[p][i].offset = offset_port;
                            DPRINTF("[%d] Port %d Offset %d Count %d\n", swing_common->get_rank(), p, offset_port, count_port);
                        }
                        count_so_far += recvcount;
                    }

                    int res = MPI_SUCCESS;
                    if(env.scatter_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        res = swing_common->swing_scatter_utofu(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, blocks_info, comm);
                    }else{
                        res = swing_common->swing_scatter_mpi(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, blocks_info, comm);
                    }

                    // Free blocks_info
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        free(blocks_info[p]);
                    }
                    free(blocks_info);        
                    return res;
                }
                default:
                    assert("Invalid value for LIBSWING_SCATTER_ALGO" && 0);
            }
        }
        default:
            assert("Invalid value for LIBSWING_SCATTER_ALGO" && 0);
    }
    return MPI_ERR_OTHER;
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                MPI_Comm comm){    
    read_env(comm);
    switch(env.gather_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:{
            switch(env.gather_config.algo){
                case SWING_GATHER_ALGO_BINOMIAL_TREE_CONT_PERMUTE:{
                    // We first split the data by block, and then by port (i.e., we split each block in num_ports parts). 
                    // This is the opposite of what we do in allreduce.
                    // Allocate blocks_info
                    DPRINTF("Creating block infos.\n");
                    assert(recvcount >= swing_common->get_num_ports());
                    BlockInfo** blocks_info = (BlockInfo**) malloc(sizeof(BlockInfo*)*swing_common->get_num_ports());
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        blocks_info[p] = (BlockInfo*) malloc(sizeof(BlockInfo)*swing_common->get_size());
                    }
                    size_t count_so_far = 0;
                    int dtsize;
                    MPI_Type_size(sendtype, &dtsize);
                    size_t partition_size = recvcount / swing_common->get_num_ports();
                    size_t remaining = recvcount % swing_common->get_num_ports();     
                    
                    for(size_t i = 0; i < (size_t) swing_common->get_size(); i++){           
                        size_t block_offset = count_so_far*dtsize;
                        size_t block_count_so_far = 0;
                        for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                            size_t count_port = partition_size + (p < remaining ? 1 : 0);
                            size_t offset_port = block_offset + block_count_so_far*dtsize;
                            block_count_so_far += count_port;
                            blocks_info[p][i].count = count_port;
                            blocks_info[p][i].offset = offset_port;
                            DPRINTF("[%d] Port %d Offset %d Count %d\n", swing_common->get_rank(), p, offset_port, count_port);
                        }
                        count_so_far += recvcount;
                    }

                    int res = MPI_SUCCESS;                    
                    if(env.gather_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        res = swing_common->swing_gather_utofu(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, blocks_info, comm);
                    }else{
                        res = swing_common->swing_gather_mpi(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, blocks_info, comm);
                    }

                    // Free blocks_info
                    for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                        free(blocks_info[p]);
                    }
                    free(blocks_info);        
                    return res;
                }
                default:
                    assert("Invalid value for LIBSWING_GATHER_ALGO" && 0);
            }
        }
        default:
            assert("Invalid value for LIBSWING_GATHER_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
    read_env(comm);
    switch(env.reduce_config.algo_family){
        case SWING_ALGO_FAMILY_DEFAULT:
            return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
        case SWING_ALGO_FAMILY_SWING:
        case SWING_ALGO_FAMILY_RECDOUB:
            switch(env.reduce_config.algo){
                case SWING_REDUCE_ALGO_BINOMIAL_TREE:
                    if(env.reduce_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        return swing_common->swing_reduce_utofu(sendbuf, recvbuf, count, datatype, op, root, comm);
                    }else{
                        return swing_common->swing_reduce_mpi(sendbuf, recvbuf, count, datatype, op, root, comm);
                    }
                case SWING_REDUCE_ALGO_REDUCE_SCATTER_GATHER:
                    if(env.reduce_config.algo_layer == SWING_ALGO_LAYER_UTOFU){
                        int dtsize;
                        MPI_Type_size(datatype, &dtsize);
                        BlockInfo** blocks_info = get_blocks_info(count, swing_common, dtsize);
                        int r = swing_common->swing_reduce_redscat_gather_utofu(sendbuf, recvbuf, count, datatype, op, root, comm, blocks_info);
                        // Free blocks_info
                        for(size_t p = 0; p < swing_common->get_num_ports(); p++){
                            free(blocks_info[p]);
                        }
                        free(blocks_info);
                        return r;
                    }else{
                        return swing_common->swing_reduce_redscat_gather_mpi(sendbuf, recvbuf, count, datatype, op, root, comm);
                    }
                default:
                    assert("Invalid value for LIBSWING_REDUCE_ALGO" && 0);
            }
        default:
            assert("Invalid value for LIBSWING_REDUCE_ALGO_FAMILY" && 0);
    }
    return MPI_ERR_OTHER;
}

// TODO: Don't use Swing for non-continugous non-native datatypes (tedious implementation)
