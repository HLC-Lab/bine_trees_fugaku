#!/bin/bash
pushd ..
source conf.sh
popd

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

NODES=""
EXTRA=""
DIMENSIONS=""
PORTS_LIST=""
while getopts n:e:d:p:c: flag
do
    case "${flag}" in
        n) NODES=($(echo "${OPTARG}" | tr ',' '\n'));;
        e) EXTRA="_"${OPTARG};;
        d) DIMENSIONS=${OPTARG};;
        p) PORTS_LIST=${OPTARG};;
        c) COLLECTIVES=($(echo "${OPTARG}" | tr ',' '\n'));;
    esac
done

OUT_PATH=../data/${SYSTEM}
TIMESTAMP=$(TZ=":Europe/Rome" date +%Y_%m_%d_%H_%M_%S)
OUT_FOLDER=${OUT_PATH}/${TIMESTAMP}
mkdir -p ${OUT_FOLDER}

OUT_PREFIX="swing_out"
ERR_PREFIX="swing_err"
MPIEXEC_OUT="-stdout-proc /vol0004/mdt1/home/u12936/swing-allreduce/bench/${OUT_PREFIX} -stderr-proc /vol0004/mdt1/home/u12936/swing-allreduce/bench/${ERR_PREFIX}"

DATATYPE="INT32"
SIZEOF_DATATYPE=4
for COLLECTIVE in "${COLLECTIVES[@]}"
do
    for p in "${NODES[@]}"
    do
        echo ${SYSTEM}${EXTRA},${p},${OUT_FOLDER} >> ../data/description.csv
        if [ ${COLLECTIVE} == "MPI_Allreduce" ]; then
            for n in 1 8 64 512 2048 16384 131072 1048576 8388608 67108864
            do
                msg_size=$((n * SIZEOF_DATATYPE))
                iterations=0
                if [ $n -le 512 ]
                then
                    iterations=100 #00
                elif [ $n -le 1048576 ]
                then
                    iterations=100 #0
                elif [ $n -le 8388608 ]
                then
                    iterations=100
                elif [ $n -le 67108864 ]
                then
                    iterations=10
                else
                    iterations=4
                fi
                echo -n "Running on "${p}" nodes with count="${n}"..."

                coll_tuned_prealloc_size=1024 # This is in MiB

                DEFAULT_ALGO="default"
                # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
                EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"
                # TODO: Maybe I should prealloc only for large allreduce?
                LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                
                # Decision rules can be found at /opt/FJSVxtclanga/.common/MECA030/etc/fjmpi-dectree.conf (but you can access it only on computing nodes after doing pjsub)

                if [ $n -le 512 ]; then
                    for DEFAULT_ALGO in "basic_linear"
                    do
                        LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                        mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                        #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                        #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                    done
                fi


                # All those that do not have segsize as parameter
                for DEFAULT_ALGO in "rdbc" "ring" "recursive_doubling" "nonoverlapping"
                do
                    LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                    #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                    #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                done

                # For segmented ring we just use 1MiB as segment size (is never used/selected by fjmpi anyway)
                for DEFAULT_ALGO in "segmented_ring"
                do
                    for coll_select_allreduce_algorithm_segmentsize in 1048576
                    do
                        LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_select_allreduce_algorithm_segmentsize ${coll_select_allreduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                        mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                        #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                        #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                    done            
                done

                # Those that have segsize as parameter
                for DEFAULT_ALGO in "trix6" "trix3"
                do
                    # Same rules as in the fjmpirules file
                    if [ $msg_size -le 393216 ]
                    then
                        coll_select_allreduce_algorithm_segmentsize=4096
                    elif [ $msg_size -le 6291456 ]
                    then
                        coll_select_allreduce_algorithm_segmentsize=16384
                    else
                        coll_select_allreduce_algorithm_segmentsize=65536
                    fi

                    LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_select_allreduce_algorithm_segmentsize ${coll_select_allreduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                    #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                    #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                done
            
                PREALLOC_SIZE=536870912
                
                for PORTS in ${PORTS_LIST//,/ }
                do
                    # Run lat optimal swing and lat optimal recdoub
                    if [ $n -le 1048576 ]; then
                        for SEGMENT_SIZE in 4096 65536 1048576
                        do
                            if [ $SEGMENT_SIZE -lt $msg_size ]; then
                                LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_lat_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                                #mkdir ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_e/
                                #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_e/	       
                                #mkdir ${OUT_FOLDER}/${p}_${n}_lat_stats/
                                #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_lat_stats/

                                if [ $PORTS -eq 1 ]; then
                                    if [ $SEGMENT_SIZE -eq 1048576 ]; then
                                        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="RECDOUB_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                                        mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_lat_rd_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                                    fi
                                fi
                            fi
                        done
                    fi
                    #LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                    #mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
                    #mkdir ${OUT_FOLDER}/${p}_${n}_bw_${PORTS}_ports_stats/
                    #mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_${PORTS}_ports_stats/	
                
                    #LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_COALESCE" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                    #mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_coalesce_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
                    #mkdir ${OUT_FOLDER}/${p}_${n}_bw_coalesce_${PORTS}_ports_stats/
                    #mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_coalesce_${PORTS}_ports_stats/
                
                    #LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_CONT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                    #mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_cont_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
                    #mkdir ${OUT_FOLDER}/${p}_${n}_bw_cont_${PORTS}_ports_stats/
                    #mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_cont_${PORTS}_ports_stats/	
                
                    for SEGMENT_SIZE in 4096 65536 1048576
                    do
                        if [ $SEGMENT_SIZE -lt $msg_size ]; then
                            MIN_ELEMS=$((PORTS * p))
                            if [ "$n" -ge "$MIN_ELEMS" ]; then
                                LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_UTOFU" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
                                #mkdir ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports_stats/
                                #mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports_stats/	

                                if [ $PORTS -eq 1 ]; then
                                    if [ $SEGMENT_SIZE -eq 1048576 ]; then
                                        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="RECDOUB_B_UTOFU" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                                        mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_rd_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
                                    fi
                                fi
                            fi
                        fi
                    done
                    echo " ${GREEN}[Done]${NC}"
                done
            done
        elif [ ${COLLECTIVE} == "MPI_Bcast" ]; then
            #for n in 1 8 64 512 2048 16384 131072 1048576 8388608 67108864
            for n in 64 16384 8388608
            do
                msg_size=$((n * SIZEOF_DATATYPE))
                iterations=0
                if [ $n -le 512 ]
                then
                    iterations=100 #00
                elif [ $n -le 1048576 ]
                then
                    iterations=100 #0
                elif [ $n -le 8388608 ]
                then
                    iterations=100
                elif [ $n -le 67108864 ]
                then
                    iterations=10
                else
                    iterations=4
                fi
                echo -n "Running BCAST on "${p}" nodes with count="${n}"..."


                coll_tuned_prealloc_size=1024 # This is in MiB

                DEFAULT_ALGO="default"
                # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
                EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"
                # TODO: Maybe I should prealloc only for large allreduce?
                LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bcast_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                
                PREALLOC_SIZE=536870912
                
                for PORTS in ${PORTS_LIST//,/ }
                do
                    for SEGMENT_SIZE in 1024 8192 16384 65536 1048576
                    do
                        if [ $SEGMENT_SIZE -lt $msg_size ]; then
                            LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="RECDOUB_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                            mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bcast_lat_rd_utofu_${PORTS}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                            
                            LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                            mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bcast_lat_utofu_${PORTS}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*

                            LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L_UTOFU" LIBSWING_UTOFU_ADD_AG=1 LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                            mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bcast_lat_utofu_ag_${PORTS}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                        fi
                    done
                done
            done
        elif [ ${COLLECTIVE} == "MPI_Alltoall" ]; then
            for n in 1 8 64 512 2048 16384 131072 1048576
            do
                msg_size=$((n * SIZEOF_DATATYPE))
                iterations=0
                if [ $n -le 512 ]
                then
                    iterations=100 #00
                elif [ $n -le 1048576 ]
                then
                    iterations=100 #0
                elif [ $n -le 8388608 ]
                then
                    iterations=100
                elif [ $n -le 67108864 ]
                then
                    iterations=10
                else
                    iterations=4
                fi
                echo -n "Running A2A on "${p}" nodes with count="${n}"..."

                coll_tuned_prealloc_size=1024 # This is in MiB
                PREALLOC_SIZE=536870912

                DEFAULT_ALGO="default"
                # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
                EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"
                # TODO: Maybe I should prealloc only for large allreduce?
                LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_a2a_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*

                DEFAULT_ALGO="modified_bruck"
                LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_select_alltoall_algorithm ${DEFAULT_ALGO} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_a2a_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                                 

                LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="BRUCK" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_a2a_bruck.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*

                LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="BRUCK" LIBSWING_PREALLOC_SIZE=0 LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_a2a_bruck_no_prealloc.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                
                SEGMENT_SIZE=0
                for PORTS in ${PORTS_LIST//,/ }
                do
                    LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_a2a_lat_${PORTS}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*

                    LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L" LIBSWING_PREALLOC_SIZE=0 LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE}  ${DATATYPE} ${n} ${iterations}
                    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_a2a_lat_no_prealloc_${PORTS}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                done
            done
        fi
    done
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
#rm -rf ${TIMESTAMP}
popd
