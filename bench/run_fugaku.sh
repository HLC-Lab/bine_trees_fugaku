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
PORTS=""
while getopts n:e:d:p: flag
do
    case "${flag}" in
        n) NODES=($(echo "${OPTARG}" | tr ',' '\n'));;
        e) EXTRA="_"${OPTARG};;
        d) DIMENSIONS=${OPTARG};;
        p) PORTS=${OPTARG};;
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

for p in "${NODES[@]}"
do
    echo ${SYSTEM}${EXTRA},${p},${OUT_FOLDER} >> ../data/description.csv
    COLLECTIVE="MPI_Allreduce"    
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


        if [ ${PORTS} -eq 1 ]; then
            coll_tuned_prealloc_size=1024 # This is in MiB

            DEFAULT_ALGO="default"
            # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
            EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"
            # TODO: Maybe I should prealloc only for large allreduce?
            LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} DATATYPE ${n} ${iterations}
            mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
            #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
            #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
            
            # Decision rules can be found at /opt/FJSVxtclanga/.common/MECA030/etc/fjmpi-dectree.conf (but you can access it only on computing nodes after doing pjsub)

            if [ $n -le 512 ]; then
                for DEFAULT_ALGO in "basic_linear"
                do
                    LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} DATATYPE ${n} ${iterations}
                    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
                    #mkdir ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                    #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_ALGO}_stats/
                done
            fi


            # All those that do not have segsize as parameter
            for DEFAULT_ALGO in "rdbc" "ring" "recursive_doubling" "nonoverlapping"
            do
                LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} DATATYPE ${n} ${iterations}
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
        fi

	
        PREALLOC_SIZE=536870912
		    
	    # Run lat optimal swing and lat optimal recdoub
        if [ $n -le 1048576 ]; then
        for SEGMENT_SIZE in 4096 65536 1048576
        do
		    LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
		    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_lat_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
		    #mkdir ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_e/
		    #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_e/	       
		    #mkdir ${OUT_FOLDER}/${p}_${n}_lat_stats/
		    #mv ${ERR_PREFIX}* ${OUT_FOLDER}/${p}_${n}_lat_stats/

		    LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="RECDOUB_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
		    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_lat_rd_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
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
            MIN_ELEMS=$((PORTS * p))
            if [ "$n" -ge "$MIN_ELEMS" ]; then
                LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_UTOFU" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
                #mkdir ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports_stats/
                #mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports_stats/	

                LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="RECDOUB_B_UTOFU" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_rd_utofu_${PORTS}_${SEGMENT_SIZE}.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
            fi
        done
        echo " ${GREEN}[Done]${NC}"
    done


    COLLECTIVE="MPI_Bcast"
    #for n in 1 8 64 512 2048 16384 131072 1048576 8388608 67108864
    for n in 64 16384 8388608
    do
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

        if [ ${PORTS} -eq 1 ]; then
            coll_tuned_prealloc_size=1024 # This is in MiB

            DEFAULT_ALGO="default"
            # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
            EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"
            # TODO: Maybe I should prealloc only for large allreduce?
            LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations}
            mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bcast_default_${DEFAULT_ALGO}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
        fi

        
        PREALLOC_SIZE=536870912
        
        for SEGMENT_SIZE in 4096 65536 1048576
        do
            LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_ALGO="SWING_L_UTOFU" LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} LIBSWING_NUM_PORTS=${PORTS} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations}
            mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bcast_lat_utofu_${PORTS}.csv; rm -f ${OUT_PREFIX}* #${ERR_PREFIX}*
        done
    done
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
#rm -rf ${TIMESTAMP}
popd
