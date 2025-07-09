#!/bin/bash
COLLECTIVE="MPI_Reduce"

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
EXP_ID=""
while getopts d:p:o:i: flag
do
    case "${flag}" in
        d) DIMENSIONS=${OPTARG};;
        p) PORTS_LIST=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
        i) EXP_ID=${OPTARG};;
    esac
done

OUT_PREFIX="bine_out_${TIMESTAMP}"
ERR_PREFIX="bine_err_${TIMESTAMP}"
MPIEXEC_OUT="-stdout-proc /vol0004/mdt1/home/u12936/bine-allreduce/bench/${OUT_PREFIX} -stderr-proc /vol0004/mdt1/home/u12936/bine-allreduce/bench/${ERR_PREFIX}"

DATATYPE="INT32"
SIZEOF_DATATYPE=4
DATATYPE_lc=$(echo ${DATATYPE} | tr '[:upper:]' '[:lower:]')

# Split the values in DIMENSIONS (by x), and multiply them
IFS='x' read -r -a DIMENSIONS_ARRAY <<< "$DIMENSIONS"
p=1
for i in "${DIMENSIONS_ARRAY[@]}"
do
    p=$((p * i))
done

export MPI_OP="MPI_SUM"
python3 generate_metadata.py ${EXP_ID} || exit 1

#for n in 1 8 64 512 4096 32768 262144 2097152 16777216 134217728
for n in 262144 2097152 16777216 134217728
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
        iterations=10
    fi
    echo -n "Running on "${DIMENSIONS}" (${p} nodes) with count="${n}"..."
    msg_size=$((n * SIZEOF_DATATYPE))        

    #########################
    # Run the default algos #
    #########################
    export LIBBINE_REDUCE_ALGO_FAMILY="DEFAULT" 
    export LIBBINE_REDUCE_ALGO_LAYER="MPI" 

    coll_tuned_prealloc_size=1539 # This is in MiB (1.5 GiB + 3MiB)
    PREALLOC_SIZE=1610612736 # 1.5 GiB
    
    # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
    # TODO: Maybe I should prealloc only for large REDUCE?
    EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"

    # Do a run just to print the decision process
    DEFAULT_ALGO="default"
    LIBBINE_REDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} -mca coll_select_show_decision_process 2 -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} 1
    ALGO_FNAME=default-${DEFAULT_ALGO}
    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.decision; rm -f ${OUT_PREFIX}* 
    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.decision.err; rm -f ${ERR_PREFIX}*; fi

    ## Run the actual benchmark
    start_time=$(date +%s)
    DEFAULT_ALGO="default"    
    LIBBINE_REDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    ALGO_FNAME=default-${DEFAULT_ALGO}
    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    end_time=$(date +%s)
    max_duration=$(( (end_time - start_time) * 10 ))
    # If max duration is less than 1 seconds, set it to 3 seconds
    if [ $max_duration -le 1 ]; then
        max_duration=10
    fi
    #echo "Running defaults for at most ${max_duration} seconds"    
    
    # Disable uTofu barrier for non-default-default algos
    if [ $n -le 12 ]
    then
        # Append -mca coll ^tbi to EXTRA_MCAS
        EXTRA_MCAS="${EXTRA_MCAS} -mca coll ^tbi"
    fi

    # Always slower than the others
    #for DEFAULT_ALGO in "linear" "chain" "pipeline"
    #do        
    #    export LIBBINE_REDUCE_ALGO_FAMILY="DEFAULT" 
    #    ${MPIRUN} ${EXTRA_MCAS}  -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_reduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    #    ALGO_FNAME=default-$(echo ${DEFAULT_ALGO} | tr '_' '-')
    #    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    #    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    #done

    for DEFAULT_ALGO in "binary" "binomial" "trinaryx6" "trinaryx3"
    do        
        # Same rules as in the fjmpirules file
        if [ $msg_size -le 24577 ]
        then
            coll_select_reduce_algorithm_segmentsize=0
        elif [ $msg_size -le 49152 ]
        then
            coll_select_reduce_algorithm_segmentsize=1024
        elif [ $msg_size -le 393216 ]
        then
            coll_select_reduce_algorithm_segmentsize=4096
        elif [ $msg_size -le 6291456 ]
        then
            coll_select_reduce_algorithm_segmentsize=16384
        else
            coll_select_reduce_algorithm_segmentsize=65536
        fi

        export LIBBINE_REDUCE_ALGO_FAMILY="DEFAULT" 
        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_select_reduce_algorithm_segmentsize ${coll_select_reduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_reduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
        ALGO_FNAME=default-$(echo ${DEFAULT_ALGO} | tr '_' '-')
        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    done

    for DEFAULT_ALGO in "in-order_binary"
    do        
        # Same rules as in the fjmpirules file
        coll_select_reduce_algorithm_segmentsize=65536

        export LIBBINE_REDUCE_ALGO_FAMILY="DEFAULT" 
        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_select_reduce_algorithm_segmentsize ${coll_select_reduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_reduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
        ALGO_FNAME=default-$(echo ${DEFAULT_ALGO} | tr '_' '-')
        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    done


    #######################
    # Run the Bine algos #
    #######################
    export LIBBINE_DIMENSIONS=${DIMENSIONS} 
    export LIBBINE_PREALLOC_SIZE=${PREALLOC_SIZE} 
    export LIBBINE_UTOFU_ADD_AG=1
    for PORTS in ${PORTS_LIST//,/ }
    do
        export LIBBINE_NUM_PORTS=${PORTS}
        if [ $n -ge $PORTS ]; then        
            # Run bine binomial tree
            export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
            export LIBBINE_REDUCE_ALGO_LAYER="UTOFU" 
            export LIBBINE_REDUCE_ALGO="BINOMIAL_TREE"    
            for SEGMENT_SIZE in 0 4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
                    ALGO_FNAME=${LIBBINE_REDUCE_ALGO_FAMILY}-${LIBBINE_REDUCE_ALGO}-${LIBBINE_REDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done

            if [ $PORTS -eq 1 ]; then
                export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
                export LIBBINE_REDUCE_ALGO_LAYER="MPI" 
                export LIBBINE_REDUCE_ALGO="BINOMIAL_TREE"
                timeout ${max_duration} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                sleep 2 # To avoid running the next job to early in the case we killed this one
                ALGO_FNAME=${LIBBINE_REDUCE_ALGO_FAMILY}-${LIBBINE_REDUCE_ALGO}-${LIBBINE_REDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
            fi
        fi

        actual_count=$((n / p))
        if [ $actual_count -ge $PORTS ]; then
            # Run bine redscat gather
            export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
            export LIBBINE_REDUCE_ALGO_LAYER="UTOFU" 
            export LIBBINE_REDUCE_ALGO="REDUCE_SCATTER_GATHER"
            export LIBBINE_REDUCE_DISTANCE="INCREASING"
            for SEGMENT_SIZE in 0 #16384 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
                    ALGO_FNAME=${LIBBINE_REDUCE_ALGO_FAMILY}-${LIBBINE_REDUCE_ALGO}-${LIBBINE_REDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
            unset LIBBINE_REDUCE_DISTANCE


            if [ $PORTS -eq 1 ]; then
                export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
                export LIBBINE_REDUCE_ALGO_LAYER="MPI" 
                export LIBBINE_REDUCE_ALGO="REDUCE_SCATTER_GATHER"
                timeout ${max_duration} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
                sleep 2 # To avoid running the next job to early in the case we killed this one
                ALGO_FNAME=${LIBBINE_REDUCE_ALGO_FAMILY}-${LIBBINE_REDUCE_ALGO}-${LIBBINE_REDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
            fi
        fi
    done

    #PORTS=1
    #export LIBBINE_NUM_PORTS=${PORTS}
    ## Run recdoub binomial tree
    #export LIBBINE_REDUCE_ALGO_FAMILY="RECDOUB" 
    #export LIBBINE_REDUCE_ALGO_LAYER="UTOFU" 
    #export LIBBINE_REDUCE_ALGO="BINOMIAL_TREE"    
    #for SEGMENT_SIZE in 0 4096 65536 1048576
    #do                
    #    if [ $SEGMENT_SIZE -lt $msg_size ]; then
    #        LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
    #        ALGO_FNAME=${LIBBINE_REDUCE_ALGO_FAMILY}-${LIBBINE_REDUCE_ALGO}-${LIBBINE_REDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
    #        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    #        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    #    fi
    #done                


#    # Run recdoub redscat gather
#    actual_count=$((n / p))
#    if [ $actual_count -ge $PORTS ]; then
#        export LIBBINE_REDUCE_ALGO_FAMILY="RECDOUB" 
#        export LIBBINE_REDUCE_ALGO_LAYER="UTOFU" 
#        export LIBBINE_REDUCE_ALGO="REDUCE_SCATTER_GATHER"
#        export LIBBINE_REDUCE_DISTANCE="INCREASING"
#        for SEGMENT_SIZE in 0 #4096 65536 1048576
#        do                
#            if [ $SEGMENT_SIZE -lt $msg_size ]; then
#                LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
#                ALGO_FNAME=${LIBBINE_REDUCE_ALGO_FAMILY}-${LIBBINE_REDUCE_ALGO}-${LIBBINE_REDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
#                mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
#                if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
#            fi
#        done
#        unset LIBBINE_REDUCE_DISTANCE
#    fi    
    
    echo " ${GREEN}[Done]${NC}"
done

