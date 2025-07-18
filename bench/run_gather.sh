#!/bin/bash
COLLECTIVE="MPI_Gather"

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

export MPI_OP="null"
python3 generate_metadata.py ${EXP_ID} || exit 1

for n in 1 8 64 512 4096 32768 262144 2097152 16777216 134217728
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

    actual_count=$((n / p))
    msg_size=$((actual_count * SIZEOF_DATATYPE))
    total_msg_size=$((n * SIZEOF_DATATYPE))

    # Skip if the actual count is less than 1
    if [ $actual_count -lt 1 ]; then
        continue
    fi

    echo -n "Running on "${DIMENSIONS}" (${p} nodes) with count="${n}" (actual count=${actual_count})..."

    #########################
    # Run the default algos #
    #########################
    export LIBBINE_GATHER_ALGO_FAMILY="DEFAULT" 
    export LIBBINE_GATHER_ALGO_LAYER="MPI" 

    coll_tuned_prealloc_size=1539 # This is in MiB (1.5 GiB + 3MiB)
    PREALLOC_SIZE=1610612736 # 1.5 GiB
    
    # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
    # TODO: Maybe I should prealloc only for large GATHER?
    EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"

    # Do a run just to print the decision process
    DEFAULT_ALGO="default"
    LIBBINE_GATHER_ALGO_FAMILY="DEFAULT" ${MPIRUN} -mca coll_select_show_decision_process 2 -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} 1
    ALGO_FNAME=default-${DEFAULT_ALGO}
    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.decision; rm -f ${OUT_PREFIX}* 
    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.decision.err; rm -f ${ERR_PREFIX}*; fi

    ## Run the actual benchmark
    start_time=$(date +%s)
    DEFAULT_ALGO="default"    
    LIBBINE_GATHER_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} ${iterations}
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
    
    for DEFAULT_ALGO in "simple" "binomial" #"basic_linear" "linear_sync" 
    do        
        export LIBBINE_GATHER_ALGO_FAMILY="DEFAULT" 
        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_gather_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} ${iterations}
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
        if [ $actual_count -ge $PORTS ]; then
            export LIBBINE_NUM_PORTS=${PORTS}
            # Run bine (INC)
            export LIBBINE_GATHER_ALGO_FAMILY="BINE" 
            export LIBBINE_GATHER_ALGO_LAYER="UTOFU" 
            export LIBBINE_GATHER_ALGO="BINOMIAL_TREE_CONT_PERMUTE" 
            export LIBBINE_GATHER_DISTANCE=INCREASING     
            for SEGMENT_SIZE in 0 #4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $total_msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} ${iterations}                    
                    ALGO_FNAME=${LIBBINE_GATHER_ALGO_FAMILY}-${LIBBINE_GATHER_ALGO}-INC-${LIBBINE_GATHER_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
            unset LIBBINE_GATHER_DISTANCE

            # Run bine (DEC)
            export LIBBINE_GATHER_ALGO_FAMILY="BINE" 
            export LIBBINE_GATHER_ALGO_LAYER="UTOFU" 
            export LIBBINE_GATHER_ALGO="BINOMIAL_TREE_CONT_PERMUTE" 
            export LIBBINE_GATHER_DISTANCE=DECREASING     
            for SEGMENT_SIZE in 0 #4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $total_msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} ${iterations}                    
                    ALGO_FNAME=${LIBBINE_GATHER_ALGO_FAMILY}-${LIBBINE_GATHER_ALGO}-DEC-${LIBBINE_GATHER_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
            unset LIBBINE_GATHER_DISTANCE

            if [ $PORTS -eq 1 ]; then
                export LIBBINE_GATHER_ALGO_FAMILY="BINE" 
                export LIBBINE_GATHER_ALGO_LAYER="MPI" 
                export LIBBINE_GATHER_ALGO="BINOMIAL_TREE_CONT_PERMUTE"
                timeout ${max_duration} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} ${iterations}
                sleep 2 # To avoid running the next job to early in the case we killed this one
                ALGO_FNAME=${LIBBINE_GATHER_ALGO_FAMILY}-${LIBBINE_GATHER_ALGO}-${LIBBINE_GATHER_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
            fi

        fi
    done

    #PORTS=1
    #export LIBBINE_NUM_PORTS=${PORTS}
    #if [ $actual_count -ge $PORTS ]; then
    #    # Run recdoub
    #    export LIBBINE_GATHER_ALGO_FAMILY="RECDOUB" 
    #    export LIBBINE_GATHER_ALGO_LAYER="UTOFU" 
    #    export LIBBINE_GATHER_ALGO="BINOMIAL_TREE_CONT_PERMUTE"    
    #    for SEGMENT_SIZE in 0 #4096 65536 1048576
    #    do                
    #        if [ $SEGMENT_SIZE -lt $total_msg_size ]; then
    #            LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${actual_count} ${iterations}                    
    #            ALGO_FNAME=${LIBBINE_GATHER_ALGO_FAMILY}-${LIBBINE_GATHER_ALGO}-${LIBBINE_GATHER_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
    #            mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    #            if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    #        fi
    #    done  
    #fi
    echo " ${GREEN}[Done]${NC}"
done

