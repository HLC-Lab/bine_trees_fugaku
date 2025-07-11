#!/bin/bash

# Define function to get Bine optimal segment size based on msg size
#function get_bine_optimal_segment_size() {
#    local msg_size=$1
#    local ports=1
#    local segment_size=0
#    if [ $msg_size -le 16384 ]
#    then
#        segment_size=0
#    elif [ $msg_size -le 131072 ]
#    then
#        segment_size=4096
#    else
#        segment_size=65536
#    fi
#    echo $segment_size
#}
#
## Define function to get Bine optimal segment size based on msg size
#function get_bine_optimal_ports() {
#    local msg_size=$1
#    local ports=1
#    if [ $msg_size -le 131072 ]
#    then
#        ports=1
#    else
#        ports=6
#    fi
#    echo $ports
#}


COLLECTIVE="MPI_Allreduce"

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
    echo -n "Running on "${DIMENSIONS}" (${p} nodes) with count="${n}"..."
    msg_size=$((n * SIZEOF_DATATYPE))
    
    #########################
    # Run the default algos #
    #########################
    export LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" 
    export LIBBINE_ALLREDUCE_ALGO_LAYER="MPI" 

    coll_tuned_prealloc_size=1539 # This is in MiB (1.5 GiB + 3MiB)
    PREALLOC_SIZE=1610612736 # 1.5 GiB
    
    # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
    # TODO: Maybe I should prealloc only for large allreduce?
    EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"

    ## Do a run just to print the decision process
    DEFAULT_ALGO="default"
    LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_select_show_decision_process 2 -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    ALGO_FNAME=default-${DEFAULT_ALGO}
    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.decision; rm -f ${OUT_PREFIX}* 
    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.decision.err; rm -f ${ERR_PREFIX}*; fi

    ## Run the actual benchmark
    start_time=$(date +%s)
    DEFAULT_ALGO="default"
    LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    ALGO_FNAME=default-${DEFAULT_ALGO}
    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    end_time=$(date +%s)
    max_duration=$(( (end_time - start_time) * 2 ))
    # If max duration is less than 1 seconds, set it to 3 seconds
    if [ $max_duration -le 1 ]; then
        max_duration=3
    fi
    #echo "Running defaults for at most ${max_duration} seconds"        
    
    # Disable uTofu barrier for non-default-default algos
    if [ $n -le 12 ]
    then
        # Append -mca coll ^tbi to EXTRA_MCAS
        EXTRA_MCAS="${EXTRA_MCAS} -mca coll ^tbi"
    fi

    # Decision rules can be found at /opt/FJSVxtclanga/.common/MECA030/etc/fjmpi-dectree.conf (but you can access it only on computing nodes after doing pjsub)
    # We do not use uTofu barrier when we force the algo (we use it only for default_default)
    #if [ $n -le 512 ]; then
    #    for DEFAULT_ALGO in "basic_linear"
    #    do
    #        export LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" 
    #        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    #        ALGO_FNAME=default-${DEFAULT_ALGO}
    #        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    #        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    #    done
    #fi

    # All those that do not have segsize as parameter
    for DEFAULT_ALGO in "rdbc" "recursive_doubling" "nonoverlapping" #"ring" 
    do
        export LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" 
        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
        ALGO_FNAME=default-${DEFAULT_ALGO}
        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    done

    ## For segmented ring we just use 1MiB as segment size (is never used/selected by fjmpi anyway)
    #for DEFAULT_ALGO in "segmented_ring"
    #do
    #    for coll_select_allreduce_algorithm_segmentsize in 1048576
    #    do
    #        export LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" 
    #        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_select_allreduce_algorithm_segmentsize ${coll_select_allreduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    #        ALGO_FNAME=default-${DEFAULT_ALGO}
    #        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    #        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    #    done            
    #done

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

        export LIBBINE_ALLREDUCE_ALGO_FAMILY="DEFAULT" 
        ${MPIRUN} ${EXTRA_MCAS}  -mca coll_select_allreduce_algorithm_segmentsize ${coll_select_allreduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
        ALGO_FNAME=default-${DEFAULT_ALGO}
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
        # Run lat optimal bine 
        export LIBBINE_ALLREDUCE_ALGO_FAMILY="BINE" 
        export LIBBINE_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBBINE_ALLREDUCE_ALGO="L"    
        MIN_ELEMS=$((PORTS))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 #4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then                    
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
                    ALGO_FNAME=${LIBBINE_ALLREDUCE_ALGO_FAMILY}-${LIBBINE_ALLREDUCE_ALGO}-${LIBBINE_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi

        # Run bw optimal bine
        export LIBBINE_ALLREDUCE_ALGO_FAMILY="BINE" 
        export LIBBINE_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBBINE_ALLREDUCE_ALGO="B_CONT"    
        MIN_ELEMS=$((PORTS * p))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 65536 1048576 #4096
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                    ALGO_FNAME=${LIBBINE_ALLREDUCE_ALGO_FAMILY}-${LIBBINE_ALLREDUCE_ALGO}-${LIBBINE_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi

        # Run bucket
        export LIBBINE_SKIP_VALIDATION=1
        export LIBBINE_ALLREDUCE_ALGO_FAMILY="RING" 
        export LIBBINE_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBBINE_ALLREDUCE_ALGO="B_CONT"    
        MIN_ELEMS=$((PORTS * p))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                    ALGO_FNAME=${LIBBINE_ALLREDUCE_ALGO_FAMILY}-${LIBBINE_ALLREDUCE_ALGO}-${LIBBINE_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi  
        unset LIBBINE_SKIP_VALIDATION      

        #########################
        # Run the Recdoub algos #
        #########################
        # Run lat optimal recdoub
        export LIBBINE_ALLREDUCE_ALGO_FAMILY="RECDOUB" 
        export LIBBINE_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBBINE_ALLREDUCE_ALGO="L"    
        MIN_ELEMS=$((PORTS))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 #4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
                    ALGO_FNAME=${LIBBINE_ALLREDUCE_ALGO_FAMILY}-${LIBBINE_ALLREDUCE_ALGO}-${LIBBINE_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi

        # Run bw optimal recdoub
        export LIBBINE_ALLREDUCE_ALGO_FAMILY="RECDOUB" 
        export LIBBINE_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBBINE_ALLREDUCE_ALGO="B_CONT"    
        MIN_ELEMS=$((PORTS * p))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 65536 1048576 #4096
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBBINE_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                    ALGO_FNAME=${LIBBINE_ALLREDUCE_ALGO_FAMILY}-${LIBBINE_ALLREDUCE_ALGO}-${LIBBINE_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi
    done
    echo " ${GREEN}[Done]${NC}"
done

