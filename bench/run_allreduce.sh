#!/bin/bash
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

OUT_PREFIX="swing_out_${TIMESTAMP}"
ERR_PREFIX="swing_err_${TIMESTAMP}"
MPIEXEC_OUT="-stdout-proc /vol0004/mdt1/home/u12936/swing-allreduce/bench/${OUT_PREFIX} -stderr-proc /vol0004/mdt1/home/u12936/swing-allreduce/bench/${ERR_PREFIX}"

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

for n in 1 8 64 512 2048 16384 131072 1048576 8388608 67108864
do
    # DATATYPE to lowercase
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
    echo -n "Running on "${DIMENSIONS}" (${p} nodes) with count="${n}"..."

    
    #########################
    # Run the default algos #
    #########################
    export LIBSWING_ALLREDUCE_ALGO_FAMILY="DEFAULT" 
    export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI" 

    coll_tuned_prealloc_size=1024 # This is in MiB (1 GiB)
    PREALLOC_SIZE=1073741824 # 1 GiB
    
    # ATTENTION: Showing decision process adds non-negligible overhead (for small vectors). Use it with care.
    # TODO: Maybe I should prealloc only for large allreduce?
    EXTRA_MCAS="" #"-mca mpi_print_stats 1 -mca coll_select_show_decision_process 2" #"-mca coll_base_reduce_commute_safe 1"

    DEFAULT_ALGO="default"
    
    LIBSWING_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
    ALGO_FNAME=default-${DEFAULT_ALGO}
    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    
    # Decision rules can be found at /opt/FJSVxtclanga/.common/MECA030/etc/fjmpi-dectree.conf (but you can access it only on computing nodes after doing pjsub)
    # We do not use uTofu barrier when we force the algo (we use it only for default_default)
    if [ $n -le 512 ]; then
        for DEFAULT_ALGO in "basic_linear"
        do
            LIBSWING_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll ^tbi -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
            ALGO_FNAME=default-${DEFAULT_ALGO}
            mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
            if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
        done
    fi

    # All those that do not have segsize as parameter
    for DEFAULT_ALGO in "rdbc" "ring" "recursive_doubling" "nonoverlapping"
    do
        LIBSWING_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll ^tbi -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
        ALGO_FNAME=default-${DEFAULT_ALGO}
        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    done

    # For segmented ring we just use 1MiB as segment size (is never used/selected by fjmpi anyway)
    for DEFAULT_ALGO in "segmented_ring"
    do
        for coll_select_allreduce_algorithm_segmentsize in 1048576
        do
            LIBSWING_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll ^tbi -mca coll_select_allreduce_algorithm_segmentsize ${coll_select_allreduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
            ALGO_FNAME=default-${DEFAULT_ALGO}
            mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
            if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
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

        LIBSWING_ALLREDUCE_ALGO_FAMILY="DEFAULT" ${MPIRUN} ${EXTRA_MCAS} -mca coll ^tbi -mca coll_select_allreduce_algorithm_segmentsize ${coll_select_allreduce_algorithm_segmentsize} -mca coll_tuned_prealloc_size ${coll_tuned_prealloc_size} -mca coll_select_allreduce_algorithm ${DEFAULT_ALGO} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}
        ALGO_FNAME=default-${DEFAULT_ALGO}
        mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
        if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
    done


    ###################################
    # Run the Swing and Recdoub algos #
    ###################################
    export LIBSWING_DIMENSIONS=${DIMENSIONS} 
    export LIBSWING_PREALLOC_SIZE=${PREALLOC_SIZE} 
    for PORTS in ${PORTS_LIST//,/ }
    do
        export LIBSWING_NUM_PORTS=${PORTS}        
        # Run lat optimal swing 
        export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
        export LIBSWING_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBSWING_ALLREDUCE_ALGO="L"    
        MIN_ELEMS=$((PORTS))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then                    
                    LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
                    ALGO_FNAME=${LIBSWING_ALLREDUCE_ALGO_FAMILY}-${LIBSWING_ALLREDUCE_ALGO}-${LIBSWING_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi

        # Run lat optimal recdoub
        export LIBSWING_ALLREDUCE_ALGO_FAMILY="RECDOUB" 
        export LIBSWING_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBSWING_ALLREDUCE_ALGO="L"    
        MIN_ELEMS=$((PORTS))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations}                    
                    ALGO_FNAME=${LIBSWING_ALLREDUCE_ALGO_FAMILY}-${LIBSWING_ALLREDUCE_ALGO}-${LIBSWING_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi

        # Run bw optimal swing
        export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
        export LIBSWING_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBSWING_ALLREDUCE_ALGO="B_CONT"    
        MIN_ELEMS=$((PORTS * p))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                    ALGO_FNAME=${LIBSWING_ALLREDUCE_ALGO_FAMILY}-${LIBSWING_ALLREDUCE_ALGO}-${LIBSWING_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi

        # Run bw optimal recdoub
        export LIBSWING_ALLREDUCE_ALGO_FAMILY="RECDOUB" 
        export LIBSWING_ALLREDUCE_ALGO_LAYER="UTOFU" 
        export LIBSWING_ALLREDUCE_ALGO="B_CONT"    
        MIN_ELEMS=$((PORTS * p))
        if [ "$n" -ge "$MIN_ELEMS" ]; then
            for SEGMENT_SIZE in 0 4096 65536 1048576
            do                
                if [ $SEGMENT_SIZE -lt $msg_size ]; then
                    LIBSWING_SEGMENT_SIZE=${SEGMENT_SIZE} ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} ${DATATYPE} ${n} ${iterations} 
                    ALGO_FNAME=${LIBSWING_ALLREDUCE_ALGO_FAMILY}-${LIBSWING_ALLREDUCE_ALGO}-${LIBSWING_ALLREDUCE_ALGO_LAYER}-${SEGMENT_SIZE}-${PORTS}
                    mv ${OUT_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.csv; rm -f ${OUT_PREFIX}* 
                    if [ -f ${ERR_PREFIX}*.0 ]; then mv ${ERR_PREFIX}*.0 ${OUTPUT_DIR}/${EXP_ID}/${n}_${ALGO_FNAME}_${DATATYPE_lc}.err; rm -f ${ERR_PREFIX}*; fi
                fi
            done
        fi
    done
    echo " ${GREEN}[Done]${NC}"
done

