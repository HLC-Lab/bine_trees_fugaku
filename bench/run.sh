#!/bin/bash
pushd ..
source conf.sh
popd

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

NODES=""
EXTRA=""
while getopts n:e: flag
do
    case "${flag}" in
        n) NODES=($(echo "${OPTARG}" | tr ',' '\n'));;
        e) EXTRA="_"${OPTARG};;
    esac
done

OUT_PATH=../data/${SYSTEM}
TIMESTAMP=$(TZ=":Europe/Rome" date +%Y_%m_%d_%H_%M_%S)
OUT_FOLDER=${OUT_PATH}/${TIMESTAMP}
mkdir -p ${OUT_FOLDER}

# TODO: On Piz Daint, also change routing
for p in "${NODES[@]}"
do
    echo ${SYSTEM}${EXTRA},${p},${OUT_FOLDER} >> ../data/description.csv

    # Run EBB
    if [ -n "$EBB" ]; then
        ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ${EBB}  -s 16777216:16777216 -m mpi -x ebb > ${OUT_FOLDER}/ebb_${p}.txt
    fi

    ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} hostname > ${OUT_FOLDER}/hostnames_${p}.txt
    
    case $SYSTEM in
    daint)
        ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./get_coord_daint > ${OUT_FOLDER}/coord_${p}.txt
        ;;
    alps)
	    ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} cat /etc/cray/xname > ${OUT_FOLDER}/coord_${p}.txt
	    ;;
    lumi)
	    ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} cat /etc/cray/xname > ${OUT_FOLDER}/coord_${p}.txt
	    ;;
    esac
    TEMP_SOURCE_FILE=$(mktemp)
    for n in 1 8 64 512 2048 16384 131072 1048576 8388608 67108864
    do
        iterations=0
        if [ $n -le 512 ]
        then
            iterations=10000
        elif [ $n -le 1048576 ]
        then
            iterations=1000
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
        
        # Run all the default algorithms
        for DEFAULT_NAME in ${DEFAULT_ALGOS_ALLREDUCE//|/ }
        do            
            echo ${EXTRA_VARIABLES} | tr '|' '\n' > ${TEMP_SOURCE_FILE}
            echo ${EXTRA_VARIABLES_DEFAULT} | tr '|' '\n' >> ${TEMP_SOURCE_FILE}
            echo ${DEFAULT_NAME} >> ${TEMP_SOURCE_FILE}
            DEFAULT_IDX=$(echo ${DEFAULT_NAME} | cut -d'=' -f2)
            (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_default_${DEFAULT_IDX}.csv)
        done

        # Run manual sota algorithms
        echo ${EXTRA_VARIABLES} | tr '|' '\n' > ${TEMP_SOURCE_FILE}
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="RECDOUB_L" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_recdoub_l.csv)
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="RECDOUB_B" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_recdoub_b.csv)
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="RING" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_ring.csv)
        # Run Swing algorithms
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="SWING_L" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_lat.csv)
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="SWING_B" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_bw.csv)
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="SWING_B_COALESCE" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_bw_coalesce.csv)
        (source ${TEMP_SOURCE_FILE}; LIBSWING_ALGO="SWING_B_CONT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench INT32 ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_bw_cont.csv)
        echo " ${GREEN}[Done]${NC}"
    done
    rm ${TEMP_SOURCE_FILE}
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
rm -rf ${TIMESTAMP}
popd
