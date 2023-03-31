#!/bin/bash
pushd ..
source conf.sh
popd

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

NODES=""
while getopts n: flag
do
    case "${flag}" in
        n) NODES=($(echo "${OPTARG}" | tr ',' '\n'));;
    esac
done

OUT_PATH=../data/${SYSTEM}
TIMESTAMP=$(TZ=":Europe/Rome" date +%Y_%m_%d_%H_%M_%S)
OUT_FOLDER=${OUT_PATH}/${TIMESTAMP}
mkdir -p ${OUT_FOLDER}

echo ${SYSTEM},${NODES[-1]},${OUT_FOLDER} >> ../data/description.csv

# TODO: On Piz Daint, also change routing

for p in "${NODES[@]}"
do
    case $SYSTEM in
    daint)
        ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./get_coord_daint > ${OUT_FOLDER}/coord_${p}.txt
        ;;
    esac
    for n in 1 8 64 512 2048 16384 131072 1048576 8388608 67108864
    do
        iterations=0
        if [ $n -le 512 ]
        then
            iterations=100000
        elif [ $n -le 1048576 ]
        then
            iterations=10000
        elif [ $n -le 8388608 ]
        then
            iterations=1000
        else
            iterations=10
        fi
        echo -n "Running on "${collective}" nodes with count="${n}"..."
        LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_default.csv
        for SWINGTYPE in "BBB" "BBBN"
        do
            # Run bandwidth optimal
            LIBSWING_LATENCY_OPTIMAL_THRESHOLD=0 LIBSWING_SENDRECV_TYPE="${SWINGTYPE}" LIBSWING_ALGO="SWING" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_bw_${SWINGTYPE}.csv
            # If msg small enough, run latency optimal
            if [ $n -le 1048576 ]
            then
                LIBSWING_LATENCY_OPTIMAL_THRESHOLD=99999999 LIBSWING_SENDRECV_TYPE="${SWINGTYPE}" LIBSWING_ALGO="SWING" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_lat_${SWINGTYPE}.csv
            fi
        done
        echo " ${GREEN}[Done]${NC}"
    done
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
#rm -rf ${TIMESTAMP}
popd
