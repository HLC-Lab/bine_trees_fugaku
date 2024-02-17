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
while getopts n:e:d: flag
do
    case "${flag}" in
        n) NODES=($(echo "${OPTARG}" | tr ',' '\n'));;
        e) EXTRA="_"${OPTARG};;
        d) DIMENSIONS=${OPTARG};;
    esac
done

OUT_PATH=../data/${SYSTEM}
TIMESTAMP=$(TZ=":Europe/Rome" date +%Y_%m_%d_%H_%M_%S)
OUT_FOLDER=${OUT_PATH}/${TIMESTAMP}
mkdir -p ${OUT_FOLDER}

for p in "${NODES[@]}"
do
    echo ${SYSTEM}${EXTRA},${p},${OUT_FOLDER} >> ../data/description.csv
    ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} hostname > ${OUT_FOLDER}/hostnames_${p}.txt
    
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
        LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench VOID ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_default.csv
        # Run bandwidth optimal and lat optimal swing
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_MULTIPORT=1 LIBSWING_ALGO="SWING_B" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench VOID ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_bw_multiport.csv
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_MULTIPORT=0 LIBSWING_ALGO="SWING_B" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench VOID ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_bw.csv
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_MULTIPORT=1 LIBSWING_ALGO="SWING_L" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench VOID ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_lat_multiport.csv
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_MULTIPORT=0 LIBSWING_ALGO="SWING_L" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench VOID ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}_lat.csv
        echo " ${GREEN}[Done]${NC}"
    done
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
rm -rf ${TIMESTAMP}
popd
