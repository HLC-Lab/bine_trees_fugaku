#!/bin/bash
source conf.sh

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
        else
            iterations=1000
        fi
        echo -n "Running on "${collective}" nodes with count="${n}"..."
        ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${n} ${iterations} > ${OUT_FOLDER}/${p}_${n}.csv
        echo " ${GREEN}[Done]${NC}"
    done
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
#rm -rf ${TIMESTAMP}
popd