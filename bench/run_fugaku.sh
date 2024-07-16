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

COLLECTIVE="MPI_Allreduce"
OUT_PREFIX="swing_out"
ERR_PREFIX="swing_err"
MPIEXEC_OUT="-stdout-proc /vol0004/mdt1/home/u12936/swing-allreduce/bench/${OUT_PREFIX} -stderr-proc /vol0004/mdt1/home/u12936/swing-allreduce/bench/${ERR_PREFIX}"

for p in "${NODES[@]}"
do
    echo ${SYSTEM}${EXTRA},${p},${OUT_FOLDER} >> ../data/description.csv
    
    for n in 1048576 8388608 67108864 #1 8 64 512 2048 16384 131072 1048576 8388608 67108864
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
        LIBSWING_ALGO="DEFAULT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations}
	    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_default.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
        mkdir ${OUT_FOLDER}/${p}_${n}_default_stats/
        mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_default_stats/
	
        # Run bandwidth optimal and lat optimal swing
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_L" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations}
	    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
	    mkdir ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_ports_stats/
	    mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_lat_${PORTS}_ports_stats/
	
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations}
	    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
        mkdir ${OUT_FOLDER}/${p}_${n}_bw_${PORTS}_ports_stats/
        mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_${PORTS}_ports_stats/	
	
        #LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_COALESCE" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations}
	    #mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_coalesce_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
	    #mkdir ${OUT_FOLDER}/${p}_${n}_bw_coalesce_${PORTS}_ports_stats/
        #mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_coalesce_${PORTS}_ports_stats/
	
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_CONT" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations} 
	    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_cont_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
        mkdir ${OUT_FOLDER}/${p}_${n}_bw_cont_${PORTS}_ports_stats/
        mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_cont_${PORTS}_ports_stats/	
	
        LIBSWING_DIMENSIONS=${DIMENSIONS} LIBSWING_NUM_PORTS=${PORTS} LIBSWING_ALGO="SWING_B_UTOFU" ${MPIRUN} ${MPIRUN_MAP_BY_NODE_FLAG} ${MPIEXEC_OUT} -n ${p} ${MPIRUN_ADDITIONAL_FLAGS} ./bench ${COLLECTIVE} INT ${n} ${iterations} 
	    mv ${OUT_PREFIX}*.0 ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports.csv; rm -f ${OUT_PREFIX}* ${ERR_PREFIX}*
        mkdir ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports_stats/
        mv ./tnr_stats_*.csv ${OUT_FOLDER}/${p}_${n}_bw_utofu_${PORTS}_ports_stats/	
		
        echo " ${GREEN}[Done]${NC}"
    done
done

echo "Compressing "${OUT_FOLDER}" ..."
pushd ${OUT_PATH}
tar vcfJ ${TIMESTAMP}.tar.xz ${TIMESTAMP}
#rm -rf ${TIMESTAMP}
popd
