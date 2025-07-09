#!/bin/bash
pushd ..
source conf.sh
popd

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

TIMESTAMP=$(TZ=":Europe/Rome" date +"%Y_%m_%d___%H_%M_%S")

NODES=""
EXTRA=""
DIMENSIONS=""
PORTS_LIST=""
while getopts e:d:p:c: flag
do
    case "${flag}" in
        e) EXTRA="_"${OPTARG};;
        d) DIMENSIONS=${OPTARG};;
        p) PORTS_LIST=${OPTARG};;
        c) COLLECTIVES=($(echo "${OPTARG}" | tr ',' '\n'));;
    esac
done

export LOCATION=${SYSTEM}
export N_NODES=${DIMENSIONS}
export TIMESTAMP=${TIMESTAMP}
export NOTES=${EXTRA}
export CUDA="false"
export OUTPUT_LEVEL="summarized"
export MPI_LIB="FJMPI"
export MPI_LIB_VERSION="4.0.1"
export LIBBINE_VERSION="1.0.0"

i=0
for COLLECTIVE in "${COLLECTIVES[@]}"
do
    OUTPUT_DIR=../results/${SYSTEM}/${TIMESTAMP}/
    export DATA_DIR="$OUTPUT_DIR/$i"
    mkdir -p ${DATA_DIR}
    export COLLECTIVE_TYPE=${COLLECTIVE^^}

    ./run_${COLLECTIVE}.sh -d "${DIMENSIONS}" -p "${PORTS_LIST}" -o "${OUTPUT_DIR}" -i "${i}"
    i=$((i+1))
done

DELETE="no"
echo "Compressing "${OUTPUT_DIR}/" ..."
tarball_path="$(dirname "$OUTPUT_DIR")/$(basename "$OUTPUT_DIR").tar.gz"
if tar -czf "$tarball_path" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"; then
    if [[ "$DELETE" == "yes" ]]; then
        rm -rf "$OUTPUT_DIR"
    fi
fi
