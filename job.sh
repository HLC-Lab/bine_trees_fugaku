#!/bin/sh
#PJM -L  "node=4x4x4"
#PJM -L  "rscgrp=small"
#PJM -L  "elapse=0:10:00"
#PJM -g hp240454
#PJM --mpi  "rank-map-bynode=XZY"
#PJM --mpi "max-proc-per-node=1"
#

DIMENSIONS="4,4,4" # Must be equal to the one specified above

pushd bench
./run_fugaku.sh -n 64 -d ${DIMENSIONS}

