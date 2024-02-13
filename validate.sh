#!/bin/bash
LD_PRELOAD="./lib/libswing.so" mpirun -n $1 --oversubscribe ./lib/test