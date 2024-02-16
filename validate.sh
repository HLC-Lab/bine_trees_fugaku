#!/bin/bash
if [ "$1" = "ALL" ]; then
    for N in 4 8 64 6 10 12 14 18
    do
        echo "Running with n=${N}..."
        LD_PRELOAD="./lib/libswing.so" mpirun -n ${N} --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
    done

    echo "Running with n=2x8x2.."
    LIBSWING_DIMENSIONS=2,8,2 LD_PRELOAD="./lib/libswing.so" mpirun -n 32 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=6x6.."
    LIBSWING_DIMENSIONS=6,6 LD_PRELOAD="./lib/libswing.so" mpirun -n 36 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=10x10.."
    LIBSWING_DIMENSIONS=10,10 LD_PRELOAD="./lib/libswing.so" mpirun -n 36 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=4x6x4.."
    LIBSWING_DIMENSIONS=4,6,4 LD_PRELOAD="./lib/libswing.so" mpirun -n 144 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
else
    rm -rf out/*
    LD_PRELOAD="./lib/libswing.so" mpirun --output-filename out -n $1 --oversubscribe  ./lib/test
    #LD_PRELOAD="./lib/libswing.so" mpirun -n $1 --oversubscribe ./lib/test
fi