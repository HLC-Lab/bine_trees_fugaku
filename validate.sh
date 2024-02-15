#!/bin/bash
if [ "$1" = "ALL" ]; then
    echo "Running with n=4..."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 4 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
    
    echo "Running with n=8.."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 8 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=64.."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 64 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; } 

    echo "Running with n=6.."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 6 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=10.."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 10 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=12.."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 12 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=18.."
    LD_PRELOAD="./lib/libswing.so" mpirun -n 18 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=2x8x2.."
    LIBSWING_DIMENSIONS=2,8,2 LD_PRELOAD="./lib/libswing.so" mpirun -n 32 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

    echo "Running with n=6x6.."
    LIBSWING_DIMENSIONS=6,6 LD_PRELOAD="./lib/libswing.so" mpirun -n 36 --oversubscribe ./lib/test 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
else
    LD_PRELOAD="./lib/libswing.so" mpirun -n $1 --oversubscribe ./lib/test
fi