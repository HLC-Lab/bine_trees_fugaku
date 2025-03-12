#!/bin/bash
declare -a COLLECTIVES=("MPI_Reduce_scatter" "MPI_Allgather" "MPI_Reduce" "MPI_Gather" "MPI_Scatter" "MPI_Bcast" "MPI_Allreduce" )
#declare -a COLLECTIVES=()
COUNT=0

for COLLECTIVE in "${COLLECTIVES[@]}"
do
    #Algos
    if [ ${COLLECTIVE} = "MPI_Bcast" ]; then
        #declare -a ALGORITHMS=("SWING_L" "SWING_B" "SWING_B_COALESCE" "SWING_B_CONT")
        declare -a ALGORITHMS=("SWING_L")
        declare -a COUNTS=("131072")
    fi
    
    if [ ${COLLECTIVE} = "MPI_Scatter" ]; then
        declare -a ALGORITHMS=("SWING_L")
        declare -a COUNTS=("1024")
    fi

    if [ ${COLLECTIVE} = "MPI_Reduce" ]; then
        declare -a ALGORITHMS=("SWING_L")
        declare -a COUNTS=("1024")
    fi

    if [ ${COLLECTIVE} = "MPI_Gather" ]; then
        declare -a ALGORITHMS=("SWING_L")
        declare -a COUNTS=("1024")
    fi    

    if [ ${COLLECTIVE} = "MPI_Allreduce" ]; then
        declare -a ALGORITHMS=("SWING_L" "SWING_B" "SWING_B_COALESCE" "SWING_B_CONT")
        #declare -a ALGORITHMS=("SWING_B_CONT")
        declare -a COUNTS=("131072")
    fi

    if [ ${COLLECTIVE} = "MPI_Reduce_scatter" ]; then
        declare -a COUNTS=("1024")
        if [[ ${LIBSWING_DIMENSIONS} = "" && ${LIBSWING_MULTIPORT} = "" ]]; then
            declare -a ALGORITHMS=("SWING_B" "SWING_B_COALESCE")
        else
            declare -a ALGORITHMS=("SWING_B")
        fi        
    fi

    if [ ${COLLECTIVE} = "MPI_Allgather" ]; then
        declare -a COUNTS=("1024")
        declare -a ALGORITHMS=("SWING_B")
    fi
    
    for ALGO in "${ALGORITHMS[@]}"
    do
        for TYPE in "INT32"
        do
            for COUNT in ${COUNTS[@]}
            do
                for ITERATIONS in 4
                do
                    for N in 4 8 64 #6 10 12 14 18
                    do
                        echo "Running ${COLLECTIVE} ${ALGO} with n=${N}..."
                        LIBSWING_ALGO=${ALGO} mpirun -n ${N} --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                    done

                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x8x2.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2,8,2  mpirun -n 32 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                    #echo "Running ${COLLECTIVE} ${ALGO} with n=6x6.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=6,6  mpirun -n 36 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
#
                    #echo "Running ${COLLECTIVE} ${ALGO} with n=10x10.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=10,10  mpirun -n 100 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
#
                    #echo "Running ${COLLECTIVE} ${ALGO} with n=4x6x4.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=4,6,4  mpirun -n 96 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
#
                    #echo "Running ${COLLECTIVE} ${ALGO} with n=6x2x6.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=6,2,6  mpirun -n 72 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
#
                    #echo "Running ${COLLECTIVE} ${ALGO} with n=2x6x2.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2,6,2  mpirun -n 24 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
#
                    #echo "Running ${COLLECTIVE} ${ALGO} with n=2x2x6.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2,2,6  mpirun -n 24 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
#
                    #echo "Running ${COLLECTIVE} ${ALGO} with n=6x2x2.."
                    #LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=6,2,2  mpirun -n 24 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                done
            done
        done
    done
done

for COLLECTIVE in "MPI_Alltoall"
do
    if [ ${COLLECTIVE} = "MPI_Alltoall" ]; then
        declare -a COUNTS=("8")
        declare -a ALGORITHMS=("SWING_L")
    fi    
    for ALGO in "${ALGORITHMS[@]}"
    do
        for TYPE in "INT32"
        do
            for COUNT in ${COUNTS[@]}
            do
                for ITERATIONS in 4
                do
                    for N in 4 8 16 32 64 128 256
                    do
                        echo "Running ${COLLECTIVE} ${ALGO} with n=${N}..."
                        LIBSWING_ALGO=${ALGO} mpirun -n ${N} --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                    done
                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x8x2.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2,8,2  mpirun -n 32 --oversubscribe ./bench/bench ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                done
            done
        done
    done
done