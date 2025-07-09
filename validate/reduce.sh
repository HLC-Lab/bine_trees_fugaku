#!/bin/bash
declare -a ALGORITHMS=("REDUCE_SCATTER_GATHER" "BINOMIAL_TREE")
declare -a FAMILIES=("BINE")
declare -a COUNTS=("131072")
COLLECTIVE="MPI_Reduce"

# Check that I call the right functions
export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
export LIBBINE_REDUCE_ALGO="BINOMIAL_TREE" 
export LIBBINE_REDUCE_ALGO_LAYER="MPI" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "bine_reduce_mpi" ] || { echo "ERROR: Wrong function called for ${LIBBINE_REDUCE_ALGO_FAMILY} ${LIBBINE_REDUCE_ALGO} ${LIBBINE_REDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

# Check that I call the right functions
export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
export LIBBINE_REDUCE_ALGO="BINOMIAL_TREE" 
export LIBBINE_REDUCE_ALGO_LAYER="UTOFU" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "bine_reduce_utofu_noomp" ] || { echo "ERROR: Wrong function called for ${LIBBINE_REDUCE_ALGO_FAMILY} ${LIBBINE_REDUCE_ALGO} ${LIBBINE_REDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

# Check that I call the right functions
export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
export LIBBINE_REDUCE_ALGO="REDUCE_SCATTER_GATHER" 
export LIBBINE_REDUCE_ALGO_LAYER="MPI" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "bine_reduce_redscat_gather_mpi" ] || { echo "ERROR: Wrong function called for ${LIBBINE_REDUCE_ALGO_FAMILY} ${LIBBINE_REDUCE_ALGO} ${LIBBINE_REDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

# Check that I call the right functions
export LIBBINE_REDUCE_ALGO_FAMILY="BINE" 
export LIBBINE_REDUCE_ALGO="REDUCE_SCATTER_GATHER" 
export LIBBINE_REDUCE_ALGO_LAYER="UTOFU" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "bine_reduce_redscat_gather_utofu" ] || { echo "ERROR: Wrong function called for ${LIBBINE_REDUCE_ALGO_FAMILY} ${LIBBINE_REDUCE_ALGO} ${LIBBINE_REDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

# Now check correctness
for ALGO in "${ALGORITHMS[@]}"
do        
    for TYPE in "INT32"
    do
        for COUNT in ${COUNTS[@]}
        do
            for FAMILY in ${FAMILIES[@]}
            do  
                echo "=== ALGO: ${ALGO} -- FAMILY: ${FAMILY} ==="
                for ITERATIONS in 4
                do
                    export LIBBINE_REDUCE_ALGO=${ALGO}
                    export LIBBINE_REDUCE_ALGO_FAMILY=${FAMILY}
                    export LIBBINE_REDUCE_ALGO_LAYER="MPI"
                    for N in 4 8 64 
                    do
                        echo "Running ${COLLECTIVE} ${ALGO} with n=${N}..."
                        LIBBINE_ALGO=${ALGO} mpirun -n ${N} --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                    done

                    if [ $ALGO == "BINOMIAL_TREE" ]; then
                        for N in 6 10 12 14 18
                        do
                            echo "Running ${COLLECTIVE} ${ALGO} with n=${N}..."
                            LIBBINE_ALGO=${ALGO} mpirun -n ${N} --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                        done
                    fi

                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x8x2.."
                    LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=2x8x2  mpirun -n 32 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                    echo "Running ${COLLECTIVE} ${ALGO} with n=4x4x4.."
                    LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=4x4x4  mpirun -n 64 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                    if [ $ALGO == "BINOMIAL_TREE" ]; then
                        echo "Running ${COLLECTIVE} ${ALGO} with n=6x6.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=6x6  mpirun -n 36 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                        echo "Running ${COLLECTIVE} ${ALGO} with n=10x10.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=10x10  mpirun -n 100 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
        
                        echo "Running ${COLLECTIVE} ${ALGO} with n=4x6x4.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=4x6x4  mpirun -n 96 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
        
                        echo "Running ${COLLECTIVE} ${ALGO} with n=6x2x6.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=6x2x6  mpirun -n 72 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
        
                        echo "Running ${COLLECTIVE} ${ALGO} with n=2x6x2.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=2x6x2  mpirun -n 24 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                        echo "Running ${COLLECTIVE} ${ALGO} with n=2x2x6.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=2x2x6  mpirun -n 24 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
        
                        echo "Running ${COLLECTIVE} ${ALGO} with n=6x2x2.."
                        LIBBINE_ALGO=${ALGO} LIBBINE_DIMENSIONS=6x2x2  mpirun -n 24 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                    fi
                done
            done
        done
    done
done