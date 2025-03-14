#!/bin/bash
COLLECTIVE="MPI_Allreduce"

# Check that I call the right functions
###########
# SWING_L #
###########
export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
export LIBSWING_ALLREDUCE_ALGO="L" 
export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "swing_coll_l_mpi" ] || { echo "ERROR: Wrong function called for ${LIBSWING_ALLREDUCE_ALGO_FAMILY} ${LIBSWING_ALLREDUCE_ALGO} ${LIBSWING_ALLREDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
export LIBSWING_ALLREDUCE_ALGO="L" 
export LIBSWING_ALLREDUCE_ALGO_LAYER="UTOFU" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "swing_coll_l_utofu" ] || { echo "ERROR: Wrong function called for ${LIBSWING_ALLREDUCE_ALGO_FAMILY} ${LIBSWING_ALLREDUCE_ALGO} ${LIBSWING_ALLREDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

###########
# SWING_B #
###########
export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
export LIBSWING_ALLREDUCE_ALGO="B" 
export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "swing_coll_step_b" ] || { echo "ERROR: Wrong function called for ${LIBSWING_ALLREDUCE_ALGO_FAMILY} ${LIBSWING_ALLREDUCE_ALGO} ${LIBSWING_ALLREDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

####################
# SWING_B_COALESCE #
####################
export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
export LIBSWING_ALLREDUCE_ALGO="B_COALESCE" 
export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "swing_coll_step_coalesce" ] || { echo "ERROR: Wrong function called for ${LIBSWING_ALLREDUCE_ALGO_FAMILY} ${LIBSWING_ALLREDUCE_ALGO} ${LIBSWING_ALLREDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

################
# SWING_B_CONT #
################
export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
export LIBSWING_ALLREDUCE_ALGO="B_CONT" 
export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "swing_coll_step_cont" ] || { echo "ERROR: Wrong function called for ${LIBSWING_ALLREDUCE_ALGO_FAMILY} ${LIBSWING_ALLREDUCE_ALGO} ${LIBSWING_ALLREDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

export LIBSWING_ALLREDUCE_ALGO_FAMILY="SWING" 
export LIBSWING_ALLREDUCE_ALGO="B_CONT" 
export LIBSWING_ALLREDUCE_ALGO_LAYER="UTOFU" 
FUNC_NAME=$(mpirun -n 4 --oversubscribe ./bench/bench_validate ${COLLECTIVE} "INT32" "131072" "4" 2>/dev/null | grep "func_called" | cut -d ':' -f 2 | head -n 1 | tr -d ' ')
[ "${FUNC_NAME}" == "swing_coll_b_cont_utofu" ] || { echo "ERROR: Wrong function called for ${LIBSWING_ALLREDUCE_ALGO_FAMILY} ${LIBSWING_ALLREDUCE_ALGO} ${LIBSWING_ALLREDUCE_ALGO_LAYER}: ${FUNC_NAME}"; exit 1; }

declare -a ALGORITHMS=("B_CONT")
declare -a FAMILIES=("SWING")
declare -a COUNTS=("131072")
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
                    export LIBSWING_ALLREDUCE_ALGO=${ALGO}
                    export LIBSWING_ALLREDUCE_ALGO_FAMILY=${FAMILY}
                    export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI"
                    for N in 4 8 64
                    do
                        echo "Running ${COLLECTIVE} ${ALGO} with n=${N}..."
                        LIBSWING_ALGO=${ALGO} mpirun -n ${N} --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                    done

                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x8x2.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2x8x2  mpirun -n 32 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                done
            done
        done
    done
done

declare -a ALGORITHMS=("B" "B_COALESCE" "L")
declare -a FAMILIES=("SWING")
declare -a COUNTS=("131072")
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
                    export LIBSWING_ALLREDUCE_ALGO=${ALGO}
                    export LIBSWING_ALLREDUCE_ALGO_FAMILY=${FAMILY}
                    export LIBSWING_ALLREDUCE_ALGO_LAYER="MPI"
                    for N in 4 8 64 6 10 12 14 18
                    do
                        echo "Running ${COLLECTIVE} ${ALGO} with n=${N}..."
                        LIBSWING_ALGO=${ALGO} mpirun -n ${N} --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                    done

                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x8x2.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2x8x2  mpirun -n 32 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                    echo "Running ${COLLECTIVE} ${ALGO} with n=6x6.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=6x6  mpirun -n 36 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                    echo "Running ${COLLECTIVE} ${ALGO} with n=10x10.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=10x10  mpirun -n 100 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
    
                    echo "Running ${COLLECTIVE} ${ALGO} with n=4x6x4.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=4x6x4  mpirun -n 96 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
    
                    echo "Running ${COLLECTIVE} ${ALGO} with n=6x2x6.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=6x2x6  mpirun -n 72 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
    
                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x6x2.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2x6x2  mpirun -n 24 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }

                    echo "Running ${COLLECTIVE} ${ALGO} with n=2x2x6.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=2x2x6  mpirun -n 24 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
    
                    echo "Running ${COLLECTIVE} ${ALGO} with n=6x2x2.."
                    LIBSWING_ALGO=${ALGO} LIBSWING_DIMENSIONS=6x2x2  mpirun -n 24 --oversubscribe ./bench/bench_validate ${COLLECTIVE} ${TYPE} ${COUNT} ${ITERATIONS} 2>&1 > /dev/null || { echo 'FAIL' ; exit 1; }
                done
            done
        done
    done
done