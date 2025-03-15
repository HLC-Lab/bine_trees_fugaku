source conf.sh

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

rm -f ./lib/*.so ./lib/*.o ./lib/fugaku/*.o ./bench/bench ./bench/bench_dummy_utofu

EXTRA_LIBS=""

# Compile system-specific stuff
if [ ${SYSTEM} = "fugaku" ]; then
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} ./bench/get_coord_fugaku.c -o ./bench/get_coord_fugaku
    # Compile uTofu helpers
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -c -fPIC ./lib/fugaku/swing_utofu.cc -o ./lib/fugaku/swing_utofu.o
    EXTRA_LIBS="-ltofucom ./lib/fugaku/swing_utofu.o"
fi

# Collective impls
# Bash list of collectives to compile
# Define a list of collectives
collectives=("libswing" "libswing_common" "libswing_coll" "libswing_coll_reduce" "libswing_coll_reduce_scatter" "libswing_coll_bcast" "libswing_coll_allgather" "libswing_coll_gather" "libswing_coll_scatter" "libswing_coll_alltoall")
collective_objects=""
# Compile each collective
for collective in "${collectives[@]}"; do
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -c -fPIC -pthread ./lib/${collective}.cc -o ./lib/${collective}.o ${MPI_COMPILER_FLAGS}
    if [ ! -f "./lib/${collective}.o" ]; then
        echo "${RED}[Error] ${collective}.o compilation failed, please check error messages above.${NC}"
        exit 1
    fi
    collective_objects="${collective_objects} ./lib/${collective}.o"
done

# VALIDATE compilation
# Collective impls
# Bash list of collectives to compile
collective_objects_validate=""

if [ ${SYSTEM} = "fugaku" ]; then
    echo "Skipping validation compilation for Fugaku"
else
    # Compile each collective
    for collective in "${collectives[@]}"; do
        ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -DVALIDATE -D${SYSTEM^^} -c -fPIC -pthread ./lib/${collective}.cc -o ./lib/${collective}_validate.o ${MPI_COMPILER_FLAGS}
        if [ ! -f "./lib/${collective}_validate.o" ]; then
            echo "${RED}[Error] ${collective}_validate.o compilation failed, please check error messages above.${NC}"
            exit 1
        fi
        collective_objects_validate="${collective_objects_validate} ./lib/${collective}_validate.o"
    done
fi

${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -shared -pthread -pthread -o ./lib/libswing.so ${collective_objects} ${EXTRA_LIBS} ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.so" ]; then
    echo "${RED}[Error] libswing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

# Profiling compilation
#if [ ${SYSTEM} != "fugaku" ]; then
#    FLAGS_PROFILE="-O0 -g -pg -DPROFILE" # To profile
#    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -c -fPIC -pthread ./lib/libswing.cc -o ./lib/libswing_profile.o ${FLAGS_PROFILE}
#    if [ ! -f "./lib/libswing_profile.o" ]; then
#        echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
#        exit 1
#    fi
#    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -c -fPIC -pthread ./lib/libswing_common.cc -o ./lib/libswing_common_profile.o ${FLAGS_PROFILE}
#    if [ ! -f "./lib/libswing_profile.o" ]; then
#        echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
#        exit 1
#    fi
#    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -pthread ./bench/bench.cc ./lib/libswing_common_profile.o ./lib/libswing_profile.o -o ./bench/bench_profile ${FLAGS_PROFILE}
#fi


# Bench
#${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread ./bench/bench.cc ./lib/libswing.o -o ./bench/bench ${MPI_COMPILER_FLAGS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/bench.cc ${collective_objects} -o ./bench/bench ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/bench.cc ${collective_objects_validate} -o ./bench/bench_validate ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread ./bench/get_coord_daint.c -o ./bench/get_coord_daint
#${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/bench_dummy_utofu.c -o ./bench/bench_dummy_utofu ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}

# Tree benchmark
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/bench_tree.cc ${collective_objects} -o ./bench/bench_tree ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}

# Compile system-specific stuff
if [ ${SYSTEM} = "fugaku" ]; then
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/check_coord.cc ${collective_objects} -o ./bench/check_coord ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}
fi