source conf.sh

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

rm -f ./lib/libswing.o ./lib/libswing_common.o ./lib/libswing.so ./lib/libswing_common_profile.o ./lib/libswing_profile.o ./lib/fugaku/*.o ./bench/bench ./bench/bench_dummy_utofu

EXTRA_LIBS=""

# Compile system-specific stuff
if [ ${SYSTEM} = "fugaku" ]; then
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} ./bench/get_coord_fugaku.c -o ./bench/get_coord_fugaku
    # Compile uTofu helpers
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -c -fPIC ./lib/fugaku/swing_utofu.cc -o ./lib/fugaku/swing_utofu.o
    EXTRA_LIBS="-ltofucom ./lib/fugaku/swing_utofu.o"
fi

# Normal compilation
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -c -fPIC -pthread ./lib/libswing_common.cc -o ./lib/libswing_common.o ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing_common.o" ]; then
    echo "${RED}[Error] libswing_common.o compilation failed, please check error messages above.${NC}"
    exit 1
fi
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -c -fPIC -pthread ./lib/libswing.cc -o ./lib/libswing.o ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.o" ]; then
    echo "${RED}[Error] libswing.o compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -shared -pthread -pthread -o ./lib/libswing.so ./lib/libswing.o ./lib/libswing_common.o ${EXTRA_LIBS} ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.so" ]; then
    echo "${RED}[Error] libswing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

# Profiling compilation
if [ ${SYSTEM} != "fugaku" ]; then
    FLAGS_PROFILE="-O0 -g -pg -DPROFILE" # To profile
    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -c -fPIC -pthread ./lib/libswing.cc -o ./lib/libswing_profile.o ${FLAGS_PROFILE}
    if [ ! -f "./lib/libswing_profile.o" ]; then
        echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
        exit 1
    fi
    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -c -fPIC -pthread ./lib/libswing_common.cc -o ./lib/libswing_common_profile.o ${FLAGS_PROFILE}
    if [ ! -f "./lib/libswing_profile.o" ]; then
        echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
        exit 1
    fi
    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -pthread ./bench/bench.cc ./lib/libswing_common_profile.o ./lib/libswing_profile.o -o ./bench/bench_profile ${FLAGS_PROFILE}
fi


# Bench
#${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread ./bench/bench.cc ./lib/libswing.o -o ./bench/bench ${MPI_COMPILER_FLAGS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/bench.cc ./lib/libswing.o ./lib/libswing_common.o -o ./bench/bench ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread ./bench/get_coord_daint.c -o ./bench/get_coord_daint
#${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -pthread  -D${SYSTEM^^} ./bench/bench_dummy_utofu.c -o ./bench/bench_dummy_utofu ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}