source conf.sh

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

rm -f ./lib/libswing.o ./lib/libswing.so ./lib/libswing_profile.o ./lib/fugaku/*.a ./lib/fugaku/*.o ./bench/bench ./bench/bench_dummy_utofu

EXTRA_LIBS=""

# Compile system-specific stuff
if [ ${SYSTEM} = "fugaku" ]; then
    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} ./bench/get_coord_fugaku.c -o ./bench/get_coord_fugaku

    # Compile uTofu helpers
    pushd ./lib/fugaku
    CC=${MPI_COMPILER} make
    popd
    EXTRA_LIBS="-ltofucom -L./lib/fugaku/ -lrdma_comlib"
fi

# Normal compilation
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -c -fPIC -fopenmp ./lib/libswing.cc -o ./lib/libswing.o ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.o" ]; then
    echo "${RED}[Error] libswing.o compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -D${SYSTEM^^} -shared -pthread -fopenmp -o ./lib/libswing.so ./lib/libswing.o ${EXTRA_LIBS} ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.so" ]; then
    echo "${RED}[Error] libswing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

# Profiling compilation
if [ ${SYSTEM} != "fugaku" ]; then
    FLAGS_PROFILE="-O0 -g -pg -DPROFILE" # To profile
    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -c -fPIC -fopenmp ./lib/libswing.cc -o ./lib/libswing_profile.o ${FLAGS_PROFILE}
    if [ ! -f "./lib/libswing_profile.o" ]; then
        echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
        exit 1
    fi
    ${MPI_COMPILER} ${FLAGS_PROFILE} -D${SYSTEM^^} -fopenmp ./bench/bench.cc ./lib/libswing_profile.o -o ./bench/bench_profile ${FLAGS_PROFILE}
fi


# Bench
#${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -fopenmp ./bench/bench.cc ./lib/libswing.o -o ./bench/bench ${MPI_COMPILER_FLAGS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -fopenmp  -D${SYSTEM^^} ./bench/bench.cc ./lib/libswing.o -o ./bench/bench ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -fopenmp ./bench/get_coord_daint.c -o ./bench/get_coord_daint
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -fopenmp  -D${SYSTEM^^} ./bench/bench_dummy_utofu.c -o ./bench/bench_dummy_utofu ${MPI_COMPILER_FLAGS} ${EXTRA_LIBS}