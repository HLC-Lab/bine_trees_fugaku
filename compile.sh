source conf.sh

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

rm -f ./lib/libswing.o ./lib/libswing.so ./lib/libswing_profile.o ./bench/bench

# Compile system-specific stuff
#if [ ${SYSTEM} = "fugaku" ]; then
#    ${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -c -fPIC -fopenmp ./lib/fugaku_tnr_stats.c -o ./lib/fugaku_tnr_stats.o
#fi

# Normal compilation
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -c -fPIC -fopenmp ./lib/libswing.cc -o ./lib/libswing.o ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.o" ]; then
    echo "${RED}[Error] libswing.o compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -shared -pthread -fopenmp -o ./lib/libswing.so ./lib/libswing.o ${MPI_COMPILER_FLAGS}
if [ ! -f "./lib/libswing.so" ]; then
    echo "${RED}[Error] libswing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

# Profiling compilation
FLAGS_PROFILE="-O0 -g -pg -DPROFILE" # To profile
${MPI_COMPILER} ${FLAGS_PROFILE} -c -fPIC -fopenmp ./lib/libswing.cc -o ./lib/libswing_profile.o ${FLAGS_PROFILE}
if [ ! -f "./lib/libswing_profile.o" ]; then
    echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
    exit 1
fi
${MPI_COMPILER} ${FLAGS_PROFILE} -fopenmp ./bench/bench.cc ./lib/libswing_profile.o -o ./bench/bench_profile ${FLAGS_PROFILE}


# Bench
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -fopenmp  -D${SYSTEM^^} ./bench/bench.cc ./lib/libswing.o -o ./bench/bench ${MPI_COMPILER_FLAGS}
${MPI_COMPILER} ${MPI_COMPILER_FLAGS} -fopenmp ./bench/get_coord_daint.c -o ./bench/get_coord_daint
