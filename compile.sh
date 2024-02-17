source conf.sh

GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

# Normal compilation
FLAGS="-O3 -g -Wall -pedantic"
${MPI_COMPILER} ${FLAGS} -c -fPIC -fopenmp ./lib/libswing.cc -o ./lib/libswing.o ${FLAGS}
if [ ! -f "./lib/libswing.o" ]; then
    echo "${RED}[Error] libswing.o compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${FLAGS} -shared -pthread -fopenmp -o ./lib/libswing.so ./lib/libswing.o ${FLAGS}
if [ ! -f "./lib/libswing.so" ]; then
    echo "${RED}[Error] libswing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${FLAGS} -fopenmp ./lib/test.cc -o ./lib/test ${FLAGS}

# Profiling compilation
FLAGS_PROFILE="-O0 -g -pg -DPROFILE" # To profile
${MPI_COMPILER} ${FLAGS_PROFILE} -c -fPIC -fopenmp ./lib/libswing.cc -o ./lib/libswing_profile.o ${FLAGS_PROFILE}
if [ ! -f "./lib/libswing_profile.o" ]; then
    echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
    exit 1
fi
${MPI_COMPILER} ${FLAGS_PROFILE} -fopenmp ./lib/test.cc ./lib/libswing_profile.o -o ./lib/test_profile ${FLAGS_PROFILE}


# Bench
${MPI_COMPILER} ${FLAGS} -fopenmp ./bench/bench.cc ./lib/libswing.o -o ./bench/bench ${FLAGS}
${MPI_COMPILER} ${FLAGS} -fopenmp ./bench/get_coord_daint.c -o ./bench/get_coord_daint
