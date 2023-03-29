GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

MPI_COMPILER=mpic++

# Normal compilation
FLAGS="-O3 -g"
${MPI_COMPILER} ${FLAGS} -c -fPIC swing.cc -o swing.o ${FLAGS}
if [ ! -f "swing.o" ]; then
    echo "${RED}[Error] swing.o compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${FLAGS} -shared -pthread -o swing.so swing.o ${FLAGS}
if [ ! -f "swing.so" ]; then
    echo "${RED}[Error] swing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${FLAGS} test.cc -o test ${FLAGS}
${MPI_COMPILER} ${FLAGS} bench.cc swing.o -o bench ${FLAGS}


# Profiling compilation
FLAGS_PROFILE="-O0 -g -pg -DPROFILE" # To profile
${MPI_COMPILER} ${FLAGS_PROFILE} -c -fPIC swing.cc -o swing_profile.o ${FLAGS_PROFILE}
if [ ! -f "swing_profile.o" ]; then
    echo "${RED}[Error] swing_profile.o compilation failed, please check error messages above.${NC}"
    exit 1
fi
${MPI_COMPILER} ${FLAGS_PROFILE} test.cc swing_profile.o -o test_profile ${FLAGS_PROFILE}