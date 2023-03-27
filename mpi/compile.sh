GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)

MPI_COMPILER=mpic++
#FLAGS="-O0 -pg"
FLAGS="-O3"

${MPI_COMPILER} ${FLAGS} -c -fPIC swing.cc -o swing.o
if [ ! -f "swing.o" ]; then
    echo "${RED}[Error] swing.o compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${FLAGS} -shared -pthread -o swing.so swing.o 
if [ ! -f "swing.so" ]; then
    echo "${RED}[Error] swing.so compilation failed, please check error messages above.${NC}"
    exit 1
fi

${MPI_COMPILER} ${FLAGS} test.cc -o test