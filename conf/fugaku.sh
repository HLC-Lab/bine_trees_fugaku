MPI_COMPILER=mpiFCCpx # MPI Compiler
#MPI_COMPILER=mpiFCC # MPI Compiler
MPI_COMPILER_FLAGS="-Kfast,openmp,parallel -DBINE_USE_UTOFU" # -Koptmsg=2 # To print what the compiler parallelizes
MPIRUN="mpiexec" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="" # Flag to force ranks to be mapped by node
MPIRUN_ADDITIONAL_FLAGS="" # Any additional flag that must be used by mpirun
EBB="" # Leave empty
