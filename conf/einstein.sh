MPIRUN="mpirun" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="" # Flag to force ranks to be mapped by node (on this system we do it with "export MPICH_RANK_REORDER_METHOD=0" -- see below)
MPIRUN_ADDITIONAL_FLAGS=""    # Any additional flag that must be used by mpirun
MPI_COMPILER=mpiCC # MPI Compiler
MPI_COMPILER_FLAGS="-O3 -g -Wall -pedantic"
EBB=""
