MPIRUN="srun" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="--cpu-bind=map_cpu=57,25,9,41,49,17,1,33" 
MPIRUN_ADDITIONAL_FLAGS=""    # Any additional flag that must be used by mpirun
MPI_COMPILER=CC # MPI Compiler
MPI_COMPILER_FLAGS="-O3 -g -Wall -pedantic"
EBB=""
