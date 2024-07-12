MPIRUN="srun" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="-m plane=1" # Flag to force ranks to be mapped by node (srun)
MPIRUN_PINNING_FLAGS="--cpu-bind=map_cpu=2,2" # Pinning flags
MPIRUN_ADDITIONAL_FLAGS=""    # Any additional flag that must be used by mpirun
MPI_COMPILER=CC # MPI Compiler
MPI_COMPILER_FLAGS="-O3 -g -Wall -pedantic"
EBB="/project/g34/desensi/netgauge-2.4.6/netgauge"
