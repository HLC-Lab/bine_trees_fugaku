MPIRUN="srun" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="-m plane=1" # Flag to force ranks to be mapped by node (srun)
MPIRUN_ADDITIONAL_FLAGS=""    # Any additional flag that must be used by mpirun
MPI_COMPILER=mpicc # MPI Compiler
EBB=""

module load intel-para/2021b
#module load Intel/2021.4.0
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export GOMP_CPU_AFFINITY="0-23"
#module load mpi-settings/plain

export I_MPI_FABRIC=shm:ofa
