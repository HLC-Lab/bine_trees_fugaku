MPIRUN="srun" # Command for running MPI applications
MPIRUN_MAP_BY_NODE_FLAG="" # Flag to force ranks to be mapped by node (on this system we do it with "export MPICH_RANK_REORDER_METHOD=0" -- see below)
MPIRUN_ADDITIONAL_FLAGS=""    # Any additional flag that must be used by mpirun
MPI_COMPILER=mpiCC # MPI Compiler
MPI_COMPILER_FLAGS="-O3 -g -Wall -pedantic"
EBB=""
EXTRA_VARIABLES="OMPI_MCA_coll_hcoll_enable=0|UCX_IB_SL=1" # Run on SL 1 (more stable), and do not use hcoll
EXTRA_VARIABLES_DEFAULT="OMPI_MCA_coll_tuned_use_dynamic_rules=1" # Only to be used for default algorithms
DEFAULT_ALGOS_ALLREDUCE="OMPI_MCA_coll_tuned_allreduce_algorithm=1|OMPI_MCA_coll_tuned_allreduce_algorithm=2|OMPI_MCA_coll_tuned_allreduce_algorithm=3|OMPI_MCA_coll_tuned_allreduce_algorithm=4|OMPI_MCA_coll_tuned_allreduce_algorithm=5|OMPI_MCA_coll_tuned_allreduce_algorithm=6"
