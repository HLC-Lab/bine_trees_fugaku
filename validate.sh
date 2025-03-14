for COLLECTIVE in allreduce allgather reduce_scatter bcast reduce scatter gather alltoall
do
    echo "###############################"
    echo "###############################"
    echo "Validating $COLLECTIVE"
    echo "###############################"
    echo "###############################"
    ./validate/$COLLECTIVE.sh
done