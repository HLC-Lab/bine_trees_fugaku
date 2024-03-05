NODES=8
rm -f gmon.out-*
export GMON_OUT_PREFIX=gmon.out-
mpirun -x GMON_OUT_PREFIX -np ${NODES} ./test_profile &> log.txt
gprof -s ./test_profile gmon.out-*
gprof ./test_profile gmon.sum > profile.out
rm -f gmon.out-* gmon.sum