#!/bin/bash
SYSTEM="einstein"

case $SYSTEM in
  daint)
    source conf/daint.sh
    ;;
  
  alps)
    source conf/alps.sh
    ;;

  deep-est)
    source conf/deep-est.sh
    ;;
  
  einstein)
    source conf/einstein.sh
    ;;

  leonardo)
    source conf/leonardo.sh
    ;;

  fugaku)
    source conf/fugaku.sh
    ;;

  *)
    echo -n "Unknown SYSTEM "$SYSTEM
    ;;
esac
