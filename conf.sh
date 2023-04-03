#!/bin/bash
SYSTEM="deep-est"

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

  *)
    echo -n "Unknown SYSTEM "$SYSTEM
    ;;
esac
