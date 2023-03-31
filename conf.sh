#!/bin/bash
SYSTEM="daint"

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

  *)
    echo -n "Unknown SYSTEM "$SYSTEM
    ;;
esac