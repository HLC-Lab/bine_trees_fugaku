#!/bin/bash
SYSTEM="fugaku"

case $SYSTEM in
  fugaku)
    source conf/fugaku.sh
    ;;

  local)
    source conf/local.sh
    ;;
  
  *)
    echo -n "Unknown SYSTEM "$SYSTEM
    ;;
esac
