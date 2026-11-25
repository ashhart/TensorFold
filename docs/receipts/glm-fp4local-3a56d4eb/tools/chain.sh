#!/bin/bash
# usage: chain.sh QUEUE; runs each nonblank line of QUEUE as run.sh's arguments, one launch after another.
cd ~/fl-receipt
while read -r line; do [ -n "$line" ] && eval ./run.sh $line < /dev/null; done < "$1"
echo "$(date '+%F %T') chain $1 END" >> ~/fl-receipt/runs.log
