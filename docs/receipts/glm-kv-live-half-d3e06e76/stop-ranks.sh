#!/usr/bin/env bash
# Stop both native ranks: TERM, wait for the processes to exit (drain is up to 20 s), KILL leftovers, then confirm
# the GPUs and the bench lock are free on both nodes.
pkill -TERM -f "[t]ensorfold-native serve"; ssh -o BatchMode=yes RANK1_RING_IP "pkill -TERM -f \"[t]ensorfold-native serve\""
for i in $(seq 1 30); do
  a=$(pgrep -f "[t]ensorfold-native serve" | wc -l); b=$(ssh -o BatchMode=yes RANK1_RING_IP "pgrep -f \"[t]ensorfold-native serve\" | wc -l")
  [[ "$a" == 0 && "$b" == 0 ]] && break; sleep 2
done
pkill -KILL -f "[t]ensorfold-native serve" 2>/dev/null; ssh -o BatchMode=yes RANK1_RING_IP "pkill -KILL -f \"[t]ensorfold-native serve\"" 2>/dev/null
for i in $(seq 1 30); do
  g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); h=$(ssh -o BatchMode=yes RANK1_RING_IP "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
  l0=$(flock -n $GPU_LOCK true && echo free || echo held); l1=$(ssh -o BatchMode=yes RANK1_RING_IP "flock -n $GPU_LOCK true && echo free || echo held")
  [[ "$g" == 0 && "$h" == 0 && "$l0" == free && "$l1" == free ]] && { echo "both ranks stopped, GPUs and locks free after $((i*2 + 2)) s"; exit 0; }; sleep 2
done
echo "ranks not fully stopped: gpu0=$g gpu1=$h lock0=$l0 lock1=$l1"; exit 1
