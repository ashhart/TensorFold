#!/bin/bash
# run the serial engine on both nodes: run2.sh [extra args...]
set -e
cd /home/urtho/dev/_experiments/ai/tensorfold
rsync -a --exclude .venv --exclude .git --exclude notes/ref --exclude out ./ aiai:tensorfold/
rsync -a --exclude .venv --exclude .git --exclude notes/ref --exclude out ./ aiai2:tensorfold/
M=/models/dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw; G=/models/dsv41/DeepSeek-V4.1-Flash-engram
timeout 30 ssh aiai2 "docker exec -d -e ROWS=$ROWS -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 -e NCCL_IB_HCA=roceP2p1s0f1 -e NCCL_IB_GID_INDEX=3 -e NCCL_DEBUG=ERROR tf-dev bash -c 'cd /tf && python tools/dsv41_rowdiff.py $M $G --rank 1 --master 10.42.0.1 $* > /tf/serial-r1.log 2>&1'"
timeout 3000 ssh aiai "docker exec -e ROWS=$ROWS -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 -e NCCL_IB_HCA=roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_DEBUG=ERROR tf-dev bash -c 'cd /tf && python tools/dsv41_rowdiff.py $M $G --rank 0 --master 10.42.0.1 $* 2>&1 | grep -v -E \"Warning|USDT\"'" | tail -40
echo "--- rank 1 tail"; timeout 20 ssh aiai2 'tail -4 ~/tensorfold/serial-r1.log'
