#!/bin/bash
# serve DeepSeek-V4.1-Flash from both nodes (rank 1 first, detached; rank 0 in the foreground):
#   tools/dsv41_serve2.sh [extra serve args...]      e.g. --port 8899 --context 65536
set -e
cd /home/urtho/dev/_experiments/ai/tensorfold
rsync -a --exclude .venv --exclude .git --exclude notes/ref --exclude out ./ aiai:tensorfold/
rsync -a --exclude .venv --exclude .git --exclude notes/ref --exclude out ./ aiai2:tensorfold/
for h in aiai aiai2; do timeout 30 ssh $h "docker exec tf-dev pkill -f 'tensorfold serve|tools/dsv41_' ; true" >/dev/null 2>&1; done
sleep 3
M=/models/dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw
ENV="-e PYTHONPATH=/tf/src -e TF_DSV41_ENGRAM_DIR=/models/dsv41/DeepSeek-V4.1-Flash-engram -e NCCL_IB_GID_INDEX=3 -e NCCL_DEBUG=ERROR"
timeout 30 ssh aiai2 "docker exec -d $ENV -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 -e NCCL_IB_HCA==roceP2p1s0f1,rocep1s0f1 tf-dev bash -c 'cd /tf && python -m tensorfold serve $M --tp 2 --rank 1 --master 10.42.0.1 $* > /tf/serve-r1.log 2>&1'"
exec ssh aiai "docker exec $ENV -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 -e NCCL_IB_HCA==roceP2p1s0f0,rocep1s0f0 tf-dev bash -c 'cd /tf && python -m tensorfold serve $M --tp 2 --rank 0 --master 10.42.0.1 --host 0.0.0.0 $* 2>&1 | grep -v -E \"Warning|USDT\"'"
