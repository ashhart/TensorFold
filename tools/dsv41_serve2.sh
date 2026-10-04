#!/bin/bash
# serve DeepSeek-V4.1-Flash from both nodes (rank 1 first, detached; rank 0 in the foreground):
#   tools/dsv41_serve2.sh [extra serve args...]      e.g. --port 8899 --context 65536
set -e
cd /home/urtho/dev/_experiments/ai/tensorfold
rsync -a --exclude .venv --exclude .git --exclude notes/ref --exclude out --exclude __pycache__ ./ aiai:tensorfold/
rsync -a --exclude .venv --exclude .git --exclude notes/ref --exclude out --exclude __pycache__ ./ aiai2:tensorfold/
for h in aiai aiai2; do timeout 30 ssh $h "docker exec tf-dev pkill -f 'tensorfold serve|tools/dsv41_' ; true" >/dev/null 2>&1; done
# wait until the old ranks have exited (their memory is back), at most 2 minutes
for h in aiai aiai2; do
  for i in $(seq 1 60); do
    timeout 10 ssh $h "docker exec tf-dev pgrep -f 'tensorfold serve|tools/dsv41_'" >/dev/null 2>&1 || break
    sleep 2
  done
done
sleep 2
M=/models/dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw
ENV="-e TF_DSV41_FIXED_GIB=${TF_DSV41_FIXED_GIB:-} -e TF_CARVEOUT=${TF_CARVEOUT:-} -e TF_MULTI_CHECK=${TF_MULTI_CHECK:-} -e TF_DSV41_WARM=${TF_DSV41_WARM:-} -e TF_DSV41_KV_FP8=${TF_DSV41_KV_FP8:-} -e TF_STEP_THREADS=${TF_STEP_THREADS:-} -e TF_MULTI_DRAFTS=${TF_MULTI_DRAFTS:-} -e TF_MULTI_PROF=${TF_MULTI_PROF:-} -e TF_SPLIT_READS=${TF_SPLIT_READS:-} -e TF_SKIP_SHARED=${TF_SKIP_SHARED:-} -e TF_FOLD_SHARED=${TF_FOLD_SHARED:-} -e TF_ROUTER_SLICES=${TF_ROUTER_SLICES:-} -e TF_L2_PREFETCH=${TF_L2_PREFETCH:-} -e TF_COPY_DRAFTS=${TF_COPY_DRAFTS:-} -e TF_COPY_MAX=${TF_COPY_MAX:-} -e TF_COMM=${TF_COMM:-} -e TF_RDMA_MAX_KB=${TF_RDMA_MAX_KB:-} -e TF_ONE_GRAPH=${TF_ONE_GRAPH:-} -e TF_DSV41_SHARED_POOL=${TF_DSV41_SHARED_POOL:-} -e TF_DSV41_POOL_TOKENS=${TF_DSV41_POOL_TOKENS:-} -e TF_DSV41_KEPT_ENTRIES=${TF_DSV41_KEPT_ENTRIES:-} -e TF_DSV41_KEEP_MIN=${TF_DSV41_KEEP_MIN:-} -e TF_DSV41_POOL_KEEP=${TF_DSV41_POOL_KEEP:-} -e TF_DSV41_GROW_AHEAD=${TF_DSV41_GROW_AHEAD:-} -e TF_DSV41_MEMLOG=${TF_DSV41_MEMLOG:-} -e TF_DSV41_WIDTHS=${TF_DSV41_WIDTHS:-} -e TF_DSV41_SHARED_GRAPHS=${TF_DSV41_SHARED_GRAPHS:-} -e PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-} -e NUMPY_MADVISE_HUGEPAGE=0 -e MIMALLOC_ALLOW_THP=0 -e TF_DSV41_BOUNDED_TAIL=${TF_DSV41_BOUNDED_TAIL:-} -e TF_DSV41_FAST_TOPK=${TF_DSV41_FAST_TOPK:-} -e TF_DSV41_PREPARED=${TF_DSV41_PREPARED:-} -e TF_DSV41_TOOL_GRAMMAR=${TF_DSV41_TOOL_GRAMMAR:-} -e TF_DSV41_DISK=${TF_DSV41_DISK:-} -e TF_DSV41_DISK_GIB=${TF_DSV41_DISK_GIB:-} -e TF_DSV41_DISK_MIN=${TF_DSV41_DISK_MIN:-} -e TF_DSV41_DISK_GAIN=${TF_DSV41_DISK_GAIN:-} -e TF_DSV41_DISK_STAGE_MIB=${TF_DSV41_DISK_STAGE_MIB:-} -e TF_EXPERT_LOADS=${TF_EXPERT_LOADS:-} -e TF_EXPERT_LOADS_CFG=${TF_EXPERT_LOADS_CFG:-} -e TF_L2_PREFETCH=${TF_L2_PREFETCH:-} -e PYTHONPATH=/tf/src -e TF_DSV41_ENGRAM_DIR=/models/dsv41/DeepSeek-V4.1-Flash-engram -e NCCL_IB_GID_INDEX=3 -e NCCL_DEBUG=ERROR"
timeout 30 ssh aiai2 "docker exec -d $ENV -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 -e NCCL_IB_HCA==roceP2p1s0f1,rocep1s0f1 tf-dev bash -c 'cd /tf && python -m tensorfold serve $M --tp 2 --rank 1 --master 10.42.0.1 $* > /tf/serve-r1.log 2>&1'"
exec ssh aiai "docker exec $ENV -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 -e NCCL_IB_HCA==roceP2p1s0f0,rocep1s0f0 tf-dev bash -c 'cd /tf && python -m tensorfold serve $M --tp 2 --rank 0 --master 10.42.0.1 --host 0.0.0.0 $* 2>&1 | grep --line-buffered -v -E \"Warning|USDT\"'"
