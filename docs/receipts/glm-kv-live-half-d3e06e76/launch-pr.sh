#!/usr/bin/env bash
# The PR branch's binary (upstream main plus TF_GLM_KV_LIVE_HALF) on a GB10 rank: launch-pr.sh RANK [extra serve args]
# Same ring, model and ports as launch.sh; none of the fork's codec, ledger or dump settings (upstream has none).
set -euo pipefail
rank=${1:?rank}; shift || true
root=$WORK
binary=${PROBE_BINARY:-$root/TensorFold-pr/zig-out/native/bin/tensorfold-native}
model=$MODEL
export LD_LIBRARY_PATH=$NCCL_LIB_DIR:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TF_GLM_RANK=$rank TF_GLM_WORLD=2 NCCL_COMM_ID=RANK0_RING_IP:29813
export NCCL_IB_DISABLE=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if [ "$rank" = 0 ]; then
  export NCCL_SOCKET_IFNAME=enp1s0f0np0 TF_ROCE_HCA=rocep1s0f0,roceP2p1s0f0
else
  export NCCL_SOCKET_IFNAME=enp1s0f1np1 TF_ROCE_HCA=rocep1s0f1,roceP2p1s0f1
fi
export NCCL_IB_HCA=$TF_ROCE_HCA TF_ROCE=1 TF_ROCE_MAX_KB=512
export TENSORFOLD_CUDA_KERNELS=native TF_GLM_PROF=${TF_GLM_PROF:-0} TF_HEAT_HIGH=90 TF_HEAT_LOW=85
export TF_GLM_HOST_CACHE=${TF_GLM_HOST_CACHE:-1}
echo "rank $rank launch-pr.sh start $(date -u +%H:%M:%S) binary=$binary live_half=${TF_GLM_KV_LIVE_HALF:-unset}"
if [[ $(ulimit -Sl) != unlimited || $(ulimit -Hl) != unlimited ]]; then sudo -n prlimit --pid $$ --memlock=unlimited:unlimited; fi
echo "rank $rank memlock soft=$(ulimit -Sl) hard=$(ulimit -Hl)"
flock -w 60 $GPU_LOCK taskset -c 0-4,10-14 "$binary" serve "$model" \
  --backend cuda --host 127.0.0.1 --port 18080 --name kvprobe-native --context "${PROBE_CONTEXT:-32768}" \
  --parallel 1 --prompt-cache-gib "${PROBE_PROMPT_CACHE_GIB:-16}" --keep-warm 0 --idle-unload 0 --no-update-check "$@"
rc=$?; echo "rank $rank: engine exited rc=$rc at $(date -u +%H:%M:%S)"; exit $rc
