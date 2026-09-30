#!/bin/bash
# Full GLM-5.3 on four DGX Sparks with TensorFold (run from the orchestrating machine).
#   tools/tp4_run.sh comm            NCCL fabric check (tools/tp4_comm_check.py) on all four ranks
#   tools/tp4_run.sh serve [ARGS]    tensorfold serve --tp 4 (rank 0 serves HTTP on :$PORT); extra ARGS go to serve
#   tools/tp4_run.sh stop            remove the containers
#   tools/tp4_run.sh logs [RANK]     follow a rank's log
set -u
# Configure (env): NODES = the four Sparks' fabric addresses, rank 0 first (it serves HTTP); IMAGE = a container with
# CUDA 13 + torch (+ optionally cuda-exl3 and vLLM for the shared expert layout's prompt GEMM); CKPT = the EXL3
# checkpoint directory, at the same path on every node (e.g. an NFS export). IF / HCA / GIDS: the ConnectX-7 RoCE port
# NCCL uses (a DGX Spark's QSFP port is enp1s0f1np1 / rocep1s0f1; GIDS = the RoCE v2 GID index of each rank).
: "${NODES:?set NODES to the fabric addresses of the four nodes, rank 0 first}"
: "${IMAGE:?set IMAGE to a CUDA 13 + torch container image}"
: "${CKPT:?set CKPT to the GLM-5.3 EXL3 checkpoint directory (same path on every node)}"
NODES=($NODES)
GIDS=(${GIDS:-3 3 3 3})
MASTER=${MASTER:-${NODES[0]}}
PORT=${PORT:-8890}
SRC=${SRC:-$HOME/tf-glm53}
IF=${IF:-enp1s0f1np1}
HCA=${HCA:-rocep1s0f1}
NAME=tf-glm53

sync_src() {
  for n in "${NODES[@]}"; do
    rsync -a --delete --exclude .git --exclude '.torch_ext*' --exclude __pycache__ "$SRC/" "$n:tf-glm53/" &
  done
  wait
}

run_rank() {   # rank node cmd
  local r=$1 n=$2 cmd=$3
  ssh "$n" "docker rm -f $NAME-r$r >/dev/null 2>&1; docker run -d --name $NAME-r$r --gpus all --network host --ipc=host \
    --device=/dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
    -v \$HOME/tf-glm53:/tf -v $(dirname "$CKPT"):$(dirname "$CKPT"):ro -w /tf \
    -e PYTHONPATH=/tf/src -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/tf/.torch_ext \
    -e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=$HCA -e NCCL_IB_GID_INDEX=${GIDS[$r]} -e NCCL_DEBUG=WARN -e NCCL_PROTO=${NCCL_PROTO:-Simple} -e NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-2} ${DOCKER_ENV:-} \
    -e TENSORFOLD_NO_UPDATE_CHECK=1 -e HF_HUB_OFFLINE=1 \
    --entrypoint bash $IMAGE -c '
      P=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; X=/tmp/xinc; mkdir -p \$X
      for f in \$P/*.h; do b=\$(basename \$f); [ -e /usr/local/cuda/include/\$b ] || ln -s \$f \$X/\$b; done
      export CPATH=\$X NVCC_APPEND_FLAGS=-I\$X
      $cmd'" >/dev/null && echo "rank $r started on $n"
}

case "${1:-}" in
  comm)
    sync_src
    for r in 3 2 1 0; do
      run_rank $r "${NODES[$r]}" "python3 tools/${CHECK:-tp4_comm_check.py} --rank $r --master $MASTER"
    done
    ssh "${NODES[0]}" "docker wait $NAME-r0 >/dev/null; docker logs $NAME-r0 2>&1 | grep -E 'tp4-comm|roce-big|overlap\]|Error|error' | tail -12"
    ;;
  serve)
    shift
    sync_src
    for r in 3 2 1 0; do
      run_rank $r "${NODES[$r]}" "python3 -m tensorfold.cli serve $CKPT --tp 4 --rank $r --master $MASTER \
        --port $PORT --host 0.0.0.0 --name glm-5.3-tf $*"
    done
    ;;
  stop)
    for r in 0 1 2 3; do ssh "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1"; done; echo stopped ;;
  logs)
    r=${2:-0}; ssh "${NODES[$r]}" "docker logs -f $NAME-r$r" ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
