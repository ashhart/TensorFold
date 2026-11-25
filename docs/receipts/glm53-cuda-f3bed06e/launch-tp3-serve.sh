#!/usr/bin/env bash
# conformance.py --binary for the TP=3 server: `tp3serve.sh serve MODEL ARGS...` runs ranks 1 and 2 on .38/.39 and
# rank 0 here, each in container <container> with ARGS; stops all three when terminated. MODEL is ignored (the volume's).
shift 2
ARGS="$*"
IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
declare -A VOL=([37]=<hf-cache> [38]=<hf-volume> [39]=<hf-volume>)
stop() { for h in 37 38 39; do ssh -o BatchMode=yes <user>@<lan>.$h 'docker rm -f <container>' >/dev/null 2>&1; done; }
trap 'stop; exit 0' TERM INT
stop
for h in 37 38 39; do ssh <user>@<lan>.$h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'; done
r=0
for h in 37 38 39; do
  ssh -o BatchMode=yes <user>@<lan>.$h "docker run -d --name <container> --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 --device /dev/infiniband --memory 120g -e NCCL_COMM_ID=<link-a>:29651 -e NCCL_SOCKET_IFNAME=<roce-if> -e NCCL_DEBUG=WARN -e TF_GLM_RANK=$r -e TF_GLM_WORLD=3 -e TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE:-6} ${XENV:-} -v ${VOL[$h]}:/hf:ro -v \$HOME/glm53-b/${BIN:-binE}:/bin2:ro --entrypoint sh $IMAGE -c 'S=\$(ls -d /hf/hub/models--Mia-AiLab--GLM-5.3-EXL3-2.75bpw-TensorFold/snapshots/*); exec /bin2/tensorfold-native serve \$S $ARGS --tp 3 --rank $r'" >/dev/null || { stop; exit 1; }
  r=$((r + 1))
done
docker logs -f <container> 2>&1 | grep --line-buffered -v SafeAllocator &
while [ "$(docker inspect -f '{{.State.Running}}' <container> 2>/dev/null)" = true ]; do sleep 5 & wait $!; done
stop; exit 1
