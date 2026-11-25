#!/usr/bin/env bash
# tf-cuda-test at TP=3 (.37 .38 .39): cudatest.sh CMD [args after <ckpt> <dc> <rank> <world> <id>]. Drops page cache
# first. Binary and dc.json in ~/glm53-b/ct on every node; container <container>; logs to ~/glm53-b/ct/<CMD>.r<rank>.log
set -u
CMD=$1; shift
IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
declare -A VOL=([37]=<hf-cache> [38]=<hf-volume> [39]=<hf-volume>)
HOSTS=(37 38 39)
for h in "${HOSTS[@]}"; do ssh <user>@<lan>.$h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; docker rm -f <container> >/dev/null 2>&1; rm -f ~/glm53-b/ct/id'; done
run() { # host rank
  ssh <user>@<lan>.$1 "docker run -d --name <container> --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 --device /dev/infiniband --memory 120g -e NCCL_SOCKET_IFNAME=<roce-if> -e NCCL_DEBUG=WARN -e TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE:-9} ${XENV:-} -v ${VOL[$1]}:/hf:ro -v \$HOME/glm53-b/ct:/ct -w /ct --entrypoint sh $IMAGE -c 'S=\$(ls -d /hf/hub/models--Mia-AiLab--GLM-5.3-EXL3-2.75bpw-TensorFold/snapshots/*); exec /ct/tf-cuda-test $CMD \$S /ct/dc.json $2 3 /ct/id $* > /ct/$CMD.r$2.log 2>&1'" >/dev/null
}
run 37 0
for _ in $(seq 60); do ssh <user>@host-a 'test -s ~/glm53-b/ct/id' && break; sleep 2; done
for h in 38 39; do ssh <user>@host-a "scp -q ~/glm53-b/ct/id <user>@<lan>.$h:glm53-b/ct/id"; done
run 38 1; run 39 2
while :; do
  n=0; for h in "${HOSTS[@]}"; do [ "$(ssh <user>@<lan>.$h "docker inspect -f '{{.State.Running}}' <container>" 2>/dev/null)" = true ] && n=$((n+1)); done
  [ $n = 0 ] && break; sleep 15
done
for h in "${HOSTS[@]}"; do echo "== .$h exit $(ssh <user>@<lan>.$h "docker inspect -f '{{.State.ExitCode}}' <container>")"; done
ssh <user>@host-a "grep -E 'PASS|FAIL|RESULT|error|REFUSE' ~/glm53-b/ct/$CMD.r0.log | tail -30"
