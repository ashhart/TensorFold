#!/bin/bash
# The GLM-5.3-Flash two-rank capture: rank 0 (this machine) drives the prompt matrix, rank 1
# (TF_PEER, ssh) follows; both record their Triton and extension launches. Pattern:
# ../nemotron/box/capture.sh, extended for a two-machine family — the engine pair needs network
# (--network host) and the peer's rank must be up before rank 0's startup handshake completes.
#
# Usage: TF_NEMO=<work dir: src/ out/ aot/> TF_MODEL_ROOT=<HF model dir whose snapshots/<rev> holds
# the checkpoint> [TF_PEER=host] [TF_MASTER=ip] [TF_PORT=29561] [TF_JOURNAL=<file>] \
#   flock <GPU lock> bash -u capture.sh RUN [--bench] [capture.py options]
#
# TF_NEMO/src is a checkout of this repository THAT ALSO CARRIES the Python engine's src/ tree
# (the 1.0.0 split moved it to the python-0.6 branch; a union checkout serves both — see README).
set -u
RUN="${1:?run id}"
MODE="${2:-}"
shift $(( $# < 2 ? $# : 2 ))
MORE=("$@")
TF="${TF_NEMO:?set TF_NEMO}"
MODELROOT="${TF_MODEL_ROOT:?set TF_MODEL_ROOT (the models--org--name directory, not the snapshot)}"
SNAP="$(ls -d "${MODELROOT:?}"/snapshots/*/ | head -1)"
WSNAP="/modelroot/snapshots/$(basename "$(dirname "$SNAP")")"
PEER="${TF_PEER:-}"
MASTER="${TF_MASTER:-127.0.0.1}"
PORT="${TF_PORT:-29561}"
OUT="${TF:?}/out/${RUN:?}"
CACHE="${TF:?}/aot/${RUN:?}"
WCACHE="${TF:?}/aot/${RUN:?}-r1"          # peer-side cache, synced back after the run
WOUT="${TF:?}/out/${RUN:?}-r1"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
JOURNAL="${TF_JOURNAL:-/dev/null}"
WHO="${TF_WHO:-zig-glm}"
TREE=$(cat "${TF:?}/src/TREE" 2>/dev/null || echo unknown)
NAME="tf-zig-glm-${RUN:?}"

journal() { printf '{"utc": "%s", "event": "%s", "who": "%s", "run": "glm-%s", "tree": "%s"%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$WHO" "$RUN" "$TREE" "${2:-}" >> "$JOURNAL"; }
cleanup() { for c in $(docker ps -q --filter "label=tensorfold.zig=glm-${RUN}"); do docker stop --time 15 "$c" >/dev/null; done; }
trap 'cleanup; journal END ", \"rc\": 143"; exit 143' TERM INT

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
cons=$(docker ps -q | wc -l)
wapps=0; wcons=0
if [ -n "$PEER" ]; then
  wapps=$(ssh "$PEER" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l')
  wcons=$(ssh "$PEER" 'docker ps -q | wc -l')
fi
if [ "$apps" != 0 ] || [ "$cons" != 0 ] || [ "$wapps" != 0 ] || [ "$wcons" != 0 ]; then
  echo "PREFLIGHT-FAIL: containers or compute apps present (local $apps/$cons, peer $wapps/$wcons)"; exit 3
fi

journal START ", \"cmd\": \"glm/box/capture.sh ${MODE}\", \"model\": \"$(basename "$MODELROOT")\""
grep MemAvailable /proc/meminfo
mkdir -p "$OUT/rank0" "$OUT/rank1" "$CACHE/triton" "$CACHE/torch_ext" "$CACHE/cuda_cache"
COMMON=(--gpus all --ipc host --network host --memory 80g --memory-swap 80g --read-only
  --tmpfs /tmp:size=8g --log-driver none
  -v "$TF/src:/tensorfold:ro" -v "$MODELROOT:/modelroot:ro"
  -e HOME=/tmp -e PYTHONPATH=/tensorfold/src -e PYTHONDONTWRITEBYTECODE=1
  -e TORCH_CUDA_ARCH_LIST=12.1 -w /tensorfold --entrypoint python3)
CAP="/tensorfold/zig/tests/cuda/glm/capture.py --out /out --rank RANK --master $MASTER --port $PORT --tools /tensorfold/tools/zig --model $WSNAP"

rc=0
if [ -n "$PEER" ]; then
  ssh "$PEER" "mkdir -p $WCACHE/triton $WCACHE/torch_ext $WCACHE/cuda_cache $WOUT" || rc=1
  echo "== rank 1 ($PEER) =="
  ssh "$PEER" "docker run -d --name $NAME-r1 --label tensorfold.zig=glm-${RUN} ${COMMON[*]} \
    -v $WCACHE:/aot -v $WOUT:/out \
    -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/aot/torch_ext -e CUDA_CACHE_PATH=/aot/cuda_cache \
    $IMAGE -B ${CAP/RANK/1}" || rc=1
fi

echo "== rank 0 (drives the matrix) =="
timeout 3600 docker run --rm --name "$NAME-r0" --label "tensorfold.zig=glm-${RUN}" "${COMMON[@]}" \
  -v "$OUT/rank0:/out" -v "$CACHE:/aot" \
  -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/aot/torch_ext -e CUDA_CACHE_PATH=/aot/cuda_cache \
  "$IMAGE" -B ${CAP/RANK/0} "${MORE[@]}" > "$OUT/rank0-capture.log" 2>&1 || rc=$?
echo "rank0 rc $rc"

if [ -n "$PEER" ]; then
  ssh "$PEER" "docker logs $NAME-r1 > $WOUT/../rank1-capture.log 2>&1; docker cp $NAME-r1:/out/. $WOUT/ >/dev/null 2>&1; docker stop -t 15 $NAME-r1 >/dev/null 2>&1; docker rm $NAME-r1 >/dev/null 2>&1" || true
  scp -q -r "$PEER:$WOUT/." "$OUT/rank1/" || echo "warn: rank1 output copy failed"
  mkdir -p "$CACHE-r1/triton" && (rsync -a "$PEER:$WCACHE/triton/" "$CACHE-r1/triton/" 2>/dev/null || scp -q -r "$PEER:$WCACHE/triton/." "$CACHE-r1/triton/") || true
fi

if [ "$rc" = 0 ] && [ "$MODE" != "--bench" ]; then
  for R in rank0 rank1; do
    [ -f "$OUT/$R/launches.json" ] && python3 -B "${TF:?}/src/tools/zig/triton_aot_manifest.py" \
      --cache "$CACHE/triton" --launches "$OUT/$R/launches.json" --mount /aot/triton --out "$OUT/$R/manifest.json" || rc=$?
  done
  [ -f "$OUT/rank0/manifest.json" ] && [ -f "$OUT/rank1/manifest.json" ] && \
    python3 -B "${TF:?}/src/tools/zig/aot_pack.py" --manifest "$OUT/rank0/manifest.json" --cache "$CACHE/triton" \
      --manifest "$OUT/rank1/manifest.json" --cache "$CACHE-r1/triton" --jit "$OUT/rank0/jit.json" --out "$CACHE" || rc=$?
  echo "packed: $(ls "$CACHE" | head -3)"
fi
cleanup
journal END ", \"rc\": $rc"
exit $rc
