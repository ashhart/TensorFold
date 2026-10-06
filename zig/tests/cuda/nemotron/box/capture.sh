#!/bin/bash
# The Python engine's oracle run for the Zig Nemotron engine: Triton cubins, weight digests, dumps, tokens and timings. Usage: TF_NEMO=<work dir: src/ out/> TF_MODEL=<checkpoint dir> [TF_JOURNAL=<file>] flock <GPU lock> bash -u capture.sh RUN [--bench|--record] [capture.py options]
set -u
RUN="${1:?run id}"
MODE="${2:-}"
shift $(( $# < 2 ? $# : 2 ))
MORE=("$@")
TF="${TF_NEMO:?set TF_NEMO}"
MODEL="${TF_MODEL:?set TF_MODEL}"
OUT="${TF:?}/out/${RUN:?}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
JOURNAL="${TF_JOURNAL:-/dev/null}"
WHO="${TF_WHO:-zig-nemotron}"
TREE=$(cat "${TF:?}/src/TREE" 2>/dev/null || echo unknown)
NAME="tf-zig-nemo-${RUN:?}"
mkdir -p "${OUT:?}" "${TF:?}/aot"

journal() { printf '{"utc": "%s", "event": "%s", "who": "%s", "run": "nemo-%s", "tree": "%s"%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$WHO" "$RUN" "$TREE" "${2:-}" >> "$JOURNAL"; }
cleanup() { for c in $(docker ps -q --filter "label=tensorfold.zig=nemo-${RUN}"); do docker stop --time 15 "$c" >/dev/null; done; }
trap 'cleanup; journal END ", \"rc\": 143"; exit 143' TERM INT

apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v -x -F "${TF_RESIDENT:-none}" || true)
if [ -n "$(docker ps -q)" ] || [ -n "$apps" ]; then
  echo "PREFLIGHT-FAIL: containers or compute apps present"; docker ps; echo "$apps"; exit 3
fi
journal START ", \"cmd\": \"zig/tests/cuda/nemotron/box/capture.sh ${MODE}\", \"model\": \"$(basename "$MODEL")\""
grep MemAvailable /proc/meminfo
extra=(--tools /tensorfold/tools/zig)
cache="${TF:?}/aot/${RUN:?}"
if [ "$MODE" = "--bench" ]; then extra=(--bench); cache="${TF:?}/aot/bench"; fi
if [ "$MODE" != "--bench" ] && [ "$MODE" != "--record" ]; then echo "mode --bench or --record"; exit 2; fi
mkdir -p "${cache:?}/triton" "${cache:?}/torch_ext" "${cache:?}/cuda_cache"
rc=0
timeout 3600 docker run --rm --name "$NAME" --label "tensorfold.zig=nemo-${RUN}" --gpus all --ipc host --network none \
  --memory 80g --memory-swap 80g --read-only --tmpfs /tmp:size=8g --log-driver none \
  -v "${MODEL:?}:/model:ro" -v "${TF:?}/src:/tensorfold:ro" -v "${OUT:?}:/out" \
  -v "${cache:?}:/aot" -e HOME=/tmp -e PYTHONPATH=/tensorfold/src -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e TORCH_CUDA_ARCH_LIST=12.1 -e TRITON_CACHE_DIR=/aot/triton \
  -e TORCH_EXTENSIONS_DIR=/aot/torch_ext -e CUDA_CACHE_PATH=/aot/cuda_cache -w /tensorfold \
  --entrypoint python "$IMAGE" -B /tensorfold/zig/tests/cuda/nemotron/capture.py --model /model --out /out \
  "${extra[@]}" "${MORE[@]}" > "${OUT:?}/capture.log" 2>&1 || rc=$?
echo "container rc $rc"
if [ "$rc" = 0 ] && [ "$MODE" != "--bench" ]; then
  python3 -B "${TF:?}/src/tools/zig/triton_aot_manifest.py" --cache "${cache:?}/triton" --launches "${OUT:?}/launches.json" \
    --mount /aot/triton --out "${OUT:?}/manifest.json" || rc=$?
fi
cleanup
journal END ", \"rc\": $rc"
exit $rc
