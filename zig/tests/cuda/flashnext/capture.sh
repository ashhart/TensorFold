#!/usr/bin/env bash
# Flash Next (qwen4_exp) capture -> manifest -> kernel set for zig's flashnext-prompt and the native lanes engine.
# Usage: capture.sh WORK SNAPSHOT_DIR [IMAGE]
#   WORK          a host dir this script owns: WORK/cache (Triton + torch extension caches), WORK/out/cap, WORK/kernels
#   SNAPSHOT_DIR  the checkpoint's HF snapshot dir (its blobs must sit at ../../blobs, as in an HF cache)
#   IMAGE         TensorFold 0.6.1 image, default tensorfold-qwen38:v0.6.1
# The Python engine runs drafts off, greedy, --context 1535 (1536 cache slots: _chunks/_merge NCH=3, QSA=0, the dense
# attention the port launches), --ple-on-ssd, bf16 KV. Needs the GPU free (~78 GiB).
set -euo pipefail
WORK=$(readlink -f "$1"); SNAP=$(readlink -f "$2"); IMAGE=${3:-tensorfold-qwen38:v0.6.1}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HF=$(dirname "$(dirname "$(dirname "$SNAP")")")          # .../hub/models--X/snapshots/REV -> .../hub (holds the blobs)
REL=${SNAP#"$HF"/}
mkdir -p "$WORK/cache" "$WORK/out/cap"
docker run --rm --name fn-capture --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  -e TENSORFOLD_MEMORY_RESERVE_GIB=2 \
  -v "$HF":/hf:ro -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out -w /out \
  "$IMAGE" python -u /src/zig/tests/cuda/flashnext/capture.py --model "/hf/$REL" --out /out/cap \
  --tools /src/tools/zig --tokens 32 --context 1535
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/cap/launches.json" --mount /cache/triton --out "$WORK/out/cap/manifest.json"
python3 -B "$REPO/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/cap/manifest.json" \
  --cache "$WORK/cache/triton" --jit "$WORK/out/cap/jit.json" --out "$WORK/kernels"
echo "kernel set: $WORK/kernels; prompt ids and references: $WORK/out/cap/{short,mid,long}.ids, results.json"
