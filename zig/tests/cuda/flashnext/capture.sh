#!/usr/bin/env bash
# Usage: capture.sh WORK SNAPSHOT_DIR [IMAGE]; captures Flash Next's launches, references and a kernel set under WORK.
set -euo pipefail
WORK=$(readlink -f "$1"); SNAP=$(readlink -f "$2"); IMAGE=${3:-tensorfold-qwen38:v0.6.1}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HF=$(dirname "$(dirname "$(dirname "$SNAP")")")
REL=${SNAP#"$HF"/}
# CONTEXT=16383 ONLY=long3k,long12k TEACHERS=teacher32 OUT=sparse: the prompts past the indexer's 2,048-key budget
CONTEXT=${CONTEXT:-1535}; ONLY=${ONLY:-}; TEACHERS=${TEACHERS-teacher,teacher32}; OUT=${OUT:-cap}
mkdir -p "$WORK/cache" "$WORK/out/$OUT"
KSET=$WORK/kernels; [ "$OUT" = cap ] || KSET=$WORK/kernels-$OUT
# SNAPSHOT_DIR is the HF snapshot (blobs at ../../blobs); the default context 1535 keeps attention dense (QSA 0)
docker run --rm --name fn-capture --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  -e TENSORFOLD_MEMORY_RESERVE_GIB=2 \
  -v "$HF":/hf:ro -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out -w /out \
  "$IMAGE" python -u /src/zig/tests/cuda/flashnext/capture.py --model "/hf/$REL" --out "/out/$OUT" \
  --tools /src/tools/zig --tokens 32 --context "$CONTEXT" --only "$ONLY" --teachers "$TEACHERS"
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/$OUT/launches.json" --mount /cache/triton --out "$WORK/out/$OUT/manifest.json"
"$(dirname "$0")/pack.sh" "$KSET" "$WORK/out/$OUT"
echo "prompt ids (*.ids) and references (results.json) in $WORK/out/$OUT"
