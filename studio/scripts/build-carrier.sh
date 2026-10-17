#!/bin/bash
# Build (PUSH=1: and push) the GHCR carrier: Mac wheel, lock, scripts, Zig engine; Spark CUDA engine from $PREFIX/tf-h3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-studio}"
TAG="${TAG:-2.8.0}"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="a2b592fef5366bf16d4729651b6549094981d3fe"
cd "$HERE"
mkdir -p payload
python3 -m pip wheel --no-deps -q -w payload "tensorfold @ git+$TF_REPO@$TF_COMMIT"
cp requirements.lock h3_generate.py qwen_image_generate.py zig_fasth3.py fasth3_q8.py h3_text.py h3_media.py h3_condition.py h3_decode.py payload/
( cd payload && sha256sum tensorfold-*.whl requirements.lock h3_generate.py qwen_image_generate.py zig_fasth3.py fasth3_q8.py h3_text.py h3_media.py h3_condition.py h3_decode.py > SHA256SUMS )
if [ -x payload/zig-engine/tf-h3-dit ]; then ( cd payload/zig-engine && sha256sum -c SHA256SUMS ) || { echo "payload/zig-engine does not match its checksums" >&2; exit 1; }
else echo "NOTE: no payload/zig-engine: this carrier will not hold the native engine" >&2; fi
# the DGX Spark's engine, when this machine has built it
SPARK_ENGINE="${PREFIX:-$HOME/.local/opt/tensorfold-studio}/tf-h3"
if [ -s "$SPARK_ENGINE/libtf_h3.so" ]; then
  mkdir -p payload/spark-engine
  cp "$SPARK_ENGINE/libtf_h3.so" "$SPARK_ENGINE/commit" payload/spark-engine/
  ( cd payload/spark-engine && sha256sum libtf_h3.so commit > SHA256SUMS )
  echo "spark-engine: libtf_h3.so at $(cut -c1-8 payload/spark-engine/commit) ($(uname -m), sm_121)"
else echo "NOTE: no engine at $SPARK_ENGINE: this carrier will not hold the DGX Spark engine" >&2; fi
cat payload/SHA256SUMS
if [ "${PUSH:-0}" = 1 ]; then
  docker buildx build --platform linux/arm64,linux/amd64 -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" --push .
  docker buildx imagetools inspect "$IMAGE:$TAG" | grep -E '^Digest' || true
else
  docker build -f Dockerfile.ghcr -t "$IMAGE:$TAG" -t "$IMAGE:latest" .
fi
