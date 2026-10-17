#!/bin/bash
# Build TensorFold's H3 CUDA engine (libtf_h3.so) into $PREFIX/tf-h3, or take the carrier's; a build needs nvcc >= 12.9
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
TF_COMMIT="${TF_COMMIT:-1f7ecc75ded5df6da151d4d5eeee87b35161e3c2}"
ZIG_VERSION=0.17.0
OUT="$PREFIX/tf-h3"
if [ -s "$OUT/libtf_h3.so" ] && [ "$(cat "$OUT/commit" 2>/dev/null)" = "$TF_COMMIT" ] && [ -z "${TF_SOURCE:-}" ]; then
  echo "[tensorfold] engine already built at ${TF_COMMIT:0:8}"; exit 0
fi
# the prebuilt engine from the carrier image, when Docker is here and the image holds this commit's build
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-studio:2.8.0}"
if [ -z "${TF_SOURCE:-}" ] && [ "${ENGINE_FROM_SOURCE:-0}" != 1 ] && [ "$(uname -m)" = aarch64 ] && command -v docker >/dev/null 2>&1 \
   && docker info >/dev/null 2>&1 && docker pull -q "$IMAGE" >/dev/null 2>&1; then
  TMP="$PREFIX/tf-h3.fetch"
  rm -rf "$TMP" 2>/dev/null || true
  mkdir -p "$TMP"
  docker run --rm -v "$TMP":/out "$IMAGE" sh -c 'cp /payload/spark-engine/* /out/ 2>/dev/null; chmod a+rw /out/*' || true
  if [ "$(cat "$TMP/commit" 2>/dev/null)" = "$TF_COMMIT" ] && ( cd "$TMP" && sha256sum -c SHA256SUMS >/dev/null 2>&1 ); then
    mkdir -p "$OUT"
    cp "$TMP/libtf_h3.so" "$TMP/commit" "$OUT/"
    rm -rf "$TMP" 2>/dev/null || true
    echo "[tensorfold] engine from $IMAGE (checksums verified) -> $OUT/libtf_h3.so"; exit 0
  fi
  rm -rf "$TMP" 2>/dev/null || true
fi
NVCC="${NVCC:-}"
[ -n "$NVCC" ] || for candidate in /usr/local/cuda/bin/nvcc "$(command -v nvcc || true)"; do
  [ -x "$candidate" ] && { NVCC="$candidate"; break; }
done
[ -n "$NVCC" ] || { echo "no nvcc: install the CUDA toolkit (12.9 or newer) or set NVCC" >&2; exit 1; }
version="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
read -r major minor <<<"$version"
[ "$major" -gt 12 ] || { [ "$major" = 12 ] && [ "$minor" -ge 9 ]; } || { echo "$NVCC is CUDA $major.$minor; the GB10 needs 12.9 or newer" >&2; exit 1; }
case "$(uname -m)" in aarch64) arch=aarch64;; x86_64) arch=x86_64;; *) echo "no Zig build for $(uname -m)" >&2; exit 1;; esac
ZIG="$PREFIX/zig-$ZIG_VERSION/zig"
if [ ! -x "$ZIG" ]; then
  echo "[tensorfold] fetching Zig $ZIG_VERSION"
  mkdir -p "$PREFIX/zig-$ZIG_VERSION"
  curl -sfL "https://ziglang.org/download/$ZIG_VERSION/zig-$arch-linux-$ZIG_VERSION.tar.xz" | tar -xJ --strip-components=1 -C "$PREFIX/zig-$ZIG_VERSION"
fi
SRC="${TF_SOURCE:-$PREFIX/tensorfold-src}"
if [ -z "${TF_SOURCE:-}" ]; then
  [ -d "$SRC/.git" ] || git init -q "$SRC"
  git -C "$SRC" fetch -q --depth 1 https://github.com/drowzeys/TensorFold.git "$TF_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
echo "[tensorfold] building the engine with $NVCC"
# two compile jobs: the kernel image's compiler takes a few GB
(cd "$SRC" && nice -n 19 "$ZIG" build tf-h3 -j2 -Doptimize=ReleaseFast -Dnvcc="$NVCC")
mkdir -p "$OUT"
cp "$SRC/zig-out/lib/libtf_h3.so" "$OUT/libtf_h3.so"
[ -n "${TF_SOURCE:-}" ] && git -C "$SRC" rev-parse HEAD > "$OUT/commit" || echo "$TF_COMMIT" > "$OUT/commit"
echo "[tensorfold] engine -> $OUT/libtf_h3.so"
