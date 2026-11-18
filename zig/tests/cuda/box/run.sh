#!/bin/bash
# One run of the Zig CUDA runtime checks on a CUDA machine: oracle in the PyTorch image, zig build with the image's nvcc, GPU tests. Usage: TF_ZIG_ROOT=<work dir holding src/, toolchain/, cache/, runs/> [TF_ZIG_JOURNAL=<file>] flock <GPU lock> bash -u run.sh RUN_ID
set -u
RUN="${1:?run id}"
TF="${TF_ZIG_ROOT:?set TF_ZIG_ROOT}"
OUT="${TF:?}/runs/${RUN:?}"
SRC="${TF:?}/src"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
JOURNAL="${TF_ZIG_JOURNAL:-/dev/null}"
WHO="${TF_ZIG_WHO:-zig-cuda}"
TREE=$(cat "${SRC:?}/TREE" 2>/dev/null || echo unknown)
mkdir -p "${OUT:?}"
rc=0

journal() { printf '{"utc": "%s", "event": "%s", "who": "%s", "run": "zig-%s", "tree": "%s"%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$WHO" "$RUN" "$TREE" "${2:-}" >> "$JOURNAL"; }
step() { echo "=== $(date -u +%H:%M:%S) $*"; }
container() {  # name, then docker run arguments; the container is ours by name and label, removed on exit
  local name="tf-zig-${RUN}-$1"; shift
  timeout 1800 docker run --rm --name "$name" --label "tensorfold.zig=${RUN}" "$@"
}
cleanup() { for c in $(docker ps -q --filter "label=tensorfold.zig=${RUN}"); do docker stop --time 10 "$c" >/dev/null; done; }
trap 'cleanup; journal END ", \"rc\": 143"; exit 143' TERM INT

journal START ", \"cmd\": \"zig/tests/cuda/box/run.sh\""
step "kernel copies match their Python sources"
python3 -B "${SRC:?}/zig/tests/cuda/copies.py" || rc=1
step "preflight"
docker ps --format '{{.Names}}' ; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
grep -E 'MemAvailable' /proc/meminfo

step "oracle: Python engine ops in ${IMAGE}"
rm -rf "${OUT:?}/torch_ext" "${OUT:?}/triton" "${OUT:?}/fixtures"
container oracle --gpus all --ipc host --network none --memory 32g --memory-swap 32g --read-only \
  --tmpfs /tmp:size=4g -e HOME=/tmp -v "${SRC:?}:/tensorfold:ro" -v "${OUT:?}:/out" \
  -e PYTHONPATH=/tensorfold/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e TORCH_EXTENSIONS_DIR=/out/torch_ext -e TRITON_CACHE_DIR=/out/triton -e CUDA_CACHE_PATH=/out/cuda_cache \
  -e TORCH_CUDA_ARCH_LIST=12.1 -w /tensorfold "$IMAGE" python -B /tensorfold/zig/tests/cuda/oracle/oracle.py /out \
  > "${OUT:?}/oracle.log" 2>&1 || { rc=1; echo "oracle failed"; tail -30 "${OUT:?}/oracle.log"; }

step "zig build with the image's nvcc"
container build --network none --memory 16g -v "${TF:?}:${TF:?}" -w "${SRC:?}" \
  -e ZIG_LOCAL_CACHE_DIR="${TF:?}/cache/zig-local" -e ZIG_GLOBAL_CACHE_DIR="${TF:?}/cache/zig-global" \
  --entrypoint "${TF:?}/toolchain/zig" "$IMAGE" build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=safe \
  --prefix "${OUT:?}/zig-out" -j8 fatbins install > "${OUT:?}/build.log" 2>&1 || { rc=1; echo "build failed"; tail -40 "${OUT:?}/build.log"; }

step "SASS: the extension's against our fatbins"
container sass --network none -v "${OUT:?}:/out" -v "${SRC:?}:/tensorfold:ro" --entrypoint bash "$IMAGE" -c '
  so=$(ls /out/torch_ext/tensorfold_gdn_v2/tensorfold_gdn_v2*.so | head -1)
  cuobjdump -sass "$so" > /out/sass-python-gdn.txt
  cuobjdump -sass /out/zig-out/fatbin/gdn.fatbin > /out/sass-zig-gdn.txt
  cuobjdump -symbols /out/zig-out/fatbin/gdn.fatbin > /out/symbols-zig-gdn.txt
  cuobjdump -symbols /out/zig-out/fatbin/probe.fatbin > /out/symbols-zig-probe.txt
  python -B /tensorfold/zig/tests/cuda/oracle/sass_compare.py /out/sass-python-gdn.txt /out/sass-zig-gdn.txt' \
  > "${OUT:?}/sass.log" 2>&1 || rc=1
cat "${OUT:?}/sass.log"

T="${OUT:?}/zig-out/bin/tf-cuda-test"
F="${OUT:?}/fixtures"
step "GPU tests on the host (no Python, no PyTorch)"
for t in "info" "smoke" "graph" "launch-ex" "ptx" "symbols" "overhead 1000 20" "overhead 10000 10" \
         "overhead-pdl 1000 20" "overhead-pdl 10000 10" "cublaslt" \
         "gdn-replay $F/gdn_replay" "gdn-tree $F/gdn_chain" "gdn-tree $F/gdn_tree" \
         "triton $F/triton_swiglu" "triton $F/triton_add_rmsnorm"; do
  echo "--- $t"
  # shellcheck disable=SC2086
  timeout 600 "$T" $t || { rc=1; echo "rc=$? for $t"; }
done 2>&1 | tee "${OUT:?}/tests.log"
grep -q '^FAIL\|^rc=' "${OUT:?}/tests.log" && rc=1

step "NCCL in the image (the host has no libnccl)"
container nccl --gpus all --network none -e NCCL_DEBUG=WARN -v "${OUT:?}:/out:ro" \
  --entrypoint /out/zig-out/bin/tf-cuda-test "$IMAGE" nccl > "${OUT:?}/nccl.log" 2>&1 || rc=1
cat "${OUT:?}/nccl.log"

cleanup
step "done rc=$rc"
journal END ", \"rc\": $rc"
exit $rc
