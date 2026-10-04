#!/bin/bash
# Builds the Zig CUDA engine with the PyTorch image's nvcc, then proves each fatbin's SASS against the Python extension's. Usage: TF_NEMO=<work dir: src/ out/ aot/> TF_ZIG=<zig binary> bash -u build.sh RUN CAPTURE_RUN
set -u
RUN="${1:?run id}"
CAP="${2:?capture run whose torch_ext holds the Python builds}"
TF="${TF_NEMO:?set TF_NEMO}"
ZIG="${TF_ZIG:?set TF_ZIG}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
OUT="${TF:?}/out/${RUN:?}"
SRC="${TF:?}/src"
EXT="${TF:?}/aot/${CAP:?}/torch_ext"
mkdir -p "${OUT:?}" "${TF:?}/cache/zig-local" "${TF:?}/cache/zig-global"
rc=0
python3 -B "${SRC:?}/zig/tests/cuda/copies.py" || rc=1
timeout 1800 docker run --rm --name "tf-zig-nemo-build-${RUN}" --network none --memory 16g -v "${TF:?}:${TF:?}" \
  -v "$(dirname "${ZIG:?}"):/zigtool:ro" -w "${SRC:?}" \
  -e ZIG_LOCAL_CACHE_DIR="${TF:?}/cache/zig-local" -e ZIG_GLOBAL_CACHE_DIR="${TF:?}/cache/zig-global" \
  --entrypoint /zigtool/zig "$IMAGE" build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=safe \
  --prefix "${OUT:?}/zig-out" -j8 ${TF_BUILD_STEPS:-fatbins install} > "${OUT:?}/build.log" 2>&1 || { rc=1; tail -60 "${OUT:?}/build.log"; }
timeout 600 docker run --rm --network none -v "${OUT:?}:/out" -v "${EXT:?}:/ext:ro" -v "${SRC:?}:/tensorfold:ro" \
  --entrypoint bash "$IMAGE" -c '
  set -u; bad=0
  for pair in qmm_group:tensorfold_qmm_v5/qmm_group qmm_prefill:tensorfold_qmm_v5/qmm_prefill \
              experts:tensorfold_experts_v7/experts experts_prefill:tensorfold_experts_v7/experts_prefill \
              experts_pack:tensorfold_experts_v7/experts_pack prefill_attention:tensorfold_prefill_attention_v1/prefill_attention \
              scan_rows:tensorfold_nemotron_scan_rows/scan_rows; do
    zig=${pair%%:*}; py=${pair#*:}
    [ -f /out/zig-out/fatbin/$zig.fatbin ] || { echo "MISSING $zig"; bad=1; continue; }
    cuobjdump -sass /ext/$py.cuda.o > /out/sass-python-$zig.txt
    cuobjdump -sass /out/zig-out/fatbin/$zig.fatbin > /out/sass-zig-$zig.txt
    cuobjdump -symbols /out/zig-out/fatbin/$zig.fatbin > /out/symbols-$zig.txt
    echo "--- $zig"; python -B /tensorfold/zig/tests/cuda/nemotron/sass_compare.py /out/sass-python-$zig.txt /out/sass-zig-$zig.txt || bad=1
  done
  cuobjdump -symbols /out/zig-out/fatbin/nemotron_ops.fatbin > /out/symbols-nemotron_ops.txt
  exit $bad' > "${OUT:?}/sass.log" 2>&1 || rc=1
cat "${OUT:?}/sass.log"
grep -h 'STT_FUNC.*STO_ENTRY\|STO_ENTRY' "${OUT:?}"/symbols-*.txt | awk '{print $NF}' | sort -u > "${OUT:?}/entries.txt"
echo "build rc=$rc"
exit $rc
