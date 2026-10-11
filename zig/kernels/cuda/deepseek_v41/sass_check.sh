#!/bin/bash
# Offline SASS bit-identity check (no GPU): reference extension builds vs our copies, kernel by kernel (sass_check.py).
set -euo pipefail
# usage: OUT=<dir> SM=121 TF_DSV41_REF_SRC=<repo> [TF_DSV41_REF_REV=<ref>] [FATBINS=<dir>] bash sass_check.sh [name ...]
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:?set OUT}"
SM="${SM:-121}"
# each copy's reference source at its git ref (py_sources.py), exported per source key under $OUT/src-<key>
declare -A SRCKEY DONE
mkdir -p "$OUT"
while read -r copy key repo ref; do
  SRCKEY[${copy%.*}]=$key
  if [ -z "${DONE[$key]:-}" ]; then            # fresh each run: a ref may have moved
    rm -rf "$OUT/src-$key" && mkdir -p "$OUT/src-$key"
    git -C "$repo" archive "$ref" src/tensorfold | tar -x -C "$OUT/src-$key"
    DONE[$key]=1
  fi
done < <(python3 "$HERE/py_sources.py" --sources)
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
X=cuda/exl3
F=families/deepseek_v41/cuda
# name|reference sources (under src/tensorfold)|extra_cuda_cflags (comma separated)
TABLE="exl3_experts|$X/experts.cu $X/experts_cb2.cu|-O3,-lineinfo
x3ld|$F/x3ld.cu|-O3,-lineinfo
x3gm|$F/x3gm.cu|-O3,-lineinfo
linear|$X/linear.cu|-O3,--expt-relaxed-constexpr
attn_cuda|$F/csa2/attn_cuda.cu|-O3,-lineinfo,--fmad=false
topk_cuda|$F/csa2/topk_cuda.cu|-O3,-lineinfo,--fmad=false
mhc_cuda|$F/mhc_cuda.cu|-O3,-lineinfo,--fmad=false
router_gemv|$F/router_gemv.cu|-O3,-lineinfo"
names=("$@")
[ ${#names[@]} -eq 0 ] && names=($(echo "$TABLE" | cut -d'|' -f1))
mkdir -p "$OUT/py" "$OUT/zig"
script='set -e
TI=$(python -c "import torch.utils.cpp_extension as c; print(\" \".join(\"-I\"+p for p in c.include_paths()))" 2>/dev/null)
PY=$(python -c "import sysconfig;print(sysconfig.get_paths()[\"include\"])")
TORCH="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20"
GEN="-gencode=arch=compute_${SM},code=sm_${SM}"
build() {  # name, sources, flags, the Python tree
  local name=$1 srcs=$2 flags=${3//,/ } t=$4
  : > /out/py/$name.sass
  for s in $srcs; do
    nvcc -c $TI -I$PY -I$t/cuda/exl3 -I$t/families/deepseek_v41/cuda $TORCH -DTORCH_API_INCLUDE_EXTENSION_H \
      -DTORCH_EXTENSION_NAME=$name -D_GLIBCXX_USE_CXX11_ABI=1 --compiler-options -fPIC $flags $GEN \
      -o /out/py/$name.$(basename $s).o $t/$s
    cuobjdump -sass /out/py/$name.$(basename $s).o >> /out/py/$name.sass
    rm -f /out/py/$name.$(basename $s).o
  done
  nvcc -fatbin $TORCH $flags $GEN -Xptxas -v -o /out/zig/$name.fatbin /k/$name.cu 2> /out/zig/$name.ptxas
  cuobjdump -sass /out/zig/$name.fatbin > /out/zig/$name.sass
  rm -f /out/zig/$name.fatbin
  echo built $name
}
'
for n in "${names[@]}"; do
  line=$(echo "$TABLE" | grep "^$n|" || true)
  [ -n "$line" ] || { echo "sass_check.sh: no copy named $n" >&2; exit 2; }
  srcs=$(echo "$line" | cut -d'|' -f2); flags=$(echo "$line" | cut -d'|' -f3)
  key=${SRCKEY[$n]:-base}
  [ "$n" = topk_cuda ] && key=${SRCKEY[attn_cuda]:-base}   # one extension, one source
  script="$script
build $n \"$srcs\" \"$flags\" /src/src-$key/src/tensorfold &
while [ \$(jobs -rp | wc -l) -ge 4 ]; do wait -n; done"
done
script="$script
wait"
docker run --rm --network none --memory "${TF_ZIG_DOCKER_MEMORY:-16g}" --memory-swap "${TF_ZIG_DOCKER_MEMORY:-16g}" -u "$(id -u):$(id -g)" -e SM="$SM" -e HOME=/tmp \
  -v "$OUT:/src:ro" -v "$HERE:/k:ro" -v "$OUT:/out" --entrypoint bash "$IMAGE" -c "$script" 2>&1 | grep -v "No CUDA runtime" || true
fat() { case $1 in attn_cuda) echo attn;; topk_cuda) echo topk;; mhc_cuda) echo mhc;; *) echo "$1";; esac; }
if [ -n "${FATBINS:-}" ]; then
  mkdir -p "$OUT/fat"
  dump=""
  for n in "${names[@]}"; do dump="$dump cuobjdump -sass /f/dsv41_$(fat "$n").fatbin > /out/fat/$n.sass;"; done
  docker run --rm --network none --memory "${TF_ZIG_DOCKER_MEMORY:-16g}" --memory-swap "${TF_ZIG_DOCKER_MEMORY:-16g}" \
    -u "$(id -u):$(id -g)" -v "$FATBINS:/f:ro" -v "$OUT:/out" --entrypoint bash "$IMAGE" \
    -c "set -e; $dump" 2>&1 | grep -v "No CUDA runtime" | grep -iE "error|fatal" || true
fi
rc=0
for n in "${names[@]}"; do
  echo "== $n"
  python3 "$HERE/sass_check.py" "$OUT/py/$n.sass" "$OUT/zig/$n.sass" --sm "$SM" | tail -4 || rc=1
  if [ -n "${FATBINS:-}" ]; then
    echo -n "built fatbin: "
    python3 "$HERE/sass_check.py" "$OUT/py/$n.sass" "$OUT/fat/$n.sass" --sm "$SM" | tail -1 || rc=1
  fi
  spill=$(grep -E "bytes spill" "$OUT/zig/$n.ptxas" | grep -vc " 0 bytes spill stores, 0 bytes spill loads" || true)
  echo "$n: $(grep -c 'Compiling entry' "$OUT/zig/$n.ptxas") entry functions, $spill with spills (sm_$SM)"
done
exit $rc
