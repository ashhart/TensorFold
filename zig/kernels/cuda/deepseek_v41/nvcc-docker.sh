#!/bin/bash
# nvcc from the NGC PyTorch image for hosts without CUDA (`zig build -Dnvcc=<this script>`); paths mount 1:1.
set -euo pipefail
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# the container's memory and memory+swap cap: the Docker daemon runs it outside the caller's own limits
MEM="${TF_ZIG_DOCKER_MEMORY:-8g}"
# the mount of a directory: its repository's root (a source's headers sit beside it), else the directory (a cache)
root() { git -C "$1" rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "$1"; }
roots=("$PWD" "$(root "$PWD")")
for a in "$@"; do
  [ "${a:0:1}" = / ] || continue
  d="$(dirname "$a")"
  mkdir -p "$d"
  roots+=("$(root "$d")")
done
mapfile -t roots < <(printf '%s\n' "${roots[@]}" | sort -u)
mounts=()
for r in "${roots[@]}"; do mounts+=(-v "$r:$r"); done
docker run --rm -i --network none --memory "$MEM" --memory-swap "$MEM" -u "$(id -u):$(id -g)" "${mounts[@]}" -w "$PWD" \
  --entrypoint /usr/local/cuda/bin/nvcc "$IMAGE" "$@"
# depfilter FILE CWD ROOT...: keep the -MF file's dependencies under a mounted root (image headers drop out)
depfilter() {
  local path=$1 cwd=$2 text target d full keep=() out=() reals k deps=()
  shift 2
  for r in "$@"; do keep+=("$(realpath -m -- "$r")/"); done
  text=$(sed -e ':a' -e '/\\$/{N;s/\\\n/ /;ba' -e '}' "$path")
  target=${text%%:*}
  read -r -d "" -a deps <<< "${text#*:}" || true
  [ "${#deps[@]}" -gt 0 ] && for d in "${deps[@]}"; do
    if [ "${d:0:1}" = / ]; then full=$d; else full=$cwd/$d; fi
    reals=$(realpath -m -- "$full")
    for k in "${keep[@]}"; do [[ "$reals" == "$k"* ]] && { out+=("$d"); break; }; done
  done
  { printf '%s: ' "$target"; local first=1; for d in "${out[@]}"; do [ $first = 1 ] && printf '%s' "$d" || printf ' \\\n  %s' "$d"; first=0; done; printf '\n'; } > "$path"
}
dep=""
prev=""
for a in "$@"; do
  [ "$prev" = "-MF" ] && dep="$a"
  prev="$a"
done
if [ -n "$dep" ] && [ -f "$dep" ]; then
  depfilter "$dep" "$PWD" "${roots[@]}"
fi
