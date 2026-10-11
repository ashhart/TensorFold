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
dep=""
prev=""
for a in "$@"; do
  [ "$prev" = "-MF" ] && dep="$a"
  prev="$a"
done
if [ -n "$dep" ] && [ -f "$dep" ]; then
  python3 - "$dep" "$PWD" "${roots[@]}" <<'EOF'
import os, sys
path, cwd, keep = sys.argv[1], sys.argv[2], [os.path.join(os.path.realpath(r), "") for r in sys.argv[3:]]
text = open(path).read().replace("\\\n", " ")
target, _, deps = text.partition(":")
out = []
for d in deps.split():
    full = d if os.path.isabs(d) else os.path.join(cwd, d)
    # the image's own headers are not inputs on the host; every mounted root's file is
    if any(os.path.realpath(full).startswith(k) for k in keep):
        out.append(d)
open(path, "w").write(target + ": " + " \\\n  ".join(out) + "\n")
EOF
fi
