#!/bin/bash
# usage: tools/zig/xpu_guard.sh SECONDS COMMAND [ARGS...]; one GPU program at a time under flock /tmp/b70.lock
limit_mb=${XPU_OTHERS_MB:-8192}
secs=${1:?seconds}; shift
vram_others_mb() {
  local kb=0 f v
  for f in /proc/[0-9]*/fdinfo/[0-9]*; do
    case "$f" in "/proc/$$/"*|"/proc/self/"*) continue;; esac
    v=$(awk '/^drm-driver:/{xe=($2=="xe")} /^drm-total-vram0:/{if(xe) s+=$2} END{print s+0}' "$f" 2>/dev/null) || continue
    kb=$((kb + ${v:-0}))
  done
  echo $((kb / 1024))
}
# waits while other processes hold over 8 GB VRAM (xe fdinfo); stops the command only with SIGINT after SECONDS
exec 9>/tmp/b70.lock
flock 9
while [ "$(vram_others_mb)" -gt "$limit_mb" ]; do
  echo "waiting: other processes hold $(vram_others_mb) MB of VRAM (limit $limit_mb)" >&2
  sleep 20
done
timeout -s INT -k 0 "$secs" "$@"
