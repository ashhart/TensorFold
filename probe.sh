#!/bin/bash
# probe.sh: why do these compiles miss on another runner? Fixed zig path or not, and which inputs are missing.
set -u
bash tools/zig/zig_cache_contents.sh
real=$(dirname "$(command -v zig)")
echo "PROBE zig at $real"
cp -R "$real" "$HOME/zig-fixed"
grep -m1 'model name' /proc/cpuinfo | sed 's/^/PROBE cpu /'
uname -r | sed 's/^/PROBE kernel /'
PY=$(command -v python3 || command -v python)
[ -d .zig-cache/h ] && "$PY" probe_manifests.py "$real/lib"
STEPS="test-window-fallback-host test-qwen35-page-runs-host test-qwen35-whole-host test-runtime-libraries-host"
go() {
  local name=$1 dir=$2 s=$(date +%s)
  PATH="$dir:$PATH"
  echo "PROBE $name uses $(command -v zig)"
  zig build -Dfixed-cpu -Dname-tests $STEPS --summary all 2>&1 | grep -E 'error:|Build Summary|CC |compile lib' | sed -E 's/MaxRSS:[^ ]+//; s/^[ |+`-]+//' | cut -c1-84 | sort | uniq -c | sort -rn | head -24
  echo "PROBE $name $(( $(date +%s) - s )) s"
}
go fixed "$HOME/zig-fixed"
exit 0
