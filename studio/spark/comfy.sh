#!/bin/bash
# The ComfyUI server behind the Studio on a DGX Spark. Usage: bash spark/comfy.sh start|stop|status (127.0.0.1:8190)
set -euo pipefail
# The memory guard always runs beside it: GPU and system share one pool, and running out can leave a Spark unreachable.
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
PORT="${SPARK_COMFY_PORT:-8190}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATCH="[m]ain.py --listen 127.0.0.1 --port $PORT"
case "${1:-start}" in
  stop) pkill -f "$MATCH" 2>/dev/null && echo "stopped" || echo "not running"; rm -f "$PREFIX/comfy.kind";;
  status) pgrep -f "$MATCH" >/dev/null && echo "running on 127.0.0.1:$PORT" || echo "not running";;
  run)
    export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" MAX_JOBS="${MAX_JOBS:-2}"
    # TensorFold's H3 engine as a ComfyUI node, when it has been built (spark/build-engine.sh)
    NODE="$PREFIX/ComfyUI/custom_nodes/tensorfold_h3"
    rm -rf "$NODE"
    if [ -s "$PREFIX/tf-h3/libtf_h3.so" ]; then
      mkdir -p "$NODE" && cp "$HERE/tensorfold_h3/__init__.py" "$NODE/"
      export TF_H3_LIB="$PREFIX/tf-h3/libtf_h3.so"
    fi
    # --use-ck-attention: ComfyUI's own flash-attention kernel, faster a pass than PyTorch's on a GB10
    cd "$PREFIX/ComfyUI"
    # shellcheck disable=SC2086
    nice -n 19 "$PREFIX/venv/bin/python" main.py --listen 127.0.0.1 --port "$PORT" ${SPARK_COMFY_ARGS:---use-ck-attention} &
    pid=$!
    trap 'kill $pid 2>/dev/null' TERM INT
    # the guard watches the server until it ends (3: it stopped it); python3 fallback, else a plain loop holds the floor
    GUARD=0
    for PY in "$PREFIX/venv/bin/python" python3; do
      RC=0; "$PY" "$HERE/memory_guard.py" watch --pid "$pid" || RC=$?
      if [ "$RC" = 0 ] || [ "$RC" = 3 ]; then GUARD=1; break; fi
    done
    if [ "$GUARD" = 0 ]; then
      FLOOR="${SPARK_MEMORY_FLOOR_MB:-12000}"
      echo "[tensorfold] memory floor: the guard could not start; holding ${FLOOR} MB with a plain loop"
      while kill -0 $pid 2>/dev/null; do
        read -r _ free _ < <(grep MemAvailable /proc/meminfo)
        if [ "$((free / 1024))" -lt "$FLOOR" ]; then echo "[tensorfold] memory floor: $((free / 1024)) MB available, under the floor of ${FLOOR} MB: stopping ComfyUI"; kill -9 $pid; fi
        sleep 0.2
      done
    fi
    wait $pid 2>/dev/null || true;;
  start)
    pgrep -f "$MATCH" >/dev/null && { echo "already running"; exit 0; }
    [ -x "$PREFIX/venv/bin/python" ] && [ -f "$PREFIX/ComfyUI/main.py" ] || { echo "run oneshot-setup-spark.sh first: nothing at $PREFIX" >&2; exit 1; }
    rm -f "$PREFIX/comfy.kind"
    nohup setsid bash "$0" run > "$PREFIX/comfy.log" 2>&1 < /dev/null &
    echo "ComfyUI starting on 127.0.0.1:$PORT (log: $PREFIX/comfy.log)";;
  *) echo "usage: comfy.sh start|stop|status" >&2; exit 2;;
esac
