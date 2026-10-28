#!/bin/bash
# One image on a DGX Spark (CUDA): scripts/image.sh hands over here on Linux with the same interface (env: README).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup-spark.sh first: no environment at $PREFIX" >&2; exit 1; }
PROMPT="${1:-}"; OUT="${2:-outputs/image.png}"
ARGS=(image -o "$OUT" --width "${WIDTH:-1344}" --height "${HEIGHT:-768}" --seed "${SEED:-0}")
[ "${TURBO:-1}" = 1 ] || ARGS+=(--base --steps "${STEPS:-40}")
[ "${UPSAMPLE:-1}" = 1 ] || ARGS+=(--no-upsample)
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
# shellcheck disable=SC2086
PREFIX="$PREFIX" exec "$PREFIX/venv/bin/python" "$HERE/spark_generate.py" "${ARGS[@]}" ${EXTRA:-}
