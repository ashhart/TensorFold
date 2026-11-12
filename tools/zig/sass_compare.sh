#!/usr/bin/env bash
set -euo pipefail
sass_tool_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "$sass_tool_dir/cuda_sass_compare.py" "$@"
