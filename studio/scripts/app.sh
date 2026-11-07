#!/bin/bash
# The Studio web app on http://127.0.0.1:7870 (HOST=0.0.0.0: LAN, no login). Env: PREFIX, HOST, PORT, STUDIO_HOME.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
export PATH="/opt/homebrew/bin:$PATH"
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup.sh first: no environment at $PREFIX" >&2; exit 1; }
echo "Wasserman Slate AI Studio (powered by TensorFold) on http://${HOST:-127.0.0.1}:${PORT:-7870}"
# the app starts without ffmpeg and says what it then cannot do (long videos, export, frames, reference clips)
for TOOL in ffmpeg ffprobe; do
  command -v "$TOOL" >/dev/null || echo "  ! $TOOL is not installed on this machine: $([ "$(uname -s)" = Linux ] && echo "sudo apt install ffmpeg" || echo "brew install ffmpeg")" >&2
done
exec "$PREFIX/venv/bin/python" "$HERE/app/server.py"
