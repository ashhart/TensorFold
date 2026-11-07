#!/bin/bash
# Fetch the essential LoRA bundle from its authors' pages and install it into the hub (Mac or Spark). Options: README.
set -uo pipefail
# Exit 0: all present; 1: some unavailable (the rest installed); 2: usage or disk. Each LoRA keeps its author's licence.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
INSTALL=1; ARGS=(); DEST=""; CHECK=0; GROUPS_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-install) INSTALL=0;;
    --dest) DEST="$2"; ARGS+=("$1" "$2"); shift;; --dest=*) DEST="${1#--dest=}"; ARGS+=("$1");;
    --categories) GROUPS_ARG="$2"; ARGS+=("$1" "$2"); shift;; --categories=*) GROUPS_ARG="${1#--categories=}"; ARGS+=("$1");;
    --verify|--dry-run) CHECK=1; ARGS+=("$1");;
    *) ARGS+=("$1");;
  esac; shift
done
# One package per platform holds the 14 of 22 LoRAs whose licences allow passing on; LORA_PACKAGE=none|<file or URL>.
PKG_BASE="https://huggingface.co/drowzeys/keys-studio-lora-essential/resolve/main"
PKG_FILE="keys-studio-lora-essential-$([ "$(uname -s)" = Darwin ] && echo mac || echo spark)-2026-10-13-r3.tar"
case " ${ARGS[*]-} " in *" --package"*|*" --source"*|*" --make-package"*|*" --verify "*|*" --dry-run "*) ;; *)
  if [ "${LORA_PACKAGE:-}" != none ] && [ -z "${LORA_BUNDLE:-}" ] && [ ! -e "${DEST:-${STUDIO_HOME:-$HOME/TensorFoldStudio}/lora-bundle}/package.json" ]; then
    ARGS+=(--package "${LORA_PACKAGE:-$PKG_BASE/$PKG_FILE}")
  fi;; esac
PY="$PREFIX/venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)" || { echo "python3 is needed" >&2; exit 2; }
"$PY" "$HERE/scripts/lora-bundle.py" ${ARGS[@]+"${ARGS[@]}"}; RC=$?
[ $RC -le 1 ] || exit $RC
if [ "$INSTALL" = 1 ] && [ "$CHECK" = 0 ]; then
  DEST="${DEST:-${STUDIO_HOME:-$HOME/TensorFoldStudio}/lora-bundle}"
  if [ -x "$PREFIX/venv/bin/python" ]; then
    # into the hub from the bundle only: nothing here goes to the network
    LORA_BUNDLE="$DEST" LORA_OFFLINE=1 H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}" "$PREFIX/venv/bin/python" "$HERE/app/loras.py" install "${GROUPS_ARG:-all}" >&2 || true
  else
    echo "the Studio is not installed at $PREFIX yet: the bundle is ready at $DEST, and the hub installs from it at its first start" >&2
  fi
fi
exit $RC
