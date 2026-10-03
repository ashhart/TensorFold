#!/bin/sh
# Read-only qualification by default. Native launchd smoke requires an explicit second argument.
set -eu
PYTHON=${1:-python3}
"$PYTHON" -m compileall -q src/tensorfold/control
"$PYTHON" -m pytest tests/control -q -m 'not macos' --junitxml=control-results.xml
umask 077
tmp=$(mktemp -d "${TMPDIR:-/tmp}/tensorfold-control.XXXXXX")
trap 'rm -f "$tmp/preview.plist"; rmdir "$tmp"' EXIT
"$PYTHON" -m tensorfold.control service install Example/Cached-Model --dry-run > "$tmp/preview.plist"
if [ "$(uname -s)" = Darwin ]; then
  /usr/bin/plutil -lint "$tmp/preview.plist"
fi
if [ "${2:-}" = --native-launchd ]; then
  [ "$(uname -s)" = Darwin ] || { echo 'Native launchd check requires macOS' >&2; exit 1; }
  TENSORFOLD_TEST_LAUNCHD=1 "$PYTHON" -m pytest tests/control/test_macos_smoke.py -v
fi
