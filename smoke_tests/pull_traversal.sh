#!/bin/sh
# F2 (security): a hostile repo's file tree can write outside the cache.
# hub's tree API returns paths the client trusts blindly:
#   - pull_file.zig linkIntoSnapshot: symlink at snapshot_dir/<path> with path
#     "../../../../../pwned.txt" escapes the cache root entirely
#   - pull_file.zig blobName's fallback also joins the raw path onto blobs/
# Expected for a fixed client: the pull fails (or sanitizes), and no file
# appears outside the cache dir. Today it succeeds and escapes.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/../zig-out/bin/tensorfold}
PORT=${PORT:-8931}
# The cache sits nested so ../../../../../pwned.txt escapes it but stays in /tmp.
TMP=$(mktemp -d /tmp/tf-traverse-XXXX)
mkdir -p "$TMP/a/b/cache"
CACHE=$TMP/a/b/cache
ESCAPED=$TMP/a/pwned.txt

python3 "$HERE/malicious_hub.py" $PORT &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 0.5
kill -0 $SRV 2>/dev/null || { echo "FAIL: mock hub did not start (port $PORT in use?); set PORT=<free>"; exit 1; }

HF_ENDPOINT="http://127.0.0.1:$PORT" HF_HUB_CACHE="$CACHE" "$BIN" pull Localhost/evil@main >/dev/null 2>&1 || true
kill $SRV 2>/dev/null || true

if [ -e "$ESCAPED" ] || [ -L "$ESCAPED" ]; then
    echo "FAIL: the hostile repo wrote outside the cache:"
    ls -la "$ESCAPED" | sed 's/^/    /'
    echo "    (cache was $CACHE)"
    exit 1
fi
# Also show the pull refused the hostile tree (post-fix: error.SnapshotPath).
HF_ENDPOINT="http://127.0.0.1:$PORT" HF_HUB_CACHE="$CACHE" "$BIN" pull Localhost/evil@main 2>&1 | tail -1 | sed 's/^/    pull: /'
echo "PASS: nothing escaped the cache ($CACHE)"
rm -rf "$TMP"
