#!/bin/bash
# img_session.sh BIN TAG CLIENT [client args]: one --vision server, one client run over $IMG, the server killed.
# Env: CKPT (checkpoint dir), IMG (the images), WORK (output dir), TENSORFOLD_LIBTURBOJPEG (the pinned libjpeg-turbo).
# Run it under `flock /tmp/tf-gpu.lock`; the client is read from this script's directory.
BIN=$1; TAG=$2; CLIENT=$3; shift 3
"$BIN" serve "$CKPT" --host 127.0.0.1 --port 8123 --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking --vision > "$WORK/$TAG.serve.txt" 2>&1 &
S=$!
for i in $(seq 300); do curl -s localhost:8123/v1/models >/dev/null && break; sleep 2; done
python3 "$(dirname "$0")/$CLIENT" http://127.0.0.1:8123/v1 m "$IMG" "$WORK/$TAG.json" "$@" > "$WORK/$TAG.log" 2>&1
kill $S; wait $S 2>/dev/null
