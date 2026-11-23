#!/bin/bash
# Images: CKPT, BIN (tensorfold-native), WORK (holds img/), PY (oracle python), JPEG (libturbojpeg.0.dylib, the pinned libjpeg-turbo)
export CKPT BIN WORK PY JPEG
export TENSORFOLD_LIBTURBOJPEG="$JPEG"
flock /tmp/tf-gpu.lock bash -c '
"$BIN" serve "$CKPT" --host 127.0.0.1 --port 8123 --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking --vision > "$WORK/img.native.serve.log" 2>&1 &
S=$!
for i in $(seq 300); do curl -s localhost:8123/v1/models >/dev/null && break; sleep 2; done
"$PY" img_client.py http://127.0.0.1:8123/v1 m "$WORK/img" "$WORK/img.native.json" > "$WORK/img.native.log" 2>&1
kill $S; wait $S
"$PY" -m mlx_lm.server --model "$CKPT" --host 127.0.0.1 --port 8124 > "$WORK/img.mlx.serve.log" 2>&1 &
S=$!
for i in $(seq 300); do curl -s localhost:8124/v1/models >/dev/null && break; sleep 2; done
"$PY" img_client.py http://127.0.0.1:8124/v1 default_model "$WORK/img" "$WORK/img.mlx.json" > "$WORK/img.mlx.log" 2>&1
kill $S; wait $S'
