#!/bin/bash
# Teacher-forced fidelity, native side: CKPT (checkpoint path), BIN (tensorfold-native), WORK (holds prompts.json, mlx.json), PY (oracle python)
export CKPT BIN WORK PY
S_ARGS="--host 127.0.0.1 --port 8123 --context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking"
flock /tmp/tf-gpu.lock bash -c '
"$BIN" serve "$CKPT" '"$S_ARGS"' > "$WORK/native.serve.log" 2>&1 &
S=$!
for i in $(seq 300); do curl -s localhost:8123/v1/models >/dev/null && break; sleep 2; done
"$PY" fid_native.py http://127.0.0.1:8123/v1 "$CKPT" "$WORK/prompts.json" "$WORK/mlx.json" "$WORK/native.json" > "$WORK/native.log" 2>&1
kill $S; wait $S'
