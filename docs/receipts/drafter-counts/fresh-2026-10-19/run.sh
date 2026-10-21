#!/bin/bash
# #1001 fresh gfx1151 run, step 2. Stops llm-server, runs R1-R6 with the binary prep.sh built,
# and ALWAYS restarts llm-server (EXIT trap), then checks it answers a chat request.
#   bash ~/tf-1001-fresh-runs/run.sh > ~/tf-1001-fresh-runs/run.log 2>&1
set -u
R=$HOME/tf-1001-fresh-runs; D=$R/out; mkdir -p $D
B=$HOME/tf-1001-fresh/zig-out/native-hip/bin/tensorfold-native
M27=TensorFold/Qwen3.8-27B-MLX-4bit
M35=TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP
gtt() { echo "gtt_used=$(cat /sys/class/drm/card*/device/mem_info_gtt_used | head -1) busy=$(cat /sys/class/drm/card*/device/gpu_busy_percent | head -1) t=$(date -u +%T)"; }
restart_llm() {
  for f in $R/serve-*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  sudo -n systemctl start llm-server.service
  echo "llm-server: $(systemctl is-active llm-server) t=$(date -u +%T)"
  for i in $(seq 1 60); do
    out=$(curl -sS -m 60 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $(cat $HOME/.llm-api-key 2>/dev/null)" \
      -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say ok."}],"max_tokens":8}' 2>&1)
    if echo "$out" | grep -q '"choices"'; then echo "llm-server chat OK after ${i}0s t=$(date -u +%T)"; echo LLMBACK; return; fi
    sleep 10
  done
  echo "llm-server chat check FAILED: $out" | cut -c1-300; echo LLMFAILED
}
[ "$(sha256sum $B | cut -d' ' -f1)" = "$(cut -d' ' -f1 $R/binary.sha256)" ] || { echo "binary changed since prep"; exit 1; }
cat $R/heads.txt; df -h / | tail -1; free -g | sed -n 2p
env | grep -E '^(HIP|CUDA|ROCR)_VISIBLE_DEVICES=' || echo "no *_VISIBLE_DEVICES set"
trap restart_llm EXIT
sudo -n systemctl stop llm-server.service || exit 1
echo "llm-server: $(systemctl is-active llm-server)"; gtt
export HF_HUB_OFFLINE=1
# R1: 27B, default drafting (copy drafts; no MTP head at 70ae7fac)
gtt; timeout 1200 $B benchmark $M27 --output $D/27b-auto.json > $D/27b-auto.log 2>&1; echo "R1 exit=$?"; gtt
# R2a/R2b: --drafter refused before any weights load
timeout 300 $B benchmark $M27 --drafter z-lab/Qwen3.8-27B-DFlash2 --output $D/27b-dflash2.json > $D/27b-dflash2.log 2>&1; echo "R2a exit=$?"; ls $D/27b-dflash2.json 2>&1
gtt; timeout 180 $B serve $M27 --backend rocm --host 127.0.0.1 --port 8093 --drafter z-lab/Qwen3.8-27B-DFlash2 > $D/27b-serve-drafter.log 2>&1; echo "R2b exit=$?"; gtt
# R3: 35B-MTP, MTP drafting
gtt; timeout 1200 $B benchmark $M35 --output $D/35b.json > $D/35b.log 2>&1; echo "R3 exit=$?"; gtt
# R4: 35B-MTP --serial (plain decode)
timeout 1200 $B benchmark $M35 --serial --output $D/35b-serial.json > $D/35b-serial.log 2>&1; echo "R4 exit=$?"; gtt
# R5: attached to a 35B serve: manifest, receipt and per-request counts against the server log
$B serve $M35 --backend rocm --host 127.0.0.1 --port 8092 --name amd-35b --no-thinking > $D/35b-serve.log 2>&1 &
echo $! > $R/serve-35b.pid
for i in $(seq 1 120); do grep -q "serving" $D/35b-serve.log && break; sleep 2; done
curl -sS http://127.0.0.1:8092/v1/benchmark/manifest > $D/35b-manifest.json
timeout 1200 $B benchmark $M35 --server http://127.0.0.1:8092 --model-id amd-35b --backend rocm --output $D/35b-attached.json > $D/35b-attached.log 2>&1; echo "R5 exit=$?"
kill "$(cat $R/serve-35b.pid)"; wait "$(cat $R/serve-35b.pid)" 2>/dev/null; rm -f $R/serve-35b.pid; gtt
# R6: the visible-device fixes on hardware: 35B-MTP with HIP_VISIBLE_DEVICES=0 set
gtt; HIP_VISIBLE_DEVICES=0 timeout 1200 $B benchmark $M35 --output $D/35b-hipvis0.json > $D/35b-hipvis0.log 2>&1; echo "R6 exit=$?"; gtt
echo RUNDONE
