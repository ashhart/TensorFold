#!/bin/bash
# #1001 gfx1151 run on the Bosgame M5. Run ON THE BOX as the box user, after the pulls finished:
#   bash ~/tf-1001-runs/pr-1001-run.sh > ~/tf-1001-runs/run.log 2>&1
# Builds the merged tree in ~/tf-1001, stops llm-server, runs R1-R5 one model at a time,
# and ALWAYS restarts llm-server (EXIT trap), then checks it answers a chat request.
set -u
R=$HOME/tf-1001-runs; D=$R/out; mkdir -p $D
Z=$HOME/zig/zig-x86_64-linux-0.17.0/zig
B=$HOME/tf-1001/zig-out/native-hip/bin/tensorfold-native
M27=TensorFold/Qwen3.8-27B-MLX-4bit
M35=TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP
gtt() { echo "gtt_used=$(cat /sys/class/drm/card*/device/mem_info_gtt_used | head -1) busy=$(cat /sys/class/drm/card*/device/gpu_busy_percent | head -1) t=$(date -u +%T)"; }

restart_llm() {
  for f in $R/serve-*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  sudo -n systemctl start llm-server.service
  echo "llm-server: $(systemctl is-active llm-server)"
  for i in $(seq 1 60); do
    out=$(curl -sS -m 60 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $(cat $HOME/.llm-api-key 2>/dev/null)" \
      -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say ok."}],"max_tokens":8}' 2>&1)
    if echo "$out" | grep -q '"choices"'; then echo "llm-server chat OK after ${i}0s: $out" | cut -c1-300; return; fi
    sleep 10
  done
  echo "llm-server chat check FAILED: $out" | cut -c1-300
}

# 1. checkout from the bundle (copied to $R/tf-1001.bundle)
if [ ! -d $HOME/tf-1001/.git ]; then
  git clone -q $R/tf-1001.bundle $HOME/tf-1001 || exit 1
fi
cd $HOME/tf-1001 || exit 1
git checkout -q -B tf1001-gpu origin/benchmark-rocm-host-identity
git -c user.name=ThinkOffApp -c user.email=233579950+ThinkOffApp@users.noreply.github.com merge -q --no-edit origin/hip-refuses-drafter origin/benchmark-drafter-counts || exit 1
git rev-parse HEAD origin/main origin/benchmark-rocm-host-identity origin/hip-refuses-drafter origin/benchmark-drafter-counts > $R/heads.txt

# 2. preflight
df -h / ; free -g | head -2
ls $HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots $HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.6-35B-A3B-MLX-4bit-MTP/snapshots || exit 1

# 3. from here on llm-server is restarted whatever happens
trap restart_llm EXIT
sudo -n systemctl stop llm-server.service || exit 1
echo "llm-server: $(systemctl is-active llm-server)"; gtt

# 4. build (CPU)
$Z build native-hip -Dhip-arch=gfx1151 -Dhipcc=/usr/bin/hipcc -Dhip-include=/usr/include \
  -Dhip-device-lib=/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode -Doptimize=ReleaseFast -j16 > $R/build.log 2>&1
echo "build_exit=$?"; sha256sum $B | tee $R/binary.sha256 || exit 1

export HF_HUB_OFFLINE=1
# R1: 27B, --drafter auto (DFlash2 cached)
gtt; $B benchmark $M27 --output $D/27b-auto.json > $D/27b-auto.log 2>&1; echo "R1 exit=$?"; gtt
# R2a/R2b: refusals
$B benchmark $M27 --drafter z-lab/Qwen3.8-27B-DFlash2 --output $D/27b-dflash2.json > $D/27b-dflash2.log 2>&1; echo "R2a exit=$?"; ls $D/27b-dflash2.json 2>&1
gtt; timeout 180 $B serve $M27 --backend rocm --host 127.0.0.1 --port 8093 --drafter z-lab/Qwen3.8-27B-DFlash2 > $D/27b-serve-drafter.log 2>&1; echo "R2b exit=$?"; gtt
# R3: 35B-MTP drafting
gtt; $B benchmark $M35 --output $D/35b.json > $D/35b.log 2>&1; echo "R3 exit=$?"; gtt
# R4: 35B-MTP --serial
$B benchmark $M35 --serial --output $D/35b-serial.json > $D/35b-serial.log 2>&1; echo "R4 exit=$?"; gtt
# R5: attached to a 35B serve: manifest + counts cross-check
$B serve $M35 --backend rocm --host 127.0.0.1 --port 8092 --name amd-35b --no-thinking > $D/35b-serve.log 2>&1 &
echo $! > $R/serve-35b.pid
for i in $(seq 1 120); do grep -q "serving" $D/35b-serve.log && break; sleep 2; done
curl -sS http://127.0.0.1:8092/v1/benchmark/manifest > $D/35b-manifest.json
$B benchmark $M35 --server http://127.0.0.1:8092 --model-id amd-35b --backend rocm --output $D/35b-attached.json > $D/35b-attached.log 2>&1; echo "R5 exit=$?"
kill "$(cat $R/serve-35b.pid)"; wait "$(cat $R/serve-35b.pid)" 2>/dev/null; rm -f $R/serve-35b.pid; gtt
echo RUNDONE
# EXIT trap restarts llm-server and checks a chat request
