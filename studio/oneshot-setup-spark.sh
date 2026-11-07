#!/bin/bash
# One-shot DGX Spark install (GB10, CUDA 13) under $PREFIX and $SPARK_MODELS. Options and sizes: README.
set -euo pipefail
# Licences: Qwen-Image-2.1-Turbo non-commercial (Qwen Research); FastH3 and H3 under the MiniMax H3 Community License.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Credit: FastVideo (Hao AI Lab), MiniMax, the Qwen team, ComfyUI; Ash Hart (TensorFold), whose Studio this is for CUDA.
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
MODELS="${SPARK_MODELS:-$HOME/spark-studio-models}"
COMFY_COMMIT=5a665890ded5020c282f63c4ea2dd9e083e025ff  # v0.41.0 and later: ComfyUI's FLUX 3 nodes
IMAGE_ONLY=0; RENDER=1; VERIFY=0; REF2VA=0; MOTION=0; VOICE=0; UPGRADE=0; LORAS=""; H3_BASE=0; WORKFLOWS=1; IMAGE_EDIT=0; FLUX3=0
for arg in "$@"; do
  case "$arg" in
    --ref2va) REF2VA=1;; --h3-base) H3_BASE=1;; --image-edit) IMAGE_EDIT=1;; --flux3) FLUX3=1;; --motion) MOTION=1;; --voice) VOICE=1;; --image-only) IMAGE_ONLY=1;; --no-render) RENDER=0;; --verify) VERIFY=1; RENDER=0;; --upgrade) UPGRADE=1; RENDER=0; MOTION=1;;
    --no-workflows) WORKFLOWS=0;;
    --loras) LORAS=all;; --loras=*) LORAS="${arg#--loras=}";; --no-loras) LORAS=none;; --lora-bundle=*) LORA_BUNDLE="${arg#--lora-bundle=}";;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
say() { printf '==> %s\n' "$*"; }
ok() { printf '  \342\234\223 %s\n' "$*"; }
die() { printf '  \342\234\227 %s\n' "$*" >&2; exit 1; }

say "Preflight"
if [ "$(uname -s)" = Darwin ]; then echo "macOS detected: installing the Mac (Metal) build"; exec bash "$HERE/oneshot-setup.sh" "$@"; fi
[ "$(uname -s)" = Linux ] || die "this installs on a DGX Spark (Linux, CUDA) or an Apple-silicon Mac"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs an NVIDIA GPU with its driver"
# --upgrade moves an install to this version, fetches no weights, keeps its kind (image-only, Ref2VA) and studio folder.
INSTALLED="$(cat "$PREFIX/VERSION" 2>/dev/null || echo "2.6 or earlier")"
if [ "$UPGRADE" = 1 ]; then
  [ -x "$PREFIX/venv/bin/python" ] || die "nothing to upgrade at $PREFIX: run oneshot-setup-spark.sh without --upgrade"
  [ -s "$MODELS/FastH3-Comfy/diffusion_models/fastvideo_fasth3_8step_v2_pruned_bf16.safetensors" ] || IMAGE_ONLY=1
  [ ! -s "$MODELS/MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ] || REF2VA=1
  [ ! -s "$MODELS/FLUX3-K-Comfy/diffusion_models/flux3_image_klein_sd_bf16.safetensors" ] || FLUX3=1
  ok "upgrade: installed $INSTALLED -> $(cat "$HERE/VERSION")$([ "$IMAGE_ONLY" = 0 ] || echo " (image-only install)")"
else ok "version $(cat "$HERE/VERSION")$([ ! -f "$PREFIX/VERSION" ] || echo " (installed: $INSTALLED)")"; fi
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
MEM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
ok "$GPU, $(uname -m), ${MEM_GB} GB"
case "$GPU" in *GB10*) ;; *) echo "  ! measured only on a DGX Spark (GB10, 128 GB unified memory); other GPUs are untested";; esac
[ "$MEM_GB" -ge 100 ] || [ "$IMAGE_ONLY" = 1 ] || die "video needs about 90 GB of memory at 1344x768 (128 GB machine); this one has ${MEM_GB} GB. --image-only fits."
for tool in git ffmpeg ffprobe; do command -v "$tool" >/dev/null || die "install $tool first (sudo apt install $([ "$tool" = ffprobe ] && echo ffmpeg || echo "$tool"))"; done
if ! command -v uv >/dev/null; then
  [ "$VERIFY" = 0 ] || die "no uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
NEED=$([ "$IMAGE_ONLY" = 1 ] && echo 40 || echo 125)
if [ "$VERIFY" = 0 ]; then
  mkdir -p "$PREFIX" "$MODELS"
  FREE=$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc 0-9)
  HAVE=$(du -sBG "$MODELS" 2>/dev/null | cut -f1 | tr -dc 0-9)
  [ $((FREE + HAVE)) -ge "$NEED" ] || die "$MODELS has ${FREE} GB free; the install needs about ${NEED} GB"
fi

say "Environment: Python 3.12 and PyTorch for CUDA 13 -> $PREFIX/venv"
if [ ! -x "$PREFIX/venv/bin/python" ]; then
  [ "$VERIFY" = 0 ] || die "no environment at $PREFIX/venv"
  uv venv "$PREFIX/venv" --python 3.12 --python-preference only-managed --seed -q
fi
PY="$PREFIX/venv/bin/python"
pipi() { UV_TORCH_BACKEND=cu130 uv pip install -q --python "$PY" "$@"; }
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  [ "$VERIFY" = 0 ] || die "PyTorch does not see the GPU"
  pipi torch torchvision torchaudio
fi
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "PyTorch does not see the GPU"
ok "torch $("$PY" -c 'import torch; print(torch.__version__, "on", torch.cuda.get_device_name(0))')"

say "ComfyUI @ ${COMFY_COMMIT:0:8} -> $PREFIX/ComfyUI"
if [ "$(git -C "$PREFIX/ComfyUI" rev-parse HEAD 2>/dev/null)" != "$COMFY_COMMIT" ]; then
  [ "$VERIFY" = 0 ] || die "ComfyUI is missing or not at the pinned commit"
  [ -d "$PREFIX/ComfyUI/.git" ] || git clone -q https://github.com/comfyanonymous/ComfyUI "$PREFIX/ComfyUI"
  git -C "$PREFIX/ComfyUI" fetch -q origin "$COMFY_COMMIT" 2>/dev/null || git -C "$PREFIX/ComfyUI" fetch -q origin
  git -C "$PREFIX/ComfyUI" checkout -q "$COMFY_COMMIT"
fi
if [ "$VERIFY" = 0 ]; then
  pipi -r "$PREFIX/ComfyUI/requirements.txt" huggingface_hub fastapi uvicorn python-multipart pillow
  # speech-to-text for the QA gate of the Wasserman Sharp Spec (app/qa.py); without it that gate is shown as skipped
  pipi faster-whisper || echo "  note: faster-whisper was not installed; the QA gate's speech check will be skipped"
  rm -rf "$PREFIX/ComfyUI/custom_nodes/fasth3_sparse"
  cat > "$PREFIX/ComfyUI/extra_model_paths.yaml" <<YAML
studio:
  base_path: $MODELS
  diffusion_models: |
    FastH3-Comfy/diffusion_models
    Qwen-Image-2.1-Comfy/diffusion_models
    MiniMax-H3-Comfy/diffusion_models
    FLUX3-K-Comfy/diffusion_models
  text_encoders: |
    FastH3-Comfy/text_encoders
    Qwen-Image-2.1-Comfy/text_encoders
    FLUX3-K-Comfy/text_encoders
  vae: |
    FastH3-Comfy/vae
    Qwen-Image-2.1-Comfy/vae
    FLUX3-K-Comfy/vae
  loras: |
    loras
    MiniMax-H3-Comfy/loras
  checkpoints: |
    Workflows-Comfy/checkpoints
  latent_upscale_models: |
    Workflows-Comfy/latent_upscale_models
YAML
fi
"$PY" -c "import comfy_kitchen, aiohttp, fastapi, uvicorn, PIL" || die "the environment is missing packages"
ok "ComfyUI and its requirements"

# fetch <repo> <folder under $MODELS> <file in the repo>...: download what is not there yet
fetch() {
  local repo="$1" folder="$2" missing=(); shift 2
  for file in "$@"; do [ -s "$MODELS/$folder/$file" ] || missing+=("$file"); done
  [ ${#missing[@]} = 0 ] && return 0
  [ "$VERIFY" = 0 ] || die "missing in $MODELS/$folder: ${missing[*]}"
  if [ "$UPGRADE" = 1 ]; then echo "  ! missing in $MODELS/$folder: ${missing[*]}: --upgrade fetches no weights (run oneshot-setup-spark.sh without it)"; return 0; fi
  "$PREFIX/venv/bin/hf" download "$repo" "${missing[@]}" --local-dir "$MODELS/$folder" >/dev/null
  for file in "${missing[@]}"; do [ -s "$MODELS/$folder/$file" ] || die "$file did not download"; done
}

say "Qwen-Image-2.1-Turbo for ComfyUI (24 GB) -> $MODELS"
fetch Comfy-Org/Qwen-Image-2.1 Qwen-Image-2.1-Comfy diffusion_models/qwen_image_2.1_turbo_bf16.safetensors \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors vae/qwen_image_2.1_vae_bf16.safetensors
ok "Qwen-Image-2.1-Turbo (8 steps), its text encoder and decoder"

if [ "$IMAGE_ONLY" = 0 ]; then
  say "FastH3 8-Step V2 for ComfyUI (75 GB) -> $MODELS"
  fetch FastVideo/FastVideo-FastH3-Comfy FastH3-Comfy diffusion_models/fastvideo_fasth3_8step_v2_pruned_bf16.safetensors \
    text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors vae/minimax_h3_video_vae_int8_convrot.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors
  ok "FastH3 transformer (bf16), H3 text encoder (8-bit), video decoder (8-bit) and audio decoder"
fi

if [ "$REF2VA" = 1 ] && [ "$IMAGE_ONLY" = 0 ]; then
  say "MiniMax H3 Ref2VA for ComfyUI and its Turbo adapter (39 GB) -> $MODELS"
  fetch Comfy-Org/MiniMax-H3 MiniMax-H3-Comfy diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
    loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
  ok "Ref2VA transformer (bf16) and LightX2V's Ref2VA Turbo adapter (4 passes)"
  # a running ComfyUI read its model folders when it started
  if bash "$HERE/spark/comfy.sh" status | grep -q running; then echo "  restart ComfyUI to see them: bash spark/comfy.sh stop (the next render starts it)"; fi
fi

if [ "$IMAGE_EDIT" = 1 ] && [ "$VERIFY" = 0 ]; then
  say "Qwen-Image-2.1's 40-step base model for ComfyUI (14 GB) -> $MODELS"
  echo "  Qwen-Image-2.1 is under the Qwen Research License Agreement: NON-COMMERCIAL use only. Read it first:"
  echo "  https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"
  fetch Comfy-Org/Qwen-Image-2.1 Qwen-Image-2.1-Comfy diffusion_models/qwen_image_2.1_bf16.safetensors
  ok "Qwen-Image-2.1 base model (bf16): BASE=1 bash scripts/edit.sh, and the 40 steps choice of Edit a picture"
  if bash "$HERE/spark/comfy.sh" status | grep -q running; then echo "  restart ComfyUI to see it: bash spark/comfy.sh stop (the next render starts it)"; fi
fi

if [ "$FLUX3" = 1 ]; then
  say "FLUX3-K for ComfyUI: Fast and Full, text encoder, prompt upsampler, decoder (41 GB) -> $MODELS"
  [ "$VERIFY" = 1 ] || { echo "  FLUX 3 Image Klein is under the FLUX Kommunity License: non-commercial use, and commercial use of the outputs"
    echo "  only for Qualifying Users (under US\$5M revenue). Read it, and accept it on the model page, first:"
    echo "  https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy"; }
  fetch black-forest-labs/flux-3-image-klein-comfy FLUX3-K-Comfy diffusion_models/flux3_image_klein_sd_bf16.safetensors \
    diffusion_models/flux3_image_klein_base_bf16.safetensors text_encoders/qwen3vl_4b_bf16.safetensors \
    text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors vae/flux2-vae.safetensors
  fetch black-forest-labs/flux-3-image-klein-pup FLUX3-K-Comfy system_prompts.json
  ok "FLUX3-K Fast and Full (bf16), Qwen3-VL-4B text encoder, BFL's prompt upsampler (8-bit) and the decoder"
  if [ "$VERIFY" = 0 ] && bash "$HERE/spark/comfy.sh" status | grep -q running; then echo "  restart ComfyUI to see them: bash spark/comfy.sh stop (the next render starts it)"; fi
fi

if [ "$H3_BASE" = 1 ] && [ "$IMAGE_ONLY" = 0 ] && [ "$VERIFY" = 0 ]; then
  say "MiniMax H3's own first-and-last-frame transformer for ComfyUI (40 GB) -> $MODELS"
  fetch Comfy-Org/MiniMax-H3 MiniMax-H3-Comfy diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors
  ok "MiniMax H3 FL2VA transformer (bf16): SPARK_BASE=h3, and keyframe LoRAs from the hub"
  if bash "$HERE/spark/comfy.sh" status | grep -q running; then echo "  restart ComfyUI to see it: bash spark/comfy.sh stop (the next render starts it)"; fi
fi

# Library actions run their authors' ComfyUI nodes on Ref2VA: installed wherever Ref2VA is (spark/setup-workflows.sh).
if [ "$IMAGE_ONLY" = 0 ] && [ "$WORKFLOWS" = 1 ] && [ -s "$MODELS/MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ]; then
  say "Library actions: three node packs at pinned commits, SAM 3.1 and the H3 latent upscaler (2.5 GB) -> $PREFIX/ComfyUI, $MODELS/Workflows-Comfy"
  PREFIX="$PREFIX" SPARK_MODELS="$MODELS" bash "$HERE/spark/setup-workflows.sh" $([ "$VERIFY" = 0 ] || echo --verify) $([ "$UPGRADE" = 0 ] || echo --no-models) \
    || echo "  ! the Library actions are not fully set up: the Library says which one is missing what"
fi

if [ "$IMAGE_ONLY" = 0 ]; then
  say "TensorFold's H3 engine for CUDA -> $PREFIX/tf-h3"
  if [ "$VERIFY" = 1 ]; then
    [ -s "$PREFIX/tf-h3/libtf_h3.so" ] && ok "libtf_h3.so" || echo "  not built: clips render on ComfyUI's own blocks (slower)"
  elif PREFIX="$PREFIX" bash "$HERE/spark/build-engine.sh"; then
    ok "libtf_h3.so (FastH3's blocks in int8 on the tensor cores)"
  else
    echo "  the engine did not build: clips will render on ComfyUI's own blocks (slower). Fix and rerun spark/build-engine.sh"
  fi
fi

if [ "$MOTION" = 1 ] || [ -x "$PREFIX/motion-venv/bin/python" ]; then
  # Motion Previs (port of Sam Wasserman's Motion Previs Studio analysis) in its own env; uninstallable layers skipped
  say "Motion Previs: MediaPipe, OpenCV, ONNX Runtime and the pose and depth models (0.5 GB) -> $PREFIX/motion-venv"
  PREFIX="$PREFIX" bash "$HERE/scripts/setup-motion.sh" $([ "$VERIFY" = 1 ] && echo --verify) | sed 's/^/  /' \
    || echo "  ! Motion Previs is not fully set up: its tab says which layers are not available on this machine"
fi

[ "$VERIFY" = 1 ] || cp "$HERE/VERSION" "$PREFIX/VERSION"
if [ "$UPGRADE" = 1 ]; then
  say "Upgraded: $INSTALLED -> $(cat "$PREFIX/VERSION"). Checking the install (restart the app to serve the new pages: scripts/app.sh)"
  exec bash "$HERE/oneshot-setup-spark.sh" --verify $([ "$IMAGE_ONLY" = 0 ] || echo --image-only) $([ "$REF2VA" = 0 ] || echo --ref2va) $([ "$FLUX3" = 0 ] || echo --flux3)
fi
# Essential LoRA bundle: --loras=GROUPS takes some groups, --lora-bundle=PATH installs offline, --no-loras skips it.
if [ "$LORAS" != none ] && [ "$IMAGE_ONLY" = 0 ] && [ "$VERIFY" != 1 ]; then
  echo; echo "== Essential LoRA bundle${LORA_BUNDLE:+ from $LORA_BUNDLE}"
  PREFIX="$PREFIX" H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}" bash "$HERE/oneshot-essential-bundle.sh" \
    $([ -z "$LORAS" ] || [ "$LORAS" = all ] || echo "--categories=$LORAS") $([ -z "${LORA_BUNDLE:-}" ] || echo "--source=$LORA_BUNDLE") 2>&1 | sed 's/^/  /' \
    || echo "  ! some LoRAs could not be fetched (listed above): the rest are installed, and bash oneshot-essential-bundle.sh tries again"
fi
if [ "$VOICE" = 1 ]; then  # Fish Audio Drama 3: checked, and the hand steps printed when something is missing
  PREFIX="$PREFIX" bash "$HERE/scripts/setup-voice.sh" $([ "$VERIFY" = 1 ] && echo --verify) || [ "$VERIFY" = 0 ]
fi
if [ "$RENDER" = 1 ]; then
  say "Test render"
  mkdir -p "$HERE/outputs"
  PREFIX="$PREFIX" bash "$HERE/scripts/image.sh" "A lighthouse on a rocky coast at sunrise, waves breaking, warm light." "$HERE/outputs/test.png"
  ok "outputs/test.png"
  if [ "$IMAGE_ONLY" = 0 ]; then
    PREFIX="$PREFIX" bash "$HERE/scripts/video.sh" 'A woman in a yellow raincoat stands on a pier, looks at the camera and says in a clear voice: "The tide is coming in." Gulls and waves.' "$HERE/outputs/test.mp4"
    ok "outputs/test.mp4 (864x480, 124 frames, 8 passes)"
  fi
fi
echo "DONE. Web app: bash $HERE/scripts/app.sh   (http://127.0.0.1:7870)"
echo "      Image:   bash $HERE/scripts/image.sh \"a prompt\" out.png"
echo "      Video:   bash $HERE/scripts/video.sh \"a prompt\" out.mp4"
echo "      Free the models' memory: bash $HERE/spark/comfy.sh stop"
