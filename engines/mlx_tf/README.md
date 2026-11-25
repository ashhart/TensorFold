# TensorFold `mlx_tf`

**Universal Bare-Metal Apple Silicon LLM & VLM Runtime, Quantizer & OpenAI Server — Zero Apple MLX.**

`mlx_tf` is a standalone, drop-in replacement for Apple `mlx_lm` built directly on **Zig + handwritten Metal shaders + C ABI + Python SDK**. It compiles to a self-contained native binary (`mlx_tf`) and shared library (`libmlxtf.dylib`) with **zero runtime dependency on Apple MLX (`mlx` / `mlx_lm`)**.

---

## Supported Quantization Formats

| Format | Precision & Group Sizes | Notes |
|---|---|---|
| **Native TFQ3 (`.tffpw`)** | `3-bit`, **`3.5-bit` mixed**, `4-bit`, `5-bit` (2D FWHT-128 + 2D-AffineCoset/half4) | Built-in data-free Metal GPU quantizer (`mlx_tf convert --format tfq3 --profile 3b\|3.5b\|4b\|5b`), zero-copy 16 KiB-aligned `.tffpw` loader, cross-platform Metal + CUDA bit-exact format |
| **MLX Affine (`safetensors`)** | `2-bit`, `3-bit`, `4-bit`, `5-bit`, `6-bit`, `8-bit` × `gs ∈ {32, 64, 128}` + **mixed precision** | Ultra-fast `qk4_g64` tile path (M5 hardware `tensor_ops` + M1–M4 SIMD) plus universal `qmv_affine_any` / `qmm_affine_any` & mixed per-tensor bit-width support; built-in GCD `mlx_tf convert --bits 2..8 --group-size 32\|64\|128` |
| **Lossless BF16 / FP16 / F32** | Raw HuggingFace `bfloat16` / `float16` / `float32` | Lossless inference via `MLX_TF_BF16=1` (or automatic parallel GCD 4-bit on-the-fly quantization when unset) |

---

## Supported Architecture Families (LLMs & VLMs)

| Architecture Family | Models & Variants | Core Primitives |
|---|---|---|
| **Qwen3.5 / Qwen3.8** | Dense & MoE (`qwen3_5`, `qwen3_5_moe`, `qwen3_6_moe`, `qwen3_next`) | Hybrid Gated DeltaNet + GQA (output gate, partial RoPE), SwiGLU MoE + gated shared expert, DFlash2 block-diffusion speculative decoding, Hybrid KV+GDN prefix caching, `/v1/stream-parallel`, TF-Rewind |
| **Qwen2 / Qwen2.5 / Qwen3** | Dense & MoE (`qwen2`, `qwen3`, `qwen3_moe`) | GQA + QKV bias (`Qwen2/2.5`), Q/K RMSNorm (`Qwen3`), softmax SwiGLU MoE (`Qwen3-MoE`) |
| **Llama 3 & Mistral / Mixtral** | Llama 3 / 3.1 / 3.2 / 3.3, Mistral, Mixtral (`llama`, `mistral`, `mixtral`) | Piecewise Llama 3 RoPE scaling, sliding-window GQA, Mixtral `w1/w3/w2` + MLX `switch_mlp` MoE |
| **Gemma 2 / Gemma 3 / Gemma 4** | Dense & VLM (`gemma2`, `gemma3`, `gemma4`) | `√H` embed scaling, `+1`-centered 4-norm blocks, GeGLU (`gelu_pytorch_tanh`), attn/logit softcapping, dual local/global RoPE, Gemma 4 `k_eq_v` + `v_norm` + mixed `head_dim=256/512` + proportional RoPE + `layer_scalar` |
| **DeepSeek-V2 / V3 / R1 / V3.2** | `deepseek_v2`, `deepseek_v3`, `deepseek_v32` | Multi-Head Latent Attention (MLA: HF `kv_b_proj` + MLX-Community split `embed_q`/`unembed_out`), YaRN `mscale`/`mscale_all_dim`, grouped top-K sigmoid MoE + `e_score_correction_bias` + shared experts |
| **GLM-4 / GLM-4.5 / GLM-4.7 / GLM-5** | `glm`, `glm4`, `glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`, `glm5` | Interleaved & split-half partial RoPE, 4-norm blocks, fused `gate_up_proj`, grouped sigmoid MoE, GLM-5 DSA MLA (`qk_nope=192, qk_rope=64, v_dim=256`) |
| **LiquidAI LFM2 / LFM2.5 / LFM2-MoE** | `lfm2`, `lfm2_vl`, `lfm2_moe` | Single-dispatch gated causal depthwise `ShortConv` (`L_cache ≤ 8`) + GQA hybrid, `operator_norm`/`ffn_norm`, `feed_forward.{w1,w3,w2}` + sigmoid MoE with `expert_bias` |
| **NVIDIA Nemotron-H / 3.5** | `nemotron_h` (e.g. Nemotron-3.5-Lightning-30B-A3B) | 100% GPU-resident Mamba-2 SSD + ReLU² sigmoid MoE (`128× top-6` + M5 hardware `moe_expert_qk16_batched`) + NoPE GQA |
| **VLM Vision Encoders** | **Qwen3.5-VL / 3.8-VL** & **Gemma 3-VL (SigLIP)** | Native macOS `ImageIO`/`CoreGraphics` PNG/JPEG/WebP/HEIC + base64 `data:` URI decoding, 3D-collapsed patch embed, 2D bilinear `pos_embed`, 2D Vision RoPE, arbitrary `head_dim ≤ 256` (including `64` & `72`) bidirectional ViT, 2×2 patch merger, SigLIP `AvgPool2d` + `Gemma3RMSNorm` projector |

---

## Highlights

- **Zero Apple MLX:** `import mlx_tf` never imports `mlx` or `mlx_lm`. Every GEMM/GEMV, TFQ3 FWHT/beam kernel, RMSNorm, LayerNorm, RoPE, FlashAttention, MLA, Gated DeltaNet, ShortConv, Mamba-2 SSD, Sparse MoE, and Vision Transformer kernel runs on handwritten Metal compute shaders via direct Objective-C runtime dispatch.
- **Zero-Copy `.mlxtf` Pre-Packed Tile Cache (`newBufferWithBytesNoCopy`):** Persists pre-packed `1152B` `qk` Metal buffers at Apple Silicon `16 KiB` VM page boundaries in `~/.cache/tensorfold/pack_cache/` (disable with `MLX_TF_NO_CACHE=1`), mapping `15–17 GB` models in **`144–199 ms`** (**`8.8×–13.0×` faster** than `mlx_lm`, **`17.7×` faster** than `llama.cpp`).
- **Universal Apple Silicon (M1–M5) + Auto-Hybrid Metal 4/3:** Automatically uses Metal 4.1 hardware `mpp::tensor_ops` on M5+ and seamlessly falls back to pure Metal 3.1 SIMD kernels on M1–M4 / macOS 14–15 (verified `100%` token-exact with `--force-simd`).
- **Built-in Bare-Metal Quantizers (`mlx_tf convert`):**
  - `--format tfq3`: Data-free Metal GPU TFQ3 quantizer (`--profile 3b|3.5b|4b|5b`, `--bits 3|4|5`, `--head-bits`, `--keep-head-bf16`).
  - `--format affine`: Parallel GCD BF16/FP16/F32 → MLX `2..8`-bit affine `g32/64/128` safetensors converter (`100%` cross-compatible with `mlx_lm`).
- **Production OpenAI HTTP/SSE Server (`mlx_tf serve`):** `/health`, `/v1/models`, `/v1/chat/completions` (including multimodal `content: [{"type":"text",...}, {"type":"image_url",...}]`), `/v1/completions`, and `/v1/stream-parallel` with `stop` sequence streaming buffering, penalties, and `stream_options.include_usage`.

---

### Build, Install & Verify

```bash
# 1. Build native binary + libmlxtf.dylib (requires Zig 0.17.0 on macOS Apple Silicon)
cd engines/mlx_tf
zig build --release=fast

# 2. Install drop-in Python SDK & CLI entrypoints (mlx-tf, mlx-tf-serve, mlx-tf-convert)
python3 -m pip install -e ./python --no-build-isolation --no-deps

# 3. Run the 7-part zero-MLX production test suite (~4s)
python3 tests/test_suite.py
```

---

## Usage

### Python SDK (Drop-in `mlx_lm` API + Multimodal VLM)

```python
import sys
import mlx_tf

model, tok = mlx_tf.load("mlx-community/Qwen3.5-0.8B-4bit")
assert "mlx" not in sys.modules  # Zero Apple MLX!

# Text generation
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain Gated DeltaNet in two sentences."}],
    add_generation_prompt=True,
    enable_thinking=False,
)
for resp in mlx_tf.stream_generate(model, tok, prompt, max_tokens=128, temp=0.0):
    print(resp.text, end="", flush=True)

# Multimodal VLM generation (local path, file:// URI, or data:image/...;base64,...)
if model.has_vision:
    reply = mlx_tf.generate(
        model, tok,
        prompt="Describe what is in this image.",
        images=["photo.png"],
        max_tokens=128,
        temp=0.0,
    )
    print(reply)
```

### CLI, Quantizers (TFQ3 & Affine) & Multimodal Server

```bash
# Text or VLM generation (also supports --force-simd on any M1-M5 Mac)
./zig-out/bin/mlx_tf generate /path/to/Qwen3.5-0.8B-4bit --prompt "Write a haiku about Metal."
./zig-out/bin/mlx_tf generate /path/to/Qwen3.5-0.8B-4bit --image photo.png --prompt "What is in this image?"

# Quantize raw HF weights to native TFQ3 (3 / 3.5 / 4 / 5-bit, data-free on Metal GPU)
./zig-out/bin/mlx_tf convert /path/to/HF-Model ./model-tfq3-3.5b --format tfq3 --profile 3.5b

# Quantize raw HF weights to MLX affine (2..8-bit, g32/64/128, zero Apple MLX)
./zig-out/bin/mlx_tf convert /path/to/Qwen2.5-7B ./qwen25-7b-4bit --format affine --bits 4 --group-size 64

# Run raw BF16 weights losslessly (without on-the-fly 4-bit quantization)
MLX_TF_BF16=1 ./zig-out/bin/mlx_tf generate /path/to/BF16-Model --prompt "Hello!"

# OpenAI-compatible HTTP/SSE + multimodal + /v1/stream-parallel server
./zig-out/bin/mlx_tf serve /path/to/Qwen3.5-0.8B-4bit --port 8080
```

### C ABI

Link against `zig-out/lib/libmlxtf.dylib` and include [`include/mlx_tf.h`](include/mlx_tf.h):
- Core: `mlxtf_version`, `mlxtf_load`, `mlxtf_load_ex`, `mlxtf_free`, `mlxtf_device_name`, `mlxtf_arch_name`, `mlxtf_uses_m5_tensor_ops`
- Generation, Server & Quantization: `mlxtf_generate`, `mlxtf_stream_parallel`, `mlxtf_serve`, `mlxtf_serve_host`, `mlxtf_convert`, `mlxtf_convert_tfq3`
- Vision / VLM: `mlxtf_has_vision`, `mlxtf_encode_image`, `mlxtf_encode_patches`, `mlxtf_generate_with_images`

---

## Verified Idle-GPU Benchmarks (`Apple M5 Max`, 128 GB)

All benchmarks run on an **idle Apple M5 Max** in isolated subprocesses. `mlx_tf` and stock Apple `mlx_lm` evaluate the **exact same 4-bit affine group-64 safetensors checkpoints** (`Qwen3.8-0.8B`, `Qwen2.5-1.5B`, and `Qwen2.5-7B` converted once via `mlx_tf convert`, zero Apple MLX). `llama.cpp` runs `Q4_K_M` GGUF checkpoints via `llama-bench` (`-p 512 -n 128 -r 3 -ngl 99 -fa 1`) with Metal 4 + M5 tensor API enabled.

### 1. Three-Way Clean Sweep: `mlx_tf` vs Stock Apple `mlx_lm` vs Metal-4 `llama.cpp`

Reproducible via [`bench/compare_3way.py`](bench/compare_3way.py) (raw JSON: [`bench/results_3way_idle.json`](bench/results_3way_idle.json)).

| Model (4-bit) | Decode `mlx_tf / mlx_lm / llama.cpp` | vs `mlx_lm` / `llama.cpp` | Prefill `mlx_tf / mlx_lm / llama.cpp` | vs `mlx_lm` / `llama.cpp` | Load `mlx_tf / mlx_lm / llama.cpp` | vs `mlx_lm` / `llama.cpp` |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen3.5-0.8B` | **571.2** / 559.5 / 345.7 tok/s | **1.02×** / **1.65×** | **23,425** / 16,770 / 15,745 tok/s | **1.40×** / **1.49×** | **149** / 767 / 999 ms | **5.1×** / **6.7×** |
| `Qwen3.5-2B` | **352.5** / 336.2 / 237.4 tok/s | **1.05×** / **1.48×** | **12,732** / 10,151 / 9,348 tok/s | **1.25×** / **1.36×** | **143** / 823 / 830 ms | **5.8×** / **5.8×** |
| `Qwen2.5-1.5B` | **408.9** / 384.7 / 297.5 tok/s | **1.06×** / **1.37×** | **12,270** / 12,194 / 11,152 tok/s | **1.01×** / **1.10×** | **81** / 358 / 511 ms | **4.4×** / **6.3×** |
| `Qwen2.5-7B` | **127.1** / 123.0 / 107.3 tok/s | **1.03×** / **1.18×** | **3,215** / 3,077 / 2,850 tok/s | **1.04×** / **1.13×** | **84** / 597 / 999 ms | **7.1×** / **11.9×** |
| `Qwen3.8-27B` | **35.3** / 33.6 / 25.4 tok/s | **1.05×** / **1.39×** | **833** / 764 / 665 tok/s | **1.09×** / **1.25×** | **144** / 1,867 / 2,550 ms | **13.0×** / **17.7×** |
| `Qwen3.8-27B + DF2` | **108.7** / 33.6 / 25.4 tok/s | **3.24×** / **4.28×** | 734 / 764 / 665 tok/s | 0.96× / **1.10×** | **1,289** / 1,867 / 2,550 ms | **1.4×** / **2.0×** |

### 2. Full 7-Model 4-bit vs 4-bit Sweep Across Core Architectures (`mlx_tf` vs `mlx_lm`)

Reproducible via [`bench/compare_mlx_lm.py`](bench/compare_mlx_lm.py) (raw JSONs: [`bench/results_7models_idle_4bit.json`](bench/results_7models_idle_4bit.json) & [`bench/results_3way_idle.json`](bench/results_3way_idle.json)). **7/7 Decode win · 7/7 Prefill win · 7/7 Load win · 7/7 Exact greedy parity (`prefix80_exact=True`)**.

| Model (4-bit vs 4-bit) | Family | Decode `mlx_tf / mlx_lm` | Decode | Prefill `mlx_tf / mlx_lm` | Prefill | Load `mlx_tf / mlx_lm` | Load | Parity |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `Qwen3.5-0.8B-4bit` | Hybrid GDN | **577.8** / 561.1 tok/s | **1.03×** | **23,600** / 17,304 tok/s | **1.36×** | **112** / 739 ms | **6.6×** | Exact |
| `Qwen3.8-0.8B-4bit` | Hybrid GDN | **574.5** / 565.3 tok/s | **1.02×** | **23,555** / 17,448 tok/s | **1.35×** | **122** / 744 ms | **6.1×** | Exact |
| `Qwen3.5-2B-4bit` | Hybrid GDN | **357.0** / 344.5 tok/s | **1.04×** | **12,774** / 10,104 tok/s | **1.26×** | **135** / 785 ms | **5.8×** | Exact |
| `Qwen2.5-1.5B-4bit` | Dense GQA | **408.4** / 384.8 tok/s | **1.06×** | **12,283** / 12,053 tok/s | **1.02×** | **72** / 346 ms | **4.8×** | Exact |
| `Qwen2.5-7B-4bit` | Dense GQA | **127.7** / 123.7 tok/s | **1.03×** | **3,221** / 3,098 tok/s | **1.04×** | **71** / 565 ms | **7.9×** | Exact |
| `Qwen3.8-27B-4bit` | Hybrid GDN | **35.3** / 33.6 tok/s | **1.05×** | **833** / 764 tok/s | **1.09×** | **144** / 1,867 ms | **13.0×** | Exact |
| `Qwen3.8-27B-4bit + DF2` | Hybrid + DFlash2 | **108.7** / 33.6 tok/s | **3.24×** | 734 / 764 tok/s | 0.96× | **1,289** / 1,867 ms | **1.4×** | Exact |
| `Nemotron-3.5-30B-A3B` | MoE + Mamba-2 | **183.0** / 164.4 tok/s | **1.11×** | **3,278** / 2,431 tok/s | **1.35×** | **199** / 1,757 ms | **8.8×** | Exact |

### Additional Verified Capabilities

- **Universal Architecture & Quantization Parity:** Bit-exact / token-exact verification across **TFQ3** (`3b/3.5b/4b/5b`), **MLX affine** (`2/3/4/5/6/8-bit`, `g32/64/128`, mixed precision), **BF16** (`Meta-Llama-3.1-8B-Instruct` `224/224` exact), **Llama 3 / Mistral / Mixtral / Qwen3 Dense & MoE / Qwen3.5 MoE**, **Gemma 2 / 3 / 4**, **DeepSeek-V2 / V3 / V3.2 (HF & MLX split MLA)**, **GLM-4 / 4.5 MoE / 5 (`glm_moe_dsa`)**, **LiquidAI LFM2 / LFM2.5 / LFM2-MoE**, and **VLM Vision Encoders** (**Qwen3.5-VL** `cos=0.999178` vs PyTorch `Qwen3_5VisionModel` on real `Qwen3.5-0.8B-4bit` + live multimodal CLI & Python image understanding; **Gemma 3-VL SigLIP** `hd=72` + `AvgPool2d` + `Gemma3RMSNorm` `8/8` exact).
- **Universal M1–M4 SIMD (`--force-simd`):** `100%` token-exact greedy parity vs M5 tensor ops (`21/21` branching tree speculative verification + `92/92 (100% BIT-EXACT)` adaptive copy lookahead on `Qwen3.5-0.8B-4bit`).
- **OpenAI Server (`/v1/stream-parallel` & Multimodal `/v1/chat/completions`):** 4-stream zero-copy prefix-forked wave on `Qwen3.5-0.8B-4bit` (`1,018.7 tok/s` aggregate decode), full OpenAI `image_url` vision support, streaming/non-streaming `stop` sequences, penalties, and `stream_options.include_usage`.
