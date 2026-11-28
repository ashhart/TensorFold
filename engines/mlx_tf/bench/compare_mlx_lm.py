#!/usr/bin/env python3
"""Strict Apples-to-Apples 4-bit Benchmark: TensorFold mlx_tf (Zero Apple MLX) vs Stock Apple mlx_lm.

Runs every model on an IDLE Apple Silicon GPU in isolated subprocesses using the EXACT SAME
4-bit affine group-64 safetensors weights for both engines across all 7 models & 3 architectures:
  1. Qwen3.5 / Qwen3.8 Hybrid Gated DeltaNet + GQA (0.8B, 2B, 0.8B-Agentic, 27B + DFlash2)
  2. Qwen2.5 Dense GQA Transformers (1.5B, 7B — converted via `mlx_tf convert`, zero Apple MLX)
  3. NVIDIA Nemotron-3.5-Lightning-30B-A3B Sparse MoE (128x top-6) + Mamba-2 + NoPE GQA
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

CONV = os.path.expanduser("~/.cache/tensorfold/converted_4bit")

MODELS = [
    {
        "label": "Qwen3.5-0.8B-4bit",
        "repo": "mlx-community/Qwen3.5-0.8B-4bit",
        "family": "Hybrid GDN (4-bit)",
        "chat": True,
        "prompt": "Explain what a GPU compute shader does and list 3 practical applications in detail.",
        "max_tokens": 128,
    },
    {
        "label": "Qwen3.5-2B-4bit",
        "repo": "mlx-community/Qwen3.5-2B-4bit",
        "family": "Hybrid GDN (4-bit)",
        "chat": True,
        "prompt": "Explain what a GPU compute shader does and list 3 practical applications in detail.",
        "max_tokens": 128,
    },
    {
        "label": "Qwen3.8-0.8B-4bit",
        "repo": os.path.join(CONV, "Qwen3.8-0.8B-Agentic-4bit"),
        "family": "Hybrid GDN (4-bit)",
        "chat": True,
        "prompt": "Explain what a GPU compute shader does and list 3 practical applications in detail.",
        "max_tokens": 128,
    },
    {
        "label": "Qwen2.5-1.5B-4bit",
        "repo": os.path.join(CONV, "Qwen2.5-1.5B-4bit"),
        "family": "Dense GQA (4-bit)",
        "chat": False,
        "prompt": "Write a detailed technical overview of how unified memory works on Apple Silicon:\n\n",
        "max_tokens": 128,
    },
    {
        "label": "Qwen2.5-7B-4bit",
        "repo": os.path.join(CONV, "Qwen2.5-7B-4bit"),
        "family": "Dense GQA (4-bit)",
        "chat": False,
        "prompt": "Write a detailed technical overview of how unified memory works on Apple Silicon:\n\n",
        "max_tokens": 128,
    },
    {
        "label": "Qwen3.8-27B-4bit",
        "repo": "TensorFold/Qwen3.8-27B-MLX-4bit",
        "family": "Hybrid GDN (4-bit)",
        "chat": True,
        "prompt": "Explain how speculative decoding accelerates autoregressive LLMs without changing the output distribution.",
        "max_tokens": 128,
        "dflash_repo": "z-lab/Qwen3.8-27B-DFlash2",
    },
    {
        "label": "Nemotron-3.5-30B-A3B",
        "repo": "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",
        "family": "MoE+Mamba-2 (4-bit)",
        "chat": True,
        "prompt": "Explain how Mamba-2 state space models differ from Transformer attention and why hybrid MoE architectures combine them.",
        "max_tokens": 128,
    },
]

WORKER_MLX_TF = r"""
import json, sys, time, mlx_tf
assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules
cfg = json.loads(sys.argv[1])
t0 = time.perf_counter()
model, tok = mlx_tf.load(cfg["repo"], draft_model=cfg.get("dflash_repo"))
load_ms = (time.perf_counter() - t0) * 1000.0
assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules

prompt = cfg["prompt"]
if cfg["chat"]:
    prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)

# Warmup decode & prefill (2 prefill passes ensure OS page-cache is resident on 14+ GiB models)
_ = mlx_tf.generate(model, tok, prompt, max_tokens=16, temp=0.0)
long_prompt = ("Apple Silicon unified memory enables zero-copy GPU compute across CPU and GPU cores. " * 32).strip()
for _ in range(2):
    _ = mlx_tf.generate(model, tok, long_prompt, max_tokens=1, temp=0.0)

# Run 3 passes and record peak (minimum-contention) decode & prefill throughput
best_gen_tps = 0.0
best_pre_tps = 0.0
out = []
last = None
p_toks = 0
for trial in range(3):
    cur_out = []
    cur_last = None
    for r in mlx_tf.stream_generate(model, tok, prompt, max_tokens=cfg["max_tokens"], temp=0.0):
        cur_out.append(r.text)
        cur_last = r
    if cur_last and cur_last.generation_tps > best_gen_tps:
        best_gen_tps = cur_last.generation_tps
        out = cur_out
        last = cur_last

    p_last = None
    for r in mlx_tf.stream_generate(model, tok, long_prompt, max_tokens=1, temp=0.0):
        p_last = r
    if p_last:
        p_toks = p_last.prompt_tokens
        if p_last.prompt_tps > best_pre_tps:
            best_pre_tps = p_last.prompt_tps

assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules
print(json.dumps({
    "engine": "mlx_tf",
    "device": model.device_name,
    "arch": model.arch_name,
    "m5_tensor_ops": model.uses_m5_tensor_ops,
    "load_ms": round(load_ms, 1),
    "prefill_tokens": p_toks,
    "prefill_tps": round(best_pre_tps, 1),
    "prompt_tokens": last.prompt_tokens if last else 0,
    "gen_tokens": last.generation_tokens if last else 0,
    "gen_tps": round(best_gen_tps, 1),
    "text": "".join(out).strip(),
}))
"""

WORKER_MLX_LM = r"""
import json, sys, time, mlx_lm, mlx_tf
cfg = json.loads(sys.argv[1])
path = mlx_tf.resolve_model_path(cfg["repo"], allow_download=False)
t0 = time.perf_counter()
model, tok = mlx_lm.load(path)
load_ms = (time.perf_counter() - t0) * 1000.0

prompt = cfg["prompt"]
if cfg["chat"]:
    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Warmup decode & prefill symmetrically (2 prefill passes for OS page-cache parity)
for _ in mlx_lm.stream_generate(model, tok, prompt=prompt, max_tokens=16):
    pass
long_prompt = ("Apple Silicon unified memory enables zero-copy GPU compute across CPU and GPU cores. " * 32).strip()
for _ in range(2):
    for _ in mlx_lm.stream_generate(model, tok, prompt=long_prompt, max_tokens=1):
        pass

# Run 3 passes and record peak (minimum-contention) decode & prefill throughput
best_gen_tps = 0.0
best_pre_tps = 0.0
out = []
last = None
p_toks = 0
for trial in range(3):
    cur_out = []
    cur_last = None
    for r in mlx_lm.stream_generate(model, tok, prompt=prompt, max_tokens=cfg["max_tokens"]):
        cur_out.append(r.text)
        cur_last = r
    if cur_last and cur_last.generation_tps > best_gen_tps:
        best_gen_tps = cur_last.generation_tps
        out = cur_out
        last = cur_last

    p_last = None
    for r in mlx_lm.stream_generate(model, tok, prompt=long_prompt, max_tokens=1):
        p_last = r
    if p_last:
        p_toks = p_last.prompt_tokens
        if p_last.prompt_tps > best_pre_tps:
            best_pre_tps = p_last.prompt_tps

print(json.dumps({
    "engine": "mlx_lm",
    "load_ms": round(load_ms, 1),
    "prefill_tokens": p_toks,
    "prefill_tps": round(best_pre_tps, 1),
    "prompt_tokens": last.prompt_tokens if last else 0,
    "gen_tokens": last.generation_tokens if last else 0,
    "gen_tps": round(best_gen_tps, 1),
    "text": "".join(out).strip(),
}))
"""


def run_worker(code: str, cfg: dict) -> dict:
    raw = subprocess.check_output([sys.executable, "-c", code, json.dumps(cfg)], text=True)
    for line in reversed(raw.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from worker:\n{raw}")


def main() -> int:
    results = []
    hdr = (
        f"{'Model (4-bit vs 4-bit)':24s} | {'Decode tf/lm (tok/s)':22s} | {'Dec X':6s} | "
        f"{'Prefill tf/lm (tok/s)':22s} | {'Pre X':6s} | {'Load tf/lm (ms)':17s} | {'Ld X':5s}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for m in MODELS:
        cfg_std = {k: v for k, v in m.items() if k != "dflash_repo"}
        r_tf = run_worker(WORKER_MLX_TF, cfg_std)
        r_df = run_worker(WORKER_MLX_TF, m) if "dflash_repo" in m else None
        r_lm = run_worker(WORKER_MLX_LM, cfg_std)

        d_x = r_tf["gen_tps"] / max(r_lm["gen_tps"], 0.1)
        p_x = r_tf["prefill_tps"] / max(r_lm["prefill_tps"], 0.1)
        l_x = r_lm["load_ms"] / max(r_tf["load_ms"], 0.1)
        print(
            f"{m['label']:24s} | {r_tf['gen_tps']:8.1f} / {r_lm['gen_tps']:8.1f}   | {d_x:5.2f}x | "
            f"{r_tf['prefill_tps']:8.0f} / {r_lm['prefill_tps']:8.0f}   | {p_x:5.2f}x | "
            f"{r_tf['load_ms']:6.0f} / {r_lm['load_ms']:6.0f}   | {l_x:4.1f}x"
        )
        if r_df is not None:
            df_x = r_df["gen_tps"] / max(r_lm["gen_tps"], 0.1)
            df_px = r_df["prefill_tps"] / max(r_lm["prefill_tps"], 0.1)
            df_lx = r_lm["load_ms"] / max(r_df["load_ms"], 0.1)
            print(
                f"{m['label'] + '+DF2':24s} | {r_df['gen_tps']:8.1f} / {r_lm['gen_tps']:8.1f}   | {df_x:5.2f}x | "
                f"{r_df['prefill_tps']:8.0f} / {r_lm['prefill_tps']:8.0f}   | {df_px:5.2f}x | "
                f"{r_df['load_ms']:6.0f} / {r_lm['load_ms']:6.0f}   | {df_lx:4.1f}x"
            )

        results.append({
            "model": m["label"],
            "family": m["family"],
            "mlx_tf": r_tf,
            "mlx_tf_dflash2": r_df,
            "mlx_lm": r_lm,
        })

    print("=" * len(hdr))

    out_path = os.path.join(os.path.dirname(__file__), "results_7models_idle_4bit.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full idle 4-bit vs 4-bit metrics to: {out_path}")
    for row in results:
        same = row["mlx_tf"]["text"][:80] == row["mlx_lm"]["text"][:80]
        print(f"\n[{row['model']}] prefix80_exact={same}")
        print(f"  mlx_tf : {row['mlx_tf']['text'][:140]}")
        print(f"  mlx_lm : {row['mlx_lm']['text'][:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
