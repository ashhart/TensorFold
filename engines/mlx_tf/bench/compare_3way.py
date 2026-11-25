#!/usr/bin/env python3
"""3-Way Bare-Metal Benchmark: TensorFold mlx_tf vs Stock Apple mlx_lm vs llama.cpp (Metal 4).

Compares decode throughput (tok/s), prompt prefill throughput (tok/s, ~480-512 tok), and
cold model load time (ms) across Hybrid GDN (Qwen3.5 0.8B, 2B, 27B) and Dense GQA
(Qwen2.5 1.5B, 7B) models on Apple Silicon.

Notes on fairness:
  - mlx_tf and mlx_lm load the EXACT SAME 4-bit affine group-64 safetensors checkpoints.
  - llama.cpp loads standard Q4_K_M GGUF checkpoints via `llama-bench` (`-p 512 -n 128 -fa 1 -ngl 99`)
    and `llama-cli` cold load timing, running with Metal 4 / M5 tensor API enabled.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compare_mlx_lm as cm

GGUF_DIR = os.path.expanduser("~/.cache/tensorfold/gguf")

GGUF_MAP = {
    "Qwen3.5-0.8B-4bit": os.path.join(GGUF_DIR, "Qwen3.5-0.8B-Q4_K_M.gguf"),
    "Qwen3.5-2B-4bit": os.path.join(GGUF_DIR, "Qwen3.5-2B-Q4_K_M.gguf"),
    "Qwen2.5-1.5B-4bit": os.path.join(GGUF_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "Qwen2.5-7B-4bit": os.path.join(GGUF_DIR, "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
    "Qwen3.8-27B-4bit": os.path.join(GGUF_DIR, "Qwen3.5-27B-Q4_K_M.gguf"),
}


def run_llama_bench(gguf_path: str) -> dict | None:
    if not os.path.exists(gguf_path):
        return None
    # Measure llama-bench (pp512 + tg128, -r 3, Metal flash-attn enabled)
    cmd = [
        "llama-bench",
        "-m", gguf_path,
        "-p", "512",
        "-n", "128",
        "-r", "3",
        "-ngl", "99",
        "-fa", "1",
        "-o", "json",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    rows = json.loads(raw)
    pp_tps = 0.0
    tg_tps = 0.0
    for row in rows:
        samples = row.get("samples_ts") or []
        best = max(samples) if samples else float(row.get("avg_ts", 0.0))
        if int(row.get("n_prompt", 0)) > 0 and int(row.get("n_gen", 0)) == 0:
            pp_tps = best
        elif int(row.get("n_gen", 0)) > 0 and int(row.get("n_prompt", 0)) == 0:
            tg_tps = best

    # Measure cold wall-clock load + 1-token startup via llama-cli (-n 1 -no-cnv --no-warmup)
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [
                "llama-cli", "-m", gguf_path, "-p", "Hi", "-n", "1",
                "-ngl", "99", "-fa", "on", "-no-cnv", "--no-warmup",
                "--single-turn", "--no-display-prompt", "--simple-io",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        load_ms = (time.perf_counter() - t0) * 1000.0
    except Exception:
        load_ms = 0.0

    return {
        "engine": "llama.cpp",
        "gguf": os.path.basename(gguf_path),
        "prefill_tps": round(pp_tps, 1),
        "gen_tps": round(tg_tps, 1),
        "load_ms": round(load_ms, 1),
    }


def main() -> int:
    results = []
    hdr = (
        f"{'Model (4-bit)':22s} | {'Decode tf / lm / llama (tok/s)':33s} | "
        f"{'Prefill tf / lm / llama (tok/s)':33s} | {'Load tf / lm / llama (ms)':27s}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for m in cm.MODELS:
        if m["label"] not in GGUF_MAP:
            continue
        gguf_path = GGUF_MAP[m["label"]]
        if not os.path.exists(gguf_path):
            continue
        cfg_std = {k: v for k, v in m.items() if k != "dflash_repo"}
        r_tf = cm.run_worker(cm.WORKER_MLX_TF, cfg_std)
        r_df = cm.run_worker(cm.WORKER_MLX_TF, m) if "dflash_repo" in m else None
        r_lm = cm.run_worker(cm.WORKER_MLX_LM, cfg_std)
        r_lc = run_llama_bench(gguf_path)

        lc_dec = r_lc["gen_tps"] if r_lc else 0.0
        lc_pre = r_lc["prefill_tps"] if r_lc else 0.0
        lc_ld = r_lc["load_ms"] if r_lc else 0.0
        print(
            f"{m['label']:22s} | "
            f"{r_tf['gen_tps']:8.1f} / {r_lm['gen_tps']:8.1f} / {lc_dec:8.1f}      | "
            f"{r_tf['prefill_tps']:8.0f} / {r_lm['prefill_tps']:8.0f} / {lc_pre:8.0f}      | "
            f"{r_tf['load_ms']:6.0f} / {r_lm['load_ms']:6.0f} / {lc_ld:6.0f}"
        )
        if r_df is not None:
            print(
                f"{m['label'] + '+DF2':22s} | "
                f"{r_df['gen_tps']:8.1f} / {r_lm['gen_tps']:8.1f} / {lc_dec:8.1f}      | "
                f"{r_df['prefill_tps']:8.0f} / {r_lm['prefill_tps']:8.0f} / {lc_pre:8.0f}      | "
                f"{r_df['load_ms']:6.0f} / {r_lm['load_ms']:6.0f} / {lc_ld:6.0f}"
            )

        results.append({
            "model": m["label"],
            "mlx_tf": r_tf,
            "mlx_tf_dflash2": r_df,
            "mlx_lm": r_lm,
            "llama_cpp": r_lc,
        })

    print("=" * len(hdr))
    out_path = os.path.join(os.path.dirname(__file__), "results_3way_idle.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved 3-way results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
