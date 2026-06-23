"""A/B: NemotronH deferred decode with the fused gated-RMSNorm kernel OFF vs ON.

ONE model load + ONE fixed hot-set; the fused kernel (`fuse_ssm_norm_gate`) is
toggled between decode runs so the ONLY difference is ~6 fused dispatches per
Mamba2 layer. Headline = the runner's own decode_tok_s (excludes prefill) for
OFF (best of 2) vs ON (best of 2), plus token-equality (the fused norm is
~1 fp32 ULP and must not change emitted tokens) and route/memory telemetry.
Studio-only for the real run (loads the 550B); use --prompt-ids on a tiny
fixture to dry-run the flow locally.

  Studio: PYTHONPATH=src ~/st-venv/bin/python benchmarks/nemotron_fused_norm_ab.py
  Dry-run: PYTHONPATH=src python3 benchmarks/nemotron_fused_norm_ab.py \
             --model /tmp/tiny/<dir> --prompt-ids 1,5,9,3 --k 4 \
             --budget-gib 1 --decode 4 --warmup-tokens 4
"""
from __future__ import annotations

import argparse
import time


def _gb(x: float) -> float:
    return x / (1024.0 ** 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/path/to/models/Nemotron-3-Ultra-550B-A55B-4bit",
    )
    ap.add_argument("--k", type=int, default=200, help="fixed_hotset_experts")
    ap.add_argument("--budget-gib", type=int, default=150)
    ap.add_argument("--decode", type=int, default=64, help="decode tokens per run")
    ap.add_argument("--warmup-tokens", type=int, default=32, help="hot-set warmup")
    ap.add_argument(
        "--prompt",
        default=(
            "Explain, step by step, how a transformer language model turns a "
            "prompt into the next token, then name two ways to make it faster."
        ),
    )
    ap.add_argument(
        "--prompt-ids",
        default=None,
        help="comma-separated token ids; bypasses the tokenizer (local dry-run)",
    )
    ap.add_argument(
        "--arbitrary-hotset",
        action="store_true",
        help="skip the cold warmup-generate; seed the hot-set with experts "
        "range(K) per MoE layer. Valid for a KERNEL A/B (the gated-norm fusion "
        "acts on the Mamba2 layers regardless of which experts are resident) and "
        "avoids the >RAM full-model cold forward that exhausts the page cache "
        "during warmup discovery.",
    )
    args = ap.parse_args()

    import mlx.core as mx

    from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

    print(
        f"[ab] load model={args.model} K={args.k} budget={args.budget_gib}GiB "
        f"decode={args.decode}",
        flush=True,
    )
    runner = NemotronHStreamingForwardRunner(
        args.model,
        pin_policy="all",
        page_experts=True,
        weight_page_budget_bytes=args.budget_gib * (1024 ** 3),
        fixed_hotset_experts=args.k,
        cold_substitution=True,
        fuse_ssm_norm_gate=False,
    )

    try:
        if args.prompt_ids:
            ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
        else:
            from mlx_lm.utils import load_tokenizer

            ids = list(load_tokenizer(args.model).encode(args.prompt))
        print(f"[ab] prompt_tokens={len(ids)}", flush=True)

        t0 = time.perf_counter()
        if args.arbitrary_hotset:
            moe = list(runner._moe_layer_indices())
            override = {int(layer): list(range(args.k)) for layer in moe}
            print(
                f"[ab] OVERRIDE hot-set (no warmup): {len(moe)} MoE layers, "
                f"experts range({args.k})",
                flush=True,
            )
            runner.build_fixed_hotset(ids, override=override)
        else:
            runner.build_fixed_hotset(ids, warmup_tokens=args.warmup_tokens)
        print(
            f"[ab] build_fixed_hotset {time.perf_counter() - t0:.1f}s "
            f"peak={_gb(mx.get_peak_memory()):.1f}GB "
            f"active={_gb(mx.get_active_memory()):.1f}GB",
            flush=True,
        )

        def run(label: str) -> dict:
            t = time.perf_counter()
            out = runner.generate_greedy_deferred(ids, args.decode)
            wall = time.perf_counter() - t
            toks = out["tokens"]
            n = max(1, len(toks))
            summ = out.get("summary", {}) or {}
            wps = summ.get("weight_page_summary", {}) or {}
            d_tps = out.get("decode_tok_s")
            tele = {
                "cold_subs": out.get("cold_substitutions"),
                "cold_redos": out.get("cold_redos"),
                "hit_rate": wps.get("hit_rate"),
                "native_hits": summ.get("hotset_native_hits"),
                "cold_fallbacks": summ.get("hotset_cold_fallbacks"),
            }
            print(
                f"[ab] {label} fused={runner.fuse_ssm_norm_gate} "
                f"installed={runner._fused_ssm_norm_installed} n={len(toks)} "
                f"wall={wall:.2f}s decode_s={out.get('decode_s')} "
                f"decode_tok/s={d_tps} active={_gb(mx.get_active_memory()):.1f}GB "
                f"tele={tele}",
                flush=True,
            )
            return {
                "toks": toks,
                "wall_ms": 1000.0 * wall / n,
                "decode_tok_s": d_tps if d_tps else (1000.0 * n / (wall * 1000.0)),
            }

        print("[ab] warmup (discard)...", flush=True)
        run("warmup")

        off1 = run("OFF#1")
        off2 = run("OFF#2")

        runner.fuse_ssm_norm_gate = True
        on1 = run("ON#1")
        on2 = run("ON#2")

        off_tps = max(off1["decode_tok_s"], off2["decode_tok_s"])
        on_tps = max(on1["decode_tok_s"], on2["decode_tok_s"])
        match = off1["toks"] == on1["toks"]
        print("[ab] ===== RESULT =====", flush=True)
        print(
            f"[ab] decode_tok/s OFF(best)={off_tps:.2f} ON(best)={on_tps:.2f} "
            f"speedup={on_tps / max(off_tps, 1e-9):.3f}x tokens_match={match}",
            flush=True,
        )
        print(
            f"[ab] wall_ms/tok OFF(best)={min(off1['wall_ms'], off2['wall_ms']):.2f} "
            f"ON(best)={min(on1['wall_ms'], on2['wall_ms']):.2f}",
            flush=True,
        )
        if not match:
            pref = 0
            for a, b in zip(off1["toks"], on1["toks"]):
                if a != b:
                    break
                pref += 1
            print(
                f"[ab] token prefix-match={pref}/{len(off1['toks'])} "
                f"(a late 1-ULP argmax flip is expected drift, not a bug)",
                flush=True,
            )
        print(f"[ab] peak_mem={_gb(mx.get_peak_memory()):.1f}GB", flush=True)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
