"""NemotronH ``force_top_k`` Pareto sweep: decode speed vs routing-truncation loss.

NemotronH routes top-``num_experts_per_tok`` experts/token (22 on the 550B). The
deferred runner's ``force_top_k=K`` keeps only the K highest-gate-score experts of
that 22 (renormalized like ``norm_topk_prob`` on the K subset) and gathers K
experts instead of 22 -> fewer expert-GEMMs/token = the real BANDWIDTH lever,
lossy by design (it changes routing, so NOT token-exact).

For each K in ``--k-list`` (default 22,16,12,8,6,4) on ONE model load:

* SPEED  -- set ``runner.force_top_k=K`` and time ``generate_greedy_deferred``;
            headline = the runner's own ``decode_tok_s`` (excludes prefill).
* QUALITY -- teacher-forced ``forward_logits_deferred`` over a fixed held-out
            sequence at this K; with K=22 (== the routed count, a no-op) as the
            REFERENCE compute, per K:
              PPL      = exp(mean NLL of the true-next-token under the top-K logits)
              KL_vs_22 = mean KL(softmax(ref) || softmax(topK)) per position
              sub_rate = frac. of positions where argmax(topK) != argmax(ref)

Output = ONE Pareto table: K | decode_tok_s | PPL | KL_vs_top22 | sub_rate |
active_experts (active_experts = the per-token routed width == min(K, routed)).

==============================  HONEST CAVEAT  ==============================
The QUALITY metric is the REAL routing cliff ONLY if the routed experts are REAL
(resident, NOT buddy-substituted). On an ``--arbitrary-hotset`` (all-buddy) path
the deferred gather serves BUDDIES for non-resident experts, so KL/PPL reflect
top-K-of-BUDDIES, not real top-K routing. The harness still computes the pipeline
CORRECTLY; it prints whether the hot-set is REAL-FREQUENCY or ARBITRARY so the
numbers are read in the right frame. The real-expert quality on the 550B is
constrained by the >RAM page-cache wall (the full cold forward needed to discover
a real frequency hot-set), which is why ``--arbitrary-hotset`` exists at all.

Why no cold-redo skews this: ``forward_logits_deferred`` discards the optimistic
(truncated) forward and recomputes EXACT if any layer routed to a non-resident
expert -- which would silently erase the K-truncation. The sweep therefore runs
with ``cold_substitution=True`` (every cold id -> a resident buddy slot, so the
cold flag never fires and the TRUNCATED forward is always the one measured). With
a covering REAL hot-set there are no cold experts and no buddies, so the quality
is the true routing cliff.

  Studio (real 550B, arbitrary all-buddy hot-set to skip the >RAM warmup):
    PYTHONPATH=src ~/st-venv/bin/python benchmarks/nemotron_topk_sweep.py \
      --arbitrary-hotset --hotset-k 256 --budget-gib 200 \
      --k-list 22,16,12,8,6,4 --decode 128 --tokens 512

  Local dry-run (tiny fixture, no tokenizer):
    PYTHONPATH=src python3 benchmarks/nemotron_topk_sweep.py \
      --model /tmp/tiny/<dir> --prompt-ids 1,5,9,3,7,2 \
      --k-list 2,1 --decode 4 --tokens 6 --hotset-k 8 --budget-gib 1
"""
from __future__ import annotations

import argparse
import math
import time

# Mixed-domain held-out sample: ~40% code, ~30% math/reasoning, ~30% prose.
HELDOUT_TEXT = (
    # --- code (~40%) ---
    "def attention(q, k, v, mask=None):\n"
    "    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])\n"
    "    if mask is not None:\n"
    "        scores = scores + mask\n"
    "    weights = softmax(scores, axis=-1)\n"
    "    return weights @ v\n\n"
    "class Router(nn.Module):\n"
    "    def __init__(self, dim, n_experts, top_k):\n"
    "        super().__init__()\n"
    "        self.gate = nn.Linear(dim, n_experts, bias=False)\n"
    "        self.top_k = top_k\n"
    "    def forward(self, x):\n"
    "        logits = self.gate(x)\n"
    "        scores = logits.sigmoid()\n"
    "        idx = scores.topk(self.top_k, dim=-1).indices\n"
    "        return idx, scores.gather(-1, idx)\n\n"
    # --- math / reasoning (~30%) ---
    "Theorem. For a mixture-of-experts layer routing the top-k of n experts, the "
    "expected number of distinct experts activated over a sequence of length T is "
    "n * (1 - (1 - k/n) ** T) under a uniform routing assumption. Proof sketch: by "
    "linearity of expectation, the probability that expert i is never selected in T "
    "independent draws of k experts is (1 - k/n) ** T, so the complement summed over "
    "n experts gives the claim. Suppose k = 22 and n = 256; then for T = 512 nearly "
    "every expert is touched at least once, which is precisely why a fixed resident "
    "hot-set must be large or the cold-miss rate dominates the bandwidth budget. "
    "Consider the limit as T grows without bound: the activated fraction tends to 1. "
    "Therefore truncating k trades a strict reduction in per-token gather cost "
    "against an increase in routing error, and the Pareto frontier is what we measure.\n\n"
    # --- prose (~30%) ---
    "Long before the model could speak, it learned to listen. Each token arrived like "
    "a traveller at a crossroads, and the router, patient and unhurried, chose which "
    "few of its many counsellors should weigh in. Most counsellors sat idle; only a "
    "handful spoke for any given word, and yet the chorus, when it came, was "
    "remarkably coherent. The engineers watched the memory gauge the way sailors watch "
    "the tide, knowing that to keep every counsellor close at hand was to run aground "
    "on the hard limit of the machine. So they asked a simpler question: how few "
    "voices can we keep before the meaning frays? The answer, they suspected, lay not "
    "in theory but in the patient accounting of perplexity, token by token, K by K."
)


def _gb(x: float) -> float:
    return x / (1024.0 ** 3)


def _kl_div(ref_logits, top_logits, mx) -> float:
    """Mean over positions of KL(softmax(ref) || softmax(top)), in nats.

    ref/top are ``[positions, vocab]`` (last token's next-step logits dropped by
    the caller so each row predicts a real next token). Computed in fp32 with a
    log-softmax for numerical stability; returns a python float.
    """
    ref = ref_logits.astype(mx.float32)
    top = top_logits.astype(mx.float32)
    log_p = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    log_q = top - mx.logsumexp(top, axis=-1, keepdims=True)
    p = mx.exp(log_p)
    kl = (p * (log_p - log_q)).sum(axis=-1)  # [positions]
    return float(mx.mean(kl))


def _ppl_and_sub(top_logits, ref_logits, targets, mx) -> tuple[float, float]:
    """(PPL of ``targets`` under top-K logits, argmax-disagreement vs ref).

    ``top_logits``/``ref_logits`` = ``[positions, vocab]``; ``targets`` =
    ``[positions]`` true next-token ids. PPL = exp(mean NLL); sub_rate = fraction
    of positions where argmax(top) != argmax(ref).
    """
    top = top_logits.astype(mx.float32)
    log_q = top - mx.logsumexp(top, axis=-1, keepdims=True)
    tgt = mx.array(targets)
    nll = -mx.take_along_axis(log_q, tgt[:, None], axis=-1).squeeze(-1)  # [positions]
    ppl = float(mx.exp(mx.mean(nll)))
    sub = float(mx.mean((mx.argmax(top, axis=-1) != mx.argmax(ref_logits, axis=-1))))
    return ppl, sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/path/to/models/Nemotron-3-Ultra-550B-A55B-4bit",
    )
    ap.add_argument(
        "--k-list",
        default="22,16,12,8,6,4",
        help="comma-separated K values; the LARGEST is the no-op reference",
    )
    ap.add_argument("--decode", type=int, default=128, help="decode tokens/run (speed)")
    ap.add_argument(
        "--tokens", type=int, default=512, help="held-out tokens for the quality pass"
    )
    ap.add_argument(
        "--arbitrary-hotset",
        action="store_true",
        help="skip the >RAM cold warmup-generate; seed each MoE layer's hot-set "
        "with experts range(--hotset-k). See the module CAVEAT: KL/PPL then "
        "reflect top-K-of-BUDDIES, not real routing.",
    )
    ap.add_argument(
        "--hotset-k",
        type=int,
        default=256,
        help="fixed_hotset_experts K (resident set size); also the range() size "
        "for --arbitrary-hotset",
    )
    ap.add_argument("--budget-gib", type=int, default=200)
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="hot-set warmup tokens (only when NOT --arbitrary-hotset)",
    )
    ap.add_argument(
        "--prompt-ids",
        default=None,
        help="comma-separated token ids; bypasses the tokenizer for BOTH the "
        "speed prompt and the quality sequence (tiny local dry-run)",
    )
    args = ap.parse_args()

    k_list = [int(t) for t in args.k_list.split(",") if t.strip()]
    if not k_list:
        raise SystemExit("--k-list is empty")
    k_list = sorted(set(k_list), reverse=True)  # descending; [0] = reference
    ref_k = k_list[0]

    import mlx.core as mx

    from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

    print(
        f"[topk] load model={args.model} hotset_k={args.hotset_k} "
        f"budget={args.budget_gib}GiB decode={args.decode} tokens={args.tokens} "
        f"k_list={k_list} (ref={ref_k})",
        flush=True,
    )
    runner = NemotronHStreamingForwardRunner(
        args.model,
        pin_policy="all",
        page_experts=True,
        weight_page_budget_bytes=args.budget_gib * (1024 ** 3),
        fixed_hotset_experts=args.hotset_k,
        # Buddy-substitute every cold id so the cold flag never fires and the
        # TRUNCATED deferred forward is what we actually measure (no exact redo
        # silently erasing the K-truncation). With a covering REAL hot-set there
        # are no cold experts, so no buddies are used and quality is the true cliff.
        cold_substitution=True,
        force_top_k=None,  # flipped per-K below
    )

    try:
        # --- token sources -------------------------------------------------
        if args.prompt_ids:
            speed_ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
            quality_ids = list(speed_ids)
            tok_source = "prompt-ids (no tokenizer)"
        else:
            from mlx_lm.utils import load_tokenizer

            tok = load_tokenizer(args.model)
            speed_ids = list(tok.encode(HELDOUT_TEXT))[: max(8, args.decode)]
            quality_ids = list(tok.encode(HELDOUT_TEXT))
            tok_source = "mlx_lm tokenizer (mixed-domain held-out)"
        quality_ids = quality_ids[: max(2, args.tokens)]
        print(
            f"[topk] tokens: speed={len(speed_ids)} quality={len(quality_ids)} "
            f"src={tok_source}",
            flush=True,
        )

        # --- ONE hot-set build --------------------------------------------
        hotset_kind = "ARBITRARY (all-buddy: KL/PPL = top-K-of-BUDDIES)"
        t0 = time.perf_counter()
        if args.arbitrary_hotset:
            moe = list(runner._moe_layer_indices())
            override = {int(layer): list(range(args.hotset_k)) for layer in moe}
            print(
                f"[topk] OVERRIDE hot-set (no warmup): {len(moe)} MoE layers, "
                f"experts range({args.hotset_k})",
                flush=True,
            )
            runner.build_fixed_hotset(quality_ids, override=override)
        else:
            hotset_kind = "REAL-FREQUENCY (quality = the true routing cliff)"
            runner.build_fixed_hotset(quality_ids, warmup_tokens=args.warmup_tokens)
        subs = runner._deferred_stats.get("cold_substitutions", 0)
        print(
            f"[topk] build_fixed_hotset {time.perf_counter() - t0:.1f}s "
            f"kind={hotset_kind} cold_substitutions={subs} "
            f"peak={_gb(mx.get_peak_memory()):.1f}GB",
            flush=True,
        )

        routed = int(
            runner.session.config.get("num_experts_per_tok", ref_k)
        )
        print(f"[topk] model routes top-{routed} experts/token", flush=True)

        # Honesty gate: the K-truncation is measured on the deferred forward, so
        # the quality is REAL routing only if every expert that actually ROUTES is
        # resident (not buddy-served). On a real-frequency build the resident set
        # IS the routed union by construction, so routed experts are resident even
        # if non-routed experts carry buddies (harmless: they never fire). The
        # reference K is a full top-routed pass; ``cold_redos==0`` there confirms
        # no routed expert escaped the resident set. The arbitrary path makes NO
        # such guarantee -> its KL/PPL are top-K-of-BUDDIES.
        real_routing = not args.arbitrary_hotset

        # --- reference logits (K = ref_k, a no-op when ref_k >= routed) -----
        runner.force_top_k = ref_k
        redos_before = runner._deferred_stats.get("cold_redos", 0)
        ref_logits_full = runner.forward_logits_deferred(quality_ids)
        mx.eval(ref_logits_full)
        ref_redos = runner._deferred_stats.get("cold_redos", 0) - redos_before
        if ref_redos and real_routing:
            # A routed expert escaped the resident set on the reference pass; the
            # deferred forward was redone EXACT, so quality reflects real routing
            # but with reduced K-truncation visibility on those tokens. Flag it.
            print(
                f"[topk] NOTE: reference pass had {ref_redos} cold-redo(s) -- some "
                "routed experts were not resident; widen --hotset-k for a fully "
                "covering real hot-set.",
                flush=True,
            )
        # [1, seq, vocab] -> drop the last position (no true next token).
        ref_logits = ref_logits_full[0, :-1, :]
        targets = quality_ids[1:]
        mx.eval(ref_logits)

        rows: list[dict] = []
        for k in k_list:
            runner.force_top_k = k
            active = min(k, routed)

            # SPEED -- the runner's own decode_tok_s (excludes prefill).
            # A warmup decode (discarded) settles caches before timing.
            runner.generate_greedy_deferred(speed_ids, min(4, args.decode))
            out = runner.generate_greedy_deferred(speed_ids, args.decode)
            d_tps = out.get("decode_tok_s") or 0.0

            # QUALITY -- top-K logits over the held-out sequence.
            top_logits_full = runner.forward_logits_deferred(quality_ids)
            mx.eval(top_logits_full)
            top_logits = top_logits_full[0, :-1, :]
            ppl, sub = _ppl_and_sub(top_logits, ref_logits, targets, mx)
            kl = 0.0 if k == ref_k else _kl_div(ref_logits, top_logits, mx)

            rows.append(
                {
                    "k": k,
                    "tok_s": d_tps,
                    "ppl": ppl,
                    "kl": kl,
                    "sub": sub,
                    "active": active,
                    "cold_redos": out.get("cold_redos", 0),
                }
            )
            print(
                f"[topk] K={k:>3} decode_tok/s={d_tps:7.2f} PPL={ppl:10.3f} "
                f"KL={kl:.4e} sub_rate={sub:.4f} active={active} "
                f"cold_redos={out.get('cold_redos', 0)}",
                flush=True,
            )

        # --- Pareto TABLE --------------------------------------------------
        print("\n[topk] ===================== PARETO TABLE =====================")
        print(f"[topk] hot-set: {hotset_kind}")
        print(
            f"[topk] reference K={ref_k} (KL/sub_rate measured against it); "
            f"model routes top-{routed}; cold_substitution=ON"
        )
        if real_routing:
            print(
                "[topk] QUALITY FRAME: REAL routing -- routed experts are resident; "
                "KL/PPL/sub_rate are the true top-K routing cliff."
            )
        else:
            print(
                "[topk] QUALITY FRAME: ARBITRARY hot-set -> KL/PPL/sub_rate reflect "
                "top-K-of-BUDDIES, NOT real top-K routing (see module docstring)."
            )
        header = (
            f"{'K':>4} | {'decode_tok_s':>12} | {'PPL':>12} | "
            f"{'KL_vs_top'+str(ref_k):>14} | {'sub_rate':>9} | {'active_experts':>14}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['k']:>4} | {r['tok_s']:>12.2f} | {r['ppl']:>12.3f} | "
                f"{r['kl']:>14.4e} | {r['sub']:>9.4f} | {r['active']:>14}"
            )
        print(
            f"[topk] peak_mem={_gb(mx.get_peak_memory()):.1f}GB "
            f"active_mem={_gb(mx.get_active_memory()):.1f}GB",
            flush=True,
        )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
