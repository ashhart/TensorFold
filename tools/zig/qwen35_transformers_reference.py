#!/usr/bin/env python3
"""Public Qwen BF16 reference: qwen35_transformers_reference."""
from __future__ import annotations
from qwen35_reference_verdict import evaluate_report, public_report

from datetime import datetime, timezone
import argparse
import json
import os
import sys
import torch
from qwen35_reference_inputs import (
    build_capture_map,
    capture_self_checks,
    checkpoint_identity,
    environment_report,
    json_safe,
    load_capture,
    load_fixtures,
    load_served,
    log,
    now_iso,
    peak_rss_gb,
    sha256_file,
)
from qwen35_reference_forward import (
    bound_profile,
    load_model,
    model_load_report,
)
from qwen35_reference_fixture import (
    compare_fp64_records,
    load_head_from_safetensors,
    run_fixture,
)
from qwen35_reference_compare import (
    run_native_comparison,
)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("inputs", "load", "a", "b", "all"), default="all")
    ap.add_argument("--model", required=True,
                    help="pinned BF16 checkpoint directory")
    ap.add_argument("--fixtures",
                    default="docs/receipts/qwen35-bf16-scoring/fixtures.json",
                    help="tracked token fixtures")
    ap.add_argument("--api",
                    default="docs/receipts/qwen35-bf16-scoring/api.json",
                    help="tracked API receipt holding the served continuations")
    ap.add_argument("--fp64", required=True,
                    help="FP64 reference rows for the named projection captures")
    ap.add_argument("--captures", required=True,
                    help="provider capture tree (metas and float32 dumps)")
    ap.add_argument("--revision-record", default=None,
                    help="public model revision record, if one is available")
    ap.add_argument("--output", default="transformers-prompt-scores.json")
    ap.add_argument("--row-chunk", type=int, default=64)
    ap.add_argument("--head-vocab-chunk", type=int, default=4096)
    ap.add_argument("--skip-fp32-4097", dest="skip_fp32_4097", action="store_true", default=True)
    ap.add_argument("--fp32-4097", dest="skip_fp32_4097", action="store_false")
    ap.add_argument("--no-shard-hash", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()
    checkpoint_identity._no_hash = bool(args.no_shard_hash)

    torch.set_grad_enabled(False)
    t_start = datetime.now(timezone.utc)
    fixtures = load_fixtures(args.fixtures)
    cont, cont_agreement = load_served(args.api)
    fp64 = json.loads(open(args.fp64, "r").read())
    log("fixtures: %s" % [len(x) for x in fixtures])
    log("served continuations: %s" % {k: v[:8] for k, v in sorted(cont.items())})

    # ---- capture inventory ----------------------------------------------
    names = sorted(os.listdir(args.captures))
    indices = sorted(int(n.split("-")[0]) for n in names if n.endswith("-meta.json"))
    all_caps = []
    for i in indices:
        try:
            c = load_capture(args.captures, i)
        except FileNotFoundError:
            log("capture %d: meta only (no binaries), ignored" % i)
            continue
        all_caps.append(c)
    log("captures with binaries: %s" % [c.index for c in all_caps])
    logprob_caps = [c for c in all_caps if c.phase != "projection"]
    projection_caps = [c for c in all_caps if c.phase == "projection"]
    log("logprob captures: %s ; projection captures: %s" % (
        [c.index for c in logprob_caps], [c.index for c in projection_caps]))
    meta_checks = {c.index: capture_self_checks(c) for c in logprob_caps}

    res = {
        "schema": 1,
        "complete": False,
        "native_comparison": {},
        "qualification": {"status": "not-run", "reason": "run-in-progress"},
        "kind": "transformers-forward-prompt-logprob-reference",
        "stage": args.stage,
        "reference_source_sha256": {name: sha256_file(os.path.join(os.path.dirname(__file__), name))
                                    for name in ("qwen35_transformers_reference.py",
                                                 "qwen35_reference_inputs.py", "qwen35_reference_forward.py",
                                                 "qwen35_reference_fixture.py", "qwen35_reference_compare.py",
                                                 "qwen35_reference_verdict.py")},
        "generated_utc": now_iso(),
        "argv": sys.argv,
        "input_files": {
            "fixtures": {"path": args.fixtures, "sha256": sha256_file(args.fixtures)},
            "api": {"path": args.api, "sha256": sha256_file(args.api)},
            "fp64_reference": {"path": args.fp64, "sha256": sha256_file(args.fp64)},
            "captures_dir": args.captures,
        },
        "served_continuations": {str(k): v for k, v in sorted(cont.items())},
        "served_continuation_agreement": {str(k): v for k, v in sorted(cont_agreement.items())},
        "native_capture_inventory": {
            "all_with_binaries": [c.index for c in all_caps],
            "logprob": [c.index for c in logprob_caps],
            "projection": [c.index for c in projection_caps],
            "self_checks": {str(k): v for k, v in meta_checks.items()},
            "projection_keys": [c.checks.get("projection") for c in projection_caps],
        },
    }

    def write_out(merge=True):
        nonlocal res
        res["elapsed_s"] = (datetime.now(timezone.utc) - t_start).total_seconds()
        res["peak_rss_gb"] = peak_rss_gb()
        res["host"] = "redacted"
        if merge and os.path.exists(args.output):
            try:
                old = json.loads(open(args.output, "r").read())
                old.update(res)
                res = old
            except Exception:
                pass
        with open(args.output, "w") as f:
            json.dump(public_report(res), f, indent=1, default=json_safe)
            f.write("\n")
        log("wrote %s (%.1f MB)" % (args.output, os.path.getsize(args.output) / 1e6))

    # ---- inputs stage ----------------------------------------------------
    if args.stage == "inputs":
        W = load_head_from_safetensors(args.model)
        log("lm_head loaded from shard %s: %s" % (json.loads(open(os.path.join(
            args.model, "model.safetensors.index.json")).read())["weight_map"]["lm_head.weight"], W.shape))
        # fp64 cross-check against recorded fp64 metrics
        recomputed = []
        for c in logprob_caps:
            prof = bound_profile(torch.from_numpy(c.hidden_f32), torch.from_numpy(c.logits_f32),
                                 torch.from_numpy(c.lse), c.picks, W,
                                 chunk=args.head_vocab_chunk)
            prof["capture_name"] = "%d-meta.json" % c.index
            prof["phase"] = c.phase
            prof["rows"] = c.rows
            recomputed.append(prof)
        res["fp64_crosscheck"] = compare_fp64_records(fp64, recomputed)
        res["capture_map_candidates"] = build_capture_map(fixtures, cont, logprob_caps)
        log("fp64 crosscheck: %s" % json.dumps({k: v for k, v in res["fp64_crosscheck"].items()
                                                if k != "boolean_mismatches"}))
        write_out(merge=False)
        return 0

    # ---- model stages ----------------------------------------------------
    env = environment_report()
    res["environment"] = env
    log("env: torch %s (cuda %s, available %s) transformers %s accelerate %s" % (
        env["torch"], env["torch_version_cuda"], env["cuda_available"],
        env["transformers"], env["accelerate"]))
    ck, _cfg = checkpoint_identity(args.model, args.revision_record)
    res["checkpoint"] = ck
    log("checkpoint revision: %s (%s), %d shards, %.1f GB" % (
        ck["revision"], ck["revision_source"], len(ck.get("safetensors", [])),
        ck.get("checkpoint_bytes", 0) / 1e9))
    model, loading_info = load_model(args.model)
    res["model_load"] = model_load_report(model, loading_info)
    log("loaded: %s dtype %s, %d params, peak rss %.1f GB" % (
        res["model_load"]["class"], res["model_load"]["model_dtype"],
        res["model_load"]["parameter_count"], peak_rss_gb()))

    # Teacher-force five continuation IDs per fixture to place decode captures; causal prompt rows are unaffected.
    capmap = build_capture_map(fixtures, cont, logprob_caps)
    needed = sorted(set())
    # Resolve fixtures whose continuation or prompt tail occurs among native captures after the forwards.
    for item in capmap:
        for cand in item["candidates"]:
            needed.append(cand["fixture"])
    needed = sorted(set(needed))
    log("fixtures involved in native comparison (candidates): %s" % needed)
    tf_cache = {fi: {"fixture": fi, "prompt_len": len(fixtures[fi])} for fi in needed}

    # ---- A stage ---------------------------------------------------------
    if args.stage in ("a", "all"):
        verify_fixture = None
        for i, ids in enumerate(fixtures):
            if len(ids) >= 16:
                verify_fixture = i
                break
        fixture_results = []
        for i, ids in enumerate(fixtures):
            res_i = run_fixture(model, i, ids, cont, args, tf_cache,
                                verify_layers=[0, 3] if i == verify_fixture else [])
            fixture_results.append(res_i)
            log("fixture %d done: %d rows bf16, %d rows fp32, max|delta| %s" % (
                i, len(res_i["bf16"]["rows"]), len(res_i.get("fp32_rows") or []),
                (res_i["delta"]["stats"] or {}).get("max_abs_logprob_delta")))
            # incremental checkpoint of A results
            res["fixtures"] = fixture_results
            write_out(merge=True)
        res["fixtures"] = fixture_results
        # headline A
        deltas = [d for fr in fixture_results for d in fr["delta"]["rows"]]
        complete = [fr for fr in fixture_results if not fr["fp32"].get("skipped")]
        skipped = [fr["index"] for fr in fixture_results if fr["fp32"].get("skipped")]
        headline = {
            "fixtures_reported": [fr["index"] for fr in fixture_results],
            "fixture_lengths": [fr["length"] for fr in fixture_results],
            "fixtures_with_both_arms": [fr["index"] for fr in complete],
            "fixtures_fp32_skipped": skipped,
            "rows_compared": len(deltas),
            "max_abs_logprob_delta_bf16_vs_fp32": max(
                (abs(d["delta"]) for d in deltas), default=None),
            "mean_abs_logprob_delta_bf16_vs_fp32": (
                float(sum(abs(d["delta"]) for d in deltas) / len(deltas)) if deltas else None),
            "max_abs_logit_delta_bf16_vs_fp32": max(
                (abs(d["logit_delta"]) for d in deltas), default=None),
            "max_abs_lse_delta_bf16_vs_fp32": max(
                (abs(d["lse_delta"]) for d in deltas), default=None),
            "argmax_agreement_rows": sum(1 for d in deltas if d["argmax_agree"]),
            "rank_agreement_rows": sum(1 for d in deltas if d["rank_agree"]),
            "by_fixture": {str(fr["index"]): fr["delta"]["stats"] for fr in complete},
            "positions_per_fixture": {str(fr["index"]): len(fr["bf16"]["rows"]) for fr in fixture_results},
        }
        res["headline"] = headline
        log("A headline: %s" % json.dumps({k: v for k, v in headline.items()
                                           if k not in ("by_fixture", "positions_per_fixture")}))
        write_out(merge=True)

    # ---- B stage ---------------------------------------------------------
    if args.stage in ("b", "all"):
        if not tf_cache or any("h_bf16" not in tf_cache[fi] for fi in tf_cache):
            raise RuntimeError("B stage needs A-stage caches; run --stage all")
        W = model.lm_head.weight
        payload = run_native_comparison(model, fixtures, cont, tf_cache, args,
                                        logprob_caps, meta_checks, W, fp64)
        res["native_comparison"] = payload
        h = payload["headline"]
        log("B rows: %d (prompt %d, decode %d); max |native - tf_bf16| = %.6g" % (
            h["rows"], h["prompt_rows"], h["decode_rows"],
            h["max_abs_native_minus_tf_bf16"]))
        log("B rigorous bound holds: %s, max ratio %.6g, max bound %.6g" % (
            h["all_rows_within_bound"], h["max_bound_ratio"], h["max_bound"]))
        log("B measured bf16-vs-fp32 delta: max %.6g, within own bound: %s (max ratio %.6g)" % (
            h["max_measured_tf_bf16_minus_tf_fp32"], h["all_deltas_within_their_bound"],
            h["max_tf_bf16_vs_fp32_delta_bound"]))
        write_out(merge=True)

    res["complete"] = True
    res["qualification"] = evaluate_report(res)
    write_out(merge=False)
    log("done in %.1f s, peak rss %.1f GB" % (
        (datetime.now(timezone.utc) - t_start).total_seconds(), peak_rss_gb()))
    return {"passed": 0, "failed": 1, "not-run": 2}[res["qualification"]["status"]]


if __name__ == "__main__":
    sys.exit(main())
