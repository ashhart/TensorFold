#!/usr/bin/env python3
"""Public Qwen BF16 reference: fixture."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import torch
from qwen35_reference_inputs import (
    log,
    peak_rss_gb,
    tensor_bytes_hash,
)
from qwen35_reference_forward import (
    head_logits,
    library_forward,
    manual_forward,
    score_rows,
)

def run_fixture(model, fixture_index, ids, cont, args, tf_cache, verify_layers):
    L = len(ids)
    if args.skip_long and L > 1024:
        log("fixture %d: length %d skipped by --skip-long" % (fixture_index, L))
        return {"index": fixture_index, "length": L, "skipped": True,
                "ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
                "reason": "--skip-long: %d-token fixture not measured in this run" % L,
                "bf16": {"rows": []}, "fp32": {"skipped": True},
                "delta": {"stats": {}, "rows": []}}
    # Teacher-force the same five continuation tokens; short fixtures verify prompt invariance.
    ext = list(ids) + list(cont.get(fixture_index, [])[:5])
    log("fixture %d: prompt length %d, forward length %d (prompt + 5 served continuation tokens)" % (
        fixture_index, L, len(ext)))
    t0 = datetime.now(timezone.utc)
    ids_t = torch.tensor(ext, dtype=torch.long)
    prompt_positions = list(range(1, L))
    W = model.lm_head.weight

    # --- BF16 arm ---------------------------------------------------------
    h_bf16, st_bf16 = manual_forward(model, ids_t.view(1, -1), upcast=False)
    h_bf16 = h_bf16[0]
    rows_bf16, score_info_bf16 = score_rows(
        W, h_bf16, ids_t, prompt_positions,
        row_chunk=args.row_chunk, vocab_chunk=args.head_vocab_chunk)

    # --- prompt-only re-forward: the extension must be bitwise inert ------
    if len(ext) <= 512:
        h_prompt, _ = manual_forward(model, torch.tensor(list(ids), dtype=torch.long).view(1, -1),
                                     upcast=False)
        h_prompt = h_prompt[0]
        n_cmp = min(h_prompt.shape[0], h_bf16.shape[0])
        ext_invariance = {
            "prompt_only_rows": int(h_prompt.shape[0]),
            "forward_rows": int(h_bf16.shape[0]),
            "compared_rows": int(n_cmp),
            "max_abs_diff": float((h_prompt[:n_cmp].float() - h_bf16[:n_cmp].float()).abs().max()),
            "exact_equal": bool(torch.equal(h_prompt[:n_cmp], h_bf16[:n_cmp])),
        }
        del h_prompt
    else:
        ext_invariance = {"skipped": True,
                          "reason": "prompt-only re-forward skipped for the %d-token fixture to "
                                    "bound CPU cost; prompt rows are the first %d rows of a causal "
                                    "forward and fixtures 1/32/129/257 verify the 5-token extension "
                                    "is bitwise inert" % (L, L)}

    # --- library-equivalence re-forward (short fixtures only) -------------
    if len(ext) <= 512:
        lib_h, lib_logits = library_forward(model, ids_t.view(1, -1), logits_to_keep=0)
        f32_first_row = head_logits(h_bf16[:1], W, args.head_vocab_chunk)
        bf16_first_row = model.lm_head(h_bf16[:1])
        hidden_equiv = {
            "skipped": False,
            "hidden_max_abs_diff": float((lib_h[0].float() - h_bf16.float()).abs().max()),
            "hidden_exact_equal": bool(torch.equal(lib_h[0], h_bf16)),
            "library_logits_rows": int(lib_logits.shape[0]),
            "first_row_logits_compare": {
                "bf16_module_path_max_abs_diff": float(
                    (bf16_first_row.float() - lib_logits[0, :1].float()).abs().max()),
                "bf16_module_path_exact_equal": bool(
                    torch.equal(bf16_first_row.float(), lib_logits[0, :1].float())),
                "fp32_gemm_path_max_abs_diff": float(
                    (f32_first_row - lib_logits[0, :1].float()).abs().max()),
                "note": "library logits are the BF16 module output; the reported logprobs use the "
                        "float32 GEMM path",
            },
        }
        del lib_logits, lib_h, f32_first_row, bf16_first_row
    else:
        hidden_equiv = {"skipped": True,
                        "reason": "forward length %d > 512: library-equivalence re-forward skipped "
                                  "to bound CPU cost; bitwise equivalence is verified on fixtures "
                                  "0/1/32/129/257" % len(ext)}

    # --- head-path checks (on the smallest fixture) -----------------------
    head_checks = None
    if fixture_index == 0:
        bf16_module_head = model.lm_head(h_bf16[:2])
        fp32_gemm_head = head_logits(h_bf16[:2], W, args.head_vocab_chunk)
        head_checks = {"bf16_dtype_head_vs_fp32_gemm": {
            "rows": int(h_bf16[:2].shape[0]),
            "bf16_module_head_max_abs_logit_diff": float(
                (bf16_module_head.float() - fp32_gemm_head).abs().max()),
            "bf16_module_head_relative": float(
                ((bf16_module_head.float() - fp32_gemm_head).abs()
                 / (bf16_module_head.float().abs() + 1e-30)).max()),
            "note": "a BF16 head matmul returns BF16 logits; the reported logprobs use the float32 "
                    "GEMM over the same promoted BF16-stored weights, so the delta between the two "
                    "arms isolates activation/accumulation precision rather than logit storage",
        }}
        del bf16_module_head, fp32_gemm_head

    # --- FP32 arm ---------------------------------------------------------
    skip_fp32 = (len(ext) > 512) and args.skip_fp32_4097
    fp32 = {"skipped": False}
    h_f32 = None
    rows_fp32 = []
    verify_report = []
    if skip_fp32:
        fp32 = {"skipped": True,
                "reason": "forward length %d > 512 and --skip-fp32-4097 is set (a whole-model "
                          "float32 arm would require a second resident copy on a host whose RAM is "
                          "shared with two live GPU serves); the BF16 arm still covers this fixture"
                          % len(ext)}
        log("fixture %d: FP32 arm skipped by policy" % fixture_index)
    else:
        layer_set = set(verify_layers)
        h32, st32 = manual_forward(model, ids_t.view(1, -1), upcast=True, verify_layers=layer_set,
                                   verify_report=verify_report, track_params=True)
        h_f32 = h32[0].float()
        del h32
        fp32 = {
            "skipped": False,
            "upcast_layers": st32["layers"],
            "params_cast": st32["cast"],
            "params_already_float32": st32["kept_f32"],
            "parameter_dtypes_after": st32.get("parameter_dtypes_after"),
            "layer_float_equivalence": verify_report,
            "upcast_semantics": "per-layer streaming promotion: each decoder layer's stored BF16 "
                                "parameters/buffers are promoted to float32, the layer runs, then the "
                                "stored BF16 values are restored before the next layer; the final norm "
                                "is promoted the same way and the BF16-stored lm_head is promoted to "
                                "float32 for the scoring matmul. BF16->FP32 promotion is exact, so this "
                                "is float32 arithmetic over the checkpoint's stored values -- it is NOT "
                                "a trained float32 checkpoint and no claim of exact fp32-partial sums is "
                                "made for any single fused kernel.",
            "params_resident_fp32": "one decoder layer at a time (peak transient ~%.1f GB)" % (
                sum(p.numel() for p in model.model.language_model.layers[0].parameters()) * 2 / 1e9),
        }
        rows_fp32, score_info_fp32 = score_rows(
            W, h_f32, ids_t, prompt_positions,
            row_chunk=args.row_chunk, vocab_chunk=args.head_vocab_chunk)
        fp32["score_compare"] = score_info_fp32
        fp32["lm_head_dtype_during_scoring"] = str(W.dtype)
        if fixture_index == 0 and head_checks is not None:
            # Compare one promoted FP32 vocabulary chunk without allocating the full FP32 head.
            vchunk = min(int(args.head_vocab_chunk), int(W.shape[0]))
            w32_chunk = W.data[:vchunk].to(torch.float32)
            ref_chunk = torch.nn.functional.linear(h_f32[:2], w32_chunk)
            ours_chunk = head_logits(h_f32[:2], W, vchunk)[:, :vchunk]
            head_checks["lm_head_promotion_vs_fp32_gemm"] = {
                "rows": int(h_f32[:2].shape[0]),
                "vocab_chunk": vchunk,
                "max_abs_logit_diff": float((ref_chunk - ours_chunk).abs().max()),
                "exact_equal": bool(torch.equal(ref_chunk, ours_chunk)),
                "note": "float32 linear over an explicitly promoted vocabulary chunk vs the "
                        "chunked float32 GEMM path used for the reported logprobs; the full "
                        "float32 lm_head copy is never materialized",
            }
            del w32_chunk, ref_chunk, ours_chunk

    # --- per-position delta between the two arms --------------------------
    n = min(len(rows_bf16), len(rows_fp32)) if rows_fp32 else 0
    deltas = []
    for i in range(n):
        a, b = rows_bf16[i], rows_fp32[i]
        if a["position"] != b["position"]:
            raise RuntimeError("row order mismatch in fixture %d" % fixture_index)
        deltas.append({"position": a["position"], "token_id": a["token_id"],
                       "bf16_logprob": a["logprob"], "fp32_logprob": b["logprob"],
                       "delta": a["logprob"] - b["logprob"],
                       "logit_delta": a["logit"] - b["logit"],
                       "lse_delta": a["lse"] - b["lse"],
                       "argmax_agree": a["top_ids"][0] == b["top_ids"][0],
                       "rank_agree": a["rank"] == b["rank"]})
    delta_stats = {}
    if deltas:
        dabs = [abs(d["delta"]) for d in deltas]
        dlog = [abs(d["logit_delta"]) for d in deltas]
        dlse = [abs(d["lse_delta"]) for d in deltas]
        delta_stats = {
            "rows": len(deltas),
            "max_abs_logprob_delta": max(dabs),
            "mean_abs_logprob_delta": float(sum(dabs) / len(dabs)),
            "max_abs_logit_delta": max(dlog),
            "max_abs_lse_delta": max(dlse),
            "argmax_agreement_rows": sum(1 for d in deltas if d["argmax_agree"]),
            "rank_agreement_rows": sum(1 for d in deltas if d["rank_agree"]),
        }

    # --- cache the normalized hidden rows for the native comparison ------
    hidden_sha = tensor_bytes_hash([h_bf16])
    cache = {"fixture": fixture_index, "prompt_len": L, "ext_ids": ext,
             "h_bf16": h_bf16.float().clone()}
    if h_f32 is not None:
        cache["h_fp32"] = h_f32.clone()
    tf_cache[fixture_index] = cache
    del h_bf16, h_f32

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    result = {
        "index": fixture_index,
        "length": L,
        "ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
        "forward_ids_length": len(ext),
        "forward_ids_sha256": hashlib.sha256(json.dumps(list(ext)).encode()).hexdigest(),
        "forward_note": "prompt ids plus the first 5 served greedy continuation tokens (teacher "
                        "forced, no cache); the prompt rows are the first %d rows of a causal forward"
                        % L,
        "prompt_extension_invariance": ext_invariance,
        "rows_note": "rows cover scored prompt positions 1..%d (position p scores token ids[p] from "
                     "the hidden row at position p-1); position 0 is not scored, matching the served "
                     "prompt_logprobs row-0 null" % (L - 1),
        "bf16": {"hidden_sha256": hidden_sha,
                 "hidden_rows": int(len(ext)),
                 "forward_stats": {k: v for k, v in st_bf16.items() if k != "layers"},
                 "hidden_equivalence_vs_library_forward": hidden_equiv,
                 "rows": rows_bf16},
        "fp32": fp32,
        "fp32_rows": rows_fp32,
        "delta": {"stats": delta_stats, "rows": deltas},
        "head_path_checks": head_checks,
        "elapsed_s": elapsed,
        "peak_rss_gb": peak_rss_gb(),
    }
    return result


def load_head_from_safetensors(model_dir):
    from safetensors import safe_open
    index = json.loads(open(os.path.join(model_dir, "model.safetensors.index.json"), "r").read())
    wm = index["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in wm else "model.language_model.embed_tokens.weight"
    with safe_open(os.path.join(model_dir, wm[key]), framework="pt", device="cpu") as f:
        head = f.get_tensor(key)
    return head


def compare_fp64_records(fp64, records):
    'Compare locally recomputed FP64 bound metrics with the recorded ones'
    by_capture = {}
    for rec in fp64.get("records", []):
        by_capture[rec["capture"]] = rec
    fields = ("finite", "head_bound_passed", "reduction_bound_passed", "logprob_bound_passed",
              "argmax_differences")
    scalar = ("worst_head_absolute", "worst_head_bound_ratio", "worst_reduction_absolute",
              "worst_logprob_absolute", "worst_logprob_bound_ratio")
    out = {"records_checked": 0, "missing": [], "boolean_mismatches": [],
           "scalar_max_abs_diff": {}, "all_booleans_match": True}
    for rec in records:
        name = rec.get("capture_name")
        if name not in by_capture:
            out["missing"].append(name)
            continue
        ref = by_capture[name]
        out["records_checked"] += 1
        for f in fields:
            if bool(ref.get(f)) != bool(rec.get(f)):
                out["boolean_mismatches"].append({"capture": name, "field": f,
                                                  "recorded": ref.get(f), "recomputed": rec.get(f)})
                out["all_booleans_match"] = False
        for f in scalar:
            a = float(ref.get(f, float("nan")))
            b = float(rec.get(f, float("nan")))
            d = abs(a - b)
            prev = out["scalar_max_abs_diff"].get(f, 0.0)
            out["scalar_max_abs_diff"][f] = max(prev, d)
    out["all_scalars_match_tolerance_1e-12"] = all(
        v <= 1e-12 for v in out["scalar_max_abs_diff"].values())
    return out
