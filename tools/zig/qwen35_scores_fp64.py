"""Validate native scoring captures against FP64 dots of the same stored BF16 head and actual BF16 input rows."""
import argparse
import json
import math
from pathlib import Path


def gamma(depth):
    u = 2.0 ** -24
    return depth * u / (1 - depth * u)


def projection_reference(args, meta, meta_file, stem, index, np, torch, safe_open):
    rows, hidden, width = (meta[k] for k in ("rows", "hidden", "vocab"))
    raw = np.fromfile(args.captures / (stem + "hidden.bf16"), dtype="<u2").copy()
    x = torch.from_numpy(raw).view(torch.bfloat16).reshape(rows, hidden).to(args.device, torch.float64)
    raw_out = np.fromfile(args.captures / (stem + "output.bf16"), dtype="<u2").copy()
    native = torch.from_numpy(raw_out).view(torch.bfloat16).reshape(rows, width).to(args.device, torch.float64)
    key = meta["projection"]
    reference, magnitude = torch.empty_like(native), torch.empty_like(native)
    with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
        matrix = f.get_slice(key)
        for lo in range(0, width, 4096):
            hi = min(width, lo + 4096)
            stored = matrix[lo:hi]
            if stored.dtype != torch.bfloat16:
                raise ValueError("projection reference requires BF16 weights")
            w = stored.to(args.device, torch.float64)
            reference[:, lo:hi] = x @ w.T
            magnitude[:, lo:hi] = x.abs() @ w.abs().T
    work = gamma(math.ceil(hidden / 2048) * 8 + 13) * magnitude
    bound = work + 2.0 ** -8 * (reference.abs() + work) + 2.0 ** -134
    error = (native - reference).abs()
    ratios = error / bound
    passed = bool(torch.isfinite(native).all() and torch.isfinite(reference).all() and (error <= bound).all())
    worst = int(ratios.argmax())
    absolute = int(error.argmax())
    return {"capture": meta_file.name, "phase": meta["phase"], "projection": key, "rows": rows,
            "passed": passed, "worst_absolute": float(error.max()), "worst_bound_ratio": float(ratios.max()),
            "worst_bound_row": worst // width, "worst_bound_column": worst % width,
            "worst_absolute_row": absolute // width, "worst_absolute_column": absolute % width}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("captures", type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    import numpy as np
    import torch
    from safetensors import safe_open
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in index else "model.language_model.embed_tokens.weight"
    u = 2.0 ** -24
    records = []
    projections = []
    passed = True
    with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
        head = f.get_slice(key)
        for meta_file in sorted(args.captures.glob("*-meta.json"), key=lambda f: int(f.name.split("-")[0])):
            meta = json.loads(meta_file.read_text())
            stem = meta_file.name.removesuffix("meta.json")
            if "projection" in meta:
                result = projection_reference(args, meta, meta_file, stem, index, np, torch, safe_open)
                projections.append(result)
                passed &= result["passed"]
                continue
            rows, hidden, vocab = (meta[k] for k in ("rows", "hidden", "vocab"))
            raw = np.fromfile(args.captures / (stem + "hidden.bf16"), dtype="<u2").copy()
            x = torch.from_numpy(raw).view(torch.bfloat16).reshape(rows, hidden).to(args.device, torch.float64)
            native = torch.from_numpy(np.fromfile(args.captures / (stem + "logits.f32"), dtype="<f4").copy())
            native = native.reshape(rows, vocab).to(args.device, torch.float64)
            stats = np.fromfile(args.captures / (stem + "stats.u32"), dtype="<u4").reshape(rows, 43)
            ref = torch.empty_like(native)
            magnitude = torch.empty_like(native)
            for lo in range(0, vocab, 4096):
                hi = min(vocab, lo + 4096)
                stored = head[lo:hi]
                if stored.dtype != torch.bfloat16:
                    raise ValueError("reference requires stored BF16 weights")
                w = stored.to(args.device, torch.float64)
                ref[:, lo:hi] = x @ w.T
                magnitude[:, lo:hi] = x.abs() @ w.abs().T
            dot_depth = math.ceil(hidden / 2048) * 8 + 13
            work = gamma(dot_depth) * magnitude
            dot_bound = work + u * (ref.abs() + work) + 2.0 ** -150
            dot_error = (native - ref).abs()
            normalizer = torch.from_numpy(stats[:, 0].copy().view("<f4")).to(args.device, torch.float64)
            native_ref_lse = torch.logsumexp(native, dim=-1)
            red_work = gamma(math.ceil(vocab / 256) + 8) + 12 * u * native_ref_lse.abs()
            red_bound = red_work + u * (native_ref_lse.abs() + red_work) + 2.0 ** -150
            red_error = (normalizer - native_ref_lse).abs()
            ref_lse = torch.logsumexp(ref, dim=-1)
            native_lp = (native.float() - normalizer.float()[:, None]).double()
            ref_lp = ref - ref_lse[:, None]
            lp_bound = dot_bound + dot_bound.amax(dim=1)[:, None] + red_bound[:, None]
            lp_bound += u * (native.abs() + normalizer.abs()[:, None]) + 2.0 ** -150
            lp_error = (native_lp - ref_lp).abs()
            finite = bool(torch.isfinite(native).all() and torch.isfinite(ref).all() and torch.isfinite(normalizer).all())
            dot_ok = bool((dot_error <= dot_bound).all())
            red_ok = bool((red_error <= red_bound).all())
            lp_ok = bool((lp_error <= lp_bound).all())
            picks = torch.tensor(meta["picks"], device=args.device)
            row_ids = torch.arange(rows, device=args.device)
            selected = native[row_ids, picks].float().cpu().numpy().view("<u4")
            selected_ok = bool(np.array_equal(stats[:, 1], selected))
            ranks = (native >= native[row_ids, picks, None]).sum(dim=1).cpu().numpy()
            rank_ok = bool(np.array_equal(stats[:, 42], ranks))
            top_ids = torch.argsort(native, dim=1, descending=True, stable=True)[:, :20]
            top_ok = bool(np.array_equal(stats[:, 2:22], top_ids.cpu().numpy()))
            top_values = native.gather(1, top_ids).float().cpu().numpy().view("<u4")
            top_ok = top_ok and bool(np.array_equal(stats[:, 22:42], top_values))
            ratio = lp_error / lp_bound
            worst = int(ratio.argmax())
            head_absolute = int(dot_error.argmax())
            head_ratio = int((dot_error / dot_bound).argmax())
            lp_absolute = int(lp_error.argmax())
            native_best, ref_best = native.argmax(1), ref.argmax(1)
            top_ref = torch.topk(ref, 2, dim=1).values
            differences = [{"row": r, "native_token": int(native_best[r]), "fp64_token": int(ref_best[r]),
                            "fp64_margin": float(top_ref[r, 0] - top_ref[r, 1])}
                           for r in range(rows) if native_best[r] != ref_best[r]]
            result = {"capture": meta_file.name, "phase": meta["phase"], "rows": rows,
                      "finite": finite, "head_bound_passed": dot_ok, "reduction_bound_passed": red_ok,
                      "logprob_bound_passed": lp_ok, "selected_exact": selected_ok, "top_exact": top_ok,
                      "rank_exact": rank_ok, "worst_head_absolute": float(dot_error.max()),
                      "worst_head_bound_ratio": float((dot_error / dot_bound).max()),
                      "worst_head_absolute_row": head_absolute // vocab, "worst_head_absolute_token": head_absolute % vocab,
                      "worst_head_bound_row": head_ratio // vocab, "worst_head_bound_token": head_ratio % vocab,
                      "worst_reduction_absolute": float(red_error.max()), "worst_reduction_row": int(red_error.argmax()),
                      "worst_reduction_bound_ratio": float((red_error / red_bound).max()),
                      "worst_logprob_absolute": float(lp_error.max()), "worst_logprob_bound_ratio": float(ratio.max()),
                      "worst_logprob_bound_row": worst // vocab, "worst_logprob_bound_token": worst % vocab,
                      "worst_logprob_absolute_row": lp_absolute // vocab, "worst_logprob_absolute_token": lp_absolute % vocab,
                      "argmax_differences": len(differences), "argmax_difference_rows": differences,
                      "fp64_top_margins": (top_ref[:, 0] - top_ref[:, 1]).cpu().tolist()}
            passed &= all((finite, dot_ok, red_ok, lp_ok, selected_ok, top_ok, rank_ok))
            records.append(result)
    report = {"schema": 1, "checkpoint": "Qwen/Qwen3.5-27B", "revision": args.revision,
              "reference": "FP64 head dots and full-vocabulary logsumexp on identical native BF16 normalized rows; not a full-model FP64 forward",
              "bound": {"fp32_unit": u, "head_depth": "ceil(hidden/2048)*8+13",
                        "head": "gamma(depth)*sum(abs(x*w)) plus FP32 output rounding",
                        "reduction_depth": "ceil(vocab/256)+8", "post_reduction_ops": 12,
                        "logprob": "per-token head bound + row-max head bound + reduction bound + FP32 subtraction rounding"},
              "torch": torch.__version__, "cuda": torch.version.cuda, "records": records, "projections": projections,
              "prompt_rows": sum(r["rows"] for r in records if r["phase"] == "prompt"),
              "decode_rows": sum(r["rows"] for r in records if r["phase"] == "decode"),
              "passed": passed and bool(records)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "prompt_rows": report["prompt_rows"], "decode_rows": report["decode_rows"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
