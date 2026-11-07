#!/usr/bin/env python3
"""Evaluate the recorded BF16-distance comparison without running a model."""
import argparse
import json
import math
from pathlib import Path


def evaluate_report(report):
    """Keep numerical diagnostics distinct from the measured BF16-distance envelope."""
    comparison = report.get("native_comparison") or {}
    rows = comparison.get("rows", [])
    failures = []
    checks = []
    if not report.get("complete") or not rows or not comparison.get("fp32_available"):
        return {"status": "not-run", "reason": "complete native/BF16/FP32 rows required"}
    for row in rows:
        try:
            native = float(row["native_logprob"])
            bf16 = float(row["transformers_bf16_logprob"])
            fp32 = float(row["transformers_fp32_logprob"])
            terms = row["bound_terms"]
            allowance = float(terms["native_vs_fp64_box"]) + float(terms["rounding_slack"])
            if not all(math.isfinite(v) for v in (native, bf16, fp32, allowance)) or allowance < 0:
                raise ValueError("invalid numeric evidence")
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"status": "not-run", "reason": "missing or non-finite per-row evidence"}
        distance = abs(bf16 - fp32)
        item = {"capture": row.get("capture"), "phase": row.get("phase"),
                "fixture": row.get("fixture"), "position": row.get("position"),
                "native_to_fp32": abs(native - fp32), "bf16_to_fp32": distance,
                "existing_head_rounding_allowance": allowance,
                "fp32_envelope_passed": abs(native - fp32) <= distance + allowance,
                "bf16_envelope_passed": abs(native - bf16) <= distance + allowance}
        checks.append(item)
        if not item["fp32_envelope_passed"] or not item["bf16_envelope_passed"]:
            failures.append(item)
    numerical = comparison.get("headline", {}).get("all_rows_within_bound") is True
    return {"status": "passed" if numerical and not failures else "failed",
            "scope": "captured native rows, not a direct HTTP prompt-score comparison",
            "policy": "both reference directions reported; maintainer acceptance remains outstanding",
            "allowance": "existing native FP64 head box plus recorded subtraction slack; no hidden-state term",
            "rows": len(rows), "numerical_bound_passed": numerical,
            "fp32_envelope_failures": sum(not r["fp32_envelope_passed"] for r in checks),
            "bf16_envelope_failures": sum(not r["bf16_envelope_passed"] for r in checks),
            "outside_rows": failures}



def public_report(value):
    """Remove identifying runner metadata while preserving numeric evidence."""
    replacements = {"host": "redacted", "hostname": "redacted", "model_dir": "<checkpoint>",
                    "captures_dir": "<captures>", "executable": "python3", "path": "<local-file>"}
    if isinstance(value, dict):
        return {key: (["qwen35_transformers_reference.py", "<see recipe for inputs>"] if key == "argv"
                      else replacements[key] if key in replacements else public_report(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [public_report(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = evaluate_report(json.loads(args.report.read_text()))
    print(json.dumps(result, indent=2))
    return {"passed": 0, "failed": 1, "not-run": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
