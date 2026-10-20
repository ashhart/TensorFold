"""Compare chunked Drama prompt probes against the same oracle, without requiring cross-chunk token equality."""
import argparse
import json
import math
from pathlib import Path
import re


METRIC = re.compile(r"^PREFILL (\S+) start=(\d+) rows=(\d+) worst_cos=(\S+) max_abs=(\S+)$")
EXPECTED = ({"backbone_input_embeds", "backbone_final_norm"}
            | {f"backbone_hidden.layer{i:02}" for i in range(40)}
            | {f"tower_prefill.layer{i:02}" for i in range(13)})


def collect(text):
    metrics = {}
    for line in text.splitlines():
        if not line.startswith("PREFILL "):
            continue
        match = METRIC.fullmatch(line)
        if not match:
            raise ValueError("malformed prefill metric")
        name, start, rows, cosine, error = match.groups()
        start, rows = int(start), int(rows)
        cosine, error = float(cosine), float(error)
        if not math.isfinite(cosine) or not math.isfinite(error) or error < 0 or rows == 0:
            raise ValueError("invalid prefill metric")
        item = metrics.setdefault(name, {"rows": 0, "worst_cos": 1.0, "max_abs": 0.0})
        if start != item["rows"]:
            raise ValueError("prompt probe gap or overlap")
        item["rows"] += rows
        item["worst_cos"] = min(item["worst_cos"], cosine)
        item["max_abs"] = max(item["max_abs"], error)
    if metrics.keys() != EXPECTED:
        raise ValueError("incomplete prompt probes")
    n = metrics["backbone_input_embeds"]["rows"]
    for name, item in metrics.items():
        if item["rows"] != n - int(name.startswith("tower_prefill.")):
            raise ValueError("incomplete prompt rows")
    if "RESULT drama3 check PASS:" not in text:
        raise ValueError("upstream reference check did not pass")
    return metrics


def compare(baseline, candidate):
    before, after = collect(baseline), collect(candidate)
    results = {}
    for name in sorted(EXPECTED):
        a, b = before[name], after[name]
        if a["rows"] != b["rows"]:
            raise ValueError("different prompt vectors")
        results[name] = {"one_row": a, "chunked": b,
                         "pass": b["worst_cos"] >= a["worst_cos"] and b["max_abs"] <= a["max_abs"]}
    return {"pass": all(item["pass"] for item in results.values()), "tensors": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("one_row", type=Path)
    parser.add_argument("chunked", type=Path)
    args = parser.parse_args()
    result = compare(args.one_row.read_text(), args.chunked.read_text())
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
