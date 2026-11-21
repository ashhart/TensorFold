"""Compare two bench_concurrent receipts; missing or unequal token evidence fails closed."""
import argparse
import json
import sys


def cells(report):
    result = {}
    for cell in report["cells"]:
        key = (cell["prompt"], cell["temperature"], cell["streams"])
        if key in result:
            raise ValueError("duplicate benchmark cell")
        result[key] = cell
    if not result:
        raise ValueError("empty benchmark receipt")
    return result


def hashes(cell):
    if cell["failed"] or cell["unmeasured"] or not cell["timing_valid"]:
        raise ValueError("failed, unmeasured or invalid-timing requests")
    for name in ("alone", "serial"):
        checks = cell[name]
        if checks["unequal"] or checks["failed"] or checks["equal"] < 1:
            raise ValueError(f"{name} token equality did not pass")
    result = []
    for rep in cell["reps"]:
        pairs = rep["comparisons"]
        if len(pairs) != cell["streams"]:
            raise ValueError("missing stream comparisons")
        for pair in pairs:
            values = []
            for mode in ("serial", "alone", "concurrent"):
                run = pair[mode]
                digest = run["token_sha256"]
                if (not run["complete"] or run["error"] or not run["tokens"]
                        or not isinstance(digest, str) or len(digest) != 64):
                    raise ValueError("missing complete full-token-hash evidence")
                int(digest, 16)
                values.append((run["tokens"], digest))
            if values[0] != values[1] or values[1] != values[2]:
                raise ValueError("serial, solo and concurrent tokens differ")
            result.append(values[2])
    if not result:
        raise ValueError("no measured replies")
    return result


def timing(cell):
    return {name: cell.get(name) for name in (
        "aggregate_tps", "per_stream_tps", "end_to_end_tps", "steady_tps", "ttft_s_max")}


def compare(before, after):
    if before["tokens"] != after["tokens"]:
        raise ValueError("different requested output lengths")
    if before.get("prompts") != after.get("prompts"):
        raise ValueError("different prompt identities")
    old, new = cells(before), cells(after)
    if old.keys() != new.keys():
        raise ValueError("different benchmark cells")
    rows = []
    for key, a in old.items():
        b = new[key]
        ah, bh = hashes(a), hashes(b)
        if ah != bh:
            raise ValueError(f"baseline/candidate token mismatch: {key}")
        rows.append({"prompt": key[0], "temperature": key[1], "streams": key[2],
                     "equal_replies": len(ah), "before": timing(a), "after": timing(b)})
    return {"token_equal": True, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    try:
        with open(args.before, encoding="utf-8") as f:
            before = json.load(f)
        with open(args.after, encoding="utf-8") as f:
            after = json.load(f)
        result = compare(before, after)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
