"""Check the frozen benchmark evidence without a running model."""

import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
PATCH_SHA256 = "a4e0ec3200bce7cecb4fbb8372bcae953dd35e2d36c4a7d9475fd17667faa9ec"


def read(name):
    return json.loads((HERE / "evidence" / f"{name}.json").read_text())


def check(condition, detail):
    if not condition:
        raise SystemExit(f"Receipt mismatch: {detail}")


def near(measured, expected, detail):
    check(math.isclose(measured, expected, rel_tol=1e-8), detail)


def main():
    check(hashlib.sha256((HERE / "metal-1017.patch").read_bytes()).hexdigest() == PATCH_SHA256, "source patch digest")
    data = read("decode")
    summary = read("summary")
    check(data["health"]["server_lanes"] == 4, "decode lane count")
    counts = {1: 6, 4: 3, 8: 3, 16: 3}
    check([r["level"] for r in summary["decode"]] == list(counts), "concurrency levels")
    concurrent = 0
    for reported in summary["decode"]:
        level = reported["level"]
        bursts = [b for b in data["bursts"] if b["level"] == level]
        check(len(bursts) == counts[level], f"C{level} burst count")
        rates = []
        end_to_end = []
        for b in bursts:
            streams = b["streams"]
            check(len(streams) == b["requests"] == level, f"C{level} stream count")
            for s in streams:
                ref = data["reference"][s["name"]]
                check(s["tokens"] == 256 and len(s["output_ids_sha256"]) == 64 and
                      s["output_ids_sha256"] == ref["output_ids_sha256"], f"C{level} token identity, {s['name']}")
            check(b["exact_count"] == level, f"C{level} exact count")
            last = max(s["last"] for s in streams)
            first = min(s["first"] for s in streams)
            start = min(s["start"] for s in streams)
            rates.append(sum(s["tokens"] - 1 for s in streams) / (last - first))
            end_to_end.append(sum(s["tokens"] for s in streams) / (last - start))
        near(statistics.median(rates), reported["aggregate_decode_tps"], f"C{level} decode rate")
        near(statistics.median(end_to_end), reported["aggregate_end_to_end_tps"], f"C{level} end-to-end rate")
        if level > 1:
            concurrent += level * len(bursts)
    check(concurrent == summary["matched_solo"] == summary["concurrent_replies"] == 84, "concurrent totals")

    pp, coding = read("pp"), read("coding")
    check(len(pp) == 12 and len(coding) == 3, "prefill sample counts")
    groups = {str(size): [r for r in pp if r["target"] == size] for size in (2048, 8192, 16384, 32768)}
    groups["17639 coding"] = coding
    check([r["label"] for r in summary["prefill"]] == list(groups), "prefill labels")
    for reported in summary["prefill"]:
        label = reported["label"]
        rows = groups[label]
        check(len(rows) == 3, f"{label} sample count")
        rates = []
        for r in rows:
            check(0 <= r["cached_tokens"] <= 2, f"{label} cache bound")
            rate = (r["prompt_tokens"] - r["cached_tokens"]) / r["prefill_seconds"]
            near(rate, r["prefill_tps"], f"{label} individual rate")
            rates.append(rate)
        near(statistics.median(rates), reported["median_prefill_tps"], f"{label} prefill rate")
        near(statistics.median(r["ttft_seconds"] for r in rows), reported["median_ttft_seconds"], f"{label} TTFT")
    check(all(r["prompt_tokens"] == 17639 and r["completion_tokens"] == 1024 for r in coding), "coding shape")
    near(statistics.median(r["decode_tps"] for r in coding), summary["coding_decode_tps"], "coding decode")
    print(f"Verified {concurrent} concurrent token hashes and {len(pp) + len(coding)} prefill measurements.")


if __name__ == "__main__":
    main()
