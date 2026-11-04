#!/usr/bin/env python3
"""Collect native HTTP prompt scores and compare both pinned Transformers precision arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request


def compare_rows(ids, response, reference):
    """Require exact row/ID/cache coverage; report numerical distances without inventing a tolerance."""
    rows = response["choices"][0]["prompt_logprobs"]
    usage = response["usage"]
    if len(rows) != len(ids) or rows[0] is not None or usage["prompt_tokens"] != len(ids):
        raise ValueError("HTTP prompt row/usage coverage mismatch")
    if usage["prompt_tokens_details"]["cached_tokens"] != 0:
        raise ValueError("reference request reused prompt state")
    bf16 = {r["position"]: r for r in reference["bf16"]["rows"]}
    fp32 = {r["position"]: r for r in reference.get("fp32_rows", [])}
    expected = set(range(1, len(ids)))
    if set(bf16) != expected or set(fp32) != expected:
        raise ValueError("both Transformers arms must cover every scored prompt row")
    out = []
    for pos in range(1, len(ids)):
        token = ids[pos]
        cell = rows[pos][str(token)]
        b, f = bf16[pos], fp32[pos]
        if b["token_id"] != token or f["token_id"] != token or cell["rank"] < 1:
            raise ValueError("selected token/rank mismatch")
        native, low, high = (float(v) for v in (cell["logprob"], b["logprob"], f["logprob"]))
        if not all(math.isfinite(v) for v in (native, low, high)):
            raise ValueError("non-finite prompt score")
        distance = abs(low - high)
        out.append({"position": pos, "token_id": token, "native_logprob": native,
                    "transformers_bf16_logprob": low, "transformers_fp32_logprob": high,
                    "native_to_fp32": abs(native - high), "native_to_bf16": abs(native - low),
                    "bf16_to_fp32": distance,
                    "within_fp32_distance": abs(native - high) <= distance,
                    "within_bf16_distance": abs(native - low) <= distance,
                    "native_rank": cell["rank"], "transformers_bf16_rank": b["rank"],
                    "transformers_fp32_rank": f["rank"]})
    return out



def response_evidence(response):
    """Retain numeric score/runtime evidence without generated prose or identifying request metadata."""
    rows = response["choices"][0]["prompt_logprobs"]
    scores = [None if row is None else {token: {k: cell[k] for k in ("logprob", "rank")}
                                      for token, cell in row.items()} for row in rows]
    runtime = response.get("tensorfold", {})
    keys = ("token_ids", "token_sha256", "prefill_seconds", "tokens_per_second", "drafts")
    return {"choices": [{"prompt_logprobs": scores}], "usage": response["usage"],
            "tensorfold": {k: runtime[k] for k in keys if k in runtime}}


def summarize(rows):
    """Retain the worst observed position and separate each comparison direction."""
    if not rows:
        return {"rows": 0, "null_first_row_verified": True}
    return {"rows": len(rows),
            "worst_native_to_fp32": max(rows, key=lambda r: r["native_to_fp32"]),
            "worst_native_to_bf16": max(rows, key=lambda r: r["native_to_bf16"]),
            "worst_bf16_to_fp32": max(rows, key=lambda r: r["bf16_to_fp32"]),
            "mean_native_to_fp32": sum(r["native_to_fp32"] for r in rows) / len(rows),
            "mean_native_to_bf16": sum(r["native_to_bf16"] for r in rows) / len(rows),
            "outside_fp32_distance": sum(not r["within_fp32_distance"] for r in rows),
            "outside_bf16_distance": sum(not r["within_bf16_distance"] for r in rows),
            "rank_agreement_bf16": sum(r["native_rank"] == r["transformers_bf16_rank"] for r in rows),
            "rank_agreement_fp32": sum(r["native_rank"] == r["transformers_fp32_rank"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    oracle = json.loads(args.reference.read_text())
    if not oracle.get("complete") or oracle["checkpoint"]["revision"] != "fc05daec18b0a78c049392ed2e771dde82bdf654":
        raise ValueError("incomplete reference or wrong checkpoint revision")
    refs = {r["index"]: r for r in oracle["fixtures"]}
    records = []
    for index, ids in enumerate(fixtures):
        body = {"model": args.model, "prompt": ids, "prompt_logprobs": 20, "max_tokens": 1,
                "draft": False, "temperature": 0, "seed": 1234, "ignore_eos": True,
                "return_token_ids": True}
        request = urllib.request.Request(args.base.rstrip("/") + "/v1/completions",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=args.timeout) as stream:
            response = json.load(stream)
        elapsed = time.perf_counter() - start
        rows = compare_rows(ids, response, refs[index])
        records.append({"fixture": index, "length": len(ids), "elapsed_s": elapsed,
                        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        "summary": summarize(rows), "rows": rows,
                        "request": body, "response": response_evidence(response)})
        print(json.dumps({"fixture": index, **summarize(rows)}), flush=True)
    report = {"schema": 1, "collection_passed": True, "checkpoint": oracle["checkpoint"]["revision"],
              "scope": "actual HTTP selected prompt logprobs; both Transformers arms cover every scored row",
              "comparison_policy": "measured distances only; cross-engine tolerance is not a required pass in the agreed issue gate",
              "required_gate": "same-weight FP64 head/reduction checks and registered controls are qualified separately",
              "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "records": records}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
