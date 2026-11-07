"""Exercise native prompt_logprobs: solo/concurrent bits, drafted/plain replies, admission and recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request


def digest_rows(rows):
    h = hashlib.sha256()
    for row in rows[1:]:
        for key, cell in sorted(row.items(), key=lambda item: int(item[0])):
            if not math.isfinite(cell["logprob"]):
                raise ValueError("nonfinite_prompt_logprob")
            h.update(struct.pack("<IfI", int(key), cell["logprob"], cell["rank"]))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("fixtures", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    records = []
    failures = []

    def request(ids, draft=True, top=2, temperature=0, extra=None):
        body = {"model": args.model, "prompt": ids, "temperature": temperature, "seed": 1234,
                "max_tokens": args.tokens, "draft": draft, "ignore_eos": True, "return_token_ids": True}
        if top is not None:
            body["prompt_logprobs"] = top
        body.update(extra or {})
        start = time.perf_counter()
        req = urllib.request.Request(args.base.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            response = json.load(r)
        elapsed = time.perf_counter() - start
        runtime = response["tensorfold"]
        rows = response["choices"][0].get("prompt_logprobs")
        if top is not None:
            if len(rows) != len(ids) or rows[0] is not None:
                raise ValueError("prompt_row_count_or_null_first")
            for token, row in zip(ids[1:], rows[1:]):
                if str(token) not in row or row[str(token)]["rank"] < 1:
                    raise ValueError("selected_prompt_token_missing")
        return {"prompt_tokens": len(ids), "top": top, "drafts": runtime["drafts"],
                "temperature": temperature, "elapsed_s": elapsed,
                "prompt_score_sha256": digest_rows(rows) if rows is not None else None,
                "token_ids": runtime["token_ids"], "token_sha256": runtime["token_sha256"],
                "cached_tokens": response["usage"]["prompt_tokens_details"]["cached_tokens"],
                "prefill_s": runtime["prefill_seconds"], "decode_tps": runtime["tokens_per_second"]}

    def compare(label, got, want, scores=True):
        equal = got["token_ids"] == want["token_ids"]
        if scores:
            equal &= got["prompt_score_sha256"] == want["prompt_score_sha256"]
        if not equal:
            failures.append(label)
        records.append({"check": label, "passed": equal, **got})

    warm = request(fixtures[0])
    solo = []
    for i, ids in enumerate(fixtures):
        plain = request(ids, draft=False)
        solo.append(plain)
        compare(f"drafted_plain_{i}", request(ids), plain)
        compare(f"repeat_{i}", request(ids), plain)
        compare(f"scored_unscored_tokens_{i}", request(ids, top=None), plain, scores=False)
    for temperature in (0, 0.7):
        items = fixtures[:4]
        want = [request(ids, temperature=temperature) for ids in items]
        begin = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            got = list(pool.map(lambda ids: request(ids, temperature=temperature), items))
        elapsed = time.perf_counter() - begin
        for i, (actual, expected) in enumerate(zip(got, want)):
            compare(f"concurrent_solo_{temperature}_{i}", actual, expected)
        records.append({"check": "concurrency_workload", "temperature": temperature, "requests": len(items),
                        "elapsed_s": elapsed, "requests_per_second": len(items)/elapsed,
                        "scored_prompt_tokens_per_second": sum(len(x)-1 for x in items)/elapsed,
                        "generated_tokens_per_second": sum(len(x["token_ids"]) for x in got)/elapsed})
    for k in (0, 1, 20):
        result = request(fixtures[1], top=k)
        if result["token_ids"] != solo[1]["token_ids"]:
            failures.append(f"top_{k}_changes_tokens")
        records.append({"check": f"top_{k}", **result})
    for label, ids, extra in (("invalid_token", [4294967295], {}),
                              ("unsupported_stream", fixtures[0], {"stream": True}),
                              ("invalid_top", fixtures[0], {"prompt_logprobs": 21})):
        try:
            request(ids, extra=extra)
            failures.append(label + "_not_refused")
        except urllib.error.HTTPError as exc:
            records.append({"check": label, "http_status": exc.code, "passed": exc.code == 400})
            if exc.code != 400:
                failures.append(label + "_wrong_status")
    compare("after_refusal_recovery", request(fixtures[1]), solo[1])
    report = {"schema": 1, "warmups_excluded": 1, "cache": "server configured off; counters retained",
              "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              "fixture_lengths": [len(x) for x in fixtures], "records": records,
              "failures": failures, "passed": not failures}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(records), "failures": failures}))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
