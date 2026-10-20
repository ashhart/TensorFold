"""S5 (<plan>): B-2's sequences (ids.json from s5_export.py: 128 sequences, 144,413
tokens) through a serve's /v1/completions as token ids with prompt_logprobs 20 (1.1.1: nonstreaming only; such a
request scores its whole prompt and never resumes from the prompt cache), one request a sequence, greedy, max_tokens 1.
A reply's row t scores the sequence's token t from tokens 0..t-1 (row 0 is null): the given token's logprob and rank,
and the best 20 tokens' ids and logprobs in rank order. OUT gets one JSON line a sequence:
{"seq", "n", "given": [[logprob, rank], ...], "top": [[[id, logprob], ...], ...]} for rows 1..n-1. Then "exit 0|1".

    python3 s5_prompt_lp.py PORT IDS_JSON OUT
"""

import json
import sys
import time
import urllib.request

TOP = 20


def post(port, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main(port, ids_path, out):
    seqs = json.load(open(ids_path))
    bad = 0
    t0 = time.perf_counter()
    with open(out, "w") as f:
        for s, ids in enumerate(seqs):
            d = post(port, {"model": "local-model", "prompt": ids, "max_tokens": 1, "temperature": 0,
                            "prompt_logprobs": TOP})
            rows = d["choices"][0]["prompt_logprobs"]
            if len(rows) != len(ids) or rows[0] is not None:
                print(f"FAIL seq {s}: {len(rows)} rows for {len(ids)} tokens, row 0 {rows[0]!r}", flush=True)
                bad += 1
                continue
            given, top = [], []
            for t in range(1, len(ids)):
                row = rows[t]
                g = row[str(ids[t])]
                # each best token's rank is its place; the prompt token's counts the tokens at or above it, ties
                # included (vLLM's), so when it is among the best it takes the one place the others leave
                best = sorted((v["rank"], int(k), v["logprob"]) for k, v in row.items() if k != str(ids[t]))
                left = sorted(set(range(1, TOP + 1)) - {r for r, _, _ in best})
                if len(best) == TOP - 1 and len(left) == 1:
                    best = sorted(best + [(left[0], ids[t], g["logprob"])])
                if [r for r, _, _ in best] != list(range(1, TOP + 1)):
                    print(f"FAIL seq {s} row {t}: ranks {[r for r, _, _ in best]}, the prompt token's {g['rank']}", flush=True)
                    bad += 1
                    break
                given.append([g["logprob"], g["rank"]])
                top.append([[i, lp] for _, i, lp in best])
            f.write(json.dumps({"seq": s, "n": len(ids), "given": given, "top": top}) + "\n")
            if s % 16 == 15:
                print(f"{s + 1} sequences, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"{len(seqs)} sequences in {time.perf_counter() - t0:.0f} s, {bad} bad", flush=True)
    print("exit", 1 if bad else 0, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
