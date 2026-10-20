"""Measures that compare checkpoints on one engine with the same inputs (10-19 08:4x: the drafted tok/s of
bench_openai also measures how well each checkpoint's own text drafts, so it cannot compare checkpoints alone).

    python3 openai_aligned.py texts BASE MODEL OUT        bench_openai's two prompts, greedy with drafts, 256 tokens
                                                          and seed 1234 as it sends them; each reply's text and hashes
    python3 openai_aligned.py nodraft BASE MODEL [flags]  bench_openai itself, with "draft": false in each request body
                                                          (its prompts, ignore_eos, seeds, timing span and report)
    python3 openai_aligned.py tokens N SKIP DC_JSON OUT   the decode-check prompts' token ids repeated, the first SKIP
                                                          dropped and cut to N (the prompts share a 2,054-token preamble,
                                                          so a second file skips into it and shares no prefix)

Run beside upstream's tools/bench_openai.py, bench_evidence.py and bench_auth.py. Standard library only.
"""
import json
import sys

import bench_openai as bo


class DraftOff:
    """bench_openai's json, with "draft": false added to each request body (the one dict with max_tokens)."""

    def __getattr__(self, name):
        return getattr(json, name)

    @staticmethod
    def dumps(o, *a, **k):
        if isinstance(o, dict) and "max_tokens" in o:
            o = {**o, "draft": False}
        return json.dumps(o, *a, **k)


def texts(base, model, out):
    rows = []
    for item in bo.PROMPTS:
        r = bo.stream(base.rstrip("/"), model, item, 256, 0.0, 1234)
        rows.append({"prompt": item["name"], **{k: r.get(k) for k in ("tokens", "token_sha", "output_sha256", "decode_tps")},
                     "text": r["text"]})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "text"}), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)


def tokens(n, skip, dc, out):
    kinds = json.load(open(dc))["kinds"]
    seq = []
    while len(seq) < n + skip:
        for x in kinds:
            seq.extend(x["prompt"])
    seq = seq[skip:skip + n]
    with open(out, "w") as f:
        json.dump(seq, f)
    print(f"{out}: {n} tokens from position {skip}, first ids {seq[:4]}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "texts":
        texts(*sys.argv[2:5])
    elif cmd == "nodraft":
        bo.json = DraftOff()
        sys.argv = ["bench_openai.py"] + sys.argv[2:]
        bo.main()
    elif cmd == "tokens":
        tokens(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
