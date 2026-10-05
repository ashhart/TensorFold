#!/usr/bin/env python3
"""Paired MMLU against a running server: the same questions, prompt and parser for two server settings (a KV format,
a build), then accuracy with a 95% CI for each and an exact McNemar test on the questions they answer differently.

    python3 tools/dsv41_mmlu.py sample --test mmlu_test.jsonl --per-subject 40 --seed 0 --out sample.jsonl
    TF_API_KEY=... python3 tools/dsv41_mmlu.py run --sample sample.jsonl --out fp4.jsonl [--base http://127.0.0.1:8888]
    python3 tools/dsv41_mmlu.py compare fp4.jsonl fp8.jsonl [--json report.json]

``sample`` reads MMLU's test split as JSON lines (``subject``, ``question``, ``choices`` (4), ``answer`` (0-3); from
cais/mmlu ``all/test`` or the hendrycks CSVs) and draws ``--per-subject`` questions from every subject with a seeded
RNG over each subject's questions in file order, so a seed and a file give one sample. ``run`` asks each question
0-shot as one user message (``PROMPT``), thinking off (``chat_template_kwargs.enable_thinking`` false), greedy, and
takes the first standalone A-D letter of the reply (``parse_letter``; none = wrong); it appends a line per question
to ``--out`` and skips the ids already there, so a stopped run resumes. ``compare`` pairs two result files by id.
The key is read from TF_API_KEY only, never a flag. Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Any

LETTERS = "ABCD"
PROMPT = ("The following is a multiple choice question about {subject}. Reply with only the letter (A, B, C or D) of "
          "the correct answer.\n\n{question}\n{options}\n\nAnswer:")
FIRST_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


# -- sample, prompt, parse ------------------------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_sample(rows: list[dict], per_subject: int, seed: int) -> list[dict]:
    """``per_subject`` questions of every subject (all of a smaller one), each subject drawn by its own RNG seeded
    from (seed, subject): adding or dropping a subject leaves the others' draws as they were. Ids are
    ``<subject>/<index within the subject in file order>``; the result is sorted by subject, then index."""

    by_subject: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append((len(by_subject[r["subject"]]), r))
    out = []
    for subject in sorted(by_subject):
        items = by_subject[subject]
        rng = random.Random(f"{seed}:{subject}")
        picked = sorted(rng.sample(range(len(items)), min(per_subject, len(items))))
        for i in picked:
            r = items[i][1]
            out.append({"id": f"{subject}/{i}", "subject": subject, "question": r["question"],
                        "choices": list(r["choices"]), "answer": int(r["answer"])})
    return out


def prompt_for(q: dict) -> str:
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return PROMPT.format(subject=q["subject"].replace("_", " "), question=q["question"].strip(), options=options)


def parse_letter(text: str | None) -> str | None:
    """The first A, B, C or D standing alone (not inside a word: 'Answer' and 'CD' do not count), else None."""

    m = FIRST_LETTER.search(text or "")
    return m.group(1) if m else None


# -- statistics -----------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval of k successes in n (95% by default)."""

    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: the binomial(b + c, 1/2) probability of a split at least as uneven as b:c."""

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a: list[dict], b: list[dict], alpha: float = 0.05) -> dict[str, Any]:
    """Pair two runs by id (only ids both answered). b_count: right in A, wrong in B; c_count: wrong in A, right in B.
    Subjects are flagged when their own exact McNemar p is under alpha / (number of subjects) (Bonferroni)."""

    ra = {r["id"]: r for r in a}
    rb = {r["id"]: r for r in b}
    ids = sorted(set(ra) & set(rb))
    n = len(ids)
    ka = sum(ra[i]["correct"] for i in ids)
    kb = sum(rb[i]["correct"] for i in ids)
    bc = sum(1 for i in ids if ra[i]["correct"] and not rb[i]["correct"])
    cc = sum(1 for i in ids if not ra[i]["correct"] and rb[i]["correct"])
    subj: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        subj[ra[i]["subject"]].append(i)
    per = []
    for s in sorted(subj):
        si = subj[s]
        sb = sum(1 for i in si if ra[i]["correct"] and not rb[i]["correct"])
        sc = sum(1 for i in si if not ra[i]["correct"] and rb[i]["correct"])
        per.append({"subject": s, "n": len(si), "acc_a": sum(ra[i]["correct"] for i in si) / len(si),
                    "acc_b": sum(rb[i]["correct"] for i in si) / len(si), "b": sb, "c": sc,
                    "p": mcnemar_exact(sb, sc)})
    cut = alpha / max(1, len(per))
    # the paired difference's standard error (B - A), from the discordant counts
    d = (kb - ka) / n if n else 0.0
    se = math.sqrt(max(0.0, (bc + cc) / n - d * d) / n) if n else 0.0
    return {"n": n, "only_a": len(ra) - n, "only_b": len(rb) - n,
            "acc_a": ka / n if n else 0.0, "ci_a": wilson(ka, n), "acc_b": kb / n if n else 0.0, "ci_b": wilson(kb, n),
            "unparsed_a": sum(1 for i in ids if ra[i].get("pred") is None),
            "unparsed_b": sum(1 for i in ids if rb[i].get("pred") is None),
            "b": bc, "c": cc, "p": mcnemar_exact(bc, cc), "diff": d, "diff_ci": (d - 1.959964 * se, d + 1.959964 * se),
            "subject_cut": cut, "subjects": per, "flagged": [p["subject"] for p in per if p["p"] < cut]}


# -- run ------------------------------------------------------------------------------------------------------------
def body_for(q: dict, model: str, max_tokens: int) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": prompt_for(q)}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}


def post(base: str, body: dict, timeout: float) -> dict:
    key = os.environ.get("TF_API_KEY", "")
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(q: dict, a) -> dict:
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            resp = post(a.base, body_for(q, a.model, a.max_tokens), a.timeout)
            msg = resp["choices"][0]["message"]
            text = msg.get("content") or ""
            pred = parse_letter(text)
            return {"id": q["id"], "subject": q["subject"], "answer": LETTERS[q["answer"]], "pred": pred,
                    "correct": pred == LETTERS[q["answer"]], "reply": text,
                    "finish": resp["choices"][0].get("finish_reason"), "seconds": round(time.time() - t0, 3)}
        except Exception as e:  # noqa: BLE001 - a transport error is retried, then reported
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 + 3 * attempt)
    return {"id": q["id"], "error": last}


def run(a) -> int:
    qs = load_jsonl(a.sample)
    done = set()
    if os.path.exists(a.out):
        done = {r["id"] for r in load_jsonl(a.out) if "error" not in r}
    todo = [q for q in qs if q["id"] not in done]
    print(f"[mmlu] {len(qs)} questions, {len(done)} done, {len(todo)} to ask, {a.concurrency} at a time", flush=True)
    errors = 0
    t0 = time.time()
    with open(a.out, "a") as f, cf.ThreadPoolExecutor(a.concurrency) as pool:
        for k, r in enumerate(pool.map(lambda q: ask(q, a), todo), 1):
            if "error" in r:
                errors += 1
                print(f"[mmlu] {r['id']}: {r['error']}", file=sys.stderr, flush=True)
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 100 == 0:
                print(f"[mmlu] {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    rows = [r for r in load_jsonl(a.out) if "error" not in r]
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    print(f"[mmlu] answered {len(rows)}/{len(qs)}, accuracy {acc:.4f}, errors {errors}, {time.time() - t0:.0f}s")
    return 1 if errors else 0


def report(c: dict, la: str, lb: str) -> str:
    out = [f"paired questions: {c['n']} (only in {la}: {c['only_a']}, only in {lb}: {c['only_b']})"]
    for lab, acc, ci, un in ((la, c["acc_a"], c["ci_a"], c["unparsed_a"]), (lb, c["acc_b"], c["ci_b"], c["unparsed_b"])):
        out.append(f"{lab:>8}: accuracy {100 * acc:.2f}%  95% CI [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]  unparsed {un}")
    out.append(f"{lb} - {la}: {100 * c['diff']:+.2f} points, 95% CI [{100 * c['diff_ci'][0]:+.2f}, "
               f"{100 * c['diff_ci'][1]:+.2f}]")
    out.append(f"McNemar: b ({la} right, {lb} wrong) = {c['b']}, c ({la} wrong, {lb} right) = {c['c']}, "
               f"exact p = {c['p']:.4g}")
    out.append(f"subjects beyond noise (exact McNemar p < {c['subject_cut']:.2g}, Bonferroni): "
               f"{', '.join(c['flagged']) or 'none'}")
    out.append("largest per-subject differences:")
    for p in sorted(c["subjects"], key=lambda p: (-abs(p["acc_b"] - p["acc_a"]), p["subject"]))[:8]:
        out.append(f"  {p['subject']:<38} n={p['n']:<3} {la} {100 * p['acc_a']:5.1f}%  {lb} {100 * p['acc_b']:5.1f}%"
                   f"  b={p['b']} c={p['c']} p={p['p']:.3g}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--test", required=True)
    s.add_argument("--per-subject", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sample", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--base", default="http://127.0.0.1:8888")
    r.add_argument("--model", default="DeepSeek-v4.1-Flash-EXL3")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=16)
    r.add_argument("--timeout", type=float, default=600.0)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--labels", default="A,B")
    c.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        rows = stratified_sample(load_jsonl(a.test), a.per_subject, a.seed)
        with open(a.out, "w") as f:
            f.writelines(json.dumps(q) + "\n" for q in rows)
        print(f"[mmlu] {len(rows)} questions over {len({q['subject'] for q in rows})} subjects -> {a.out}")
        return 0
    if a.cmd == "run":
        return run(a)
    la, lb = a.labels.split(",")
    res = compare([x for x in load_jsonl(a.a) if "error" not in x], [x for x in load_jsonl(a.b) if "error" not in x])
    print(report(res, la, lb))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
