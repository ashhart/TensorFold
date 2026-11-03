#!/usr/bin/env python3
"""Gate 4 for Clef-Flash: a served native engine's /v1/systemone against the reference's BF16 pipeline, per bucket."""
# usage:
#   clef_e2e.py records <model dir> <out>                         40 seeded System One requests, 10 per length bucket
#   clef_e2e.py reference <model dir> <records> <out> [--device cuda]   load_release_model + encode_record, BF16
#   clef_e2e.py native <url> <records> <out>                      POST /v1/systemone with "raw": true
#   clef_e2e.py compare <records> <reference> <native> --model <model dir> [--report FILE]
# The pass rule (proposed): per bucket mean KL <= 1e-4 and max KL <= 1e-2; a top-choice disagreement passes only when
# both rank the same two options first and second and both top-two gaps are <= 0.02. Standard library plus the
# reference's torch for `reference`.
import argparse, importlib.util, json, math, os, random, sys, time, urllib.request
from pathlib import Path

SOURCES = {
    "torch": "2.15.0a0+rocm10.1.0a20260822",
    "transformers": "5.22.1",
    "checkpoint": "Cloudflare/clef-flash@fb4417d31c1d0f9e7e77f1141b40aaede7491e00",
}
BUCKETS = [(245, 1024), (1025, 4096), (4097, 10240), (10241, 16384)]
SEED = 1952
MEAN_KL, MAX_KL, GAP = 1e-4, 1e-2, 0.02


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rows_tool():
    return load(Path(__file__).with_name("clef_backbone_rows.py"), "clef_backbone_rows")


def questions(rng, words):
    """Two to five fields: noul (some with their own true/false text), choice of 2 to 5 ids, score of 3 to 5 levels."""
    out = {}
    for i in range(rng.randint(2, 5)):
        kind = rng.choice(["noul", "choice", "score"])
        say = lambda n: " ".join(rng.choice(words) for _ in range(n))
        q = {"type": kind, "instructions": f"Decide whether the {say(2)} is {say(3)}."}
        if kind == "noul" and rng.random() < 0.3:
            q["criteria"] = {"true": f"The {say(2)} holds.", "false": f"No {say(2)}."}
        if kind == "choice":
            q["criteria"] = {f"{rng.choice(words)}_{j}": f"{say(4)}" for j in range(rng.randint(2, 5))}
        if kind == "score":
            q["criteria"] = [f"{say(3)} (level {j})" for j in range(rng.randint(3, 5))]
        out[f"{rng.choice(words)}_{i}"] = q
    return out


def cmd_records(a):
    rt = rows_tool()
    jsm = load(Path(a.model) / "joint_schema_model.py", "joint_schema_model")
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(a.model).tokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for b, sizes in enumerate(rt.lengths()):
        for j, n in enumerate(sizes):
            rng = random.Random(SEED * 1000 + b * 100 + j)
            qs = questions(rng, rt.WORDS)
            state = rt.state_text(rng, n)
            req = {"model": "clef-flash", "state": state, "questions": qs}
            # the state cut by characters to the record's length (the last bucket's top is the release's own cut)
            lo, hi = 0, len(state)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                req["state"] = state[:mid]
                if len(jsm.encode_record(tok, req).input_ids) <= n:
                    lo = mid
                else:
                    hi = mid - 1
            req["state"] = state[:lo] if n < 16384 else state
            tokens = len(jsm.encode_record(tok, req).input_ids)
            name = f"rec_{len(index):02d}"
            (out / f"{name}.json").write_text(json.dumps(req, ensure_ascii=False))
            index.append({"name": name, "bucket": b, "target": n, "tokens": tokens})
            print(f"{name}: bucket {b}, {tokens} tokens, {len(qs)} questions", flush=True)
    (out / "index.json").write_text(json.dumps({"sources": SOURCES, "records": index}, indent=1))


def cmd_reference(a):
    import torch
    jsm = load(Path(a.model) / "joint_schema_model.py", "joint_schema_model")
    model, processor = jsm.load_release_model(a.model, device=a.device, dtype=torch.bfloat16)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in json.loads((Path(a.records) / "index.json").read_text())["records"]:
        req = json.loads((Path(a.records) / f"{r['name']}.json").read_text())
        t0 = time.time()
        with torch.inference_mode():
            enc = jsm.encode_record(processor.tokenizer, req, max_length=16384, processor=processor)
            logits = model(jsm.collate_records([enc], processor.tokenizer.pad_token_id, a.device))[0]
            probs = {q.question_id: {"options": list(q.option_ids), "probs": l.float().softmax(-1).tolist()}
                     for q, l in zip(enc.questions, logits)}
        # the reference's own answer shape over the same probabilities (its `systemone` without a second pass)
        answers = {qid: jsm.systemone_answer(req["questions"][qid], dict(zip(v["options"], v["probs"]))) for qid, v in probs.items()}
        answer = {"model": req["model"], "answers": answers, "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}
        (out / f"{r['name']}.json").write_text(json.dumps({"raw": probs, "answer": answer}))
        print(f"{r['name']}: {len(enc.input_ids)} tokens in {time.time() - t0:.1f} s", flush=True)
    (out / "meta.json").write_text(json.dumps({"sources": SOURCES, "torch": torch.__version__, "device": a.device}))


def cmd_native(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in json.loads((Path(a.records) / "index.json").read_text())["records"]:
        req = json.loads((Path(a.records) / f"{r['name']}.json").read_text())
        req["raw"] = True
        data = json.dumps(req).encode()
        t0 = time.time()
        http = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=600) as resp:
            body = json.loads(resp.read())
        (out / f"{r['name']}.json").write_text(json.dumps(body))
        print(f"{r['name']}: {body['usage']['input_tokens']} tokens in {time.time() - t0:.2f} s", flush=True)


def kl(p, q):
    """KL(p || q), strict: a term with p > 0 and q == 0 is infinite; only terms where p is 0 drop out."""
    total = 0.0
    for x, y in zip(p, q, strict=True):
        if x == 0:
            continue
        if y == 0:
            return math.inf
        total += x * (math.log(x) - math.log(y))
    return total


def probabilities(where, v):
    """A probability vector or the run is void: finite values in [0, 1] summing to 1 within 1e-3."""
    if not isinstance(v, list) or not v or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in v):
        sys.exit(f"{where}: not a probability vector; it voids the run")
    if abs(sum(v) - 1) > 1e-3:
        sys.exit(f"{where}: probabilities sum to {sum(v)}; it voids the run")
    total = sum(v)
    return [x / total for x in v]  # normalized, so rounding drift cannot make a KL negative


def top2(probs):
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    return order[:2], (probs[order[0]] - probs[order[1]]) if len(order) > 1 else 1.0


def cmd_compare(a):
    jsm = load(Path(a.model) / "joint_schema_model.py", "joint_schema_model")
    index = json.loads((Path(a.records) / "index.json").read_text())["records"]
    if not index:
        sys.exit("no records: gate 4 needs every record")
    # coverage: the planned ten a bucket, no bucket outside the gate's, every record inside its bucket, no stray file
    targets = [n for sizes in rows_tool().lengths() for n in sizes]
    for r, n in zip(index, targets, strict=True):
        r.setdefault("target", n)
    for b in range(len(BUCKETS)):
        if sum(1 for r in index if r["bucket"] == b) != 10:
            sys.exit(f"bucket {b} holds {sum(1 for r in index if r['bucket'] == b)} records, not 10")
    for r in index:
        if r["bucket"] not in range(len(BUCKETS)):
            sys.exit(f"{r['name']}: bucket {r['bucket']} is not one of the gate's")
        lo, hi = BUCKETS[r["bucket"]]
        if not lo <= r["tokens"] <= hi:
            sys.exit(f"{r['name']}: {r['target']} target, {r['tokens']} actual, outside bucket {lo}-{hi}")
    names = {r["name"] for r in index}
    if len(names) != len(index):
        sys.exit(f"the index names {len(names)} distinct records in {len(index)} entries; each must be one record")
    for d in (a.reference, a.native):
        stray = {p.stem for p in Path(d).glob("rec_*.json")} - names
        if stray:
            sys.exit(f"{d}: files of records the index does not hold: {sorted(stray)}")
    report = {"sources": SOURCES, "buckets": [], "disagreements": [], "answers_equal": 0, "records": len(index),
              "lengths": [{"record": r["name"], "target": r["target"], "actual": r["tokens"]} for r in index]}
    ok = True
    print("| bucket | records | questions | mean KL | max KL | top choice agrees | pass |")
    print("|---|---|---|---|---|---|---|")
    for b, (lo, hi) in enumerate(BUCKETS):
        kls, agree, nq, recs = [], 0, 0, [r for r in index if r["bucket"] == b]
        for r in recs:
            ref = json.loads((Path(a.reference) / f"{r['name']}.json").read_text())
            nat = json.loads((Path(a.native) / f"{r['name']}.json").read_text())
            native_answers = {k: v for k, v in nat.items() if k != "raw"}
            report["answers_equal"] += native_answers == ref["answer"]
            # the request's questions, each in both answers and both raw probabilities, none missing and none extra
            asked = json.loads((Path(a.records) / f"{r['name']}.json").read_text())["questions"]
            want_ids = set(asked)
            for side, raw, answers in (("reference", ref.get("raw"), ref["answer"].get("answers")), ("native", nat.get("raw"), nat.get("answers"))):
                if set(raw or {}) != want_ids or set(answers or {}) != want_ids:
                    sys.exit(f"{r['name']}: {side} questions {sorted(answers or {})} / raw {sorted(raw or {})} vs requested {sorted(want_ids)}")
            # both sides read the record's real length
            for side, usage in (("reference", ref["answer"].get("usage")), ("native", nat.get("usage"))):
                if not isinstance(usage, dict) or usage.get("input_tokens") != r["tokens"]:
                    sys.exit(f"{r['name']}: {side} usage {usage} vs the record's {r['tokens']} tokens")
            # each native answer is the reference's answer function over the native raw probabilities, of the asked type
            for qid, question in asked.items():
                got, want = nat["answers"][qid], ref["answer"]["answers"][qid]
                if not isinstance(got, dict) or not isinstance(want, dict) or not got.get("type") == want.get("type") == question["type"]:
                    sys.exit(f"{r['name']} {qid}: native answer {got} vs a reference answer of type {question['type']}")
                same = jsm.systemone_answer(question, dict(zip(nat["raw"][qid]["options"], nat["raw"][qid]["probs"])))
                if got != same:
                    sys.exit(f"{r['name']} {qid}: native answer {got}, its raw probabilities give {same}")
            for qid, want in ref["raw"].items():
                got = nat["raw"][qid]
                if got["options"] != want["options"]:
                    sys.exit(f"{r['name']} {qid}: option order {got['options']} vs {want['options']}")
                p = probabilities(f"{r['name']} {qid} reference", want["probs"])
                q = probabilities(f"{r['name']} {qid} native", got["probs"])
                if len(p) != len(q) or len(p) != len(want["options"]):
                    sys.exit(f"{r['name']} {qid}: {len(p)} reference and {len(q)} native probabilities for {len(want['options'])} options")
                kls.append(kl(p, q))
                nq += 1
                (rt, rg), (nt, ng) = top2(p), top2(q)
                if rt[0] == nt[0]:
                    agree += 1
                    continue
                passes = set(rt) == set(nt) and rg <= GAP and ng <= GAP
                ok &= passes
                report["disagreements"].append({"record": r["name"], "question": qid, "reference_top2": [want["options"][i] for i in rt],
                                                "reference_gap": rg, "native_top2": [want["options"][i] for i in nt], "native_gap": ng,
                                                "kl": kls[-1], "pass": passes})
        mean, worst = (sum(kls) / len(kls), max(kls)) if kls else (math.inf, math.inf)
        passes = mean <= MEAN_KL and worst <= MAX_KL
        ok &= passes
        report["buckets"].append({"bucket": f"{lo}-{hi}", "records": len(recs), "questions": nq, "mean_kl": mean, "max_kl": worst, "top_agree": agree, "pass": passes})
        print(f"| {lo}-{hi} | {len(recs)} | {nq} | {mean:.2e} | {worst:.2e} | {agree}/{nq} | {'yes' if passes else 'NO'} |")
    for d in report["disagreements"]:
        print(f"disagreement {d['record']} {d['question']}: reference {d['reference_top2']} gap {d['reference_gap']:.4f}, "
              f"native {d['native_top2']} gap {d['native_gap']:.4f}, KL {d['kl']:.2e}: {'passes' if d['pass'] else 'FAILS'}")
    print("lengths (target, actual): " + ", ".join(f"{r['name']} {r['target']} {r['tokens']}" for r in index))
    print(f"answers byte-equal to the reference's: {report['answers_equal']} of {len(index)} records")
    report["pass"] = ok
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    print("PASS gate 4" if ok else "FAIL gate 4")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("model")
    r.add_argument("out")
    f = sub.add_parser("reference")
    for x in ("model", "records", "out"):
        f.add_argument(x)
    f.add_argument("--device", default="cuda")
    n = sub.add_parser("native")
    for x in ("url", "records", "out"):
        n.add_argument(x)
    c = sub.add_parser("compare")
    for x in ("records", "reference", "native"):
        c.add_argument(x)
    c.add_argument("--model", required=True, help="the checkpoint dir: its systemone_answer rebuilds each native answer")
    c.add_argument("--report")
    a = p.parse_args()
    return {"records": cmd_records, "reference": cmd_reference, "native": cmd_native, "compare": cmd_compare}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
