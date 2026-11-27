#!/usr/bin/env python3
"""The CPU engine's teacher-forced Qwen3.5-2B logits against a float64 reference: each top-token difference, margins."""
# usage: cpu_fidelity.py freeze MODEL_DIR OUT | record URL OUT --name NAME | compare --case NAME --prompt-len N ...
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

VOCAB = 248320

# Each text's first SPLIT words are the prompt; the rest, tokenized, are the teacher ids (at most ROWS of them).
TEXTS = {
    "prose": (
        "Rivers meander because water does not flow at the same speed everywhere across a channel. On the outside of a "
        "bend the current runs faster and cuts into the bank, while on the inside it slows and drops the sand and silt it "
        "carries. Over years the outer bank retreats and the inner bank grows, so the bend grows wider and the river "
        "wanders across its valley. Sometimes a loop becomes so tight that the river breaks through its own neck and "
        "leaves the old bend behind as a curved lake, which people call an oxbow. Engineers who straighten rivers for "
        "shipping find that the water simply starts carving new bends downstream, because the shape comes from the "
        "physics of flowing water and not from any particular stretch of land. Geologists read old meanders in rock to "
        "learn where ancient rivers ran, and farmers along the Mississippi have watched whole fields move from one state "
        "to another as the channel shifted. The same pattern appears in streams of air in the upper atmosphere and in "
        "currents of the ocean, which bend and loop for the same reasons even though nothing solid holds them in place."
    ),
    "code": (
        "def parse_iso_date(text):\n    \"\"\"Parse YYYY-MM-DD into a (year, month, day) tuple without datetime.\"\"\"\n"
        "    if len(text) != 10 or text[4] != '-' or text[7] != '-':\n        raise ValueError('expected YYYY-MM-DD')\n"
        "    year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])\n    if not 1 <= month <= 12:\n"
        "        raise ValueError('month out of range')\n    days = [31, 29 if year % 4 == 0 and (year % 100 != 0 or "
        "year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]\n    if not 1 <= day <= days[month - 1]:\n"
        "        raise ValueError('day out of range')\n    return year, month, day\n\n\nassert parse_iso_date('2024-02-29') "
        "== (2024, 2, 29)\nassert parse_iso_date('1999-12-31') == (1999, 12, 31)\ntry:\n    parse_iso_date('2023-02-29')\n"
        "except ValueError as error:\n    print('refused:', error)\n"
    ),
    "list": (
        "Shopping list for the week: two litres of milk, a dozen eggs, a loaf of rye bread, three onions, a bag of "
        "carrots, half a kilo of rice, a jar of honey, two tins of tomatoes, a block of cheddar, a bunch of parsley, "
        "four apples, a lemon, olive oil, black pepper, and a packet of coffee. Things to do: call the dentist on "
        "Tuesday, return the library books before Friday, water the plants on the balcony, fix the squeaking door, "
        "write to grandmother, and book the train for the twelfth. Numbers to remember: the flat is 14B, the parcel "
        "locker code is 7731, and the meter reading was 48215 last month."
    ),
}
SPLIT = {"prose": 90, "code": 40, "list": 50}
ROWS = 128


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a.astype("<u4")).tobytes()).hexdigest()


def freeze(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    os.makedirs(args.out, exist_ok=True)
    record = {"tokenizer": os.path.basename(os.path.normpath(args.model_dir)), "rows": ROWS, "cases": {}}
    for name, text in TEXTS.items():
        words = text.split(" ")
        prompt = np.array(tok.encode(" ".join(words[: SPLIT[name]])), np.uint32)
        teacher = np.array(tok.encode(" " + " ".join(words[SPLIT[name] :])), np.uint32)[:ROWS]
        np.save(os.path.join(args.out, name + ".prompt.npy"), prompt)
        np.save(os.path.join(args.out, name + ".teacher.npy"), teacher)
        record["cases"][name] = {"prompt_len": int(len(prompt)), "teacher_len": int(len(teacher)), "prompt_sha256": sha256(prompt), "teacher_sha256": sha256(teacher)}
        print(f"{name}: {len(prompt)} prompt tokens, {len(teacher)} teacher tokens")
    with open(os.path.join(args.out, "prompt-set.json"), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")


def post(base, path, body):
    import urllib.request

    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def record(args):
    """A chat prompt as the serve renders it and its greedy reply, one more (prompt, teacher) pair of the frozen set."""
    content = open(args.file).read() if args.file else args.text
    messages = [{"role": "user", "content": content}]
    prompt = np.array(post(args.url, "/v1/tokenize", {"model": args.model, "messages": messages})["tokens"], np.uint32)
    reply = post(args.url, "/v1/chat/completions", {"model": args.model, "messages": messages, "max_tokens": args.tokens, "temperature": 0, "return_token_ids": True})
    teacher = np.array(reply["tensorfold"]["token_ids"], np.uint32)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, args.name + ".prompt.npy"), prompt)
    np.save(os.path.join(args.out, args.name + ".teacher.npy"), teacher)
    path = os.path.join(args.out, "prompt-set.json")
    sets = json.load(open(path)) if os.path.exists(path) else {"cases": {}}
    sets["cases"][args.name] = {"prompt_len": int(len(prompt)), "teacher_len": int(len(teacher)), "prompt_sha256": sha256(prompt), "teacher_sha256": sha256(teacher), "recorded": {"url": args.url, "max_tokens": args.tokens, "temperature": 0, "finish_reason": reply["choices"][0].get("finish_reason")}}
    with open(path, "w") as f:
        json.dump(sets, f, indent=1)
        f.write("\n")
    print(f"{args.name}: {len(prompt)} prompt tokens, {len(teacher)} reply tokens recorded")


def bf16_spacing(value):
    """One bf16 ulp at `value`: the near-tie policy's unit (tools/zig/qwen27_margins.py)."""
    magnitude = abs(value)
    if magnitude < 2.0 ** -126:
        return 2.0 ** -133
    return 2.0 ** (math.frexp(magnitude)[1] - 1 - 7)


def top2(row):
    first = int(np.argmax(row))
    rest = row.copy()
    rest[first] = -np.inf
    second = int(np.argmax(rest))
    return first, second, float(row[first]), float(row[second])


def compare(args):
    reference = np.load(args.reference, mmap_mode="r")
    words = np.fromfile(args.candidate, np.uint16)
    rows = words.size // VOCAB
    candidate = (words.astype(np.uint32) << 16).view(np.float32).reshape(rows, VOCAB)[args.prompt_len - 1 :]
    if candidate.shape != reference.shape:
        raise SystemExit(f"shape: candidate rows {candidate.shape} from row {args.prompt_len - 1}, reference {reference.shape}")
    out = {"case": args.case, "rows": int(reference.shape[0]), "reference_sha256": hashlib.sha256(open(args.reference, "rb").read()).hexdigest(),
           "candidate_sha256": hashlib.sha256(open(args.candidate, "rb").read()).hexdigest(), "policy": "near-tie: same two contenders, both margins <= one bf16 spacing of the top score",
           "differences": [], "max_abs": 0.0, "agree": 0}
    for g in range(reference.shape[0]):
        ref = np.asarray(reference[g], np.float64)
        can = candidate[g].astype(np.float64)
        out["max_abs"] = max(out["max_abs"], float(np.abs(can - ref).max()))
        rf, rs, rt, rr = top2(ref)
        cf, cs, ct, cr = top2(can)
        if rf == cf:
            out["agree"] += 1
            continue
        near = {rf, rs} == {cf, cs} and rt - rr <= bf16_spacing(rt) and ct - cr <= bf16_spacing(ct)
        d = {"row": g, "position": args.prompt_len - 1 + g, "reference": {"ids": [rf, rs], "logits": [rt, rr], "margin": rt - rr},
             "candidate": {"ids": [cf, cs], "logits": [ct, cr], "margin": ct - cr},
             "reference_at_candidate_top": float(ref[cf]), "candidate_at_reference_top": float(can[rf]), "verdict": "near-tie" if near else "difference"}
        out["differences"].append(d)
        print(f"{args.case} row {g:4d} (position {d['position']}): {d['verdict']}: reference top {rf} margin {rt - rr:.4f} over {rs}; candidate top {cf} margin {ct - cr:.4f} over {cs}")
    out["near_ties"] = sum(1 for d in out["differences"] if d["verdict"] == "near-tie")
    out["other_differences"] = len(out["differences"]) - out["near_ties"]
    print(f"{args.case}: {out['agree']}/{out['rows']} top tokens agree; {out['near_ties']} near-ties, {out['other_differences']} other differences; max |candidate - reference| {out['max_abs']:.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0 if out["other_differences"] == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="the prompt set as uint32 .npy pairs (<case>.prompt.npy, <case>.teacher.npy) and prompt-set.json with their sha256, before any logits exist")
    f.add_argument("model_dir", help="the checkpoint directory, for its tokenizer")
    f.add_argument("out")
    r = sub.add_parser("record", help="a chat prompt as the serve renders it (/v1/tokenize) and its greedy reply, one more (prompt, teacher) pair in OUT")
    r.add_argument("url")
    r.add_argument("out")
    r.add_argument("--model", default="qwen3.5-2b")
    r.add_argument("--name", required=True)
    r.add_argument("--tokens", type=int, default=128)
    r.add_argument("--text", default="Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.")
    r.add_argument("--file", help="a document to send as the user turn instead of --text")
    c = sub.add_parser("compare", help="the candidate's rows from prompt_len - 1 against the reference's: every top-token difference with both margins")
    c.add_argument("--case", required=True)
    c.add_argument("--prompt-len", type=int, required=True)
    c.add_argument("--reference", required=True, help="the float64 reference's logits_f64.npy, one row a teacher token")
    c.add_argument("--candidate", required=True, help="tf-qwen35-cpu-forward's raw bf16 rows over prompt + teacher[:-1]")
    c.add_argument("--json", help="the differences and digests written here")
    args = p.parse_args()
    if args.command == "freeze":
        return freeze(args)
    return record(args) if args.command == "record" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
