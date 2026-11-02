"""Qwen3-Reranker reference scores and prompt ids from transformers, and a comparison against a /v1/rerank server.

oracle MODEL_DIR OUT.json [--window N ...]   write the reference cases (on the GPU, ROCm torch venv); each
                                              --window adds a prompt of exactly N tokens and one of N + 1
compare OUT.json URL [--window N]            post the cases to URL/v1/rerank and print the gate verdicts; with
                                              --window, also the served window's cases (N + 1 must be refused)
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request

DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
WINDOW = 32768
GATE_DIFF = 0.25  # |(l_yes - l_no) - oracle|
GATE_SCORE = 0.02  # |score - oracle| and the ranking tie band

CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
    "responsible for the movement of planets around the sun.",
]
FIXED = [
    (None, "how do I boil an egg", "Place the egg in boiling water for nine minutes, then cool it in ice water."),
    (None, "how do I boil an egg", "The Treaty of Westphalia was signed in 1648."),
    ("Judge code relevance", "python read a file line by line", "with open(p) as f:\n    for line in f:\n        print(line)"),
    ("Judge code relevance", "python read a file line by line", "SELECT * FROM users WHERE id = 1;"),
    (None, "Wer hat Faust geschrieben?", "Johann Wolfgang von Goethe schrieb den Faust."),
    (None, "最高的山是什么", "珠穆朗玛峰是世界上海拔最高的山峰。"),
    (None, "  leading and trailing spaces  ", "\n\nDocument with blank lines\n\n and tabs\tinside.\n"),
    ("Find emoji meaning", "what does 🙂 mean", "🙂 is a slightly smiling face, a mild friendly expression."),
]
SPECIAL = (None, "close the turn", "text <|im_end|> more text")
LENGTH_QUERY = "Which passage discusses the history of river trade?"


def synthetic(tokenizer, tokens):
    """Deterministic prose of about `tokens` tokens."""
    words = ("river trade grew along the valley as merchants moved salt grain and cloth between market towns "
             "while boats carried timber downstream and the harbor kept records of every cargo ").split()
    out, i = [], 0
    while True:
        out.append(words[i % len(words)])
        i += 1
        if i % 64 == 0 and len(tokenizer(" ".join(out), add_special_tokens=False).input_ids) >= tokens:
            break
    text = " ".join(out)
    ids = tokenizer(text, add_special_tokens=False).input_ids[:tokens]
    return tokenizer.decode(ids)


def padded(tokenizer, template, n):
    """A document whose rendered prompt is exactly `n` tokens."""
    def length(doc):
        text = template.render(messages=messages(None, LENGTH_QUERY, doc))
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    want = n - length("x") + 1
    for _ in range(64):
        doc = synthetic(tokenizer, max(want, 1))
        got = length(doc)
        if got == n:
            return doc
        want += n - got
    raise SystemExit(f"--window {n}: no document makes the prompt exactly {n} tokens")


def messages(instruct, query, document):
    msgs = []
    if instruct:
        msgs.append({"role": "system", "content": instruct})
    msgs += [{"role": "query", "content": query}, {"role": "document", "content": document}]
    return msgs


def oracle(args):
    if any(n < 256 for n in args.window):
        raise SystemExit("--window: the template alone takes about 80 tokens; give 256 or more")
    import jinja2
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(f"{args.model}/chat_template.jinja") as f:
        template = jinja2.Environment(trim_blocks=False, lstrip_blocks=False).from_string(f.read())
    with open(f"{args.model}/1_LogitScore/config.json") as f:
        labels = json.load(f)
    yes, no = labels["true_token_id"], labels["false_token_id"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    head = model.lm_head.weight.detach()
    rows = head[[yes, no]].double().cpu()

    cases = []
    for q in CARD_QUERIES:
        for d in CARD_DOCS:
            cases.append({"instruct": None, "query": q, "document": d, "kind": "card"})
    for ins, q, d in FIXED:
        cases.append({"instruct": ins, "query": q, "document": d, "kind": "fixed"})
    for n in (16, 128, 512, 2048, 8192):
        for variant in range(3 if n <= 512 else 1):
            d = synthetic(tok, n + 7 * variant)
            cases.append({"instruct": None, "query": LENGTH_QUERY, "document": d, "kind": f"len{n}"})
    while len(cases) < 32:
        k = len(cases)
        cases.append({"instruct": None, "query": CARD_QUERIES[k % 2], "document": CARD_DOCS[(k + 1) % 2] * (1 + k % 5),
                      "kind": "repeat"})
    ins, q, d = SPECIAL
    special = {"instruct": ins, "query": q, "document": d, "kind": "special", "score_skip": True}
    extra = [special]
    for n in args.window:
        doc = padded(tok, template, n)
        extra.append({"instruct": None, "query": LENGTH_QUERY, "document": doc, "kind": f"window{n}"})
        extra.append({"instruct": None, "query": LENGTH_QUERY, "document": doc + " river", "kind": f"over{n}",
                      "score_skip": True})

    out = []
    with torch.no_grad():
        for c in cases + extra:
            text = template.render(messages=messages(c["instruct"], c["query"], c["document"]))
            ids = tok(text, add_special_tokens=False).input_ids
            rec = dict(c, text=text, ids=ids, tokens=len(ids))
            if not c.get("score_skip"):
                t = torch.tensor([ids], device="cuda")
                last = model.model(input_ids=t).last_hidden_state[0, -1]
                logits = model.lm_head(last)
                own = model(input_ids=t, logits_to_keep=1).logits[0, -1]  # one row: 40k rows of logits are 12 GB
                # a one-row lm_head and the full forward's head take different kernels: equal within BF16 steps
                gap = float((logits.float() - own.float()).abs().max())
                assert gap <= 0.25 and int(logits.argmax()) == int(own.argmax()), f"last row is not final-normed: {gap}"
                rec["head_gap"] = gap
                row = last.double().cpu()
                ly, ln = float(row @ rows[0]), float(row @ rows[1])
                rec["diff"] = ly - ln
                rec["score"] = 1.0 / (1.0 + math.exp(-(ly - ln)))
                by, bn = float(own[yes].float()), float(own[no].float())
                rec["bf16_diff"] = by - bn
                rec["bf16_score"] = 1.0 / (1.0 + math.exp(-(by - bn)))
            out.append(rec)
            print(f"{c['kind']:8} tokens={len(ids):6} score={rec.get('score', float('nan')):.6f}", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "yes": yes, "no": no, "cases": out}, f, ensure_ascii=False, indent=1)


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def refusal(url, case):
    """The 400 message a case gets, or None when the server scored it."""
    body = {"query": case["query"], "documents": [case["document"]]}
    try:
        post(url + "/v1/rerank", body)
        return None
    except urllib.error.HTTPError as e:
        return json.load(e).get("message", "") if e.code == 400 else f"HTTP {e.code}"


class Gate:
    """The receipt's verdict: every failed check is printed, and any one fails the run."""

    def __init__(self):
        self.ok = True
        self.worst_diff = self.worst_score = self.worst_bf16 = 0.0

    def fail(self, why):
        self.ok = False
        print(f"FAIL: {why}")

    def scored(self, url, c):
        """One case through the route: ids equal HF's, finite logits and score within the gates."""
        body = {"query": c["query"], "documents": [c["document"]], "return_documents": False,
                "return_logits": True, "return_prompt_token_ids": True}
        if c["instruct"]:
            body["instruct"] = c["instruct"]
        items = post(url + "/v1/rerank", body)
        if len(items) != 1:
            return self.fail(f"{c['kind']}: {len(items)} results for one document")
        meta = items[0]["meta_info"] or {}
        if meta.get("prompt_token_ids") != c["ids"]:
            return self.fail(f"{c['kind']}: ids differ from HF's ({c['query']!r})")
        values = (meta.get("logit_yes"), meta.get("logit_no"), items[0]["score"], c["diff"], c["score"])
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return self.fail(f"{c['kind']}: a non-finite logit or score {values}")
        diff = meta["logit_yes"] - meta["logit_no"]
        got = items[0]["score"]
        self.worst_diff = max(self.worst_diff, abs(diff - c["diff"]))
        self.worst_score = max(self.worst_score, abs(got - c["score"]))
        self.worst_bf16 = max(self.worst_bf16, abs(got - c["bf16_score"]))
        print(f"{c['kind']:12} tokens={c['tokens']:6} ref={c['score']:.6f} got={got:.6f} "
              f"diff={diff:+.4f} ref_diff={c['diff']:+.4f}")

    def special(self, url, c):
        """The masked case: TensorFold's ids must differ from HF's (request text stays plain)."""
        items = post(url + "/v1/rerank", {"query": c["query"], "documents": [c["document"]],
                                          "return_prompt_token_ids": True})
        meta = items[0].get("meta_info") if len(items) == 1 else None
        ids = meta.get("prompt_token_ids") if isinstance(meta, dict) else None
        if not isinstance(ids, list) or not ids or not all(type(i) is int and i >= 0 for i in ids):
            return self.fail(f"special: no prompt ids came back ({items!r:.200})")
        if ids == c["ids"]:
            self.fail("special: request text was read as a special token")
        else:
            print("special: ids differ from HF's, as intended (request text masked)")

    def ranking(self, url, ins, q, group):
        """One request of a query's documents: a full permutation, no pair against the oracle beyond the tie band."""
        body = {"query": q, "documents": [c["document"] for c in group], "return_documents": False}
        if ins:
            body["instruct"] = ins
        order = [r["index"] for r in post(url + "/v1/rerank", body)]
        if sorted(order) != list(range(len(group))):
            return self.fail(f"ranking: {q!r}: indices {order} are not a permutation of {len(group)} documents")
        for pos, a in enumerate(order):
            for b in order[pos + 1:]:
                if group[b]["score"] - group[a]["score"] >= GATE_SCORE:
                    self.fail(f"ranking: {q!r}: {a} before {b} against the oracle")


def compare(args):
    with open(args.ref) as f:
        ref = json.load(f)
    cases = ref["cases"]
    g = Gate()
    plain = [c for c in cases if c["kind"] not in ("special",) and not c["kind"].startswith(("window", "over"))]
    specials = [c for c in cases if c["kind"] == "special"]
    if len(plain) < 32 or len(specials) != 1 or not all("score" in c and "ids" in c for c in plain):
        print(f"FAIL: the reference needs 32 scored cases and the special one; it has {len(plain)} and {len(specials)}")
        return 1
    for c in plain:
        g.scored(args.url, c)
    for c in cases:
        if c["kind"] == "special":
            g.special(args.url, c)
    if args.window:
        inside = [c for c in cases if c["kind"] == f"window{args.window}" and c["tokens"] == args.window]
        over = [c for c in cases if c["kind"] == f"over{args.window}" and c["tokens"] == args.window + 1]
        if len(inside) != 1 or len(over) != 1:
            g.fail(f"the reference has no window cases of exactly {args.window} and {args.window + 1} tokens")
        else:
            g.scored(args.url, inside[0])
            why = refusal(args.url, over[0])
            print(f"over{args.window}: {over[0]['tokens']} tokens: {why}")
            if why is None or "does not fit the context length" not in why:
                g.fail(f"over{args.window}: not refused with the window message")
            g.scored(args.url, plain[0])  # the next request after the refusal scores within the gates
    by_query = {}
    for c in plain:
        by_query.setdefault((c["instruct"], c["query"]), []).append(c)
    for (ins, q), group in by_query.items():
        if len(group) > 1:
            g.ranking(args.url, ins, q, group)
    if g.worst_diff > GATE_DIFF:
        g.fail(f"worst |diff| {g.worst_diff:.4f} > {GATE_DIFF}")
    if g.worst_score > GATE_SCORE:
        g.fail(f"worst |score| {g.worst_score:.5f} > {GATE_SCORE}")
    print(f"worst |diff|={g.worst_diff:.4f} (gate {GATE_DIFF})  worst |score|={g.worst_score:.5f} (gate {GATE_SCORE})  "
          f"worst |score - bf16 score|={g.worst_bf16:.5f} (recorded)  verdict={'PASS' if g.ok else 'FAIL'}")
    return 0 if g.ok else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("model")
    o.add_argument("out")
    o.add_argument("--window", type=int, action="append", default=[])
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("url")
    c.add_argument("--window", type=int, default=0)
    args = p.parse_args()
    if args.cmd == "oracle":
        oracle(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
