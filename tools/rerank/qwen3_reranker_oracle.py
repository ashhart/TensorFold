"""Qwen3-Reranker reference scores and prompt ids from transformers, and a comparison against a /v1/rerank server.

sources MODEL_DIR                             check the snapshot and Transformers against the pins (no model loads)
oracle MODEL_DIR OUT.json [--window N ...]   write the reference cases (BF16 on the GPU by default, ROCm torch
       [--dtype float32 --device cpu]         venv; FP32 on the CPU for #1895's reference); each --window adds a
                                              prompt of exactly N tokens and one of N + 1
compare OUT.json URL [--window N]            post the cases to URL/v1/rerank and print the gate verdicts; with
                                              --window, also the served window's cases (N + 1 must be refused)
formatter MODEL_DIR URL                       the served prompt ids against the model card's own formatter, token
                                              for token, with the default and with custom instructions
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
from pathlib import Path
import urllib.error
import urllib.request

# The checkpoints the receipts were made from, and sha256 of every non-weight file of each pinned snapshot.
CHECKPOINTS = {
    "8B": ("Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912"),
    "4B": ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
}
SOURCES = {
    "8B": {
        ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
        "1_LogitScore/config.json": "73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3",
        "README.md": "5f33734e77bb56c191dfd66bd8c4c04fdba6c023cddd6052871082c411f9b608",
        "chat_template.jinja": "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d",
        "config.json": "e87a4f50c475c7276fc642eabfc986a7c8da1c8a30a4ee34b4979048b330f2be",
        "config_sentence_transformers.json": "231aa494dcabada22d73f151120560b2a72b49e5b93e1b0a8329664f2348cf2e",
        "generation_config.json": "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "model.safetensors.index.json": "8a22e7ffd6001796256c0be7c73594835b6f907eec442fde691301ebf35087d9",
        "modules.json": "6f13b6b4a89e577b591b2077bca40c67c26541a6740a8809267cb474f90806a9",
        "sentence_bert_config.json": "3234ebd224d492cbe8d55d5ec80a3f408451c4db3005bafb64fe1c51c763e01e",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
    "4B": {
        ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
        "1_LogitScore/config.json": "73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3",
        "README.md": "b7cf82b30d5e38cd1a02ca7cfd60c26c0e3664cb31b2c4b661550a93c9281bee",
        "chat_template.jinja": "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d",
        "config.json": "38bff5eac700032a185745e4076eccad7aa453473cafc2a27de412cdb7b79e19",
        "config_sentence_transformers.json": "6a153d6696f78fd588c1c728967f0b773ea869d3c6028f151ce71ebe49140762",
        "generation_config.json": "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "model.safetensors.index.json": "18a783f197360068da36d2ff9ecffce0121542a87598fd1fef3079ead1c3cc08",
        "modules.json": "6f13b6b4a89e577b591b2077bca40c67c26541a6740a8809267cb474f90806a9",
        "sentence_bert_config.json": "3234ebd224d492cbe8d55d5ec80a3f408451c4db3005bafb64fe1c51c763e01e",
        "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
VERSION = "5.22.1"
# The Qwen3 model code the oracle runs, and tree_sha256 of the whole installed Transformers package (shared code too).
LIBRARY = {
    "models/qwen3/configuration_qwen3.py": "49d9ccd0f29ccd977b93dba2004f84b867891b3d0509b71fd8ba3ed01054e64d",
    "models/qwen3/modeling_qwen3.py": "63a6213c15d6fb04e58a1c64fd05a4cdfb49476443fbeef524f0e0486043824b",
}
TRANSFORMERS_TREE = "b08cead6d1b841ef9723106a81a92ec37d67aff3e43a198e56b7dc76538d377b"
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed(root: Path, pins: dict, what: str) -> dict:
    """The hashes of the pinned files under root; refuses when any differs from its pin."""
    found = {name: digest(root / name) if (root / name).is_file() else None for name in pins}
    if wrong := sorted(name for name, value in found.items() if value != pins[name]):
        raise SystemExit(f"{what} differs from its pin: {', '.join(wrong)}")
    return found


def tree_sha256(package: Path) -> str:
    """Hash every file of an installed package (bytecode caches excluded) as sorted "path NUL sha256" lines."""
    # rglob does not descend into directory links, and a linked directory could shadow a pinned module.
    if links := sorted(path.relative_to(package).as_posix() for path in package.rglob("*")
                       if path.is_symlink() and path.is_dir()):
        raise SystemExit(f"{package.name} has linked directories: {', '.join(links)}")
    files = sorted(path for path in package.rglob("*")
                   if path.is_file() and "__pycache__" not in path.relative_to(package).parts)
    lines = (f"{path.relative_to(package).as_posix()}\0{digest(path)}\n" for path in files)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def size_of(model: Path) -> str:
    """The pinned checkpoint `model` is: by its snapshot revision; a folder of no pinned revision is refused."""
    for size, (_, revision) in CHECKPOINTS.items():
        if model.resolve().name == revision:
            return size
    raise SystemExit(f"{model}: not a pinned snapshot ({', '.join(f'{r}@{v}' for r, v in CHECKPOINTS.values())})")


def pinned(model: Path) -> dict:
    """Refuse, before any model code runs, a snapshot or Transformers install whose sources differ from the pins."""
    size = size_of(model)
    repo, revision = CHECKPOINTS[size]
    model = model.resolve()
    if links := sorted(path.relative_to(model).as_posix() for path in model.rglob("*")
                       if path.is_symlink() and path.is_dir()):
        raise SystemExit(f"{size}: snapshot has linked directories: {', '.join(links)}")
    extra = sorted(path.relative_to(model).as_posix() for path in model.rglob("*")
                   if path.is_file() and path.suffix != ".safetensors"
                   and path.relative_to(model).as_posix() not in SOURCES[size])
    if extra:
        raise SystemExit(f"{size}: snapshot has unpinned non-weight files: {', '.join(extra)}")
    if (version := importlib.metadata.version("transformers")) != VERSION:
        raise SystemExit(f"expected Transformers {VERSION}, found {version}")
    package = Path(importlib.util.find_spec("transformers").origin).parent
    if (tree := tree_sha256(package)) != TRANSFORMERS_TREE:
        raise SystemExit(f"Transformers {VERSION} package tree differs from its pin: {tree}")
    return {"repo": repo, "revision": revision, "checkpoint": hashed(model, SOURCES[size], f"{repo}@{revision}"),
            "transformers": VERSION, "transformers_files": hashed(package, LIBRARY, f"Transformers {VERSION}"),
            "transformers_tree": tree}


def weights_sha256(model: Path) -> dict:
    """Every shard's sha256, after the index names exactly the shards present."""
    model = model.resolve()
    shards = sorted(model.glob("*.safetensors"))
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != {path.name for path in shards}:
        raise SystemExit("the safetensors shards do not match the checkpoint index")
    return {path.name: digest(path) for path in shards}


def sources(args):
    """Check the snapshot and the Transformers sources against the pins and print their hashes; no model loads."""
    print(json.dumps(pinned(args.model), indent=1))
    return 0


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
    pins = pinned(args.model)  # before any model code runs
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: write each reference to a new file")
    weights = weights_sha256(args.model)  # the shards hashed before they load
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
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model, loading = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=True,
                                                          trust_remote_code=False, output_loading_info=True)
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if errors := {key: loading[key] for key in keys if loading.get(key)}:
        raise SystemExit(f"the checkpoint did not load completely: {errors}")
    if (model.config.tie_word_embeddings
            and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr()):
        raise SystemExit("the model did not tie its input and output embeddings")
    model = model.to(args.device).eval()
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
                t = torch.tensor([ids], device=args.device)
                last = model.model(input_ids=t).last_hidden_state[0, -1]
                logits = model.lm_head(last)
                own = model(input_ids=t, logits_to_keep=1).logits[0, -1]  # one row: 40k rows of logits are 12 GB
                # a one-row lm_head and the full forward's head take different kernels: equal within the dtype's steps
                gap = float((logits.float() - own.float()).abs().max())
                assert gap <= 0.25 and int(logits.argmax()) == int(own.argmax()), f"last row is not final-normed: {gap}"
                rec["head_gap"] = gap
                row = last.double().cpu()
                ly, ln = float(row @ rows[0]), float(row @ rows[1])
                rec["diff"] = ly - ln
                rec["score"] = 1.0 / (1.0 + math.exp(-(ly - ln)))
                by, bn = float(own[yes].float()), float(own[no].float())
                rec["own_diff"] = by - bn  # the reference model's own head, at its dtype
                rec["own_score"] = 1.0 / (1.0 + math.exp(-(by - bn)))
            out.append(rec)
            print(f"{c['kind']:8} tokens={len(ids):6} score={rec.get('score', float('nan')):.6f}", file=sys.stderr)
    partial = out_path.with_name(out_path.name + ".partial")
    with open(partial, "w") as f:
        json.dump({"model": str(args.model), "dtype": args.dtype, "device": args.device,
                   "torch": torch.__version__, "sources": pins, "weights_sha256": weights,
                   "yes": yes, "no": no, "cases": out}, f, ensure_ascii=False, indent=1)
    partial.replace(out_path)  # whole or not at all


# The model card's formatter, verbatim (README "Transformers Usage"): prefix, content and suffix encoded apart.
CARD_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
               "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
CARD_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CUSTOM = ["Judge code relevance", "Given a question in German, find the passage that answers it",
          "  An instruction with spaces, a tab\tand\nnew lines  "]


def card_ids(tok, instruction, query, doc):
    """The ids the card's process_inputs builds for one pair (no truncation: every case is under its 8,192)."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCT
    content = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    return (tok.encode(CARD_PREFIX, add_special_tokens=False) + tok(content).input_ids
            + tok.encode(CARD_SUFFIX, add_special_tokens=False))


def formatter(args):
    """Every case's served ids equal the card formatter's, with the default instruction, each custom one sent as
    #1895's `instruction`, and once as SGLang's `instruct`."""
    pinned(args.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    pairs = [(q, d) for q in CARD_QUERIES for d in CARD_DOCS] + [(q, d) for _, q, d in FIXED]
    pairs += [(LENGTH_QUERY, synthetic(tok, n)) for n in (16, 512, 2048)]
    cases = [(None, q, d, None) for q, d in pairs]
    cases += [(ins, q, d, "instruction") for ins in CUSTOM for q, d in pairs]
    cases += [(CUSTOM[0], q, d, "instruct") for q, d in pairs]
    bad = 0
    for ins, q, d, field in cases:
        body = {"query": q, "documents": [d], "return_documents": False, "return_prompt_token_ids": True}
        if field:
            body[field] = ins
        meta = post(args.url + "/v1/rerank", body)["results"][0]["meta_info"]
        want = card_ids(tok, ins, q, d)
        if meta["prompt_token_ids"] != want:
            bad += 1
            print(f"FAIL: {field or 'default'} {ins!r} {q!r}: served ids differ from the card's ({len(want)} tokens)")
    print(f"formatter: {len(cases) - bad}/{len(cases)} prompts equal the card's ids token for token "
          f"(default instruction: {len(pairs)}, custom: {len(cases) - len(pairs)})  "
          f"verdict={'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


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
        self.worst_diff = self.worst_score = self.worst_own = 0.0

    def fail(self, why):
        self.ok = False
        print(f"FAIL: {why}")

    def scored(self, url, c):
        """One case through the route: ids equal HF's, finite logits and score within the gates."""
        body = {"query": c["query"], "documents": [c["document"]], "return_documents": False,
                "return_logits": True, "return_prompt_token_ids": True}
        if c["instruct"]:
            body["instruct"] = c["instruct"]
        items = post(url + "/v1/rerank", body)["results"]
        if len(items) != 1:
            return self.fail(f"{c['kind']}: {len(items)} results for one document")
        meta = items[0]["meta_info"] or {}
        if meta.get("prompt_token_ids") != c["ids"]:
            return self.fail(f"{c['kind']}: ids differ from HF's ({c['query']!r})")
        values = (meta.get("logit_yes"), meta.get("logit_no"), items[0]["relevance_score"], c["diff"], c["score"])
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return self.fail(f"{c['kind']}: a non-finite logit or score {values}")
        diff = meta["logit_yes"] - meta["logit_no"]
        got = items[0]["relevance_score"]
        self.worst_diff = max(self.worst_diff, abs(diff - c["diff"]))
        self.worst_score = max(self.worst_score, abs(got - c["score"]))
        self.worst_own = max(self.worst_own, abs(got - c.get("own_score", c.get("bf16_score"))))  # older files: bf16_
        print(f"{c['kind']:12} tokens={c['tokens']:6} ref={c['score']:.6f} got={got:.6f} "
              f"diff={diff:+.4f} ref_diff={c['diff']:+.4f}")

    def special(self, url, c):
        """The masked case: TensorFold's ids must differ from HF's (request text stays plain)."""
        items = post(url + "/v1/rerank", {"query": c["query"], "documents": [c["document"]],
                                          "return_prompt_token_ids": True})["results"]
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
        order = [r["index"] for r in post(url + "/v1/rerank", body)["results"]]
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
          f"worst |score - the reference's own-head score|={g.worst_own:.5f} (recorded)  verdict={'PASS' if g.ok else 'FAIL'}")
    return 0 if g.ok else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    src = sub.add_parser("sources")
    src.add_argument("model", type=Path)
    o = sub.add_parser("oracle")
    o.add_argument("model", type=Path)
    o.add_argument("out")
    o.add_argument("--window", type=int, action="append", default=[])
    o.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    o.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("url")
    c.add_argument("--window", type=int, default=0)
    fm = sub.add_parser("formatter")
    fm.add_argument("model", type=Path)
    fm.add_argument("url")
    args = p.parse_args()
    if args.cmd == "formatter":
        return formatter(args)
    if args.cmd == "sources":
        return sources(args)
    if args.cmd == "oracle":
        oracle(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
