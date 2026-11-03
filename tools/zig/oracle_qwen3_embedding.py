"""Pinned offline Qwen3-Embedding-8B oracle: the model card's Transformers example, and a server checked against it."""

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import urllib.request
from pathlib import Path

USAGE = """
Pins checked before any model code runs (the snapshot's revision, every file's SHA256, Transformers 5.22.1's tree):
    python tools/zig/oracle_qwen3_embedding.py sources --model DIR
Reference (writes JSON with token ids, vectors and the card's 2 x 2 scores; the card's 4 inputs, then 5 long inputs
of about 2K to 32K tokens cut from pinned files):
    python tools/zig/oracle_qwen3_embedding.py ref --model DIR --dtype float32 --device cpu --out ref32.json
Compare a running server's /v1/embeddings (same texts in batches that fit its window, and one by one) with the fp32
target and the transformers BF16 bar, at full width and at MRL widths 32, 1024 and 4096:
    python tools/zig/oracle_qwen3_embedding.py check --ref ref32.json --bar ref16.json --url http://127.0.0.1:8080
Gates, at each width: cosine distance (1 - cos) to fp32 (cropped and renormalized for MRL) at most 0.001 and at most
--bar-factor times the BF16 bar's worst distance at that width; unit norm. At full width also: the card's scores
within 0.01; batch, alone and token-id vectors equal as fp32 bytes (base64).
"""

CHECKPOINT = ("Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af")
# Every non-weight file of the pinned snapshot: card, configs, sentence-transformers modules, tokenizer, licence.
SOURCES = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "1_Pooling/config.json": "2e1da26b3fd65cf7e370d2fabf28a8c59efa7edb525d1b8b50be8e5ca1048ea6",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "1fd6f3583d642a3319efb3eb8c22b6aa6320e6ab92263447bd84fd92615206d4",
    "config.json": "dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7",
    "config_sentence_transformers.json": "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
    "generation_config.json": "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors.index.json": "ceaf1429bfa18eecc75d079737e8d1fe8dc0627f5ec7e6a985514626d1ac6a77",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "tokenizer.json": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
    "tokenizer_config.json": "2f58f4bbd7bbce15d683f525954ef3a92cd82f5e06415a9c513859bf8ab72436",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
# The shards' SHA256 (the Hub's LFS object ids).
WEIGHTS = {
    "model-00001-of-00004.safetensors": "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451",
    "model-00002-of-00004.safetensors": "dff635b0f6dbbaad2a2d633ef037ec0a39bc165cc1806c712fbd6fcbcb4526c0",
    "model-00003-of-00004.safetensors": "30b1d4c53d84eb018f642cad7b373f0aabf79699872d8702c1f38577c0a59a2f",
    "model-00004-of-00004.safetensors": "36cbc9c60375693629f25743c1e77ebb1724af58e671b2376463193c7fd21ef6",
}
VERSION = "5.22.1"
# tree_sha256 of the whole installed Transformers package, so the shared modeling and tokenizer code is pinned too.
TRANSFORMERS_TREE = "b08cead6d1b841ef9723106a81a92ec37d67aff3e43a198e56b7dc76538d377b"
# The Qwen3 model, configuration and tokenizer code the card's example runs, and the files the long inputs are cut
# from (package-relative).
LIBRARY = {
    "models/qwen3/modeling_qwen3.py": "63a6213c15d6fb04e58a1c64fd05a4cdfb49476443fbeef524f0e0486043824b",
    "models/qwen3/configuration_qwen3.py": "49d9ccd0f29ccd977b93dba2004f84b867891b3d0509b71fd8ba3ed01054e64d",
    "models/qwen2/tokenization_qwen2.py": "fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628",
    "tokenization_utils_base.py": "a70a4eda47e15756cb4ddd10ecd3a4c2c92d00b556fc00c47e7ce8a6bdb15dbd",
    "generation/utils.py": "966f70ba83d94c9ce07b21150be49ff84b24eadbf9c47de052a1fb0f8ba38679",
    "modeling_utils.py": "251aaae7733affbb7cdb5aa05a84157c46cac02518d868898cd6d6282cc14571",
}
# Long inputs: (most tokens with the closing <|endoftext|>, where the file is pinned, file). Each is the file's longest
# prefix of whole lines within that many tokens, so the 32K one fits a 32,768-token window.
LONG = [
    (2048, "checkpoint", "LICENSE"),
    (4096, "checkpoint", "README.md"),
    (8192, "library", "tokenization_utils_base.py"),
    (16384, "library", "generation/utils.py"),
    (32768, "library", "modeling_utils.py"),
]
# Every fidelity input, in order (the card's 4, then LONG): its tokens, the SHA256 of its UTF-8 text and the SHA256 of
# its token IDs as little-endian uint32. A reference whose inputs differ is refused, whatever its pin record says.
INPUTS = [
    (27, "b11f8d3ae64c3c7029275c5d8d82172ef31a9a354223c12845b61047a0c972e8", "3c1537f6fc0a496f2b68c1aaa38484d8a82030338ae52bbd955b45b50c9efc24"),
    (23, "bd6e4e8bb1b5c2e15db6003175c53f084690aeb6c09b34f1808161ef3353d524", "a08f60368f1da49a05dd929f406c503bcd0864ee464f25807866753bffe3062c"),
    (8, "b6d0a6ed3a94d5e532ae83127502407102e51342d1db011caf4c5b9808a1e131", "c1e33d8973efc767e95f15a7b31af3c68eccc5ca89e0c2648f5337fec3c8f2c6"),
    (31, "ba5bd6b5de9d9f827336e76d982b5dbae8aa3cc075c115f57405ba370bba8dbc", "4bd9685251a5e3587a645c0bf96dbc76ca70998073523b5537a1513c15f4c8ea"),
    (2044, "f60c57fd5911f2c10704c555aef3a6e38543afa78604385ad5d8dde87050e0ff", "e10c109c63916e2cacd0143d8295fa5fa449c9eb09b7d3126405bfec9a6c9d92"),
    (4043, "93c6978d85521f7b15d47dfedd386b72f063fd25ea5ada5ccf38fefdc88e7766", "a87db7cfe361a40e15c9b209796f6260bfb74017bc1e897e6e5be3d1a557cdbf"),
    (8177, "3f5c76527fb4d032b31c827e5ec4d171a9feb94e7ab2ab96341028b1dd412505", "17c5b8a1cb0f288b40b9810549881b901eea58c83edb6e6d1b89ac1beab603eb"),
    (16383, "c0a043846083a03ec205a34af84d7b1053a8d1a4ea4f3d3c028bb4c048ef1366", "528fda3905d2fc0340f8d5ca34f4bd91703613d76b40e5ac0835fab641acf173"),
    (32759, "009256a9df975a6f67e5984760078c1dbcb0bd0b9aefbce0bdb8373ebd386dd7", "1dfcec470654b993960be543f8e84c461bc28d743aa5a50ee63bd81a9b8e15a8"),
]
# MRL widths checked besides the full one (the card's range is 32 to 4096).
WIDTHS = (32, 1024, 4096)

TASK = "Given a web search query, retrieve relevant passages that answer the query"
CARD_SCORES = [[0.7493016123771667, 0.0750647559762001], [0.08795969933271408, 0.6318399906158447]]


def detailed_instruct(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery:{query}"


def card_texts() -> list:
    queries = [detailed_instruct(TASK, "What is the capital of China?"), detailed_instruct(TASK, "Explain gravity")]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and "
        "is responsible for the movement of planets around the sun.",
    ]
    return queries + documents


def long_texts(model: Path, package: Path, count) -> list:
    """Each LONG file's longest prefix of whole lines whose token count (as `count` gives it) is within its bound."""
    roots = {"checkpoint": model, "library": package}
    out = []
    for most, where, name in LONG:
        lines = (roots[where] / name).read_text(encoding="utf-8").splitlines(keepends=True)
        lo, hi = 0, len(lines)  # the longest k with count(lines[:k]) <= most, by bisection on whole lines
        while lo < hi:
            k = (lo + hi + 1) // 2
            lo, hi = (k, hi) if count("".join(lines[:k])) <= most else (lo, k - 1)
        text = "".join(lines[:lo])
        if count(text) < most // 2:
            raise SystemExit(f"FAIL: {name} gives only {count(text)} of {most} tokens")
        out.append(text)
    return out


def fingerprint(text: str, ids: list) -> tuple:
    """An input's tokens, text SHA256 and token-ID SHA256 (little-endian uint32), as INPUTS pins them."""
    import struct
    raw = struct.pack(f"<{len(ids)}I", *ids)
    return len(ids), hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(raw).hexdigest()


def check_inputs(name: str, texts, ids) -> None:
    """Refuse inputs that are not the pinned fidelity set, text and token IDs both."""
    if not isinstance(texts, list) or not isinstance(ids, list) or len(texts) != len(INPUTS) or len(ids) != len(INPUTS):
        raise SystemExit(f"FAIL: {name} does not hold the {len(INPUTS)} pinned inputs")
    for i, (text, one, pin) in enumerate(zip(texts, ids, INPUTS)):
        if not isinstance(text, str) or not isinstance(one, list) or not all(type(x) is int and 0 <= x <= 0xFFFFFFFF for x in one):
            raise SystemExit(f"FAIL: {name} input {i} is not a text and its token IDs")
        if fingerprint(text, one) != pin:
            raise SystemExit(f"FAIL: {name} input {i} differs from its pin")


def groups(ids: list, window: int, most: int = 128) -> list:
    """Consecutive input indices in requests of at most `most` inputs and `window` tokens in all."""
    out, part, used = [], [], 0
    for i, one in enumerate(ids):
        if len(one) > window:
            raise SystemExit(f"FAIL: input {i} has {len(one)} tokens, more than the {window}-token window")
        if part and (used + len(one) > window or len(part) == most):
            out.append(part)
            part, used = [], 0
        part.append(i)
        used += len(one)
    return out + [part] if part else out


def crop(v, width: int) -> list:
    """An MRL vector: the first `width` values, unit length again."""
    head = list(v[:width])
    norm = math.sqrt(sum(x * x for x in head))
    return [x / norm for x in head]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed(root: Path, pins: dict, what: str) -> dict:
    """The pinned files' hashes under root, or a refusal naming every file that differs from its pin."""
    found = {name: digest(root / name) if (root / name).is_file() else None for name in pins}
    if wrong := sorted(name for name, value in found.items() if value != pins[name]):
        raise SystemExit(f"FAIL: {what} differs from its pin: {', '.join(wrong)}")
    return found


def tree_sha256(package: Path) -> str:
    """Every file of an installed package (bytecode caches excluded) as sorted "path NUL sha256" lines, hashed."""
    if links := sorted(p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_symlink() and p.is_dir()):
        raise SystemExit(f"FAIL: {package.name} has linked directories: {', '.join(links)}")
    files = sorted(p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(package).parts)
    return hashlib.sha256("".join(f"{p.relative_to(package).as_posix()}\0{digest(p)}\n" for p in files).encode()).hexdigest()


def pinned(model: Path) -> dict:
    """Refuse, before any model code runs, a snapshot or Transformers install whose sources differ from the pins."""
    repo, revision = CHECKPOINT
    model = model.resolve()
    if model.parent.name == "snapshots" and model.name != revision:
        raise SystemExit(f"FAIL: snapshot directory is not the pinned revision {revision}")
    if links := sorted(p.relative_to(model).as_posix() for p in model.rglob("*") if p.is_symlink() and p.is_dir()):
        raise SystemExit(f"FAIL: snapshot has linked directories: {', '.join(links)}")
    files = {p.relative_to(model).as_posix() for p in model.rglob("*") if p.is_file()}
    if extra := sorted(files - set(SOURCES) - set(WEIGHTS)):
        raise SystemExit(f"FAIL: snapshot has unpinned files: {', '.join(extra)}")
    if (version := importlib.metadata.version("transformers")) != VERSION:
        raise SystemExit(f"FAIL: expected Transformers {VERSION}, found {version}")
    package = Path(importlib.util.find_spec("transformers").origin).parent
    if (tree := tree_sha256(package)) != TRANSFORMERS_TREE:
        raise SystemExit(f"FAIL: Transformers {VERSION} package tree differs from its pin: {tree}")
    return {"repo": repo, "revision": revision, "checkpoint": hashed(model, SOURCES, f"{repo}@{revision}"),
            "weights": hashed(model, WEIGHTS, f"{repo}@{revision} weights"),
            "transformers": VERSION, "transformers_tree": tree,
            "library": hashed(package, LIBRARY, f"Transformers {VERSION}"),
            "torch": importlib.metadata.version("torch")}


def sources(args) -> None:
    print(json.dumps(pinned(args.model), indent=1))


def expected_pins() -> dict:
    """The pin record every reference must carry (torch's version is recorded, not pinned)."""
    return {"repo": CHECKPOINT[0], "revision": CHECKPOINT[1], "checkpoint": SOURCES, "weights": WEIGHTS,
            "transformers": VERSION, "transformers_tree": TRANSFORMERS_TREE, "library": LIBRARY}


def check_pins(name: str, made: dict) -> None:
    """Refuse a reference whose pin record is missing a field or differs from the pins in any of them."""
    pins = made.get("pins")
    if not isinstance(pins, dict):
        raise SystemExit(f"FAIL: {name} carries no pin record")
    if wrong := sorted(k for k, v in expected_pins().items() if pins.get(k) != v):
        raise SystemExit(f"FAIL: {name} was not made under the pins: {', '.join(wrong)}")


def reference(args) -> None:
    # the pins first: no Transformers or torch code runs before its tree and the snapshot are checked
    pins = pinned(args.model)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    def last_token_pool(last_hidden_states, attention_mask):
        # the card's pooling, verbatim in effect
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), lengths]

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    package = Path(importlib.util.find_spec("transformers").origin).parent
    longs = long_texts(args.model, package, lambda text: len(tokenizer(text)["input_ids"]))
    texts = card_texts() + longs
    ids = [tokenizer(t)["input_ids"] for t in texts]
    check_inputs("the tokenized inputs", texts, ids)  # before the model loads
    model = AutoModel.from_pretrained(args.model, dtype=dtype, local_files_only=True).to(args.device).eval()
    # the card's 4 inputs as the card runs them: one left-padded batch, truncated at 8,192 tokens
    batch = tokenizer(card_texts(), padding=True, truncation=True, max_length=8192, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**batch)
    emb = [F.normalize(last_token_pool(out.last_hidden_state, batch["attention_mask"]).float(), p=2, dim=1)]
    # the long inputs one by one, whole: no padding, no truncation
    for text in longs:
        one = tokenizer(text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**one)
        emb.append(F.normalize(last_token_pool(out.last_hidden_state, one["attention_mask"]).float(), p=2, dim=1))
        print(f"long input: {one['input_ids'].shape[1]} tokens", flush=True)
    emb = torch.cat(emb)
    scores = (emb[:2] @ emb[2:4].T).tolist()
    print("scores", scores)
    print("card  ", CARD_SCORES)
    json.dump({"dtype": args.dtype, "device": args.device, "pins": pins, "texts": texts, "ids": ids,
               "vectors": emb.cpu().tolist(), "scores": scores}, open(args.out, "w"))


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def vectors(resp: dict, count: int, width: int) -> list:
    """The response's fp32 vectors, in index order, after count, index, width and finiteness checks."""
    data = resp.get("data", [])
    if len(data) != count or [d.get("index") for d in data] != list(range(count)):
        raise SystemExit(f"FAIL: expected {count} vectors with indices 0..{count - 1}, got {len(data)}")
    out = []
    for d in data:
        raw = base64.b64decode(d["embedding"])
        if len(raw) != 4 * width:
            raise SystemExit(f"FAIL: vector {d['index']} has {len(raw) // 4} values, expected {width}")
        out.append(raw)
    for i, raw in enumerate(out):
        if not all(math.isfinite(x) for x in floats(raw)):
            raise SystemExit(f"FAIL: vector {i} is not finite")
    return out


def floats(raw: bytes) -> list:
    import struct
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def check_vectors(name: str, made: dict, count: int) -> int:
    """The reference's width, after checking it holds `count` finite, nonzero vectors of one width."""
    vectors = made.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != count or not vectors:
        raise SystemExit(f"FAIL: {name} does not hold {count} vectors")
    width = len(vectors[0])
    for i, v in enumerate(vectors):
        if len(v) != width or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            raise SystemExit(f"FAIL: {name} vector {i} is not {width} finite values")
        if sum(x * x for x in v) == 0:
            raise SystemExit(f"FAIL: {name} vector {i} is zero")
    return width


def check(args) -> None:
    ref = json.load(open(args.ref))
    bar = json.load(open(args.bar))
    if ref["dtype"] != "float32" or bar["dtype"] != "bfloat16" or ref["texts"] != bar["texts"]:
        raise SystemExit("FAIL: --ref must be the float32 reference and --bar the bfloat16 one, of the same texts")
    if ref["texts"][:4] != card_texts():
        raise SystemExit("FAIL: the references do not start with the card's 4 inputs")
    check_inputs("--ref", ref.get("texts"), ref.get("ids"))
    check_inputs("--bar", bar.get("texts"), bar.get("ids"))
    check_pins("--ref", ref)
    check_pins("--bar", bar)
    texts = ref["texts"]
    n = len(texts)
    width = check_vectors("--ref", ref, n)
    if check_vectors("--bar", bar, n) != width or ref.get("ids") != bar.get("ids"):
        raise SystemExit("FAIL: --ref and --bar differ in width or token ids")
    parts = groups(ref["ids"], args.window)
    enc = {"encoding_format": "base64"}

    def served(inputs: list, dims) -> list:
        """Every input's vector, from requests of the inputs in `parts`, cropped by the server to `dims` if given."""
        out = [b""] * n
        for part in parts:
            body = {"input": [inputs[i] for i in part], **enc} | ({"dimensions": dims} if dims else {})
            for i, raw in zip(part, vectors(post(args.url, body), len(part), dims or width)):
                out[i] = raw
        return out

    def limit_at(dims: int) -> float:
        worst = max(1 - cosine(crop(b, dims), crop(r, dims)) for b, r in zip(bar["vectors"], ref["vectors"]))
        if not math.isfinite(worst):
            raise SystemExit(f"FAIL: the bar's distance at width {dims} is not finite")
        limit = min(args.max_distance, args.bar_factor * worst)
        print(f"width {dims}: transformers bf16 worst distance {worst:.2e}; limit {limit:.2e}")
        return limit

    print(f"requests: {[[i for i in part] for part in parts]} (window {args.window})")
    limit = limit_at(width)
    batch = served(texts, None)
    alone = [vectors(post(args.url, {"input": t, **enc}), 1, width)[0] for t in texts]
    ids = served(ref["ids"], None)
    ok = True
    for i in range(n):
        v = floats(batch[i])
        dist = 1 - cosine(v, ref["vectors"][i])
        norm = math.sqrt(sum(x * x for x in v))
        same, by_ids = batch[i] == alone[i], batch[i] == ids[i]
        print(f"input {i} ({len(ref['ids'][i])} tokens): distance {dist:.2e}, norm {norm:.6f}, "
              f"batch == alone {same}, ids == text {by_ids}")
        ok &= math.isfinite(dist) and dist <= limit and same and by_ids and abs(norm - 1) < 1e-4
    for dims in args.widths:
        limit = limit_at(dims)
        for i, raw in enumerate(served(texts, dims)):
            v = floats(raw)
            dist = 1 - cosine(v, crop(ref["vectors"][i], dims))
            norm = math.sqrt(sum(x * x for x in v))
            print(f"width {dims} input {i}: distance {dist:.2e}, norm {norm:.6f}")
            ok &= math.isfinite(dist) and dist <= limit and abs(norm - 1) < 1e-4
    vs = [floats(b) for b in batch[:4]]
    scores = [[sum(x * y for x, y in zip(vs[q], vs[2 + d])) for d in range(2)] for q in range(2)]
    worst = max(abs(scores[q][d] - CARD_SCORES[q][d]) for q in range(2) for d in range(2))
    print("scores", scores, "card", CARD_SCORES, f"max diff {worst:.4f}")
    ok &= worst <= args.max_score_diff
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    so = sub.add_parser("sources")
    so.add_argument("--model", required=True, type=Path)
    r = sub.add_parser("ref")
    r.add_argument("--model", required=True, type=Path)
    r.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--ref", required=True, help="float32 reference JSON (the target)")
    c.add_argument("--bar", required=True, help="bfloat16 transformers reference JSON (the bar)")
    c.add_argument("--bar-factor", type=float, default=2.0)
    c.add_argument("--url", default="http://127.0.0.1:8080")
    c.add_argument("--max-distance", type=float, default=0.001)
    c.add_argument("--max-score-diff", type=float, default=0.01)
    c.add_argument("--window", type=int, default=32768, help="the server's window: the most tokens in one request")
    c.add_argument("--widths", type=int, nargs="*", default=list(WIDTHS), help="MRL widths to check")
    args = p.parse_args()
    {"sources": sources, "ref": reference, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
