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
Reference (writes JSON with token ids, vectors and the card's 2 x 2 scores):
    python tools/zig/oracle_qwen3_embedding.py ref --model DIR --dtype float32 --device cpu --out ref32.json
Compare a running server's /v1/embeddings (same texts, one batch and one by one) with the fp32 target and the
transformers BF16 bar:
    python tools/zig/oracle_qwen3_embedding.py check --ref ref32.json --bar ref16.json --url http://127.0.0.1:8080
Gates: cosine distance (1 - cos) to fp32 at most 0.001 and at most --bar-factor times the BF16 bar's worst distance;
unit norm; the card's scores within 0.01; batch, alone and token-id vectors equal as fp32 bytes (base64).
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
# The Qwen3 model, configuration and tokenizer code the card's example runs (package-relative).
LIBRARY = {
    "models/qwen3/modeling_qwen3.py": "63a6213c15d6fb04e58a1c64fd05a4cdfb49476443fbeef524f0e0486043824b",
    "models/qwen3/configuration_qwen3.py": "49d9ccd0f29ccd977b93dba2004f84b867891b3d0509b71fd8ba3ed01054e64d",
    "models/qwen2/tokenization_qwen2.py": "fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628",
}

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
    texts = card_texts()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(args.model, dtype=dtype, local_files_only=True).to(args.device).eval()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**batch)
    emb = F.normalize(last_token_pool(out.last_hidden_state, batch["attention_mask"]).float(), p=2, dim=1)
    scores = (emb[:2] @ emb[2:].T).tolist()
    ids = [tokenizer(t)["input_ids"] for t in texts]
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
    check_pins("--ref", ref)
    check_pins("--bar", bar)
    texts = ref["texts"]
    n = len(texts)
    width = check_vectors("--ref", ref, n)
    if check_vectors("--bar", bar, n) != width or ref.get("ids") != bar.get("ids"):
        raise SystemExit("FAIL: --ref and --bar differ in width or token ids")
    bar_worst = max(1 - cosine(b, r) for b, r in zip(bar["vectors"], ref["vectors"]))
    if not math.isfinite(bar_worst):
        raise SystemExit("FAIL: the bar's distance is not finite")
    limit = min(args.max_distance, args.bar_factor * bar_worst)
    print(f"bar: transformers bf16 worst distance {bar_worst:.2e}; limit {limit:.2e}")
    enc = {"encoding_format": "base64"}
    batch = vectors(post(args.url, {"input": texts, **enc}), n, width)
    alone = [vectors(post(args.url, {"input": t, **enc}), 1, width)[0] for t in texts]
    ids = vectors(post(args.url, {"input": ref["ids"], **enc}), n, width)
    ok = True
    for i in range(n):
        v = floats(batch[i])
        dist = 1 - cosine(v, ref["vectors"][i])
        norm = math.sqrt(sum(x * x for x in v))
        same, by_ids = batch[i] == alone[i], batch[i] == ids[i]
        print(f"input {i}: distance {dist:.2e}, norm {norm:.6f}, batch == alone {same}, ids == text {by_ids}")
        ok &= math.isfinite(dist) and dist <= limit and same and by_ids and abs(norm - 1) < 1e-4
    vs = [floats(b) for b in batch]
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
    args = p.parse_args()
    {"sources": sources, "ref": reference, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
