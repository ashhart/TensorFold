"""Rebuild zig/src/families/ernie5_5/fixtures from the release's files and its safetensors headers."""
import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "zig/src/families/ernie5_5/fixtures"
RELEASE = "baidu/ERNIE-5.5-Flash-Preview@ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe"
PINS = {"transformers": "5.20.0", "tokenizers": "0.23.3"}
SHA256 = {
    "config.json": "e249f88ab157f4babedfb4bf0f1557577d703c1c8d5e0c61885f0c3e4094f7aa",
    "tokenizer.json": "20e7c81f36fdfb90db664583fc6426d5c2ac6d6bcaf631c910941a8d8f30f2d9",
    "tokenizer_config.json": "3643352cfb6ea8efbd6dd2b1240570e0527ea31c36bc17f8bfdee6970fd1384b",
    "encoding_ernie.py": "0b353d1eb330574cab7704c7a09323fefb0b1b0b594db151f96b67848e3d1912",
    "tokenization_ernie.py": "50dde281b9913278cee68ba587398ebb500343447687f67e4ffe06df83098ec2",
    "decoding_ernie.py": "0ff13c80fcc8c10237c259d8bcd04e7b854ee940605bd84421c9eb89e7092b97",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "encoding_ernie.py", "tokenization_ernie.py", "decoding_ernie.py")
SHARDS = 128
PROTOCOL_IDS = 21  # added tokens 0 to 20: pad, begin, mask, the FIM markers and the chat protocol's tags
MERGES = 1500

HELLO = "Hello, I'll go 12345 times!"
UNICODE = "Café naïve — 你好，世界 \U0001f30d ñandú ١٢٣ 12½34"
CODE = "def f(x):\n    return x**2  # square\n\n\tprint(f(3))\n"
EXPLAIN = "Explain MoE routing in two sentences."
MARKERS = "Literal markers: <|ernie:im_end|> and <ernie:think> inside text."
USERS = (HELLO, UNICODE, CODE, "It's 1234567 DON'T stop;  we'll see ...\r\nnext?", "a", "", "  two  spaces  \n\n",
         EXPLAIN, MARKERS, "7" * 600 + " digits")
CHAT = ([{"user": u} for u in USERS]
        + [{"user": u, "thinking": False} for u in (HELLO, UNICODE, CODE, MARKERS)]
        + [{"user": HELLO, "effort": e} for e in ("low", "medium", "high")]
        + [{"user": EXPLAIN, "system": "You are terse."}, {"user": EXPLAIN, "system": MARKERS, "thinking": False},
           {"user": EXPLAIN, "system": "", "effort": "low"}])
RAW = (HELLO, UNICODE, CODE, MARKERS, "Version 2026.11 costs $1234567.89!", "  \r\n\t ", "")


def dump(value, depth=0):
    """JSON with one space of indent per level and every list of integers on one line."""
    pad = " " * (depth + 1)
    if isinstance(value, dict):
        rows = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {dump(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + "\n" + " " * depth + "}"
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return "[" + ", ".join(str(x) for x in value) + "]"
    if isinstance(value, list) and all(isinstance(x, list) for x in value):
        return "[" + ", ".join(dump(x) for x in value) + "]"
    if isinstance(value, list):
        return "[\n" + ",\n".join(pad + dump(v, depth + 1) for v in value) + "\n" + " " * depth + "]"
    return json.dumps(value, ensure_ascii=False)


def pinned(release, name):
    data = (release / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise SystemExit(f"{release / name} is not {RELEASE}'s {name}")
    return data


def spans(layers):
    """Layer numbers as ranges: 0-6,8-13."""
    out, run = [], []
    for n in sorted(layers):
        if run and n != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
            run = []
        run.append(n)
    out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ",".join(out)


def release_header(headers):
    """Name to (dtype, shape) over the release's model shards, read from each file's header alone."""
    tensors = {}
    for i in range(1, SHARDS + 1):
        path = headers / f"model-{i:05d}-of-{SHARDS:05d}.safetensors"
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, e in header.items():
            if name != "__metadata__":
                if name in tensors:
                    raise SystemExit(f"{name} is in two shards")
                tensors[name] = (e["dtype"], e["shape"])
    return tensors


def fold(tensors):
    """The table the host tests unfold: tag, count and sha256 of the sorted lines, then one line per name pattern."""
    names = sorted(tensors)
    shape = lambda dims: "x".join(str(d) for d in dims)  # noqa: E731 - one use
    digest = hashlib.sha256("\n".join(f"{n} {tensors[n][0]} {shape(tensors[n][1])}" for n in names).encode()).hexdigest()
    groups = {}
    for name in names:
        parts = name.split(".")
        at = [i for i, p in enumerate(parts) if p.isdigit()]
        experts = len(at) == 2 and parts[at[1] - 1] == "experts"
        if len(at) > 1 and not experts:
            raise SystemExit(f"{name}: an index past the layer's and the expert's")
        layer = int(parts[at[0]]) if at else None
        expert = int(parts[at[1]]) if experts else None
        if at:
            parts[at[0]] = "{L}"
        if experts:
            parts[at[1]] = "{E}"
        dtype, dims = tensors[name]
        groups.setdefault((".".join(parts), dtype, shape(dims)), {}).setdefault(layer, set()).add(expert)
    lines = [f"ernie55-release {len(names)} {digest} {RELEASE}"]
    for (pattern, dtype, dims), layers in sorted(groups.items()):
        line = f"{pattern} {dtype} {dims}"
        if layers.keys() != {None}:
            line += " " + spans(layers)
        counts = {len(e) for e in layers.values()}
        if "{E}" in pattern:
            if len(counts) != 1 or any(e != set(range(len(e))) for e in layers.values()):
                raise SystemExit(f"{pattern}: experts are not 0 to N-1 in every layer")
            line += f" {counts.pop()}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def byte_alphabet():
    """The 256 characters a byte-level tokenizer spells its bytes with (GPT-2's table)."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = printable + [256 + i for i, b in enumerate(b for b in range(256) if b not in printable)]
    return {chr(c) for c in chars}


def cut_tokenizer(full):
    """tokenizer.json cut to the protocol's added tokens, the byte alphabet and the first MERGES merges."""
    model = full["model"]
    merges = model["merges"][:MERGES]
    added = [t for t in full["added_tokens"] if t["id"] < PROTOCOL_IDS]
    keep = byte_alphabet() | {x for pair in merges for x in pair} | {a + b for a, b in merges} | {t["content"] for t in added}
    vocab = {t: i for t, i in sorted(model["vocab"].items(), key=lambda kv: kv[1]) if t in keep}
    return {**full, "added_tokens": added, "model": {**model, "vocab": vocab, "merges": merges}}


def load(release, tokenizer_json, tmp):
    """The release's tokenizer over `tokenizer_json`, loaded from a folder with no config.json for AutoTokenizer."""
    from transformers import AutoTokenizer
    folder = Path(tempfile.mkdtemp(dir=tmp))
    for name in TOKENIZER_FILES:
        (folder / name).write_bytes(pinned(release, name) if name != "tokenizer.json" else tokenizer_json)
    return AutoTokenizer.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True)


def chat_cases(tok):
    """Each CHAT request rendered by the release's encoding: text, the caller's byte spans, ids and decoded text."""
    out = []
    for case in CHAT:
        messages = [{"role": "system", "content": case["system"]}] if "system" in case else []
        messages.append({"role": "user", "content": case["user"]})
        kwargs = {}
        if case.get("thinking") is False:
            kwargs["enable_thinking"] = False
        if "effort" in case:
            kwargs["reasoning_effort"] = case["effort"]
        rendered = tok.render_chat(messages, add_generation_prompt=True, **kwargs)
        text, caller, at = "", [], 0
        for seg in rendered.segments:
            size = len(seg.text.encode("utf-8"))
            if not seg.allow_special and size:
                caller.append([at, at + size])
            text += seg.text
            at += size
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs))
        if tok.encode_segments(rendered.segments) != ids:
            raise SystemExit(f"segment ids differ from apply_chat_template for {case}")
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise SystemExit(f"decoded text differs from the rendered text for {case}")
        out.append({**case, "text": text, "spans": caller, "ids": ids})
    return out


def raw_cases(tokenizer_json):
    """RAW texts' ids through the tokenizers library alone, added tokens matched as in tokenizer.json."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
    out = []
    for text in RAW:
        ids = tok.encode(text, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != text:
            raise SystemExit(f"decoded text differs from the input for {text!r}")
        out.append({"text": text, "ids": ids})
    return out


def build(release, headers):
    full = pinned(release, "tokenizer.json")
    cut = json.dumps(cut_tokenizer(json.loads(full)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    calls = ("render_chat(messages, add_generation_prompt=True[, enable_thinking=False][, reasoning_effort=E]) and "
             "apply_chat_template(..., tokenize=True) of the release's ErnieChatTokenizer; raw: tokenizers' encode")
    reference = {**PINS, "files": f"{RELEASE}: " + ", ".join(TOKENIZER_FILES), "calls": calls}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["HF_MODULES_CACHE"] = str(Path(tmp) / "modules")
        os.environ["HF_HUB_OFFLINE"] = "1"
        prompt = {"reference": reference, "chat": chat_cases(load(release, full, tmp)), "raw": raw_cases(full)}
        mini = {"reference": {**reference, "tokenizer": f"mini_tokenizer.json: tokenizer.json cut to added tokens 0 to "
                              f"{PROTOCOL_IDS - 1}, the byte alphabet and the first {MERGES} merges"},
                "chat": chat_cases(load(release, cut, tmp)), "raw": raw_cases(cut)}
    return {
        "config.json": pinned(release, "config.json"),
        "prompt_cases.json": (dump(prompt) + "\n").encode("utf-8"),
        "mini_tokenizer.json": cut + b"\n",
        "mini_cases.json": (dump(mini) + "\n").encode("utf-8"),
        "release_header.txt": fold(release_header(headers)).encode("utf-8"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help=f"{RELEASE}: config.json and " + ", ".join(TOKENIZER_FILES))
    parser.add_argument("--headers", type=Path, required=True, help="its 128 model shards, or copies of their headers alone")
    parser.add_argument("--out", type=Path, default=FIXTURES, help="the fixtures directory to write or compare")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a file differs from the one in --out")
    args = parser.parse_args()
    for name, want in PINS.items():
        if version(name) != want:
            raise SystemExit(f"{name} {version(name)} is installed, {want} is pinned")
    sys.stdout.reconfigure(encoding="utf-8")
    differ = 0
    for name, data in sorted(build(args.release, args.headers).items()):
        path = args.out / name
        if args.check:
            same = path.is_file() and path.read_bytes() == data
            differ += not same
            print(("same    " if same else "DIFFERS ") + name)
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {name}, {len(data)} bytes")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
