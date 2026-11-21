"""Rebuild zig/src/families/inkling2/fixtures from the Small, tiny-random and Mini checkpoint inputs."""
import argparse
import hashlib
import json
import struct
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "zig/src/families/inkling2/fixtures"
RELEASE = "thinkingmachines/Inkling-2-Small@2ad1157016e1aac77b6ff1b770e7c05229403134"
TINY = "tiny-random/inkling-2@0a57c72e4e5b68aafe1d983dbb62980344a5fcd7"
MINI_RELEASE = "thinkingmachines/Inkling-2-Mini"
MINI_PROVENANCE = {
    "release": MINI_RELEASE,
    "revision": None,
    "revision_status": "full commit SHA absent from the supplied published metadata",
    "published": "2026-11-20",
    "license": "apache-2.0",
}
MINI_HEADER_PROVENANCE = (
    f"release={MINI_RELEASE} revision=unavailable-full-commit-SHA-absent-from-supplied-inputs "
    "published=2026-11-20 license=apache-2.0"
)
PINS = {"transformers": "5.20.0", "tokenizers": "0.23.3", "jinja2": "3.1.6"}
MERGES = 1500

HELLO = "Hello, I'll go 12345 times!"
UNICODE = "Caf\u00e9 na\u00efve \u2014 \u4f60\u597d \U0001f30d \u00f1and\u00fa"
CODE = "def f(x):\n    return x**2  # square\n\n\tprint(f(3))\n"
EXPLAIN = "Explain MoE routing in two sentences."
TERSE = "You are terse."
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", 0.5, 0.99, 0.1, 0)
USERS = (HELLO, UNICODE, CODE, "It's 1234567 DON'T stop;  we'll see ...\r\nnext?", "a", "  two  spaces  and trailing  ", EXPLAIN,
         "Literal markers: <|end_message|> and <|message_user|> inside text.", "x" * 50 + " word" * 40 + " ")
CHAT = ([{"user": u} for u in USERS]
        + [{"user": u, "effort": "none"} for u in (HELLO, UNICODE, CODE)]
        + [{"user": HELLO, "effort": e} for e in EFFORTS]
        + [{"user": EXPLAIN, "system": TERSE}, {"user": EXPLAIN, "system": TERSE, "effort": "none"}])
RAW = ("Hello world", "I'll 12345 \u00c4\u00d6\u00dc", "\n\n  \t x", "<|endoftext|>plain", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
       "e\u0301 combining")
MINI = (HELLO, "I'LL DON'T it's we've", UNICODE, CODE, "  two  spaces  and trailing  ", "a\r\nb\n\n\nc", "1234567890",
        "CamelCaseWord \u00c0\u00c9\u00ce \u01c5", "x" * 20, "<|message_user|><|content_text|>hi<|end_message|>",
        "e\u0301 combining \u01c6", "...!!!???///\n", "word's 'quoted' text", "")


def dump(value, depth=0):
    """JSON with one space of indent per level and every list of integers on one line."""
    pad = " " * (depth + 1)
    if isinstance(value, dict):
        rows = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {dump(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + "\n" + " " * depth + "}"
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return "[" + ", ".join(str(x) for x in value) + "]"
    if isinstance(value, list):
        return "[\n" + ",\n".join(pad + dump(v, depth + 1) for v in value) + "\n" + " " * depth + "]"
    return json.dumps(value, ensure_ascii=False)


def read_header(path):
    """The tensors of a safetensors file or extracted JSON header as {name: (dtype tag, shape)}."""
    if path.suffix == ".json":
        meta = json.loads(path.read_text(encoding="utf-8"))
    else:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(length))
    return {name: (v["dtype"], tuple(v["shape"])) for name, v in meta.items() if name != "__metadata__"}


def release_header(directory, stem):
    """Read the supplied extracted header when present, otherwise the release's safetensors file."""
    extracted = directory / f"{stem}.header.json"
    return read_header(extracted if extracted.is_file() else directory / f"{stem}.safetensors")


def annotated_mini_config(path):
    """The published config with explicit fixture provenance; no revision is inferred from its bytes."""
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or "_fixture_provenance" in value:
        raise SystemExit(f"{path}: expected an unannotated config object")
    provenance = json.dumps(MINI_PROVENANCE, ensure_ascii=False, indent=2).replace("\n", "\n  ")
    annotation = f'  "_fixture_provenance": {provenance},\n'.encode("utf-8")
    if not raw.startswith(b"{\n"):
        raise SystemExit(f"{path}: expected the published indented config layout")
    return raw.replace(b"{\n", b"{\n" + annotation, 1)


def spans(layers):
    """Layer numbers as comma-separated ranges, 0-3,5."""
    layers = sorted(layers)
    runs, start, prev = [], layers[0], layers[0]
    for x in layers[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    return ",".join(runs)


def fold(tag, tensors, provenance=""):
    """The table the host tests unfold: tag, count and sha256 of the sorted lines, then one line per name pattern."""
    names = sorted(tensors)

    def shape(dims):
        return "x".join(str(d) for d in dims)

    digest = hashlib.sha256("\n".join(f"{n} {tensors[n][0]} {shape(tensors[n][1])}" for n in names).encode()).hexdigest()
    groups = {}
    for name in names:
        parts = name.split(".")
        at = [i for i, p in enumerate(parts) if p.isdigit()]
        if len(at) > 1:
            raise SystemExit(f"{name}: more than one layer index")
        layer = int(parts[at[0]]) if at else None
        if at:
            parts[at[0]] = "{L}"
        dtype, dims = tensors[name]
        groups.setdefault((".".join(parts), dtype, shape(dims)), []).append(layer)
    lines = [f"{tag} {len(names)} {digest}" + (f" {provenance}" if provenance else "")]
    for (pattern, dtype, dims), layers in sorted(groups.items()):
        lines.append(f"{pattern} {dtype} {dims}" + ("" if layers == [None] else " " + spans(layers)))
    return "\n".join(lines) + "\n"


def byte_alphabet():
    """The 256 characters a byte-level tokenizer spells its bytes with (GPT-2's table)."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = printable + [256 + i for i, b in enumerate(b for b in range(256) if b not in printable)]
    return {chr(c) for c in chars}


def cut_tokenizer(full):
    """tokenizer.json cut to its added tokens, its byte alphabet and the first MERGES merges with their parts."""
    model = full["model"]
    merges = model["merges"][:MERGES]
    keep = byte_alphabet() | {x for pair in merges for x in pair} | {a + b for a, b in merges}
    vocab = {t: i for t, i in sorted(model["vocab"].items(), key=lambda kv: kv[1]) if t in keep}
    return {**full, "model": {**model, "vocab": vocab, "merges": merges}}


def mini_cases(cut_text):
    """Ids and decoded text of MINI under the cut tokenizer, from the tokenizers library."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(cut_text)
    cases = []
    for text in MINI:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
    reference = (f"tokenizers {PINS['tokenizers']} on mini_tokenizer.json (the checkpoint's tokenizer.json cut to its added "
                 f"tokens, its byte alphabet and the first {MERGES} merges)")
    return {"reference": reference, "cases": cases}


def prompt_cases(release):
    """Rendered prompts, ids and decoded text of CHAT and RAW, from transformers loading the release's files."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(release), local_files_only=True)
    chat = []
    for case in CHAT:
        messages = [{"role": "system", "content": case["system"]}] if "system" in case else []
        messages.append({"role": "user", "content": case["user"]})
        extra = {"reasoning_effort": case["effort"]} if "effort" in case else {}
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        decoded = tok.decode(ids)
        if decoded != prompt:
            raise SystemExit(f"decoded text differs from the prompt for {case}")
        chat.append({"user": case["user"], "prompt": prompt, "ids": ids, "decoded": decoded,
                     **{k: case[k] for k in ("system", "effort") if k in case}})
    raw = []
    for text in RAW:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        decoded = tok.decode(ids)
        if decoded != text:
            raise SystemExit(f"decoded text differs from the input for {text!r}")
        raw.append({"text": text, "ids": ids, "decoded": decoded})
    reference = {
        "transformers": PINS["transformers"], "tokenizers": PINS["tokenizers"], "jinja2": PINS["jinja2"],
        "files": f"{RELEASE}: tokenizer.json, tokenizer_config.json, chat_template.jinja",
        "call": "apply_chat_template(messages, tokenize=False, add_generation_prompt=True[, reasoning_effort=E]); "
                "tok(prompt, add_special_tokens=False)",
    }
    return {"reference": reference, "chat": chat, "raw": raw}


def build(release, tiny, mini):
    """Every fixture file as bytes."""
    full = json.loads((release / "tokenizer.json").read_text(encoding="utf-8"))
    cut = json.dumps(cut_tokenizer(full), separators=(",", ":"), ensure_ascii=False) + "\n"
    return {
        "config.json": (release / "config.json").read_bytes(),
        "chat_template.jinja": (release / "chat_template.jinja").read_bytes(),
        "tiny_config.json": (tiny / "config.json").read_bytes(),
        "mini_tokenizer.json": cut.encode("utf-8"),
        "mini_cases.json": (dump(mini_cases(cut)) + "\n").encode("utf-8"),
        "prompt_cases.json": (dump(prompt_cases(release)) + "\n").encode("utf-8"),
        "tiny_model_header.txt": fold("inkling2-tiny-model", read_header(tiny / "model.safetensors")).encode("utf-8"),
        "tiny_mtp_header.txt": fold("inkling2-tiny-mtp", read_header(tiny / "mtp.safetensors")).encode("utf-8"),
        "mini_config.json": annotated_mini_config(mini / "config.json"),
        "mini_model_header.txt": fold(
            "inkling2-mini-model", release_header(mini, "model-00001-of-00001"), MINI_HEADER_PROVENANCE
        ).encode("utf-8"),
        "mini_mtp_header.txt": fold(
            "inkling2-mini-mtp", release_header(mini, "mtp"), MINI_HEADER_PROVENANCE
        ).encode("utf-8"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True,
                        help=f"{RELEASE}: config.json, chat_template.jinja, tokenizer.json, tokenizer_config.json")
    parser.add_argument("--tiny", type=Path, required=True, help=f"{TINY}: config.json, model.safetensors, mtp.safetensors")
    parser.add_argument(
        "--mini", type=Path, required=True,
        help=f"{MINI_RELEASE}: config.json and model/MTP safetensors or extracted .header.json files; "
             "the supplied snapshot has no full commit SHA",
    )
    parser.add_argument("--out", type=Path, default=FIXTURES, help="the fixtures directory to write or compare")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a file differs from the one in --out")
    args = parser.parse_args()
    for name, want in PINS.items():
        if version(name) != want:
            raise SystemExit(f"{name} {version(name)} is installed, {want} is pinned")
    files = build(args.release, args.tiny, args.mini)
    differ = 0
    if not args.check:
        args.out.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(files.items()):
        path = args.out / name
        if args.check:
            same = path.is_file() and path.read_bytes() == data
            differ += not same
            print(("same    " if same else "DIFFERS ") + name)
        else:
            path.write_bytes(data)
            print(f"wrote {name}, {len(data)} bytes")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
