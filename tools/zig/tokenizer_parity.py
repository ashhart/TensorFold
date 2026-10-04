"""Compare zig/src/core/tokenizer/tokenizer.zig with Python tokenizers on cached tokenizer.json files: same ids, same decoded text."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/zig-checks/tokenizer"

PROSE = ("The quick brown fox jumps over the lazy dog. It's 9:41 on 3 Oct 2026; we'll ship v0.6.5 and they've "
         "said I'm sure you'd like it. DON'T PANIC, it'S fine, it\u017f odd, O'Neil's \"quotes\" and "
         "\u2018curly\u2019 ones.")
CODE = ('def fib(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    a, b = 0, 1\n'
        '    for _ in range(n):\n\ta, b = b, a + b\n    return a\n\nconst std = @import("std");\n'
        'pub fn main() !void {\n    std.debug.print("{d}\\n", .{42});\n}\n// x+=1; y--; z->w; a<=b && c!=d || e>=f\n'
        '<div class="x">&nbsp;&lt;tag&gt;</div>\r\nSELECT * FROM t WHERE id IN (1,2,3);\n')
JSON = json.dumps({"tool": "weather", "args": {"city": "K\xf8benhavn", "days": [1, 2, 3], "units": None, "ok": True,
                   "note": "line\nbreak\ttab \xe6\xf8\xe5 \u4e16\u754c \U0001F600"}}, ensure_ascii=False, indent=2)
CJK = ("\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u5427\u3002"
       "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3068"
       "\u30ab\u30bf\u30ab\u30ca\u3001\u3072\u3089\u304c\u306a\u3002"
       "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8\uc785\ub2c8\ub2e4. \u1100\u1161\u11a8 \u3131\u314f "
       "\U00020000\U0002A700\U00031350\U0002EBF0 \uff21\uff22\uff23\uff11\uff12\uff13")
EMOJI = ("\U0001F600\U0001F44B\U0001F3FD \U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466 "
         "\U0001F1E9\U0001F1F0 \u2764\ufe0f \U0001F9D1\U0001F3FF\u200d\U0001F4BB #\ufe0f\u20e3 "
         "\U0001FAE8\U0001FAE9 \u263a\u2603")
MARKS = ("e\u0301 a\u0300\u0301\u0302 \u1e0b\u0323 \u212b \u2126 \u0344 Vi\u1ec7t Nam Ti\u1ebfng Vi\u1ec7t "
         "\u0915\u094d\u0937 \u0939\u093f\u0928\u094d\u0926\u0940 "
         "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627 "
         "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 a\u0315\u0300\u05ae\u0300b "
         "\u0b47\u0b3e \u0dd9\u0dcf \u1100\u1161 \uac00\u11a8 Z\u0351\u036b\u0343\u036a\u0302\u036b\u033d\u034f\u0334")
SPACES = ("a  b   c\t\td\n\ne \r\n\r\nf \xa0g\u3000h\u2028i\u2029j\u0085k\u200bl\u2060m\ufeffn\xado"
          "      \n    \n\t \t\n\x0b\x0c\x1c\x1d\x1e\x1f end   ")
NUMBERS = ("0 7 42 1234567890 3.14159 -2.5e-10 1,000,000 \u0661\u0662\u0663 \u096c\u096d \u2167 \xbd \u2460 "
           "\U0001D7D8\U0001D7D9 99999999999999999999 0x1F600 1_000 v1.2.3")
SCRIPTS = ("\u0391\u03b8\u03ae\u03bd\u03b1 \u041c\u043e\u0441\u043a\u0432\u0430 "
           "\u10d7\u10d1\u10d8\u10da\u10d8\u10e1\u10d8 "
           "\u0535\u0580\u0587\u0561\u0576 \u1200\u1208 \u13a0\u13a1 \U000105C0\U00016100\U00011BC0 \U00010D50 "
           "\u01c5 \u01c8 \u1f88 \u0130stanbul \u0131 STRASSE stra\xdfe \u1e9e \ufb01 \ufb00 \u212a \xb5")
FIXED = [PROSE, CODE, JSON, CJK, EMOJI, MARKS, SPACES, NUMBERS, SCRIPTS]
ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ALPHABET += list(" \n\t\r'\".,;:!?()[]{}<>/\\|-_=+*&^%$#@~`")
ALPHABET += ["  ", "\n\n", "'s", "'S", "'ll", "'ve", "\u017f", "\xe9", "e\u0301", "\u4e2d", "\u6587", "\u3042",
             "\uac00", "\U0001F600", "\u200d", "\u0301", "\u3000", "\xa0", "\u0661", "\u2167", "\u2028", "\ufffd",
             "\u0e01", "\u0915\u094d"]


def label(path):
    parts = path.parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") - 1].removeprefix("models--").replace("--", "/")
    return path.parent.name


def discover(extra):
    """Distinct tokenizer.json files from the Hugging Face cache plus NAME=PATH or PATH extras."""
    cached = sorted((Path.home() / ".cache/huggingface/hub").glob("*/snapshots/*/tokenizer.json"))
    named = [(label(p), p) for p in cached]
    for item in extra:
        name, _, path = item.rpartition("=")
        named.append((name or label(Path(path)), Path(path)))
    found, seen = [], set()
    for name, path in named:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            found.append((name, path, digest[:16]))
    return found


def synthetic(seed):
    """Small trained tokenizers for the format features the cached models do not use."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    rng = random.Random(seed + 2)
    texts = FIXED * 30 + ["".join(rng.choice(ALPHABET) for _ in range(120)) for _ in range(300)]
    byte_tokens = [f"<0x{b:02X}>" for b in range(256)]
    specs = {
        "sentencepiece-bpe": (models.BPE(unk_token="<unk>", fuse_unk=True, byte_fallback=True),
                              normalizers.Sequence([normalizers.Prepend("\u2581"),
                                                    normalizers.Replace(" ", "\u2581")]), None,
                              decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback(),
                                                 decoders.Fuse(), decoders.Strip(" ", 1, 0)]),
                              trainers.BpeTrainer(vocab_size=700, limit_alphabet=50,
                                                  special_tokens=["<unk>", "<s>", "</s>"] + byte_tokens)),
        "metaspace-bpe": (models.BPE(unk_token="<unk>", continuing_subword_prefix="##", end_of_word_suffix="</w>"),
                          normalizers.Sequence([normalizers.Strip(right=False),
                                                normalizers.Replace(Regex("\\p{Mn}"), ""),
                                                normalizers.Replace("Q", ""), normalizers.Replace("W", "VV")]),
                          pre_tokenizers.Metaspace(prepend_scheme="first"), decoders.Metaspace(prepend_scheme="first"),
                          trainers.BpeTrainer(vocab_size=500, limit_alphabet=40, special_tokens=["<unk>"],
                                              continuing_subword_prefix="##", end_of_word_suffix="</w>")),
        "unk-bpe": (models.BPE(unk_token="<unk>", fuse_unk=False),
                    normalizers.Sequence([normalizers.NFD(), normalizers.Strip()]),
                    pre_tokenizers.Metaspace(prepend_scheme="never", split=False),
                    decoders.Metaspace(prepend_scheme="never"),
                    trainers.BpeTrainer(vocab_size=300, limit_alphabet=30, special_tokens=["<unk>"])),
        "wordpiece": (models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=12), normalizers.NFC(),
                      pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(),
                                               pre_tokenizers.Split(Regex("[\\p{P}\\p{S}]"), "isolated")]),
                      decoders.WordPiece(),
                      trainers.WordPieceTrainer(vocab_size=600, special_tokens=["[UNK]", "[CLS]", "[SEP]"])),
        "bytelevel-regex": (models.BPE(),
                            normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(" {3,}"), "  ")]),
                            pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True), decoders.ByteLevel(),
                            trainers.BpeTrainer(vocab_size=800, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())),
        "wordlevel-splits": (models.WordLevel(unk_token="[UNK]"), None,
                             pre_tokenizers.Sequence([pre_tokenizers.Split(Regex("\\p{N}"), "contiguous"),
                                                      pre_tokenizers.Split(" ", "merged_with_next"),
                                                      pre_tokenizers.Split(Regex("[,.;]"), "merged_with_previous"),
                                                      pre_tokenizers.Split(Regex("\\s+"), "removed", invert=True),
                                                      pre_tokenizers.Split("\n", "removed")]),
                             None, trainers.WordLevelTrainer(vocab_size=400, special_tokens=["[UNK]"])),
    }
    directory = OUT / "synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    built = []
    for name, (model, normalizer, pre, decoder, trainer) in specs.items():
        tok = Tokenizer(model)
        if normalizer:
            tok.normalizer = normalizer
        if pre:
            tok.pre_tokenizer = pre
        if decoder:
            tok.decoder = decoder
        tok.train_from_iterator(texts, trainer)
        tok.add_tokens([AddedToken("[INST]", lstrip=True, rstrip=True), AddedToken("<tool>", rstrip=True),
                        AddedToken("</tool>", lstrip=True), AddedToken("@@", normalized=True),
                        AddedToken("e\u0301x", normalized=True)])
        tok.add_special_tokens([AddedToken("<|eot|>", normalized=False), AddedToken("<pad>", normalized=True)])
        path = directory / f"{name}.json"
        tok.save(str(path))
        built.append(f"{name}={path}")
    return built


def random_codepoints(rng, n):
    return "".join(chr(c) for c in (rng.randrange(0x110000) for _ in range(n)) if not 0xD800 <= c < 0xE000)


def corpus(added, seed):
    rng = random.Random(seed)
    specials = added or ["<|endoftext|>"]
    texts = list(FIXED)
    texts += [" ".join(FIXED), "".join(FIXED), PROSE.lower(), PROSE.upper(), CODE * 3, SPACES * 4]
    for token in specials:
        texts += [token, token + token, f"x{token}y", f" {token} ", f"\n{token}\n", f"{token}{PROSE[:40]}{token}"]
    texts.append("".join(f"{t}hello {i}\n" for i, t in enumerate(specials)))
    for _ in range(300):
        texts.append("".join(rng.choice(ALPHABET + specials[:8]) for _ in range(rng.randrange(1, 200))))
    for _ in range(200):
        texts.append(random_codepoints(rng, rng.randrange(1, 64)))
    for start in range(0, 0x110000, 0x1000):
        chunk = [chr(c) for c in range(start, start + 0x1000) if not 0xD800 <= c < 0xE000]
        if chunk:
            texts.append("".join(chunk))
            texts.append(" ".join(rng.choice(chunk) + rng.choice(chunk) for _ in range(256)))
    texts.append((" ".join(FIXED) + "\n") * 40)
    texts.append("x" * 5000 + " " * 5000 + "\n" * 300 + "9" * 2000)
    cases = [t.encode() for t in texts]
    cases += [bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64))) for _ in range(150)]
    return cases


def decode_cases(ids_per_text, size, seed):
    rng = random.Random(seed + 1)
    cases = [ids for ids in ids_per_text if len(ids) < 4096]
    cases += [[rng.randrange(size + 8) for _ in range(rng.randrange(1, 64))] for _ in range(400)]
    cases += [list(range(i, min(i + 64, size + 8))) for i in range(0, size + 8, 64)]
    return cases


def run_native(checker, tokenizer, encode, decode, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    cases_path, out_path = OUT / f"{tag}-cases.json", OUT / f"{tag}-native.json"
    cases_path.write_text(json.dumps({"encode": [b.hex() for b in encode], "decode": decode}))
    subprocess.run([str(checker), str(tokenizer), str(cases_path), str(out_path)], check=True)
    return json.loads(out_path.read_text())


def reference(tok, cases):
    return [tok.encode(b.decode("utf-8", "replace"), add_special_tokens=False).ids for b in cases]


def shrink(checker, path, tok, case):
    """Smallest failing input reachable by deleting chunks of characters (batched delta debugging)."""
    try:
        units, join = list(case.decode()), lambda u: "".join(u).encode()
    except UnicodeDecodeError:
        units, join = [bytes([b]) for b in case], b"".join
    step = max(1, len(units) // 2)
    while step >= 1 and len(units) > 1:
        options = [units[:i] + units[i + step:] for i in range(0, len(units), step)]
        options = [u for u in options if u]
        candidates = [join(u) for u in options]
        native = run_native(checker, path, candidates, [], "shrink")["encode"]
        failing = [u for u, got, want in zip(options, native, reference(tok, candidates)) if got != want]
        if failing:
            units = min(failing, key=len)
            step = min(step, max(1, len(units) // 2))
        else:
            step //= 2
    return join(units)


def check(checker, name, path, digest, seed):
    from tokenizers import Tokenizer
    started = time.perf_counter()
    tok = Tokenizer.from_file(str(path))
    load_ms = (time.perf_counter() - started) * 1e3
    added = [t["content"] for t in json.loads(path.read_text()).get("added_tokens", [])]
    encode = corpus(added, seed)
    started = time.perf_counter()
    want = reference(tok, encode)
    py_encode_ms = (time.perf_counter() - started) * 1e3
    size = tok.get_vocab_size(with_added_tokens=True)
    decode = decode_cases(want, size, seed)
    native = run_native(checker, path, encode, decode, digest)
    enc_fail = [i for i, (got, exp) in enumerate(zip(native["encode"], want)) if got != exp]
    dec_fail = []
    for i, (ids, got) in enumerate(zip(decode, native["decode"])):
        exp = [tok.decode(ids, skip_special_tokens=s).encode() for s in (False, True)]
        if [bytes.fromhex(g) for g in got] != exp:
            dec_fail.append(i)
    result = {"name": name, "sha256_16": digest, "encode_cases": len(encode), "encode_failures": len(enc_fail),
              "decode_cases": len(decode), "decode_failures": len(dec_fail), "native_load_ms": native["load_ms"],
              "python_load_ms": round(load_ms, 1), "native_encode_ms": native["encode_ms"],
              "python_encode_ms": round(py_encode_ms, 1), "encode_mb": round(native["encode_bytes"] / 1e6, 2)}
    if enc_fail:
        small = shrink(checker, path, tok, encode[enc_fail[0]])
        result["encode_example"] = {"input": small.decode("utf-8", "replace"),
                                    "native": run_native(checker, path, [small], [], "shrink")["encode"][0],
                                    "python": reference(tok, [small])[0]}
    if dec_fail:
        ids = decode[dec_fail[0]]
        shown = bytes.fromhex(native["decode"][dec_fail[0]][0]).decode("utf-8", "replace")
        result["decode_example"] = {"ids": ids[:32], "python": tok.decode(ids, skip_special_tokens=False)[:200],
                                    "native": shown[:200]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", default=str(OUT / "tokenizer-check"), help="built zig/src/core/tokenizer/check.zig")
    parser.add_argument("--zig", default=str(ROOT / ".zig-toolchain/zig"), help="compiler used when --build is given")
    parser.add_argument("--build", action="store_true", help="build the checker first")
    parser.add_argument("--tokenizer", action="append", default=[], help="extra tokenizer.json (repeatable)")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--match", default="", help="only tokenizers whose name contains this")
    parser.add_argument("--no-synthetic", action="store_true", help="skip the trained feature-coverage tokenizers")
    args = parser.parse_args()
    if not args.no_synthetic:
        args.tokenizer += synthetic(args.seed)
    import tokenizers
    checker = Path(args.checker)
    if args.build:
        OUT.mkdir(parents=True, exist_ok=True)
        subprocess.run([args.zig, "build-exe", "zig/src/core/tokenizer/check.zig", "-O", "ReleaseSafe",
                        f"-femit-bin={checker}"], cwd=ROOT, check=True)
    found = [entry for entry in discover(args.tokenizer) if args.match in entry[0]]
    results = [check(checker, name, path, digest, args.seed) for name, path, digest in found]
    report = {"tokenizers_version": tokenizers.__version__, "seed": args.seed, "results": results}
    (OUT / "parity.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for r in results:
        status = "PASS" if not r["encode_failures"] and not r["decode_failures"] else "FAIL"
        print(f"{status} {r['name']:58s} encode {r['encode_cases'] - r['encode_failures']}/{r['encode_cases']} "
              f"decode {r['decode_cases'] - r['decode_failures']}/{r['decode_cases']}  "
              f"load {r['native_load_ms']:.0f}/{r['python_load_ms']:.0f} ms  "
              f"encode {r['encode_mb']} MB in {r['native_encode_ms']:.0f}/{r['python_encode_ms']:.0f} ms "
              "(native/python)")
        for key in ("encode_example", "decode_example"):
            if key in r:
                print("   ", key, json.dumps(r[key], ensure_ascii=False)[:600])
    print(f"tokenizers {tokenizers.__version__}; report in {(OUT / 'parity.json').relative_to(ROOT)}")
    sys.exit(0 if all(not r["encode_failures"] and not r["decode_failures"] for r in results) else 1)


if __name__ == "__main__":
    main()
