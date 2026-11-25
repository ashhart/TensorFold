"""Write the regex anchor goldens: tokenizers' pieces for anchors, atomic groups and ERNIE 5.5's digit runs."""
import argparse
import hashlib
import json
from pathlib import Path

PIN = "0.23.3"
RELEASE = "baidu/ERNIE-5.5-Flash-Preview@ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe"
TOKENIZER_SHA256 = "20e7c81f36fdfb90db664583fc6426d5c2ac6d6bcaf631c910941a8d8f30f2d9"
ANCHORED = [
    (r"b$", ["ab\ncd", "abb\n", "ab\r\n"]),
    (r"$", ["ab\ncd"]),
    (r"\Ab", ["bab", "ab"]),
    (r"b\z", ["abb", "ab\n"]),
    (r"\Gb", ["bbab", "abb"]),
    (r"\Ga|b", ["aab", "baa"]),
    (r"(?>a+)a", ["aaa"]),
    (r"(?>a|ab)c", ["abc", "ac"]),
    (r"x(?=(?>\d{2})*(?:\D|$))", ["x1234y", "x123y", "x12\n", "x1"]),
]
TEXTS = ["1", "12", "123", "1234", "12345", "123456", "1234567", "12345678", "x1234567y", "a12b345c6789",
         "12 3456", "١٢٣٤٥", "1234\n56789", "Version 2026.11 costs $1234567.89!",
         "12½345", "1234² = 1522756", "Ⅷ 1234Ⅸ"]
RUNS = [509, 510, 511, 512, 513, 1020, 1021, 1025]


def dump(path, head, sections):
    """Write the manifest with one case per line."""
    out = "{\n" + "".join(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},\n" for k, v in head.items())
    out += ",\n".join(f"  {json.dumps(name)}: [\n" + ",\n".join("    " + json.dumps(c, ensure_ascii=False) for c in cases)
                      + "\n  ]" for name, cases in sections.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out + "\n}\n", encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True, help=f"{RELEASE}'s tokenizer.json")
    p.add_argument("--output", type=Path, required=True, help="zig/src/core/tokenizer/fixtures/onig_anchors.json")
    args = p.parse_args()
    import tokenizers
    from tokenizers import Regex, Tokenizer, pre_tokenizers
    if tokenizers.__version__ != PIN:
        raise SystemExit(f"tokenizers {tokenizers.__version__} is installed, {PIN} is pinned")
    raw = args.tokenizer.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit(f"{args.tokenizer} is not {RELEASE}'s tokenizer.json")
    release = json.loads(raw)
    steps = release["pre_tokenizer"]["pretokenizers"]
    if steps[-1]["type"] != "ByteLevel":
        raise SystemExit("the release's last pre-tokenizer step is not ByteLevel")
    pre = {"type": "Sequence", "pretokenizers": steps[:-1]}
    release["pre_tokenizer"] = pre
    ernie = Tokenizer.from_str(json.dumps(release)).pre_tokenizer

    def pieces(pre_tokenizer, text):
        return [piece for piece, _ in pre_tokenizer.pre_tokenize_str(text)]

    anchored = [{"pattern": pat, "text": text, "pieces": pieces(pre_tokenizers.Split(Regex(pat), "isolated"), text)}
                for pat, texts in ANCHORED for text in texts]
    texts = [{"text": text, "pieces": pieces(ernie, text)} for text in TEXTS]
    runs = []
    for n in RUNS:
        lengths = [len(piece) for piece in pieces(ernie, "7" * n)]
        if any(length != 3 for length in lengths[1:]):
            raise SystemExit(f"a run of {n} digits split into {lengths}")
        runs.append({"digits": n, "first": lengths[0], "pieces": len(lengths)})
    head = {"tokenizers_version": PIN, "release": RELEASE, "ernie_pre_tokenizer": pre}
    dump(args.output, head, {"anchored": anchored, "texts": texts, "runs": runs})
    print("anchor goldens", len(anchored), len(texts), len(runs))


if __name__ == "__main__":
    main()
