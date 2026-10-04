"""Kimi K3's tokenizer in Zig against Moonshot's TikTokenTokenizer (tiktoken): same ids, same decoded bytes."""
import argparse
import importlib
import json
import random
import subprocess
import sys
import types
from pathlib import Path

import tokenizer_parity as tp

SPECIALS = ["[BOS]", "[EOS]", "<|im_end|>", "<|im_user|>", "<|im_assistant|>", "<|im_system|>", "<|im_middle|>",
            "[EOT]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>", "<|media_begin|>", "<|reserved_token_163700|>"]
KIMI = ["你好，世界！、。「引用」々〇〡 ⺀⼀ "
        "\U00020000\U00031350 カタカナと漢字混じり",
        "<|im_system|>system<|im_middle|>You are Kimi.<|im_end|><|im_user|>user<|im_middle|>Hi!<|im_end|>",
        "<|im_end|<|im_end|>> [BOS][EOS] <|reserved_token_163700|> <|reserved_token_1|>",
        "I'M sure it'S theirs; WE'LL see, they'VE gone, you'D know, can'T stop, itſ done.",
        "ABCdef GHI ǅabc İstanbul สวัสดี مَرْحَبًا",
        " " * 30 + "x\n\n\n   \r\n\t\t  y" + "　" * 5 + "z", "a" * 3000, "中" * 2000, " " * 26000 + "end"]


def official(k3: Path):
    """Moonshot's TikTokenTokenizer, loaded from the downloaded files as a package."""
    pkg = types.ModuleType("k3tok")
    pkg.__path__ = [str(k3)]
    sys.modules["k3tok"] = pkg
    mod = importlib.import_module("k3tok.tokenization_kimi")
    cfg = json.loads((k3 / "tokenizer_config.json").read_text())
    from tokenizers import AddedToken
    decoder = {int(k): AddedToken(**{f: v[f] for f in ("content", "lstrip", "rstrip", "normalized", "single_word", "special")})
               for k, v in cfg["added_tokens_decoder"].items()}
    return mod.TikTokenTokenizer(str(k3 / "tiktoken.model"), bos_token=cfg.get("bos_token", "[BOS]"),
                                 eos_token=cfg.get("eos_token", "[EOS]"), unk_token=cfg.get("unk_token"),
                                 pad_token=cfg.get("pad_token"), added_tokens_decoder=decoder)


def cases(seed: int, n: int):
    rng = random.Random(seed)
    out = list(tp.FIXED) + KIMI
    pieces = tp.ALPHABET + SPECIALS + ["中文", "、", "々", "'RE", "'Ll", "12345", "́̂"]
    for _ in range(n):
        out.append("".join(rng.choice(pieces) for _ in range(rng.randint(1, 80))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k3", required=True, help="folder with tiktoken.model, tokenizer_config.json, tokenization_kimi.py")
    ap.add_argument("--exe", required=True, help="tf-k3-tokenizer")
    ap.add_argument("--work", required=True, help="folder for the cases and results")
    ap.add_argument("--random", type=int, default=3000)
    a = ap.parse_args()
    tok = official(Path(a.k3))
    texts = cases(7, a.random)
    want = [tok.encode(t) for t in texts]
    rng = random.Random(11)
    dec = [[rng.randrange(163840) for _ in range(rng.randint(1, 40))] for _ in range(500)] + want[:200]
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "cases.json").write_text(json.dumps({"encode": [t.encode().hex() for t in texts], "decode": dec}))
    subprocess.run([a.exe, a.k3, str(work / "cases.json"), str(work / "out.json")], check=True)
    got = json.loads((work / "out.json").read_text())
    bad = [i for i, (g, w) in enumerate(zip(got["encode"], want)) if g != w]
    for i in bad[:5]:
        print(f"encode differs on case {i}: {texts[i][:80]!r}\n  ours  {got['encode'][i][:40]}\n  tiktoken {want[i][:40]}")
    dbad = [i for i, (g, ids) in enumerate(zip(got["decode"], dec)) if bytes.fromhex(g) != tok.model.decode_bytes(ids)]
    print(f"encode: {len(texts) - len(bad)}/{len(texts)} cases equal ({sum(len(t.encode()) for t in texts)} bytes, "
          f"{got['encode_ms']:.0f} ms in Zig); decode: {len(dec) - len(dbad)}/{len(dec)} equal; load {got['load_ms']:.0f} ms")
    return 0 if not bad and not dbad else 1


if __name__ == "__main__":
    sys.exit(main())
