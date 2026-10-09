"""Needle in a haystack through `tensorfold-xpu run`: filler text with a secret code at a depth, checks the answer."""
import os, re, subprocess, sys, time
from tokenizers import Tokenizer

# usage: xpu_needle.py MODEL --lengths 4096,32768 --depths 10,50,90 [--kv bf16|q8|q4] [--prefill ROWS]; needs tokenizers
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("TF_XPU_BIN", os.path.join(ROOT, "zig-out", "bin", "tensorfold-xpu"))
# env: TF_XPU_BIN, TF_XPU_GUARD (default tools/zig/xpu_guard.sh); a GGUF model takes TF_TOKENIZER=PATH to tokenizer.json
GUARD = os.environ.get("TF_XPU_GUARD", os.path.join(ROOT, "tools", "zig", "xpu_guard.sh"))
CODE = "7391-ALPHA"
FILLER = [
    "The river bends twice before it reaches the old stone bridge, where the market opens at dawn.",
    "Engineers measured the load on each beam and recorded the results in a long table of numbers.",
    "In autumn the orchard smells of apples, and the children carry baskets down the hill.",
    "The librarian sorted the returned books by colour first and by author second, to nobody's surprise.",
    "A small boat crossed the lake while the fog lifted slowly from the water.",
    "The committee met on Tuesday to discuss the budget, the schedule and the new parking rules.",
]


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


model = sys.argv[1]
lengths = [int(x) for x in opt("--lengths", "4096").split(",")]
depths = [int(x) for x in opt("--depths", "50").split(",")]
kv, rows = opt("--kv", "bf16"), opt("--prefill", "512")
tok_path = os.environ.get("TF_TOKENIZER") or os.path.join(model, "tokenizer.json")
tok = Tokenizer.from_file(tok_path)
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
filler = [enc(s + " ") for s in FILLER]
needle = enc(f"\nThe secret code is {CODE}. Remember it.\n")
question = enc("\nQuestion: What is the secret code?\nAnswer: The secret code is")
for L in lengths:
    for d in depths:
        want = L - len(needle) - len(question) - 64
        ids, i = [], 0
        while len(ids) < want:
            ids += filler[i % len(filler)]
            i += 1
        ids = ids[:want]
        at = want * d // 100
        ids = ids[:at] + needle + ids[at:] + question
        path = f"/tmp/needle_{L}_{d}.ids"
        open(path, "w").write(",".join(map(str, ids)))
        t0 = time.time()
        cmd = ([] if os.environ.get("TF_GUARD_HELD") else [GUARD, "7200"]) + [BIN, "run", model, "--tokens-file", path, "--max-tokens", "12", "--ignore-eos", "--no-drafts", "--prefill", rows, "--ctx", str(len(ids) + 64), "--kv", kv]
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        g = re.search(r"^generated ids:(.*)$", out, re.M)
        m = re.search(r"^tokens .*prefill ([\d.]+)s", out, re.M)
        mem = re.search(r"^device allocated [\d.]+ GB \(peak ([\d.]+) GB\)", out, re.M)
        if not g:
            print(f"L={L} depth={d}%: FAILED: {out[-300:]!r}", flush=True)
            continue
        text = tok.decode([int(x) for x in g.group(1).split()])
        print(f"L={len(ids)} depth={d}% kv={kv}: {'FOUND' if CODE in text else 'MISSED'} answer {text.strip()[:40]!r} prefill {m.group(1) if m else '?'} s ({int(rows)}-row windows), peak device {mem.group(1) if mem else '?'} GB, wall {time.time() - t0:.0f} s", flush=True)
