"""Needle-in-a-haystack for Nemotron-H on the Intel GPU: wikitext haystack of L tokens, secret code at depth D%."""
# usage: xpu_nem_needle.py [--ckpt DIR] [--bin DIR] [--rows 512] [--window] [--depths 10,50,90] [LENGTH ...]
import os, re, subprocess, sys, time, json
from tokenizers import Tokenizer
# prefill runs through `tensorfold-xpu run --prefill` in windows; --window uses tf-xpu-nem_gen-test (each step windowed)
ck = os.path.expanduser("~/models/nemotron-3.5-lightning-mlx4"); binp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "zig-out", "bin"); rows = "512"; depths = [10, 50, 90]; lens = []; window = False
# default lengths 8192 32768 65536 131072 (max 262144); reports FOUND (code in the answer before first EOS) or MISSED
a = sys.argv[1:]
while a:
    x = a.pop(0)
    if x == "--ckpt": ck = a.pop(0)
    elif x == "--bin": binp = a.pop(0)
    elif x == "--rows": rows = a.pop(0)
    elif x == "--window": window = True
    elif x == "--depths": depths = [int(v) for v in a.pop(0).split(",")]
    else: lens.append(int(x))
# /tmp/arc_stop stops it between runs; runs take xpu_guard.sh themselves: start this script bare
lens = lens or [8192, 32768, 65536, 131072]
tok = Tokenizer.from_file(f"{ck}/tokenizer.json")
eos = json.load(open(f"{ck}/config.json")).get("eos_token_id", [])
eos = set(eos if isinstance(eos, list) else [eos])
txt = open(os.path.expanduser("~/eval/wikitext-2-raw/wiki.train.raw")).read()
need = max(lens)
H = tok.encode(txt[:max(2_000_000, need * 8)]).ids
assert len(H) >= need, (len(H), need)
nd = tok.encode("\n\nThe secret code is 7391-ALPHA. Remember it.\n\n").ids
q = tok.encode("\n\nQuestion: What is the secret code mentioned in the text above?\nAnswer: The secret code is").ids
d = "/tmp/nem_needle"; os.makedirs(d, exist_ok=True)
print(f"{'L':>7} {'depth':>5}  verdict  {'prefill tok/s':>13} {'decode tok/s':>12} {'peak GB':>8} {'wall s':>7}  answer", flush=True)
for L in lens:
    for D in depths:
        if os.path.exists("/tmp/arc_stop"):
            print("stop file found: stopping between runs", flush=True); sys.exit(0)
        n = L - len(nd) - len(q); p = int(n * D / 100)
        ids = H[:p] + nd + H[p:n] + q
        f = f"{d}/n_{L}_{D}.txt"; open(f, "w").write(",".join(map(str, ids)))
        guard = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xpu_guard.sh")
        rep = f"{d}/report_{L}_{D}.json"
        if window:
            cmd = [guard, "3600", "env", f"NEM_PREFILL_ROWS={rows}", f"{binp}/tf-xpu-nem_gen-test", ck, f"@{f}", "20", "--bf16-logits", "--ctx", str(L + 64)]
        else:
            cmd = [guard, "3600", f"{binp}/tensorfold-xpu", "run", ck, "--tokens-file", f, "--max-tokens", "20", "--context", str(L + 64), "--prefill", rows, "--report", rep]
        t0 = time.time()
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        if window:
            g = re.search(r"^generated ids:(.*)$", out, re.M)
            gen = [int(v) for v in g.group(1).split()] if g else None
        else:
            gen = json.load(open(rep))["tokens"] if os.path.exists(rep) else None
        if gen is None:
            print(f"{L:>7} {D:>5}  FAILED to run: {out[-200:]!r}", flush=True); continue
        cut = next((i for i, t in enumerate(gen) if t in eos), len(gen))
        text = tok.decode(gen[:cut])
        if window:
            pf = re.search(r"prefill \d+ tokens in [\d.]+ s \(([\d.]+) tokens/s\)", out)
            dc = re.search(r"= ([\d.]+) tokens/s", out)
            pf_s, dc_s = (pf.group(1) if pf else "-"), (dc.group(1) if dc else "-")
        else:
            r_ = json.load(open(rep))
            pf_s, dc_s = f"{L / r_['prefill_seconds']:.1f}", f"{1000 / r_['ms_per_token']:.1f}"
        pk = re.search(r"device allocated [\d.]+ GB \(peak ([\d.]+) GB\)", out)
        print(f"{L:>7} {D:>4}%  {'FOUND' if '7391-ALPHA' in text else 'MISSED':<8} {pf_s:>13} {dc_s:>12} {pk.group(1) if pk else '-':>8} {time.time()-t0:>7.0f}  {text.strip()[:50]!r}", flush=True)
