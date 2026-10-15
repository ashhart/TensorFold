"""Teacher-forced and greedy runs of `tensorfold-xpu run` (bf16 logits) against the upstream CUDA reference fixture."""
import json, os, re, subprocess, sys

# usage: xpu_compare_ref.py [--family nemotron|qwen] [--model PATH] [--ref FILE] [--prompts 0,1,2,3] [--json] [-- flags]
HOME = os.path.expanduser("~")
# --family nemotron (default, ref_nemotron.json, TF_NEMOTRON_DIR) or qwen (ref_qwen27.json, TF_QWEN_DIR)
FAMILY = sys.argv[sys.argv.index("--family") + 1] if "--family" in sys.argv else "nemotron"
if FAMILY == "qwen":
    MODEL = os.environ.get("TF_QWEN_DIR", f"{HOME}/models/qwen3.8-27b-mlx4")
    REF = "ref_qwen27.json"
else:
    MODEL = os.environ.get("TF_NEMOTRON_DIR", f"{HOME}/models/nemotron-3.5-lightning-mlx4")
    REF = "ref_nemotron.json"
# TF_FIXTURES_DIR holds the ref_*.json (default: tools/zig/fixtures); models default to ~/models/*
SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
FIXTURES = os.environ.get("TF_FIXTURES_DIR") or (SHIPPED if os.path.exists(os.path.join(SHIPPED, REF)) else f"{HOME}/tensorfold-fixtures")
# TF_GPU_LOCK set: run each program through xpu_guard.sh, or run the whole script under the guard with TF_GUARD_HELD=1
LOCK = os.environ.get("TF_GPU_LOCK")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# TF_XPU_BIN: the binary (default zig-out/bin/tensorfold-xpu)
BIN = os.environ.get("TF_XPU_BIN", os.path.join(ROOT, "zig-out", "bin", "tensorfold-xpu"))
def opt(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


# --model: a checkpoint dir or .gguf (Qwen: MLX 4-bit, EXL3 or GSQ-RCO); --ref: a ref_*.json name in TF_FIXTURES_DIR
MODEL = opt("--model") or MODEL
REF = opt("--ref") or REF
# ref_qwen27.json is MLX, ref_qwen27_exl3.json EXL3, ref_qwen27_gsq.json GSQ
EXTRA = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ref = json.load(open(os.path.join(FIXTURES, REF)))


def run(p, extra):
    ids = ",".join(map(str, p["prompt_ids"]))
    n = len(p["generated_ids"])
    cmd = [BIN, "run", MODEL, "--tokens", ids, "--max-tokens", str(n), "--no-drafts", "--ignore-eos", "--logits"] + EXTRA + extra
    if LOCK and not os.environ.get("TF_GUARD_HELD"):
        # one program at a time on the card: xpu_guard.sh takes the lock, checks memory, stops only with SIGINT
        cmd = [os.path.join(ROOT, "tools", "zig", "xpu_guard.sh"), "3000"] + cmd
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=int(os.environ.get("TF_RUN_TIMEOUT", "7200"))).stdout
    steps = []
    for m in re.finditer(r"step (\d+) pos (\d+) top5: (.*)", out):
        steps.append([(int(a), float(b)) for a, b in (t.split(":") for t in m.group(3).split())])
    if len(steps) < n:
        sys.exit(f"run produced {len(steps)} of {n} steps:\n{out[-2000:]}")
    return steps[:n], out


def main():
    only = None
    if "--prompts" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--prompts") + 1].split(",")}
    tot_top1 = tot_steps = 0
    results = []
    for pi, p in enumerate(ref["prompts"]):
        if only is not None and pi not in only:
            continue
        n = len(p["generated_ids"])
        forced, _ = run(p, ["--force", ",".join(map(str, p["generated_ids"]))])
        top1_ok = set5_ok = 0
        gaps = []  # (step, |gap diff|)
        bad = []
        for s, (ours, r) in enumerate(zip(forced, p["topk"])):
            rid = [x[0] for x in r]
            rlp = [x[1] for x in r]
            tied = {i for i, lp in zip(rid, rlp) if lp == rlp[0]}
            if ours[0][0] in tied:
                top1_ok += 1
            else:
                bad.append(s)
            if set(i for i, _ in ours) == set(rid):
                set5_ok += 1
            gid = [i for i, _ in ours]
            for i in gid:
                if i in rid:
                    gaps.append((s, abs((ours[gid.index(i)][1] - ours[0][1]) - (rlp[rid.index(i)] - rlp[0]))))
        third = [[g for s, g in gaps if lo <= s < lo + 16] for lo in (0, 16, 32)]
        fmt = lambda v: f"{sum(v) / max(len(v), 1):.3f}/{max(v, default=0):.3f}"
        print(f"prompt {pi} forced: top1 {top1_ok}/{n}, top5-set {set5_ok}/{n}, gap diff mean {sum(g for _, g in gaps) / len(gaps):.3f} max {max(g for _, g in gaps):.3f}; "
              f"by step 0-15/16-31/32-47 mean/max {fmt(third[0])} {fmt(third[1])} {fmt(third[2])}; top1 misses at steps {bad}")
        greedy, _ = run(p, [])
        got = [s[0][0] for s in greedy]
        first = next((i for i, (a, b) in enumerate(zip(got, p["generated_ids"])) if a != b), None)
        match = sum(a == b for a, b in zip(got, p["generated_ids"]))
        print(f"prompt {pi} greedy: {match}/{n} ids match, first divergence {first}")
        sys.stdout.flush()
        tot_top1 += top1_ok
        tot_steps += n
        results.append(dict(prompt=pi, forced_top1=top1_ok, steps=n, greedy_match=match, first_divergence=first, misses=bad))
    print(f"teacher-forced top-1 total {tot_top1}/{tot_steps}")
    if "--json" in sys.argv:
        print(json.dumps(results))


main()
