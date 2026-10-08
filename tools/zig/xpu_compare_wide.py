"""Forced-token accuracy of Nemotron via `tensorfold-xpu run` against the CUDA reference (ref_nemotron_wide.json)."""
import json, os, re, subprocess, sys

# usage: xpu_compare_wide.py MODE [prompt_index ...]; MODE is plain, scalar (NEM_SCALAR=1), pf128 or pf512 (--prefill)
HOME = os.path.expanduser("~")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# env: TF_NEMOTRON_DIR, TF_FIXTURES_DIR, TF_XPU_BIN, TF_GPU_LOCK (run via the guard), WIDE_LIST=file (top-1 misses)
M = os.environ.get("TF_NEMOTRON_DIR", f"{HOME}/models/nemotron-3.5-lightning-mlx4")
FIX = os.environ.get("TF_FIXTURES_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
BIN = os.environ.get("TF_XPU_BIN", os.path.join(ROOT, "zig-out", "bin", "tensorfold-xpu"))
GUARD = os.path.join(ROOT, "tools", "zig", "xpu_guard.sh")
MODES = {"plain": ([], {}), "scalar": ([], {"NEM_SCALAR": "1"}), "pf128": (["--prefill", "128"], {}), "pf512": (["--prefill", "512"], {})}
mode = sys.argv[1]
extra, env = MODES[mode]
ref = json.load(open(os.path.join(FIX, "ref_nemotron_wide.json")))
ref = ref["prompts"] if isinstance(ref, dict) else ref
sel = [int(x) for x in sys.argv[2:]] or range(len(ref))
# per position: delta = our log-probability of the reference token minus the reference's chosen_logprob
res = []
for pi in sel:
    p = ref[pi]
    n = len(p["generated_ids"])
    cmd = [BIN, "run", M, "--tokens", ",".join(map(str, p["prompt_ids"])), "--max-tokens", str(n), "--ignore-eos", "--logits", "--force", ",".join(map(str, p["generated_ids"]))] + extra
    if os.environ.get("TF_GPU_LOCK") and not os.environ.get("TF_GUARD_HELD"):
        cmd = [GUARD, "3000"] + cmd
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, **env}).stdout
    lp = {}
    for m in re.finditer(r"step (\d+) lpf (\d+) (\S+) (\S+)", out):
        lp[int(m.group(1))] = (float(m.group(3)), float(m.group(4)))
    if len(lp) != n:
        print(f"prompt {pi}: only {len(lp)}/{n} scored\n{out[-800:]}", file=sys.stderr)
    for s, (ours, top) in sorted(lp.items()):
        tk = p["topk"][s]
        res.append(dict(prompt=pi, plen=len(p["prompt_ids"]), step=s, delta=ours - p["chosen_logprob"][s], top1=ours >= top - 1e-9, forced=p["generated_ids"][s], ref_top=tk[0][0], ref_margin=tk[0][1] - tk[1][1], ours=ours, top=top))
    r = [x for x in res if x["prompt"] == pi]
    print(f"prompt {pi} len {len(p['prompt_ids'])}: top1 {sum(x['top1'] for x in r)}/{len(r)} mean|d| {sum(abs(x['delta']) for x in r) / max(len(r), 1):.4f}", flush=True)
a = sorted(abs(x["delta"]) for x in res)
if a:
    t1 = sum(x["top1"] for x in res)
    print(f"MODE {mode}: positions {len(a)} top1 {t1}/{len(a)} ({100 * t1 / len(a):.1f}%) mean|d| {sum(a) / len(a):.4f} p95|d| {a[int(0.95 * (len(a) - 1))]:.4f} max|d| {a[-1]:.3f}")
if os.environ.get("WIDE_LIST"):  # every position where our top-1 is not the forced token: both margins (the reference's top-1 over its top-2, ours top-1 over the forced token)
    with open(os.environ["WIDE_LIST"], "w") as f:
        f.write("prompt\tstep\tforced_id\tref_top1_id\tref_margin\tour_gap\n")
        for x in res:
            if not x["top1"]:
                f.write(f"{x['prompt']}\t{x['step']}\t{x['forced']}\t{x['ref_top']}\t{x['ref_margin']:.4f}\t{x['top'] - x['ours']:.4f}\n")
