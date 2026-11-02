#!/usr/bin/env python3
"""Tap-on replies equal tap-off for Metal families that blit while the tap is live.

Covers GLM, GLM-DSA, Step5, MiniMax and LFM2. The blit barrier runs only when
the tap is live. Nemotron already had a live-gated blit; this script does not
cover it. Flash Next (`qwen4_exp`) and Qwen3.5 MoE on Metal never copy
and are refused by name.

Run this on a Metal host with the checkpoints. Host tests do not start a GPU server.

  python3 tools/run_metal_proofs.py plan
  python3 tools/run_metal_proofs.py run --family lfm2_moe
  python3 tools/run_metal_proofs.py run --all

Checkpoint paths come from the environment (see `plan`). The binary is
`zig-out/native/bin/tensorfold-native` unless `TF_NATIVE` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]})")

# Recipe checkpoints. None of these names is a host path; substitute a local dir after pull.
FAMILIES = (
    {
        "family": "glm5_next",
        "title": "GLM",
        "env": "GLM5_CHECKPOINT",
        "recipe": "docs/recipes/glm-5.3-flash.md",
        "default": "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP",
        "notes": "One Mac, MLX 4-bit MTP. 256 GB class in the recipe.",
    },
    {
        "family": "glm_moe_dsa",
        "title": "GLM-DSA",
        "env": "GLM_DSA_CHECKPOINT",
        "recipe": "CHANGELOG.md (GLM-5.3 glm_moe_dsa, EXLR8, two Macs)",
        "default": None,
        "notes": "Two Macs, EXLR8. Set GLM_DSA_CHECKPOINT to the local EXLR8 tree.",
    },
    {
        "family": "step5_vl",
        "title": "Step5",
        "env": "STEP5_CHECKPOINT",
        "recipe": "docs/recipes/step-5-preview.md",
        "default": None,
        "notes": "Native Metal, MLX 2-bit conversion of Step-5-Preview. 256 GB class in the recipe.",
    },
    {
        "family": "minimax_m3_vl",
        "title": "MiniMax",
        "env": "MINIMAX_CHECKPOINT",
        "recipe": "docs/recipes/minimax-m3.1-flash.md",
        "default": "BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34",
        "notes": "One 256 GB Mac. mix34 is the smaller of the two published conversions.",
    },
    {
        "family": "lfm2_moe",
        "title": "LFM2",
        "env": "LFM2_CHECKPOINT",
        "recipe": "docs/recipes/lfm2.5-24b-a2b.md",
        "default": "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
        "notes": "MLX 4-bit. A 48 GB Mac holds 16 streams at --context 40960.",
    },
)

PROMPT = "Write a short Python function that computes the Fibonacci sequence and explain it."


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def native_bin() -> Path:
    override = os.environ.get("TF_NATIVE")
    if override:
        return Path(override)
    return repo_root() / "zig-out" / "native" / "bin" / "tensorfold-native"


def checkpoint_of(row: dict) -> str | None:
    got = os.environ.get(row["env"])
    if got:
        return got
    return row["default"]


def print_plan() -> None:
    print("Metal tap-on vs tap-off proofs (blit barrier only while live)")
    print(f"binary: {native_bin()}")
    print()
    for row in FAMILIES:
        ckpt = checkpoint_of(row) or f"(set {row['env']})"
        print(f"{row['title']:10} {row['family']}")
        print(f"           env {row['env']}={ckpt}")
        print(f"           {row['recipe']}")
        print(f"           {row['notes']}")
        print()
    print("Refused on Metal (no copy, no proof): qwen4_exp, qwen3_5_moe")
    print("Nemotron already barriers; it is not in this script.")


def chat(base: str, model: str, tokens: int) -> str:
    url = base.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "max_tokens": tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def wait_ready(base: str, timeout: float = 600.0) -> None:
    url = base.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as err:
            last = err
        time.sleep(1)
    raise SystemExit(f"server at {base} did not become ready ({last})")


def serve(bin_path: Path, checkpoint: str, port: int, extra: list[str], tap: str | None) -> subprocess.Popen:
    env = os.environ.copy()
    if tap is None:
        env.pop("TF_ROUTE_TAP", None)
    else:
        env["TF_ROUTE_TAP"] = tap
    cmd = [str(bin_path), "serve", checkpoint, "--name", "local-model", "--port", str(port), "--prompt-cache-gib", "0", *extra]
    return subprocess.Popen(cmd, env=env, cwd=repo_root())


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_family(row: dict, tokens: int, port: int, extra: list[str], out_dir: Path) -> dict:
    ckpt = checkpoint_of(row)
    if not ckpt:
        return {"family": row["family"], "skipped": True, "reason": f"set {row['env']}"}
    bin_path = native_bin()
    if not bin_path.is_file():
        raise SystemExit(f"missing {bin_path}; zig build native -Dcpu=apple_m1")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"{row['family']}.tfr1"
    base = f"http://127.0.0.1:{port}"
    result: dict = {"family": row["family"], "checkpoint": ckpt, "equal": False}

    off_proc = serve(bin_path, ckpt, port, extra, None)
    try:
        wait_ready(base)
        off_text = chat(base, "local-model", tokens)
    finally:
        stop(off_proc)
    time.sleep(1)

    on_proc = serve(bin_path, ckpt, port, extra, str(trace))
    try:
        wait_ready(base)
        on_text = chat(base, "local-model", tokens)
    finally:
        stop(on_proc)

    result["equal"] = off_text == on_text
    result["tap_off"] = off_text
    result["tap_on"] = on_text
    result["trace"] = str(trace) if trace.is_file() else None
    receipt = out_dir / f"{row['family']}.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["equal"]:
        raise SystemExit(f"{row['family']}: tap-on tokens differ from tap-off")
    if trace.is_file():
        decode = subprocess.run(
            [sys.executable, str(repo_root() / "tools" / "route_tap_bench.py"), "decode", str(trace)],
            check=False,
            capture_output=True,
            text=True,
        )
        if decode.returncode != 0:
            raise SystemExit(f"{row['family']}: TFR1 decode failed\n{decode.stderr}")
        result["records"] = len(json.loads(decode.stdout).get("records") or [])
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("family", "equal", "trace", "records") if k in result}))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="print families, env vars and recipe checkpoints")
    r = sub.add_parser("run", help="serve tap-off then tap-on and compare greedy tokens")
    r.add_argument("--family", action="append", dest="families")
    r.add_argument("--all", action="store_true")
    r.add_argument("--tokens", type=int, default=64)
    r.add_argument("--port", type=int, default=8094)
    r.add_argument("--output", default="metal-proofs")
    r.add_argument("--extra", default="", help="extra serve flags, space-separated")
    args = p.parse_args()
    if args.cmd == "plan":
        print_plan()
        return 0
    wanted = {f for f in (args.families or [])}
    if args.all:
        wanted = {row["family"] for row in FAMILIES}
    if not wanted:
        p.error("pass --family NAME (repeatable) or --all")
    extra = args.extra.split() if args.extra else []
    out_dir = Path(args.output)
    ran = 0
    for row in FAMILIES:
        if row["family"] not in wanted:
            continue
        run_family(row, args.tokens, args.port, extra, out_dir)
        ran += 1
        args.port += 1
    if ran == 0:
        raise SystemExit("no matching family")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
