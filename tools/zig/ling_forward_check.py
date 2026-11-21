"""Run native Ling whole/chunk/resume checks beside the one FP64 oracle on identical routed picks."""
import argparse
import array
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def compare(native, reference):
    data = reference.read_bytes()
    if data[:8] != b"TFLINGF2":
        raise ValueError("unexpected oracle format")
    tokens, vocab, layers, topk = struct.unpack_from("<4I", data, 8)
    at = 36 + tokens * 4
    exact = array.array("d", data[at:at + tokens * vocab * 8])
    rounded = array.array("d", data[at + tokens * vocab * 8:at + tokens * vocab * 16])
    metal = array.array("f", native.read_bytes())
    if len(metal) != tokens * vocab:
        raise ValueError("native logits shape mismatch")
    rows = []
    for row in range(tokens):
        lo, hi = row * vocab, (row + 1) * vocab
        a, b, g = exact[lo:hi], rounded[lo:hi], metal[lo:hi]
        if not all(math.isfinite(v) for v in g):
            raise ValueError("non-finite native logits")
        pair = {}
        for name, values in (("metal", g), ("bf16", b)):
            errors = [x - y for x, y in zip(values, a)]
            pair[name] = {"rms": math.sqrt(math.fsum(x*x for x in errors) / vocab),
                          "max": max(map(abs, errors)), "top": max(range(vocab), key=values.__getitem__)}
        pair["top64"] = max(range(vocab), key=a.__getitem__)
        rows.append(pair)
    result = {"tokens": tokens, "moe_layers": layers, "top_k": topk, "rows": rows,
              "forced_gap": struct.unpack_from("<d", data, 28)[0]}
    for name in ("metal", "bf16"):
        result[name] = {"rms": math.sqrt(math.fsum(r[name]["rms"]**2 for r in rows) / tokens),
                        "max": max(r[name]["max"] for r in rows)}
    result["rows_above_bf16_rms"] = sum(r["metal"]["rms"] > r["bf16"]["rms"] for r in rows)
    result["top_tokens_equal"] = all(r["metal"]["top"] == r["bf16"]["top"] == r["top64"] for r in rows)
    result["accuracy_acceptance"] = "unresolved; distances reported without relaxing the reference bound"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    checkpoint = os.environ.get("TF_LING_CHECKPOINT")
    if not checkpoint or not (Path(checkpoint) / "model.safetensors.index.json").is_file():
        print("SKIP Ling full forward: set TF_LING_CHECKPOINT to the bf16 Ling-3.0-tiny checkpoint")
        return
    python = os.environ.get("TF_LING_PYTHON") or shutil.which("python3")
    if not python:
        print("SKIP Ling oracle: set TF_LING_PYTHON or install python3 on PATH")
        return
    probe = subprocess.run([python, "-c", "import torch, safetensors"], capture_output=True)
    if probe.returncode:
        print("SKIP Ling oracle: selected Python needs torch and safetensors")
        return
    with tempfile.TemporaryDirectory(prefix="ling-forward-") as temp:
        out = args.out or Path(temp)
        out.mkdir(parents=True, exist_ok=True)
        for count in args.counts:
            native, reference = out / f"native-{count}.f32", out / f"reference-{count}.bin"
            if sys.platform == "darwin":
                pressure = subprocess.check_output(["memory_pressure"], text=True)
                free = int(pressure.split("System-wide memory free percentage:")[-1].strip().rstrip("%"))
                if free <= 40:
                    raise RuntimeError("memory free is at or below 40%; refusing the Ling GPU run")
            subprocess.run([str(args.binary.resolve()), checkpoint, str(native), str(count)], check=True)
            oracle = ROOT / "zig/src/families/ling/oracle/ling_forward.py"
            subprocess.run([python, str(oracle), checkpoint, str(reference), "--tokens", str(count),
                            "--picks", str(native) + ".picks"], check=True)
            result = compare(native, reference)
            (out / f"summary-{count}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            if not result["top_tokens_equal"]:
                raise RuntimeError("top-token mismatch; inspect the numerical receipt")
        print("Ling exactness checks completed; numerical acceptance remains a separate review decision")


if __name__ == "__main__":
    main()
