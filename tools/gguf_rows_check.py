"""Row invariance of the GGUF projections on real 27B tensors: every row count gives every row the same bits.

    python tools/gguf_rows_check.py MODEL.gguf [--gguf-py DIR]

One tensor (the smallest 2D) of each quant type in the file. For each: 80 random fp32 rows through ``gguf.linear`` at
once, then every row count 1..80 and one row at a time must match those bits exactly. The run fails when the file
holds a quantized type with no kernel here, or when no tensor was checked. The reference is gguf-py's
CPU dequantizer and a float64 matmul: it must NOT match bitwise (else the check could not fail), and must stay close.
The GPU bf16 dequant (the embedding path) is checked against the same CPU weights.
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--gguf-py", default=None, help="llama.cpp's gguf-py directory, if gguf is not installed")
    ap.add_argument("--rows", type=int, default=80)
    args = ap.parse_args()
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize

    from tensorfold.cuda import gguf

    reader = GGUFReader(args.model)
    picked, unsupported = {}, set()
    for t in reader.tensors:
        if int(t.tensor_type) not in gguf.QUANT and int(t.tensor_type) not in (gguf.F32, gguf.F16, gguf.BF16):
            unsupported.add(t.tensor_type.name)                       # the smallest 2D tensor of each type keeps the run short
        q = int(t.tensor_type)
        if q in gguf.QUANT and len(t.shape) == 2 and (q not in picked or t.n_elements < picked[q].n_elements):
            picked[q] = t
    counts = range(1, args.rows + 1)
    torch.manual_seed(0)
    failed = 0
    for q, t in sorted(picked.items()):
        k, n = int(t.shape[0]), int(t.shape[1])            # GGUF lists the input dimension first
        w = torch.from_numpy(t.data.reshape(-1).view("uint8").copy()).cuda()
        x = torch.randn(args.rows, k, device="cuda") * 0.5
        ref = gguf.linear(x, w, q, n)
        bad = [m for m in counts if m <= args.rows and not torch.equal(gguf.linear(x[:m], w, q, n), ref[:m])]
        single = torch.cat([gguf.linear(x[i:i + 1], w, q, n) for i in range(args.rows)])
        if not torch.equal(single, ref):
            bad.append("one-at-a-time")
        cpu = torch.from_numpy(dequantize(t.data, t.tensor_type).reshape(n, k).astype("float32"))
        other = (x.double().cpu() @ cpu.double().T).float().cuda()
        rel = ((other - ref).norm() / other.norm()).item()
        worst = ((other - ref).abs().max() / other.abs().max()).item()     # largest element error, per the largest output
        control = "differs (ok)" if not torch.equal(other, ref) else "EQUAL (control cannot fail)"
        gpu = gguf.rows_bf16(w.view(n, -1)[:512], q, k).float().cpu()          # the embedding path, first 512 rows
        cpu = cpu[:512]
        drel = ((gpu - cpu).norm() / cpu.norm()).item()
        ok = not bad and rel < 1e-3 and drel < 5e-3 and not torch.equal(other, ref)
        failed += not ok
        print(f"{t.tensor_type.name:7} {t.name:28} N={n:6} K={k:6}  invariant={'yes' if not bad else bad}  "
              f"vs CPU fp64: L2 rel={rel:.1e} max-elem/max={worst:.1e} {control}  rows_bf16 rel={drel:.1e}  {'PASS' if ok else 'FAIL'}", flush=True)
    if unsupported:
        print(f"FAIL: quantized types with no kernel here: {sorted(unsupported)}")
        failed += 1
    if not picked:
        print("FAIL: no quantized tensor was checked")
        failed += 1
    print(f"ALL PASS ({len(picked)} types: {', '.join(t.tensor_type.name for _, t in sorted(picked.items()))}; "
          f"row counts 1..{args.rows} and one at a time)" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
