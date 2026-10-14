"""Fixtures for `tf-cuda-test exl3`: the Python engine's EXL3 linear outputs on real RED-SNOW tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

# the other fixtures: siblings in this folder, which leads sys.path when the script runs
import exl3_oracle_dense
import exl3_oracle_experts
import exl3_oracle_layer
import exl3_oracle_tp2

ROWS = (1, 3, 16)
SEED = 1234
SIZE_LIMIT = 8192  # prefer layers with N, K at or under this


def rank(t, bits: float) -> int:
    """Which layers to prefer for a width: a gate_proj for 4..6 bits, a shared-expert or attention one for 7..8."""
    if bits <= 6 and "gate_proj" in t.prefix:
        return 0
    if bits >= 7 and ("shared_experts" in t.prefix or "self_attn" in t.prefix):
        return 0
    return 1


def pick(model_dir: Path) -> list:
    """One EXL3 linear per whole bits value 4..8, first by sorted name among the preferred ones."""
    from tensorfold.cuda.exl3 import format as fmt

    cp = fmt.scan(model_dir)
    chosen = []
    for bits in (4.0, 5.0, 6.0, 7.0, 8.0):
        cands = [t for t in cp.groups.values()
                 if t.bits == bits and t.in_scales == "suh" and t.out_scales == "svh"]
        if not cands:
            print(f"no EXL3 linear at {bits:g} bits in {model_dir}")
            continue
        modest = [t for t in cands if t.k <= SIZE_LIMIT and t.n <= SIZE_LIMIT]
        t = min(modest or cands, key=lambda t: (rank(t, bits), t.prefix))
        marker = cp.markers.get(f"{t.prefix}.{t.codebook}")
        if t.codebook in fmt.MARKERS and marker is not None and marker != fmt.MARKERS[t.codebook]:
            raise ValueError(f"{t.prefix}: {t.codebook} marker is 0x{marker:08X}, want 0x{fmt.MARKERS[t.codebook]:08X}")
        chosen.append((t, marker))
    return chosen


def save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def one(out: Path, model_dir: Path, t, marker: int | None) -> dict:
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3.linear import Exl3Linear

    lid = f"python.{t.bits:g}"
    # "stored" keeps the checkpoint's trellis word order; the forward path is identical either way
    lin = Exl3Linear.load(model_dir, t.prefix, layout="stored")
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn((max(ROWS), lin.k), generator=g).to(torch.bfloat16)

    save(out, f"{lid}.trellis", lin.words)
    save(out, f"{lid}.suh", lin.suh)
    save(out, f"{lid}.svh", lin.svh)
    if lin.bias is not None:
        save(out, f"{lid}.bias", lin.bias)

    ys, xfiles, yfiles, sha = [], {}, {}, {}
    for m in ROWS:
        y = lin(x[:m].cuda())
        torch.cuda.synchronize()
        save(out, f"{lid}.x{m}", x[:m])
        save(out, f"{lid}.y{m}", y)
        xfiles[str(m)] = f"{lid}.x{m}.bin"
        yfiles[str(m)] = f"{lid}.y{m}.bin"
        sha[str(m)] = hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        ys.append(y)

    rel, ref = 0.0, fmt.forward(x.float().numpy().astype(np.float64), lin.words.view(torch.int16), lin.suh,
                                lin.svh, t.bits, t.codebook, lin.bias)
    for m, y in zip(ROWS, ys):
        want = ref[:m]
        rel = max(rel, np.linalg.norm(y.cpu().float().numpy().astype(np.float64) - want)
                  / max(np.linalg.norm(want), 1e-30))

    entry = {"id": lid, "name": t.prefix, "bits": int(t.bits), "codebook": t.codebook,
             "k": lin.k, "n": lin.n, "rows": list(ROWS), "trellis": f"{lid}.trellis.bin",
             "suh": f"{lid}.suh.bin", "svh": f"{lid}.svh.bin",
             "bias": None if lin.bias is None else f"{lid}.bias.bin",
             "x": xfiles, "y": yfiles, "x_dtype": "bf16", "y_dtype": "bf16", "sha256": sha}
    if marker is not None:
        entry["marker"] = f"0x{marker:08X}"
    print(f"{lid}: {t.prefix} EXL3 {t.bits:g}-bit {t.codebook} K={lin.k} N={lin.n} "
          f"rel={rel:.2e} sha256={sha['16'][:16]}…")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    ap.add_argument("--model", required=True, help="the EXL3 checkpoint folder")
    ap.add_argument("--out", required=True, help="the fixture folder to (over)write")
    a = ap.parse_args()
    out, model_dir = Path(a.out), Path(a.model)
    out.mkdir(parents=True, exist_ok=True)
    linears = [one(out, model_dir, t, marker) for t, marker in pick(model_dir)]
    dense = exl3_oracle_dense.write(out, model_dir)
    experts = exl3_oracle_experts.write(out, model_dir)
    layers = exl3_oracle_layer.write(out, model_dir)
    tp2 = exl3_oracle_tp2.write(out, model_dir)
    manifest = {"version": 1, "linears": linears, "dense": dense, "experts": experts, "layers": layers, "tp2": tp2,
                "model": str(model_dir)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(linears)} linears -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
