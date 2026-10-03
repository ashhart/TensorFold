"""TENSORFOLD_GGUF_FAST's kernels on real 27B tensors: Gufo's WMMA prompt GEMM and 1-row decode GEMV vs the exact one.

    TF_GGUF_PY=<llama.cpp>/gguf-py python scripts/gguf_fast_prefill_check.py [MODEL.gguf] [--per-type 3]

A few blk.* projections of each quant type in the file. For each, 4096 random bf16 rows through the fast kernel:
(a) slices of 1000/1/37/513/95/96 rows at several offsets, a permuted batch, and fp32 input must give the same bits
as the 4096-row call; (b) error against the exact kernel (``gguf.linear``, decode's bits), which must NOT match
bitwise (else the fast path did not run); (c) both kernels timed at 4096 rows. ``Gguf.prefill`` with
TENSORFOLD_GGUF_FAST=1 must return the fast kernel's bits, and with it unset (the default) the exact
kernel's (the switch is wired both ways). Decode: 1 row on ``gguf.gemv``
vs the exact kernel's 1 row (reported: equal bits keep drafted == serial), and 1-row and 16-row timings of
exact vs fast. Verify widths 2/4/8/16 per TENSORFOLD_GGUF_FAST_VERIFY option (wmma unpadded, bf16, gemv per row,
exact): time and max abs error vs exact; and 20/40-row prompt calls padded (TENSORFOLD_GGUF_FAST_PAD=1) vs not.
"""

from __future__ import annotations

import argparse
import os

import torch

MODEL = "/srv/tensorfold-strix/gguf-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
SLICES = [(1000, 0), (1000, 1000), (1000, 3096), (1, 0), (1, 4095), (1, 2049), (37, 0), (37, 1234), (37, 4059),
          (513, 0), (513, 777), (513, 3583), (95, 5), (96, 4000)]


def timed(fn, iters: int) -> float:
    """Median milliseconds of ``fn`` over ``iters`` calls after two warmups."""

    for _ in range(2):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def synced(fn):
    """``fn`` then a device sync, so an async HIP fault surfaces at the call that caused it (AMD_SERIALIZE_KERNEL=3 too)."""

    def call(*a, **kw):
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        return out

    return call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=MODEL)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--per-type", type=int, default=3, help="distinct (N, K) shapes checked per quant type")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--pad", type=int, default=0, help="prefill_linear padding under test: 0 (the production default) or 96")
    args = ap.parse_args()

    from tensorfold.cuda import gguf

    _plain = gguf.prefill_linear                     # every check below runs at the padding under test
    gguf.prefill_linear = lambda x, w, q, n, pad=args.pad: _plain(x, w, q, n, pad=pad)
    os.environ["TENSORFOLD_GGUF_FAST_PAD"] = "1" if args.pad else "0"
    print(f"prefill padding under test: {args.pad}", flush=True)
    from tensorfold.cuda.rocm import HIP
    from tensorfold.families.qwen3_5.cuda.weights import Gguf

    raw = {name: getattr(gguf, name) for name in ("linear", "gemv", "prefill_linear", "linear_bf16")}   # unsynced, for timing
    for name, fn in raw.items():                                  # sync after each launch: a fault names its call
        setattr(gguf, name, synced(fn))
    if not HIP:
        print("FAIL: the fast prompt GEMM is ROCm only")
        return 1
    picked: dict[int, list] = {}
    for t in gguf.reader(args.model).tensors:
        q = int(t.tensor_type)
        if q not in gguf.PREFILL or len(t.shape) != 2 or not t.name.startswith("blk."):
            continue
        shapes = {(int(u.shape[0]), int(u.shape[1])) for u in picked.get(q, [])}
        if len(shapes) < args.per_type and (int(t.shape[0]), int(t.shape[1])) not in shapes:
            picked.setdefault(q, []).append(t)
    torch.manual_seed(0)
    failed = checked = 0
    # each timing runs only after the same call passed once under a sync above
    for q, ts in sorted(picked.items()):
        for t in ts:
            k, n = int(t.shape[0]), int(t.shape[1])                   # GGUF lists the input dimension first
            print(f"{t.tensor_type.name:7} {t.name:28} N={n:6} K={k:6} ...", flush=True)
            checked += 1
            try:
                w = torch.from_numpy(t.data.reshape(-1).view("uint8").copy()).cuda()
                x = (torch.randn(args.rows, k, device="cuda") * 0.5).to(torch.bfloat16)
                ref = gguf.prefill_linear(x, w, q, n)
                bad = [f"{r}@{o}" for r, o in SLICES if o + r <= args.rows
                       and not torch.equal(gguf.prefill_linear(x[o:o + r], w, q, n), ref[o:o + r])]
                perm = torch.randperm(args.rows, device="cuda")
                if not torch.equal(gguf.prefill_linear(x[perm], w, q, n), ref[perm]):
                    bad.append("permuted")
                if not torch.equal(gguf.prefill_linear(x.float(), w, q, n), ref):
                    bad.append("fp32-in")
                saved = os.environ.pop("TENSORFOLD_GGUF_FAST", None)
                one = x[:1].float()
                try:
                    g = Gguf(w.view(n, -1), q, k)
                    os.environ["TENSORFOLD_GGUF_FAST"] = "1"
                    wired = torch.equal(g.prefill(x[:513]), ref[:513].to(torch.bfloat16))
                    wired &= torch.equal(g(x[:1]), gguf.gemv(one, w, q, n).to(torch.bfloat16))
                    os.environ.pop("TENSORFOLD_GGUF_FAST")   # unset = the default, which is exact
                    wired &= torch.equal(g.prefill(x[:513]), gguf.linear(x[:513], w, q, n).to(torch.bfloat16))
                finally:
                    os.environ.pop("TENSORFOLD_GGUF_FAST", None)
                    if saved is not None:
                        os.environ["TENSORFOLD_GGUF_FAST"] = saved
                e1 = gguf.linear(one.expand(2, -1), w, q, n)[:1]
                g1 = gguf.gemv(one, w, q, n)
                serial = "equal" if torch.equal(g1, e1) else f"DIFFERS max abs {(g1 - e1).abs().max().item():.1e}"
                t1_exact = timed(lambda: raw["linear"](one.expand(2, -1), w, q, n), 20)
                t1_fast = timed(lambda: raw["gemv"](one, w, q, n), 20)
                verify = {
                    "wmma": lambda r: raw["prefill_linear"](r, w, q, n, pad=0),
                    "bf16": lambda r: raw["linear_bf16"](r, w, q, n),
                    "gemv": lambda r: torch.cat([raw["gemv"](r[i:i + 1].float(), w, q, n) for i in range(r.shape[0])]),
                    "exact": lambda r: raw["linear"](r.float(), w, q, n),
                }
                widths = []
                for rows in (2, 4, 8, 16):
                    xr = x[100:100 + rows]
                    base = verify["exact"](xr)
                    torch.cuda.synchronize()
                    cells = []
                    for name, fn in verify.items():
                        y = fn(xr)
                        torch.cuda.synchronize()                    # each option once under a sync before timing
                        err = (y - base).abs().max().item()
                        cells.append(f"{name} {timed(lambda: fn(xr), 20) * 1e3:.0f}us/{err:.0e}")
                    widths.append(f"{rows}: " + " ".join(cells))
                short = []
                for rows in (20, 40):
                    xr = x[200:200 + rows]
                    gguf.prefill_linear(xr, w, q, n, pad=0), gguf.prefill_linear(xr, w, q, n, pad=96)
                    short.append(f"{rows}: unpadded {timed(lambda: raw['prefill_linear'](xr, w, q, n, pad=0), 20) * 1e3:.0f}us "
                                 f"padded {timed(lambda: raw['prefill_linear'](xr, w, q, n, pad=96), 20) * 1e3:.0f}us")
                exact = gguf.linear(x.float(), w, q, n)
                diff = (ref - exact).abs()
                rel = ((ref - exact).norm() / exact.norm()).item()
                worst = (diff.max() / exact.abs().max()).item()
                control = not torch.equal(ref, exact)
                t_fast = timed(lambda: raw["prefill_linear"](x, w, q, n), args.iters)
                t_exact = timed(lambda: raw["linear"](x.float(), w, q, n), args.iters)
                ok = not bad and wired and control and rel < 2e-2
                failed += not ok
                print(f"{t.tensor_type.name:7} {t.name:28} N={n:6} K={k:6}  chunk-independent={'yes' if not bad else bad}  "
                      f"wired={'yes' if wired else 'NO'}  vs exact: max abs={diff.max().item():.2e} max/max={worst:.1e} "
                      f"L2 rel={rel:.1e}{'' if control else ' EQUAL (control cannot fail)'}  "
                      f"{args.rows} rows: fast {t_fast:.2f} ms, exact {t_exact:.2f} ms ({t_exact / t_fast:.1f}x)  "
                      f"1 row: gemv {t1_fast * 1e3:.0f} us, exact {t1_exact * 1e3:.0f} us, gemv vs exact bits {serial}  "
                      f"verify rows (time/max abs vs exact) {' | '.join(widths)}  prompt rows {' | '.join(short)}  "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report and go on (a HIP fault is sticky: later tensors fail too)
                failed += 1
                print(f"{t.tensor_type.name:7} {t.name:28} FAIL: {type(exc).__name__}: {exc}", flush=True)
    if not checked:
        print("FAIL: no quantized projection was checked")
        failed += 1
    print(f"ALL PASS ({checked} tensors, {len(picked)} types)" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
