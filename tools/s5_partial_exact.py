"""Stream J (2026-09-27): the abliterated checkpoint's first layers on S5's GPU beside the live server — windows of
2/3/4/8/16 rows must give every row the bits the one-row (serial) step gives it, with every row kernel and fused
kernel on. Loads ``--layers`` layers (5 = dense 0-2 at 8/5/6-bit, MLA 3 at 8-bit with an 8-bit shared expert, KDA 4 at
8-bit) plus embed and the 8-bit lm_head: about 20 GB. Also times one-row and 2-row steps (kernel on vs off) for a
first read of the lever. Usage: PYTHONPATH=src python tools/s5_partial_exact.py <model_dir> --layers 5"""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--prompt", type=int, default=300)
    args = ap.parse_args()
    from tensorfold.families.glm5_next import model as glm
    from tensorfold.engine.lane_engine import LaneEngine

    assert glm.ENABLED == frozenset(glm.ROW_KERNELS), glm.ENABLED
    t0 = time.time()
    model = glm.load_backbone(args.model_dir, layers=args.layers)
    print(f"loaded {args.layers} layers in {time.time() - t0:.1f} s; active memory {mx.get_active_memory() / 2**30:.1f} GB",
          flush=True)
    kinds = []
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        if layer.is_linear:
            kinds.append(f"L{i} KDA in_proj={type(attn.in_proj).__name__}/{getattr(attn.in_proj, 'bits', 'mixed')}b "
                         f"f_b={attn.f_b.bits}b o={attn.o_proj.bits}b fused_kda={__import__('tensorfold.kernels.glm.flash.v1.kda', fromlist=['fits']).fits(attn)}")
        else:
            kinds.append(f"L{i} MLA x_proj={type(attn.x_proj).__name__} o={attn.o_proj.bits}b wk={attn.wk.bits}b")
        mlp = layer.mlp
        if hasattr(mlp, "shared"):
            kinds.append(f"   MoE shared={mlp.shared.gate_up.bits if mlp.shared else None}b fused_ok={mlp.fused_ok}")
        else:
            kinds.append(f"   dense gate_up={getattr(mlp.gate_up, 'bits', 'mixed')}b down={mlp.down.bits}b")
    print("\n".join(kinds), f"\nlm_head {model.lm_head.bits}b", flush=True)

    base = model.make_cache()
    prompt = [1000 + (37 * i) % 50_000 for i in range(args.prompt)]
    for c0 in range(0, len(prompt), 2048):
        mx.eval(model.hidden(mx.array([prompt[c0:c0 + 2048]]), base))
    tokens = [3001 + 17 * r for r in range(16)]
    one = LaneEngine.copy_single_cache(base)
    serial = mx.concatenate([model.head(model.hidden(mx.array([[t]]), one)) for t in tokens], axis=1)
    mx.eval(serial)
    ok = True
    for width in (2, 3, 4, 8, 16):
        many = LaneEngine.copy_single_cache(base)
        window = model.head(model.hidden(mx.array([tokens[:width]]), many))
        same = bool(mx.array_equal(window, serial[:, :width]).item())
        if not same:
            diff = mx.abs(window.astype(mx.float32) - serial[:, :width].astype(mx.float32))
            print(f"width {width}: DIFFERS max {diff.max().item():.4g} at {int(mx.argmax(diff.max(axis=-1)).item())}", flush=True)
        ok &= same
        print(f"width {width}: {'exact' if same else 'NOT exact'}", flush=True)
    print("RESULT", "EXACT" if ok else "NOT-EXACT", flush=True)

    # timing: one-row and two-row steps, dependent chain, kernels on (this build) — the layer subset's cost only
    def step(width: int, n: int = 20) -> float:
        cache = LaneEngine.copy_single_cache(base)
        mx.eval(model.head(model.hidden(mx.array([tokens[:width]]), cache)))
        t = time.time()
        for _ in range(n):
            out = model.head(model.hidden(mx.array([tokens[:width]]), cache))
            mx.eval(out)
        return (time.time() - t) / n * 1000
    for width in (1, 2, 4):
        print(f"{args.layers}-layer step, {width} row(s): {step(width):.2f} ms", flush=True)


if __name__ == "__main__":
    main()
