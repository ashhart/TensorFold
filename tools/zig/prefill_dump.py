"""One prompt chunk through prefill_forward.py, its rows saved for tf-nemotron-prefill-check: [layers + 2, L, D] bf16 bits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("ids", type=Path, help="a JSON object of prompt ids by name")
    ap.add_argument("name")
    ap.add_argument("out", type=Path, help="the .npy to write: the embedding, each layer's residual, the final norm")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    from prefill_forward import Prefill, State
    from tensorfold.families.nemotron_h import load

    fam, _ = load(args.model)
    run = Prefill(fam.model.backbone, fam.model.args)
    ids = json.loads(args.ids.read_text())[args.name]
    layers = fam.model.backbone.layers
    states = [State() for layer in layers if layer.block_type in "M*"]
    h = run.embed(ids)
    rows, at = [h], 0
    for layer in layers:
        hn = run.rms(h, layer.norm.weight, run.args.hidden_size)
        if layer.block_type in "M*":
            mixer = run.mamba if layer.block_type == "M" else run.attention
            out = mixer(layer.mixer, hn, states[at])
            at += 1
        else:
            out = run.moe(layer.mixer, hn)
        h = run.add(h, out)
        mx.eval(h)
        rows.append(h)
    rows.append(run.rms(h, run.b.norm_f.weight, run.args.hidden_size))
    np.save(args.out, np.array(mx.stack(rows).view(mx.uint16)))
    print(f"{args.name}: {len(ids)} ids, {len(rows)} row blocks -> {args.out}\n{','.join(map(str, ids))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
