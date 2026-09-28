"""How far each Gemma 4 decode path lands from the same 4-bit weights run with fp32 activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOP = 256


def _pairs(path: Path) -> list[tuple[list[int], list[int]]]:
    return [tuple(pair) for pair in json.loads(path.read_text())]


def truth(model_dir: Path, eval_set: Path, out: Path) -> None:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    model, _ = load(str(model_dir))
    model.set_dtype(mx.float32)          # norms, scales and biases to fp32; the 4-bit weights stay as they are
    ids_all, lps_all = [], []
    for prompt, answer in _pairs(eval_set):
        seq = prompt + answer
        k = len(prompt)
        lg = model(mx.array([seq]))[0, k - 1:-1].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        top = mx.argpartition(-lp, TOP, axis=-1)[:, :TOP]
        ids_all.append(np.array(top))
        lps_all.append(np.array(mx.take_along_axis(lp, top, axis=-1)))
        print(f"truth: {len(answer)} tokens", flush=True)
    np.savez(out, ids=np.concatenate(ids_all), lps=np.concatenate(lps_all))


def compare(model_dir: Path, eval_set: Path, truth_file: Path, backend: str | None) -> None:
    import mlx.core as mx
    import numpy as np

    from tensorfold.families.gemma4.model import load

    model, _ = load(model_dir, backend=backend, check=False)
    data = np.load(truth_file)
    t_ids, t_lps = mx.array(data["ids"]), mx.array(data["lps"])
    t_p = mx.exp(t_lps)
    rows_whole, rows_mlx, rows_decode = [], [], []
    for prompt, answer in _pairs(eval_set):
        seq = prompt + answer
        k = len(prompt)
        rows_whole.append(model.model(mx.array([seq]))[0, k - 1:-1].astype(mx.float32))
        ref_cache = model.text.make_cache()
        mlx_steps = [model.text(mx.array([prompt]), cache=ref_cache)[0, -1]]
        cache = model.make_cache()
        steps = [model.head(model.prefill(mx.array([prompt]), cache)[:, -1:])[0, -1]]
        for t in answer[:-1]:
            mlx_steps.append(model.text(mx.array([[t]]), cache=ref_cache)[0, -1])
            steps.append(model(mx.array([[t]]), cache)[0, -1])
            if len(steps) % 64 == 0:
                mx.eval(steps[-64:], mlx_steps[-64:])
        rows_mlx.append(mx.stack(mlx_steps).astype(mx.float32))
        rows_decode.append(mx.stack(steps).astype(mx.float32))
    for name, rows in (("mlx_lm whole-sequence forward", rows_whole), ("mlx_lm decode", rows_mlx),
                       (f"decode path ({model.decode.backend})", rows_decode)):
        lg = mx.concatenate(rows)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        q = mx.take_along_axis(lp, t_ids, axis=-1)
        kl = np.array((t_p * (t_lps - q)).sum(-1))
        best = mx.take_along_axis(t_ids, mx.argmax(t_lps, axis=-1, keepdims=True), axis=-1)[:, 0]
        agree = (mx.argmax(lp, axis=-1) == best).astype(mx.float32).mean().item()
        # a few knife-edge positions dominate the mean: the median and the mean without the worst 0.5% say more
        trimmed = np.sort(kl)[: int(len(kl) * 0.995)].mean()
        print(f"{name:32s} KL(truth || path) mean {kl.mean():.5f}, trimmed {trimmed:.5f}, median "
              f"{np.median(kl):.2e} nats/token; {int((kl > 1).sum())} tokens over 1 nat; argmax = truth's "
              f"{agree * 100:.2f}% ({lg.shape[0]} tokens)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="truth: the fp32-activation forward's top 256 log-probabilities at each "
                                 "answer position; compare: KL(truth || path) and argmax agreement for mlx_lm's whole "
                                 "forward, mlx_lm's decode and the family's decode path (a path's rounding error).")
    ap.add_argument("mode", choices=("truth", "compare"))
    ap.add_argument("model")
    ap.add_argument("eval_set")
    ap.add_argument("truth")
    ap.add_argument("--backend", choices=("lane", "rows"), default=None, help="default: by GPU")
    args = ap.parse_args()
    if args.mode == "truth":
        truth(Path(args.model), Path(args.eval_set), Path(args.truth))
    else:
        compare(Path(args.model), Path(args.eval_set), Path(args.truth), args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
