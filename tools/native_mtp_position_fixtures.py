"""Original speculate/settle methods isolate MTP sampling positions and retained rows.

The MTP block is an identity stub: model arithmetic is covered separately. The
unmodified Python methods choose every sampling position and retained hidden row.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_h.model import NemotronH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, cases = {}, []
    rng = np.random.default_rng(8734)
    for position in (0, 63, 10007, 262128):
        for seed, temperature in ((5678, .7), (2**64 - 1, 1.), (0, 0.)):
            settings = Sampling(seed, temperature=temperature, top_k=12, top_p=.8) if temperature else None
            for mapped in (False, True):
                hidden = mx.array(rng.standard_normal((4, 32)).astype(np.float32))
                mapping = mx.arange(32, dtype=mx.uint32) * 3 + 7 if mapped else None
                model = SimpleNamespace(
                    _last_hidden=hidden[None], _draft_ids=mapping,
                    _trim_chained=lambda cache: None, _draft_logits=lambda state: state,
                    mtp=lambda state, embedding, cache, tail=None: state if tail is None else state[:, -tail:],
                    model=SimpleNamespace(backbone=SimpleNamespace(embeddings=lambda tokens: tokens)),
                )
                for keep in (0, 1, 2, 3, 4):
                    for budget in (1, 3, 15):
                        cache = [SimpleNamespace(drafted=0, trim=lambda count: None)]
                        if keep == 0:
                            first = NemotronH.speculate(model, cache, mx.array([77], dtype=mx.uint32),
                                                       position - 1, settings, start=-1)[0]
                            expected = NemotronH.settle(model, cache, 1, first, position + 1, settings, budget)
                        else:
                            firsts = NemotronH.speculate(model, cache, mx.array([1, 2, 3, 4], dtype=mx.uint32),
                                                        position, settings)
                            expected = NemotronH.settle(model, cache, keep, firsts[keep - 1],
                                                       position + keep + 1, settings, budget)
                        key = f"c{len(cases)}"
                        arrays.update({key + ".hidden": hidden, key + ".expected": expected})
                        if mapped:
                            arrays[key + ".ids"] = mapping
                        cases.append(dict(key=key, position=position, seed=seed, temperature=temperature,
                                          keep=keep, budget=budget, mapped=mapped))
    mx.save_safetensors(str(args.output / "arrays.safetensors"), arrays)
    (args.output / "cases.json").write_text(json.dumps(cases) + "\n")
    print(f"Saved {len(cases)} original MTP position/settle fixtures", flush=True)


if __name__ == "__main__":
    main()
