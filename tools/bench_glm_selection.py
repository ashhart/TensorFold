"""Baseline/candidate GLM DSA selection and latent attention, eager and graphs.

POOL_REFERENCE must name upstream sparse.py. Inputs use real per-rank GLM
shapes, not trained weights. JSONL reports graph replay latency separately.
"""
from __future__ import annotations

import importlib.util
import json
import os

import torch
import triton

from tensorfold.families.glm5_next.cuda import latent, sparse


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    return graph, result


def main():
    spec = importlib.util.spec_from_file_location("pool_reference", os.environ["POOL_REFERENCE"])
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(128)
    for pos in (2049, 16383, 199999, 205000):
        for rows in (1, 6, 16):
            bucket = sparse.pool_bucket(pos, rows, 65536)
            qi = torch.randn((rows, 32 * 128), device="cuda", dtype=torch.bfloat16)
            weights = torch.randn((rows, 32), device="cuda", dtype=torch.bfloat16)
            keys = torch.randn((65538, 128), device="cuda", dtype=torch.bfloat16)
            position = torch.tensor([pos], device="cuda", dtype=torch.int32)

            def select(module, qi=qi, weights=weights, keys=keys, position=position, pos=pos, rows=rows):
                return module.select_tokens(qi, weights, keys, pos, rows, 65536, position)

            old = select(reference)
            new = select(sparse)
            assert all(torch.equal(a, b) for a, b in zip(old, new)), (pos, rows)
            gb, ob = capture(lambda: select(reference))
            gn, on = capture(lambda: select(sparse))
            # Replays must read changing device positions, including partial pool boundaries.
            for offset in (0, -1, -3, -127):
                position.fill_(pos + offset)
                gb.replay()
                gn.replay()
                torch.cuda.synchronize()
                assert all(torch.equal(a, b) for a, b in zip(ob, on)), (pos, rows, offset)
            position.fill_(pos)
            timings = {}
            for name, fn in (("baseline_eager", lambda: select(reference)),
                             ("candidate_eager", lambda: select(sparse)),
                             ("baseline_graph", gb.replay), ("candidate_graph", gn.replay)):
                timings[name] = triton.testing.do_bench(fn, warmup=50, rep=200)
            qa = torch.randn((rows, 32, 512), device="cuda", dtype=torch.bfloat16)
            cache = torch.randn((pos + rows + 1, 512), device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(qa)
            gb.replay()
            gn.replay()
            latent.sparse_attention(qa, cache, ob[0], ob[1], out, 256 ** -0.5)
            want = out.clone()
            latent.sparse_attention(qa, cache, on[0], on[1], out, 256 ** -0.5)
            assert torch.equal(want, out)
            ga, _ = capture(lambda cache=cache: latent.sparse_attention(qa, cache, on[0], on[1], out, 256 ** -0.5))
            timings["attention_graph"] = triton.testing.do_bench(ga.replay, warmup=50, rep=200)
            print(json.dumps({"pos": pos, "rows": rows, "bucket": bucket,
                              "bits_equal": True, "ms": timings}), flush=True)
            del ga, gb, gn, cache


if __name__ == "__main__":
    main()
