"""glm.dsa.block fixtures: one real RED-SNOW DSA layer's dsa_block at TP=2 rank 0, fp8 latent cache, kv_b bf16."""
from __future__ import annotations

import dataclasses
import gc
import os

os.environ.setdefault("TF_GLM_LATENT", "1")

from pathlib import Path

import torch

LAYER = 3                    # the first DSA layer (MoE)
DECODE = (1, 3, 8)           # decode windows
# decode positions: dense with the indexer, a window across the dense limit, all-sparse at and past it, ~8K
DECODE_POS = (100, 2048, 2051, 2052, 8000)
PROMPT = ((1800, 512), (1536, 520))   # prompt chunks across the dense limit; the second ends in an 8-row block
FILL = 8192                  # the pre-state: this many tokens written by the real block in prompt chunks
CAP = 10240                  # cache rows (cap / 4 >= the 8K window's 2048-pool bucket)
DECODE_ROWS, PREFILL_ROWS = 8, 2048   # the engine's buffers (their row counts set the dense pass's chunk cap)
SEED = 5353


def save(out: Path, name: str, t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    (out / f"{name}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return f"{name}.bin"


def _load(model_dir: Path):
    """weights.load for rank 0, cut to layers 0 .. LAYER (Config.layers patched), then just the layer kept."""
    from tensorfold.families.glm5_next.cuda import weights

    desc = weights.Config.__dict__["read"]
    orig = weights.Config.read
    weights.Config.read = classmethod(lambda cls, d: dataclasses.replace(orig(d), layers=LAYER + 1))
    try:
        w = weights.load(model_dir, rank=0, mtp=False)
    finally:
        weights.Config.read = desc
    w.layers = [w.layers[LAYER]]
    w.meta["long_context"] = True      # the recipe's 128K context: indexer caches, the indexer on every call
    gc.collect()
    torch.cuda.empty_cache()
    return w


def _weights(out: Path, a, pre: str = "dbw") -> dict:
    """Rank 0's DSA tensors exactly as the engine holds them (EXL3 words in its strips layout)."""

    def lin(name: str, q) -> dict:
        assert q.layout == "strips" and q.bias is None, name
        return {"words": save(out, f"{pre}_{name}_words", q.words), "suh": save(out, f"{pre}_{name}_suh", q.suh),
                "svh": save(out, f"{pre}_{name}_svh", q.svh), "k": q.k, "n": q.n, "k2": q.k2, "codebook": q.codebook}

    def b16(name: str, q) -> dict:
        assert q.weight.dtype == torch.bfloat16, name
        return {"w": save(out, f"{pre}_{name}", q.weight), "n": q.n, "k": q.k}

    def plain(name: str, t: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> str:
        assert t.dtype == dtype, (name, t.dtype)
        return save(out, f"{pre}_{name}", t)

    ab, ix = a.absorb, a.index
    assert type(ab).__name__ == "AbsorbW" and a.proj is None, "the recipe's bf16 AbsorbW and separate q_a/kv_a"
    return {"q_a": lin("q_a", a.q_a), "kv_a": lin("kv_a", a.kv_a), "q_b": lin("q_b", a.q_b), "o": lin("o", a.o),
            "q_norm": plain("q_norm", a.q_norm), "kv_norm": plain("kv_norm", a.kv_norm),
            "wk": plain("wk", ab.wk), "wv": plain("wv", ab.wv),
            "kw": b16("kw", ix.kw), "qb": b16("qb", ix.qb), "gate": plain("gate", ix.gate),
            "ln_w": plain("ln_w", ix.ln_w, torch.float16), "ln_b": plain("ln_b", ix.ln_b, torch.float16),
            "ape": plain("ape", ix.ape)}


def _block(w, st, b, R: int, pos: int) -> torch.Tensor:
    """forward.dsa_block as _mixer calls it for this layer (eager: host_pos set, nch from the window)."""
    from tensorfold.families.glm5_next.cuda import forward

    st.set_pos(pos)
    return forward.dsa_block(w.layers[0], w, st.kc[0], st.vc[0], st.pos_dev, b, R, forward.chunks_for(st, R),
                             st.index[0], pos, None, st.lat_scales[0])


def _caches(st) -> list[torch.Tensor]:
    ik, ig, pk = st.index[0]
    return [st.kc[0].view(torch.uint8), st.lat_scales[0], ik, ig, pk]


NAMES = ("kc", "scales", "ik", "ig", "pk")


def _case(out: Path, w, st, pre: list[torch.Tensor], tag: str, R: int, pos: int, prefill: bool) -> dict:
    """One window from the shared pre-state: rank 0's stage bits, the fp32 partial and the caches' written rows."""
    from tensorfold.families.glm5_next.cuda import forward, sparse

    c = w.cfg
    for dst, src in zip(_caches(st), pre):
        dst.copy_(src)
    b = forward.Buffers(w, PREFILL_ROWS if prefill else DECODE_ROWS, CAP, prefill=prefill)
    g = torch.Generator().manual_seed(SEED + 7 * R + pos)
    normed = torch.randn((R, c.hidden), generator=g).to(torch.bfloat16).cuda()
    b.normed[:R].copy_(normed)
    part = _block(w, st, b, R, pos).reshape(R, c.hidden).clone()
    torch.cuda.synchronize()
    s = b.lat_s
    e = {"rows": R, "pos": pos, "prefill": prefill, "buf_rows": b.rows}
    stages = {"normed": normed, "dqa": b.dqa[:R], "dkv": b.dkv[:R], "qr": b.qr[:R], "lat": b.lat[:R], "q": b.q[:R],
              "ikr": b.ikr[:R], "igr": b.igr[:R], "qa": s.qa[:R], "ol": s.ol[:R], "vn": b.vn[:R], "part": part}
    sparse_rows = pos + R - 1 >= c.dense_limit
    if sparse_rows:          # the selection again from the post-state pools it read (index_update ran before it)
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], st.index[0][2], pos, R,
                                              st.index[0][2].shape[0] - 2, st.pos_dev)
        stages.update(qi=b.qi[:R], tokens=tokens, counts=counts)
    for key, t in stages.items():
        e[key] = save(out, f"db_{tag}_{key}", t)
    # every cache row outside the window's must be untouched: the fixture keeps the written rows only
    win = {"kc": (pos, pos + R), "scales": (pos, pos + R), "ik": (pos, pos + R), "ig": (pos, pos + R),
           "pk": (pos // 4, (pos + R) // 4)}
    for name, post, before in zip(NAMES, _caches(st), pre):
        lo, hi = win[name]
        assert torch.equal(post[:lo], before[:lo]) and torch.equal(post[hi:], before[hi:]), (tag, name)
        e[name] = {"lo": lo, "hi": hi, "file": save(out, f"db_{tag}_{name}", post[lo:hi])}
    print(f"  dsa block {tag}: {R} rows at {pos}, part {part.abs().max():.3g} max, sparse {sparse_rows}")
    return e


def write(out: Path, model_dir: Path) -> dict:
    from tensorfold.families.glm5_next.cuda import forward

    out.mkdir(parents=True, exist_ok=True)
    w = _load(model_dir)
    c, a = w.cfg, w.layers[0].dsa
    assert w.layers[0].kind == "dsa" and a.index is not None and c.dense_limit == 2051
    st = forward.State(w, CAP, DECODE_ROWS, kv="fp8")
    # the true pre-state: FILL tokens through the real block in 2,048-row prompt chunks (dense, straddling, sparse)
    g = torch.Generator().manual_seed(SEED)
    b = forward.Buffers(w, PREFILL_ROWS, CAP, prefill=True)
    for p0 in range(0, FILL, PREFILL_ROWS):
        R = min(PREFILL_ROWS, FILL - p0)
        b.normed[:R].copy_(torch.randn((R, c.hidden), generator=g).to(torch.bfloat16).cuda())
        _block(w, st, b, R, p0)
    torch.cuda.synchronize()
    del b
    pre = [t.clone() for t in _caches(st)]
    man = {"layer": LAYER, "params": {"eps": c.eps, "index_eps": 1e-6, "scale": c.qk_dim ** -0.5,
                                      "index_scale": c.index_dim ** -0.5, "wscale": 1.0 / 5.656854249492381,
                                      "dense_limit": c.dense_limit, "cap": CAP, "hidden": c.hidden, "fill": FILL},
           "weights": _weights(out, a), "pre": {n: save(out, f"db_pre_{n}", t) for n, t in zip(NAMES, pre)},
           "cases": {}}
    for pos in DECODE_POS:
        for R in DECODE:
            man["cases"][f"d{pos}r{R}"] = _case(out, w, st, pre, f"d{pos}r{R}", R, pos, False)
    for pos, R in PROMPT:
        man["cases"][f"p{pos}r{R}"] = _case(out, w, st, pre, f"p{pos}r{R}", R, pos, True)
    return man
