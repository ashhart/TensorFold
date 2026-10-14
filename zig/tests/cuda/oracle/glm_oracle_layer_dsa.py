"""glm.layer.dsa fixtures: one real DSA MoE layer's whole forward at TP=2 from a real prefill's caches, fp8 latents."""
from __future__ import annotations

import os

os.environ.setdefault("TF_GLM_LATENT", "1")

import threading
from pathlib import Path

import torch

import glm_oracle_dsa as dsa_oracle
from glm_oracle_layer import Comm, Hub, _load, _weights, save

LAYER = 3                    # the first DSA layer (MoE)
DECODE = (1, 3, 8)           # decode windows
DECODE_POS = (512, 4000)     # below the dense limit (2,051) and past it
PROMPT, PROMPT_POS = 300, 1900   # one prompt chunk across the dense limit (rows 151.. are sparse)
FILL = 4096                  # the pre-state: this many tokens through the real layer_forward in prompt chunks
CAP = 4096                   # cache rows (cap / 4 = 1,024 pools: the 4K windows' bucket)
DECODE_ROWS, PREFILL_ROWS = 8, 2048   # the engine's decode and prompt buffers
SEED = 7373
NAMES = dsa_oracle.NAMES


def _caches(st) -> list[torch.Tensor]:
    ik, ig, pk = st.index[0]
    return [st.kc[0].view(torch.uint8), st.lat_scales[0], ik, ig, pk]


def _run(W, R: int, pos: int, prefill: bool, split: bool, pres, x0s, snap0=None) -> list[dict]:
    """layer_forward (or split_layers) on both ranks at once from each rank's caches; each rank's x and caches."""
    from tensorfold.families.glm5_next.cuda import forward
    from tensorfold.families.glm5_next.cuda.hcsplit import HcSplit, SplitSettings

    hub, res, errs = Hub(), [None, None], []

    def body(rank: int) -> None:
        try:
            w = W[rank]
            w.comm = Comm(hub, rank)
            st = forward.State(w, CAP, DECODE_ROWS, kv="fp8")
            if pres is not None:     # else the fill: State's zeroed caches
                for dst, src in zip(_caches(st), pres[rank]):
                    dst.copy_(src)
            b = forward.Buffers(w, PREFILL_ROWS if prefill else DECODE_ROWS, CAP, prefill=prefill)
            if split:      # TF_GLM_HC_SPLIT=1, TF_GLM_PREFILL_OVERLAP=2 (two pieces, p2p swaps on a second stream)
                b.split = HcSplit(w, b, SplitSettings(True, True, 2, 64, "p2p", True))
            if rank == 0 and snap0 is not None:
                w.comm.snap = lambda g: snap0(g, st, b)
            for p0, x0 in zip(range(pos, pos + R * len(x0s), R), x0s):   # the fill runs several chunks
                n = x0.shape[0]
                st.set_pos(p0)
                b.x[:n].copy_(x0)
                if split:
                    forward.split_layers(w, st, b, n, forward.chunks_for(st, n), p0)
                else:
                    forward.layer_forward(w.layers[0], w, st, b, n, forward.chunks_for(st, n), p0)
            torch.cuda.synchronize()
            res[rank] = {"x": b.x[:x0s[-1].shape[0]].clone(), "caches": [t.clone() for t in _caches(st)]}
        except BaseException as e:   # the peer must not wait at the barrier forever
            errs.append(e)
            hub.bar.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    for w in W:
        w.comm = None
    return res


def _case(out: Path, W, pres, tag: str, R: int, pos: int, prefill: bool) -> dict:
    """One window from the pre-state: rank 0's bits at every stage, rank 1's two partials, x and the caches after."""
    from tensorfold.families.glm5_next.cuda import sparse

    c = W[0].cfg
    g = torch.Generator().manual_seed(SEED + 7 * R + pos)
    x0 = torch.randn((R, c.streams * c.hidden), generator=g).to(torch.bfloat16).cuda()
    sparse_rows = pos + R - 1 >= c.dense_limit
    snaps: list[dict] = []

    def snap(gath: torch.Tensor, st, b) -> None:
        site = {"gath": gath.clone(), "x": b.x[:R].clone(), "normed": b.normed[:R].clone(),
                "post": b.post[:R].clone(), "comb": b.comb[:R].clone()}
        if not snaps:        # the attention site: the DSA block's stages (dsa_oracle's)
            s = b.lat_s
            for key, t in (("dqa", b.dqa), ("dkv", b.dkv), ("qr", b.qr), ("lat", b.lat), ("q", b.q),
                           ("ikr", b.ikr), ("igr", b.igr), ("qa", s.qa), ("ol", s.ol), ("vn", b.vn)):
                site[key] = t[:R].clone()
            if sparse_rows:  # the selection again from the post-state pools it read
                pk = st.index[0][2]
                tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, pos, R,
                                                      pk.shape[0] - 2, st.pos_dev)
                site.update(qi=b.qi[:R].clone(), tokens=tokens.clone(), counts=counts.clone())
        else:                # the MoE site: route, per-slot y (shared expert in slot 8)
            for key in ("mlog", "pick", "wts", "sy", "smg", "smu", "sact", "ey"):
                site[key] = getattr(b, key)[:R].clone()
        snaps.append(site)

    res = _run(W, R, pos, prefill, False, pres, [x0], snap)
    assert len(snaps) == 2, len(snaps)
    assert torch.equal(res[0]["x"], res[1]["x"]), "the two ranks' x differ"
    e = {"rows": R, "pos": pos, "prefill": prefill, "buf_rows": PREFILL_ROWS if prefill else DECODE_ROWS,
         "sparse": sparse_rows, "x0": save(out, f"ld_{tag}_x0", x0), "x": save(out, f"ld_{tag}_x", res[0]["x"])}
    n = R * c.hidden
    for i, site in enumerate(snaps):
        gath = site.pop("gath")
        e[f"s{i}"] = {"part": save(out, f"ld_{tag}_s{i}_part", gath[:n]),
                      "peer": save(out, f"ld_{tag}_s{i}_peer", gath[n:2 * n])}
        for key, t in site.items():
            e[f"s{i}"][key] = save(out, f"ld_{tag}_s{i}_{key}", t)
    # every cache row outside the window's must be untouched: the fixture keeps the written rows only
    win = {"kc": (pos, pos + R), "scales": (pos, pos + R), "ik": (pos, pos + R), "ig": (pos, pos + R),
           "pk": (pos // 4, (pos + R) // 4)}
    for name, post, before in zip(NAMES, res[0]["caches"], pres[0]):
        lo, hi = win[name]
        assert torch.equal(post[:lo], before[:lo]) and torch.equal(post[hi:], before[hi:]), (tag, name)
        e[name] = {"lo": lo, "hi": hi, "file": save(out, f"ld_{tag}_{name}", post[lo:hi])}
    if prefill:              # the same chunk row-split between the ranks: own rows of x and every cache equal
        sres = _run(W, R, pos, prefill, True, pres, [x0])
        h = R // 2
        for rank, rows in ((0, slice(0, h)), (1, slice(h, R))):
            assert torch.equal(sres[rank]["x"][rows], res[rank]["x"][rows]), f"split x rank {rank}"
            for name, a, b in zip(NAMES, sres[rank]["caches"], res[rank]["caches"]):
                assert torch.equal(a, b), f"split {name} rank {rank}"
        e["split_equal"] = True
    print(f"  layer dsa {tag}: {R} rows at {pos}, x {res[0]['x'].float().abs().max():.3g} max, sparse {sparse_rows}")
    return e


def write(out: Path, model_dir: Path) -> dict:
    W = [_load(model_dir, rank, LAYER, long=True) for rank in (0, 1)]
    c, L = W[0].cfg, W[0].layers[0]
    a = L.dsa
    assert L.kind == "dsa" and a.index is not None and L.moe is not None and L.moe.shared is not None
    assert c.dense_limit == 2051 and c.norm_topk
    # the true pre-state: FILL tokens through the real layer_forward on both ranks in 2,048-row prompt chunks
    g = torch.Generator().manual_seed(SEED)
    fill = [torch.randn((PREFILL_ROWS, c.streams * c.hidden), generator=g).to(torch.bfloat16).cuda()
            for _ in range(FILL // PREFILL_ROWS)]
    pres = [r["caches"] for r in _run(W, PREFILL_ROWS, 0, True, False, None, fill)]
    del fill
    man = {"layer": LAYER,
           "params": {"eps": c.eps, "hc_eps": c.hc_eps, "hc_iters": c.hc_iters, "lower": c.lower, "limit": c.limit,
                      "routed_scale": c.routed_scale, "top_k": c.top_k, "experts": c.experts, "hidden": c.hidden,
                      "streams": c.streams, "index_eps": 1e-6, "scale": c.qk_dim ** -0.5,
                      "index_scale": c.index_dim ** -0.5, "wscale": 1.0 / 5.656854249492381,
                      "dense_limit": c.dense_limit, "cap": CAP, "fill": FILL},
           "weights": _weights(out, L, c, "ldw"), "pre": {n: save(out, f"ld_pre_{n}", t) for n, t in zip(NAMES, pres[0])},
           "cases": {}}
    man["weights"]["dsa"] = dsa_oracle._weights(out, a, "ldw_dsa")
    for pos in DECODE_POS:
        for R in DECODE:
            man["cases"][f"d{pos}r{R}"] = _case(out, W, pres, f"d{pos}r{R}", R, pos, False)
    man["cases"][f"p{PROMPT_POS}r{PROMPT}"] = _case(out, W, pres, f"p{PROMPT_POS}r{PROMPT}", PROMPT, PROMPT_POS, True)
    return man
