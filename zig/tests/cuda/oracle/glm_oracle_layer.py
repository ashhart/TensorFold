"""glm.layer.kda/mlp fixtures: a real KDA layer's forward (MoE or dense MLP) at TP=2, both ranks in one process."""
from __future__ import annotations

import dataclasses
import gc
import threading
from pathlib import Path

import torch

LAYER = 4                    # the first KDA layer with a MoE (0-2 carry the dense MLP, 3 is DSA)
DECODE = (1, 3, 8)           # decode windows: the narrow chain, then commit(R, R) and its conv shift
KEEP = 5                     # the 8-row window also commits its first 5 rows (a draft's accepted prefix)
PROMPT, PROMPT_POS = 300, 37  # one prompt chunk at an unaligned position: its first KDA sub-chunk starts 5 rows in
DECODE_POS = 512             # a decode window's position (KDA reads none; the State carries it)
DECODE_ROWS, PREFILL_ROWS, CAP = 8, 2048, 4096   # the engine's decode and prompt buffers
SEED = 6161


def save(out: Path, name: str, t: torch.Tensor) -> str:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return f"{name}.bin"


def _load(model_dir: Path, rank: int, layer: int = LAYER, long: bool = False):
    """weights.load for one rank, cut to layers 0 .. layer, then just the layer kept; ``long``: indexer caches."""
    from tensorfold.families.glm5_next.cuda import weights

    desc = weights.Config.__dict__["read"]
    orig = weights.Config.read
    weights.Config.read = classmethod(lambda cls, d: dataclasses.replace(orig(d), layers=layer + 1))
    try:
        w = weights.load(model_dir, rank=rank, mtp=False)
    finally:
        weights.Config.read = desc
    w.layers = [w.layers[layer]]
    if long:
        w.meta["long_context"] = True
    gc.collect()
    torch.cuda.empty_cache()
    return w


class Hub:
    """Where the two rank threads meet: one barrier, one mailbox per rank."""

    def __init__(self) -> None:
        self.bar = threading.Barrier(2, timeout=1800)
        self.box: list = [None, None]


class Comm:
    """A two-rank communicator over threads: all_gather in rank order and exchange, synchronous on the host."""

    def __init__(self, hub: Hub, rank: int) -> None:
        self.hub, self.rank, self.snap = hub, rank, None

    def _post(self, sends) -> list:
        torch.cuda.synchronize()
        self.hub.box[self.rank] = [t.clone() for t in sends]
        self.hub.bar.wait()
        return self.hub.box

    def _done(self) -> None:
        torch.cuda.synchronize()
        self.hub.bar.wait()

    def all_gather(self, src: torch.Tensor, out: torch.Tensor) -> None:
        box = self._post([src.reshape(-1)])
        n = src.numel()
        for k in (0, 1):
            out[k * n:(k + 1) * n].copy_(box[k][0])
        if self.snap is not None:
            torch.cuda.synchronize()
            self.snap(out[:2 * n])
        self._done()

    def exchange(self, sends, recvs, peer: int) -> None:
        box = self._post(sends)
        for dst, src in zip(recvs, box[peer]):
            dst.copy_(src)
        self._done()


def _run(W, R: int, pos: int, prefill: bool, split: bool, states, x0, snap0=None, keep: int | None = None) -> list[dict]:
    """layer_forward (or split_layers) on both ranks at once; each rank's x, new KDA state and conv window."""
    from tensorfold.families.glm5_next.cuda import forward
    from tensorfold.families.glm5_next.cuda.hcsplit import HcSplit, SplitSettings

    hub, res, errs = Hub(), [None, None], []

    def body(rank: int) -> None:
        try:
            w = W[rank]
            w.comm = Comm(hub, rank)
            st = forward.State(w, CAP, DECODE_ROWS)
            st.conv[0].copy_(states[rank][0])
            st.rec[0, 0].copy_(states[rank][1])
            st.set_pos(pos)
            b = forward.Buffers(w, PREFILL_ROWS if prefill else DECODE_ROWS, CAP, prefill=prefill)
            if split:      # TF_GLM_HC_SPLIT=1, TF_GLM_PREFILL_OVERLAP=2 (two pieces, p2p swaps on a second stream)
                b.split = HcSplit(w, b, SplitSettings(True, True, 2, 64, "p2p", True))
            b.x[:R].copy_(x0)
            if rank == 0 and snap0 is not None:
                w.comm.snap = lambda g: snap0(g, st, b)
            if split:
                forward.split_layers(w, st, b, R, None, pos)
            else:
                forward.layer_forward(w.layers[0], w, st, b, R, host_pos=pos)
            if not prefill:
                forward.commit(w, st, b, R, keep or R)
            torch.cuda.synchronize()
            res[rank] = {"x": b.x[:R].clone(), "rec": st.rec[st.cur[0], 0].clone(), "conv": st.conv[0].clone()}
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


def _weights(out: Path, L, c, pre: str = "lkw") -> dict:
    """Rank 0's tensors exactly as the engine holds them (EXL3 words in its strips layout); a KDA layer's mixer too."""
    k, m = L.kda, L.moe

    def lin(name: str, q) -> dict:
        assert q.layout == "strips" and q.bias is None, name
        return {"words": save(out, f"{pre}_{name}_words", q.words), "suh": save(out, f"{pre}_{name}_suh", q.suh),
                "svh": save(out, f"{pre}_{name}_svh", q.svh), "k": q.k, "n": q.n, "k2": q.k2, "codebook": q.codebook}

    def b16(name: str, q) -> dict:
        assert q.weight.dtype == torch.bfloat16, name
        return {"w": save(out, f"{pre}_{name}", q.weight), "n": q.n, "k": q.k}

    def plain(name: str, t: torch.Tensor, dtype: torch.dtype) -> str:
        assert t.dtype == dtype, (name, t.dtype)
        return save(out, f"{pre}_{name}", t)

    man = {}
    if L.kind == "kda":
        man["kda"] = {"qkv": lin("qkv", k.qkv), "o": lin("o", k.o), "fa": b16("fa", k.fa), "ga": b16("ga", k.ga),
                      "b": b16("b", k.b), "fb": b16("fb", k.fb), "gb": b16("gb", k.gb),
                      "conv": plain("conv", k.conv, torch.bfloat16), "a_log": plain("a_log", k.a_log, torch.float32),
                      "dt_bias": plain("dt_bias", k.dt_bias, torch.float32),
                      "norm": plain("norm", k.norm, torch.bfloat16), "heads": k.heads, "b_off": k.b_off,
                      "fa_off": k.fa_off, "ga_off": k.ga_off, "p_stride": k.out_n}
    for site, h in (("attn_hc", L.attn_hc), ("ffn_hc", L.ffn_hc)):
        man[site] = {"fn": plain(f"{site}_fn", h.fn, torch.float32), "base": plain(f"{site}_base", h.base, torch.float32),
                     "scale": plain(f"{site}_scale", h.scale, torch.float32)}
    man["in_norm"] = plain("in_norm", L.in_norm, torch.bfloat16)
    man["post_norm"] = plain("post_norm", L.post_norm, torch.bfloat16)
    if m is None:            # a dense layer: the MLP's gate, up (the rank's columns) and down (its rows)
        d = L.mlp
        assert d.gu is None, "a fused gate|up pack"
        man["mlp"] = {"gate": lin("mlp_gate", d.gate), "up": lin("mlp_up", d.up), "down": lin("mlp_down", d.down)}
        return man
    ex = m.experts
    E = ex.count
    assert len(ex.keep) == 3 * E and ex.gu_stride == 0
    experts = {"count": E, "dims": ex.dims, "width": ex.width, "cb": ex.cb}
    for j, proj in enumerate(("gate", "up", "down")):
        mats = ex.keep[j * E:(j + 1) * E]
        offs, at = [], 0
        for t in mats:
            offs.append(at)
            at += t.numel() * t.element_size()
        experts[proj] = save(out, f"{pre}_ex_{proj}", torch.cat([t.reshape(-1) for t in mats]))
        experts[f"{proj}_offs"] = offs
    for name in ("gate_k2", "up_k2", "down_k2"):
        experts[name] = [int(v) for v in getattr(ex, name).cpu()]
    for name in ("suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d"):
        experts[name] = plain(f"ex_{name}", getattr(ex, name), torch.float16)
    s = m.shared
    man["moe"] = {"router": plain("router", m.router, torch.bfloat16), "bias": plain("bias", m.bias, torch.float32),
                  "experts": experts, "shared": {"gate": lin("sh_gate", s.gate), "up": lin("sh_up", s.up),
                                                 "down": lin("sh_down", s.down)}}
    return man


def _case(out: Path, W, tag: str, R: int, pos: int, prefill: bool, pre: str = "lk") -> dict:
    """One window from a seeded state: rank 0's bits at every stage, rank 1's two partials, the final x and state."""
    from tensorfold.families.glm5_next.cuda import kda_chunked

    c, k = W[0].cfg, W[0].layers[0].kda
    g = torch.Generator().manual_seed(SEED + 1000 * R + int(prefill))
    x0 = torch.randn((R, c.streams * c.hidden), generator=g).to(torch.bfloat16).cuda()
    C = 3 * k.heads * 128
    states = []
    for _rank in (0, 1):      # each rank's own heads: its own conv window and recurrent state
        conv = torch.randn((3, C), generator=g).to(torch.bfloat16).cuda()
        rec = (torch.randn((k.heads, 128, 128), generator=g) * 0.3).cuda()
        states.append((conv, rec))
    snaps: list[dict] = []

    def snap(gath: torch.Tensor, st, b) -> None:
        site = {"gath": gath.clone(), "x": b.x[:R].clone(), "normed": b.normed[:R].clone(),
                "post": b.post[:R].clone(), "comb": b.comb[:R].clone()}
        if not snaps:        # the attention site: the KDA block's projections, read-out and saved rows
            site["p"] = (b.kproj[0, :R] if prefill else st.proj[0, :R]).clone()
            site["ka"], site["kg"] = b.ka[:R].clone(), b.kg[:R].clone()
            sc = b.kscratch if prefill else st.scratch[0]
            site["kout"] = sc.out[:R].clone()
            if not prefill:
                for key in ("k", "v", "g", "b"):
                    site[f"s{key}"] = getattr(sc, key)[:R].clone()
        elif W[0].layers[0].moe is None:   # a dense MLP: gate, up and the SwiGLU, under the engine's names
            for key, src in (("smg", "mg"), ("smu", "mu"), ("sact", "act")):
                site[key] = getattr(b, src)[:R].clone()
        else:                # the MoE site: route, per-slot y (shared expert in slot 8)
            for key in ("mlog", "pick", "wts", "sy", "smg", "smu", "sact"):
                site[key] = getattr(b, key)[:R].clone()
            site["ey"] = b.ey[:R].clone()
        snaps.append(site)

    res = _run(W, R, pos, prefill, False, states, x0, snap)
    assert len(snaps) == 2, len(snaps)
    e = {"rows": R, "pos": pos, "prefill": prefill, "x0": save(out, f"{pre}_{tag}_x0", x0),
         "conv0": save(out, f"{pre}_{tag}_conv0", states[0][0]), "rec0": save(out, f"{pre}_{tag}_rec0", states[0][1]),
         "x": save(out, f"{pre}_{tag}_x", res[0]["x"]), "rec": save(out, f"{pre}_{tag}_rec", res[0]["rec"]),
         "conv": save(out, f"{pre}_{tag}_conv", res[0]["conv"])}
    assert torch.equal(res[0]["x"], res[1]["x"]), "the two ranks' x differ"
    n = R * c.hidden
    for i, site in enumerate(snaps):
        gath = site.pop("gath")
        e[f"s{i}"] = {"part": save(out, f"{pre}_{tag}_s{i}_part", gath[:n]),
                      "peer": save(out, f"{pre}_{tag}_s{i}_peer", gath[n:2 * n])}
        for key, t in site.items():
            e[f"s{i}"][key] = save(out, f"{pre}_{tag}_s{i}_{key}", t)
    prep = kda_chunked.prep_reference(snaps[0]["p"], k.b_off, snaps[0]["ka"], states[0][0], k.conv, k.a_log,
                                      k.dt_bias, c.lower, R)
    e["prep"] = {key: save(out, f"{pre}_{tag}_prep_{key}", t) for key, t in zip(("q", "k", "v", "lg", "beta"), prep)}
    if prefill:              # the same chunk row-split between the ranks: every rank's own rows and state equal
        sres = _run(W, R, pos, prefill, True, states, x0)
        h = R // 2
        for rank, rows in ((0, slice(0, h)), (1, slice(h, R))):
            assert torch.equal(sres[rank]["x"][rows], res[rank]["x"][rows]), f"split x rank {rank}"
            assert torch.equal(sres[rank]["rec"], res[rank]["rec"]), f"split rec rank {rank}"
            assert torch.equal(sres[rank]["conv"], res[rank]["conv"]), f"split conv rank {rank}"
        e["split_equal"] = True
    elif R == max(DECODE):   # commit(R, keep < R): the engine's replay of the kept rows, the conv shifted by keep
        kres = _run(W, R, pos, prefill, False, states, x0, keep=KEEP)
        assert torch.equal(kres[0]["x"], res[0]["x"]), "keep changed x"
        e["keep"] = {"keep": KEEP, "rec": save(out, f"{pre}_{tag}_keep_rec", kres[0]["rec"]),
                     "conv": save(out, f"{pre}_{tag}_keep_conv", kres[0]["conv"])}
    print(f"  layer {tag}: {R} rows at {pos}, x {res[0]['x'].float().abs().max():.3g} max")
    return e


def write(out: Path, model_dir: Path, layer: int = LAYER, pre: str = "lk") -> dict:
    from tensorfold.families.glm5_next.cuda import kda

    W = [_load(model_dir, rank, layer) for rank in (0, 1)]
    c, L = W[0].cfg, W[0].layers[0]
    assert L.kind == "kda" and (L.moe is None or L.moe.shared is not None) and c.norm_topk
    man = {"layer": layer, "params": {"eps": c.eps, "hc_eps": c.hc_eps, "hc_iters": c.hc_iters, "lower": c.lower,
                                      "limit": c.limit, "routed_scale": c.routed_scale, "top_k": c.top_k,
                                      "experts": c.experts, "hidden": c.hidden, "streams": c.streams},
           "weights": _weights(out, L, c, f"{pre}w"), "cases": {}}
    chunked = kda.CHUNKED
    try:
        kda.CHUNKED = True       # the recipe's TF_GLM_KDA_CHUNKED=1 (read per call)
        for R in DECODE:
            man["cases"][f"d{R}"] = _case(out, W, f"d{R}", R, DECODE_POS, False, pre)
        man["cases"][f"p{PROMPT}"] = _case(out, W, f"p{PROMPT}", PROMPT, PROMPT_POS, True, pre)
    finally:
        kda.CHUNKED = chunked
    return man
