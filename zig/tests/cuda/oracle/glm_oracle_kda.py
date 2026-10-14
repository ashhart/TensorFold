# GLM-5.3-Flash's KDA oracle fixtures: the engine's rank-0 (world 2) chain, chunks and block, bit for bit.

from __future__ import annotations

import json
from pathlib import Path

import torch

import exl3_oracle_layer  # save, sha_of, load_tensor

SEED = 4477
CHAIN_ROWS = (1, 3, 8, 64, 128)      # 1/3/8 narrow (serial), 64/128 wide (three kernels, same bits)
CHUNK_ROWS = 2048                    # one prompt chunk: 64 sub-chunks
CHUNK_SPLITS = (32, 96, 1024, 2016)  # 32-aligned cuts, the last leaving a cut segment
BLOCK_ROWS = 300                     # the block's prompt chunk (the Cut path's whole run)
CUT = 64                             # the block's cut point (first segment 64 rows, then the cut segment)


def _attn_prefix(model_dir: Path, li: int) -> str:
    """The checkpoint's attention prefix (the serve view may nest under model.language_model)."""
    index = Path(model_dir) / "model.safetensors.index.json"
    wm = json.loads(index.read_text())["weight_map"]
    for name in wm:
            if f"layers.{li}.self_attn.qkv_proj.trellis" in name:
                return name[: name.index(f"layers.{li}.self_attn.qkv_proj") + len(f"layers.{li}")]
    raise KeyError("qkv_proj.trellis")


def _save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def _pick_kda_layer(model_dir: Path) -> int:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    for i, attn in enumerate(cfg["layer_types"]):
        if attn == "linear_attention":
            return i
    raise ValueError(f"no KDA layer in {model_dir}")


def _rand(g, *shape, dtype=torch.float32, device="cuda", scale=1.0):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype).to(device)


def _front(k, qmm, x, rows):
    # The block's projection rows (kda_block's front, rank 0): p = q|k|v | f_a | g_a | b and the pair's second mats.
    dev = x.device
    p = torch.zeros((rows, k.out_n), dtype=torch.bfloat16, device=dev)
    m = x.shape[0]
    o = torch.empty((128, k.qkv.n), dtype=torch.bfloat16, device=dev)
    for r in range(0, m, 128):
        n = min(m - r, 128)
        k.qkv(x[r:r + n], out=o[:n])
        p[r:r + n, :k.qkv.n].copy_(o[:n])
    fa = torch.empty((m, k.fa.n), dtype=torch.bfloat16, device=dev)
    ga = torch.empty((m, k.ga.n), dtype=torch.bfloat16, device=dev)
    bb = torch.empty((m, k.b.n), dtype=torch.bfloat16, device=dev)
    qmm.matmul(x, k.fa, out=fa)
    qmm.matmul(x, k.ga, out=ga)
    qmm.matmul(x, k.b, out=bb)
    p[:, k.fa_off:k.fa_off + k.fa.n].copy_(fa)
    p[:, k.ga_off:k.ga_off + k.ga.n].copy_(ga)
    p[:, k.b_off:].copy_(bb)
    a = qmm.matmul(fa, k.fb, xs=qmm.group_sums(fa))
    g = qmm.matmul(ga, k.gb, xs=qmm.group_sums(ga))
    return p, a, g


def _store_entry(out: Path, tag: str, p, a, g, conv0, state0, o, state_out, prep, scratch=None) -> dict:
    keys = {"p": p, "a": a, "g": g, "conv0": conv0, "state0": state0, "out": o, "state_out": state_out,
            "q": prep[0], "k": prep[1], "v": prep[2], "lg": prep[3], "beta": prep[4]}
    e = {key: f"{tag}_{key}.bin" for key in keys}
    e["sha256"] = {"out": exl3_oracle_layer.sha_of(o), "state_out": exl3_oracle_layer.sha_of(state_out)}
    if scratch is not None:
        e["scratch"] = {}
        for key in ("k", "v", "g", "b"):
            e["scratch"][key] = f"{tag}_s{key}.bin"
            _save(out, f"{tag}_s{key}", getattr(scratch, key))
    for key, t in keys.items():
        _save(out, f"{tag}_{key}", t)
    return e


def write(out: Path, model_dir: Path) -> dict:
    from tensorfold.families.glm5_next.cuda import kda, kda_chunked, qmm, weights

    dev = "cuda"
    li = _pick_kda_layer(model_dir)
    w = weights.load(model_dir, rank=0, device=dev)
    k = w.layers[li].kda
    c = w.cfg
    H, LL, DK = k.heads, c.lin_heads // w.world, kda.DK
    hidden, b_off, eps, lower = c.hidden, k.b_off, c.eps, c.lower
    C = 3 * H * DK
    assert (H, b_off) == (c.lin_heads // w.world, 3 * (c.lin_heads // w.world) * 128 + 256), (H, b_off)

    fam: dict = {"params": {"layer": li, "H": H, "LL": LL, "dk": DK, "hidden": hidden, "b_off": b_off,
                            "eps": eps, "lower": lower, "heads_full": c.lin_heads, "conv_taps": c.conv,
                            "world": w.world, "rank": w.rank, "qkv_n": k.qkv.n, "p_stride": k.out_n},
                 "chain": {}, "chunked": {}, "block": {}}

    def inputs(rows, salt):
        g = torch.Generator().manual_seed(SEED + salt)
        x = _rand(g, rows, hidden, dtype=torch.bfloat16)
        conv0 = _rand(g, 3, C, dtype=torch.bfloat16)
        state0 = _rand(g, H, DK, DK, scale=0.3)
        return x, conv0, state0

    # -- the decode chain: narrow (serial) and wide (three kernels), the engine's own bits -------------------------
    for rows in CHAIN_ROWS:
        x, conv0, state0 = inputs(rows, rows)
        p, a, g = _front(k, qmm, x, rows)
        state_out = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
        scr = kda.KDAScratch(rows, H, dev)
        o = kda.chain(p, b_off, a, g, conv0, k.conv, state0, k.a_log, k.dt_bias, k.norm, eps, lower, rows, scr,
                      state_out)
        prep = kda_chunked.prep_reference(p, b_off, a, conv0, k.conv, k.a_log, k.dt_bias, lower, rows)
        tag = f"chain_r{rows}"
        e = _store_entry(out, tag, p, a, g, conv0, state0, o, state_out, prep, scratch=scr if rows == 8 else None)
        _save(out, f"{tag}_x", x)
        e["x"] = f"{tag}_x.bin"
        if rows == 8:  # one row at a time with the conv window shifted in between: the engine's own state
            conv_r = conv0.clone()
            st_r = state0.clone()
            for i in range(rows):
                o_r = torch.empty((1, H * DK), dtype=torch.bfloat16, device=dev)
                so_r = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
                kda.chain(p[i:i + 1], b_off, a[i:i + 1], g[i:i + 1], conv_r, k.conv, st_r, k.a_log, k.dt_bias,
                          k.norm, eps, lower, 1, kda.KDAScratch(1, H, dev), so_r)
                st_r = so_r
                if i < rows - 1:
                    conv_r = torch.cat([conv_r[1:], p[i:i + 1, :C]])
            _save(out, f"{tag}_rowwise_state", st_r)
            e["rowwise_state"] = f"{tag}_rowwise_state.bin"
        if rows == 8:  # the engine's own replay of the saved rows must reach the same state
            st = torch.empty_like(state_out)
            kda.replay(state0, scr, rows, st)
            assert torch.equal(st, state_out), "engine replay != chain state"
        fam["chain"][str(rows)] = e

    # -- the prompt chunk in chunked (WY) form, plus its 32-aligned cuts -------------------------------------------
    rows = CHUNK_ROWS
    x, conv0, state0 = inputs(rows, 100000)
    p, a, g = _front(k, qmm, x, rows)
    o = torch.empty((rows, H * DK), dtype=torch.bfloat16, device=dev)
    state_out = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
    kda_chunked.chain(p, b_off, a, g, conv0, k.conv, state0, k.a_log, k.dt_bias, k.norm, eps, lower, rows, o,
                      state_out, pos=0)
    prep = kda_chunked.prep_reference(p, b_off, a, conv0, k.conv, k.a_log, k.dt_bias, lower, rows)
    tag = "chunk_r2048"
    e = _store_entry(out, tag, p, a, g, conv0, state0, o, state_out, prep)
    splits = {}
    for n in CHUNK_SPLITS:  # the engine's own cuts must reproduce the whole run's bits (chunk invariance)
        o1 = torch.empty((n, H * DK), dtype=torch.bfloat16, device=dev)
        st1 = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
        kda_chunked.chain(p[:n], b_off, a, g, conv0, k.conv, state0, k.a_log, k.dt_bias, k.norm, eps, lower, n,
                          o1, st1, pos=0)
        conv_mid = p[n - 3:n, :C].clone()
        o2 = torch.empty((rows - n, H * DK), dtype=torch.bfloat16, device=dev)
        st2 = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
        kda_chunked.chain(p[n:], b_off, a[n:], g[n:], conv_mid, k.conv, st1, k.a_log, k.dt_bias, k.norm, eps,
                          lower, rows - n, o2, st2, pos=n)
        assert torch.equal(torch.cat((o1, o2)), o), f"engine cut {n} != whole run"
        assert torch.equal(st2, state_out), f"engine cut {n} state"
        splits[str(n)] = {"st1": f"{tag}_cut{n}_st1.bin"}
        _save(out, f"{tag}_cut{n}_st1", st1)
    fam["chunked"] = {**e, "rows": rows, "splits": splits}

    # the block starts at the layer's normed rows (the layernorm/HC mixing is the glue family's gate): seeded bf16
    g = torch.Generator().manual_seed(SEED + 200000)
    normed = _rand(g, rows, hidden, dtype=torch.bfloat16)
    p, a, g = _front(k, qmm, normed, rows)
    o = torch.empty((rows, H * DK), dtype=torch.bfloat16, device=dev)
    state_out = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
    kda_chunked.chain(p, b_off, a, g, conv0, k.conv, state0, k.a_log, k.dt_bias, k.norm, eps, lower, rows, o,
                      state_out, pos=0)
    prep = kda_chunked.prep_reference(p, b_off, a, conv0, k.conv, k.a_log, k.dt_bias, lower, rows)
    for key, t in (("q", prep[0]), ("k", prep[1]), ("v", prep[2]), ("lg", prep[3]), ("beta", prep[4])):
        _save(out, f"block_{key}", t)
    part = torch.empty((rows, hidden), dtype=torch.float32, device=dev)
    from tensorfold.cuda.exl3.linear import Exl3Linear
    if isinstance(k.o, Exl3Linear):                # the engine's exl3_mm: the kernel takes 1..128 rows
        for r in range(0, rows, 128):
            n = min(rows - r, 128)
            k.o(o[r:r + n], out=part[r:r + n])
    else:
        qmm.matmul(o, k.o, xs=qmm.group_sums(o), out=part, f32=True)
    oc1 = torch.empty((CUT, H * DK), dtype=torch.bfloat16, device=dev)
    stc1 = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
    kda_chunked.chain(p[:CUT], b_off, a, g, conv0, k.conv, state0, k.a_log, k.dt_bias, k.norm, eps, lower, CUT,
                      oc1, stc1, pos=0)
    conv_cut = p[CUT - 3:CUT, :C].clone()
    oc2 = torch.empty((rows - CUT, H * DK), dtype=torch.bfloat16, device=dev)
    stc2 = torch.empty((H, DK, DK), dtype=torch.float32, device=dev)
    kda_chunked.chain(p[CUT:], b_off, a[CUT:], g[CUT:], conv_cut, k.conv, stc1, k.a_log, k.dt_bias, k.norm, eps,
                      lower, rows - CUT, oc2, stc2, pos=CUT)
    assert torch.equal(torch.cat((oc1, oc2)), o), "engine block cut != whole run"
    assert torch.equal(stc2, state_out), "engine block cut state"
    pfx = _attn_prefix(model_dir, li)
    trellis = exl3_oracle_layer.load_tensor(model_dir, f"{pfx}.self_attn.qkv_proj.trellis")
    svh = exl3_oracle_layer.load_tensor(model_dir, f"{pfx}.self_attn.qkv_proj.svh")
    from tensorfold.families.glm5_next.cuda.split import fused_rank_runs
    t_runs = fused_rank_runs(trellis.shape[1], 3, w.rank, w.world)
    s_runs = fused_rank_runs(svh.shape[0], 3, w.rank, w.world)
    tag = "block_r300"
    blk = {"rows": rows, "cut": CUT, "p": f"{tag}_p.bin", "a": f"{tag}_a.bin", "g": f"{tag}_g.bin",
           "conv0": f"{tag}_conv0.bin", "state0": f"{tag}_state0.bin", "out": f"{tag}_out.bin",
           "state_out": f"{tag}_state_out.bin", "part": f"{tag}_part.bin",
           "sha256": {"out": exl3_oracle_layer.sha_of(o), "part": exl3_oracle_layer.sha_of(part)},
           "qkv": {"prefix": f"{pfx}.self_attn.qkv_proj", "k": k.qkv.k, "n": k.qkv.n,
                   "t_runs": [[r.start, r.stop] for r in t_runs], "s_runs": [[r.start, r.stop] for r in s_runs],
                   "codebook": c.codebook},
           "o_proj": {"prefix": f"{pfx}.self_attn.o_proj",
                      "exl3": op_trellis(model_dir, li, pfx) is not None, "n": k.o.n, "k": k.o.k}}
    assert k.fa is not None and k.ga is not None and k.b is not None, "fused qkv with no low-rank pair"
    mms = {}
    for key, q in (("fa", k.fa), ("ga", k.ga), ("b", k.b), ("fb", k.fb), ("gb", k.gb)):
        _save(out, f"{tag}_{key}", q.weight)
        mms[key] = {"n": q.n, "k": q.k, "file": f"{tag}_{key}.bin"}
    blk["mms"] = mms
    for key in ("q", "k", "v", "lg", "beta"):
        blk[f"prep_{key}"] = f"block_{key}.bin"
    if not blk["o_proj"]["exl3"]:
        _save(out, f"{tag}_o", k.o.weight)
        blk["o_proj"]["w"] = f"{tag}_o.bin"
    for key, t in (("p", p), ("a", a), ("g", g), ("conv0", conv0), ("state0", state0), ("out", o),
                   ("state_out", state_out), ("part", part)):
        _save(out, f"{tag}_{key}", t)
    _save(out, f"{tag}_normed", normed)
    blk["normed"] = f"{tag}_normed.bin"
    fam["block"] = blk

    for key, t in (("a_log", k.a_log), ("dt_bias", k.dt_bias), ("norm", k.norm), ("conv", k.conv)):
        _save(out, f"kda_{key}", t)
    fam["tensors"] = {"a_log": "kda_a_log.bin", "dt_bias": "kda_dt_bias.bin", "norm": "kda_norm.bin",
                      "conv": "kda_conv.bin"}
    print(f"  kda: layer {li}, H={H} heads/rank, chain rows {CHAIN_ROWS}, chunk {rows}, block {BLOCK_ROWS}")
    return fam


def op_trellis(model_dir: Path, li: int, pfx: str):
    from safetensors import safe_open
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    name = f"{pfx}.self_attn.o_proj.trellis"
    rel = json.loads(index.read_text())["weight_map"].get(name) if index.exists() else None
    if rel is None:
        return None
    with safe_open(str(root / rel), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


