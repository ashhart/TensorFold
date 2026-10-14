# GLM-5.3-Flash's DSA sparse-path oracle fixtures: indexer update, scores, top-pool select; fp8 cache, bit-exact.

from __future__ import annotations

import os

os.environ.setdefault("TF_GLM_LATENT", "1")

import json
from pathlib import Path

import torch

import exl3_oracle_layer  # save, sha_of

SEED = 7753
# The windows: decode rows at the count boundaries (2,048/2,051/2,052) and long contexts, one prompt chunk.
CASES = ((2048, 8), (2052, 8), (3000, 8), (16000, 128), (40000, 8), (2016, 2048))


def _save(out: Path, name: str, t: torch.Tensor) -> None:
    (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def _pick_dsa_layer(model_dir: Path) -> int:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    for i, attn in enumerate(cfg["layer_types"]):
        if attn != "linear_attention":
            return i
    raise ValueError(f"no DSA layer in {model_dir}")


def _rand(g, *shape, dtype=torch.float32, device="cuda", scale=1.0):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype).to(device)


def _cap(rows: int) -> int:
    return max(4096, (rows + 2047) // 2048 * 2048)


def write(out: Path, model_dir: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    from tensorfold.families.glm5_next.cuda import glue, latent, qmm, sparse, weights

    dev = "cuda"
    li = _pick_dsa_layer(model_dir)
    w = weights.load(model_dir, rank=0, device=dev)
    a = w.layers[li].dsa
    ix, c = a.index, w.cfg
    H, D, HD = c.index_heads, c.index_dim, c.index_heads * c.index_dim
    assert (H, D) == (32, 128), (H, D)
    scale = D ** -0.5
    wscale = 1.0 / 5.656854249492381
    _save(out, "ln_w", ix.ln_w)
    _save(out, "ln_b", ix.ln_b)
    _save(out, "ape", ix.ape)

    fam: dict = {"params": {"layer": li, "H": H, "D": D, "HP": 32, "hidden": c.hidden, "pool": sparse.POOL,
                            "topk_pools": sparse.TOPK_POOLS,
                            "width": sparse.TOPK_POOLS * sparse.POOL + sparse.POOL - 1,
                            "chunk": latent.CHUNK, "lw": latent.L, "dense_limit": c.dense_limit,
                            "scale": scale, "wscale": wscale, "world": w.world, "rank": w.rank},
                 "ln_w": "ln_w.bin", "ln_b": "ln_b.bin", "ape": "ape.bin", "cases": {}}

    for n, (pos, R) in enumerate(CASES):
        tag = f"{'p' if R > 512 else 'd'}{pos}r{R}"
        g = torch.Generator().manual_seed(SEED + n)
        cap = _cap(pos + R)
        normed = _rand(g, R, c.hidden, dtype=torch.bfloat16)
        ikr = torch.empty((R, D + H), dtype=torch.bfloat16, device=dev)
        qmm.matmul(normed, ix.kw, out=ikr)
        igr = torch.empty((R, D), dtype=torch.float32, device=dev)
        glue.router(normed, ix.gate, igr)
        torch.cuda.synchronize() # qmm.matmul runs on a side stream: index_update must see the finished k_raw
        qi = _rand(g, R, HD, dtype=torch.bfloat16)
        # the caches: rows before the window hold earlier windows' values (seeded), so partial pools re-pool them
        ik = torch.zeros((cap, D), dtype=torch.bfloat16, device=dev)
        ig = torch.zeros((cap, D), dtype=torch.bfloat16, device=dev)
        if pos:
            g2 = torch.Generator().manual_seed(SEED + 100 + n)
            ik[:pos] = _rand(g2, pos, D, dtype=torch.bfloat16)
            ig[:pos] = _rand(g2, pos, D, dtype=torch.bfloat16)
        pk = torch.zeros((cap // 4 + 2, D), dtype=torch.bfloat16, device=dev)
        if pos >= 4:
            pk[:pos // 4] = _rand(torch.Generator().manual_seed(SEED + 200 + n), pos // 4, D, dtype=torch.bfloat16)
        pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
        sparse.index_update(ikr[:, :D], igr, ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
        # the kernels must produce layer-normalised keys (loose guard: a racy or stale run drifts ~30x)
        x = ikr[:, :D].float()
        t = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-6)
        y = (t * ix.ln_w.float() + ix.ln_b.float()).to(torch.bfloat16)
        if (y - ik[pos:pos + R]).abs().max().item() > 0.1:
            raise RuntimeError("index_update window rows are not LN(ikr): racy or stale kernel run")
        e: dict = {"pos": pos, "R": R, "cap": cap, "np_max": pk.shape[0] - 2, "blocks": []}
        rows = pos + R
        for name, t in (("ikr", ikr), ("igr", igr), ("wts", ikr[:, D:]), ("qi", qi),
                        ("ik_pre", ik[:rows].clone()), ("ig_pre", ig[:rows].clone()), ("pk_pre", pk.clone()),
                        ("ik", ik[:rows].clone()), ("ig", ig[:rows].clone()), ("pk", pk.clone())):
            e[name] = f"{tag}_{name}.bin"
            _save(out, f"{tag}_{name}", t)

        # -- pool scores: the engine's own block loop (SELECT_ROWS rows at a time, one np_max for the window) -------
        B = min(R, sparse.SELECT_ROWS)
        np_max = pk.shape[0] - 2
        scores = torch.empty((B, np_max), dtype=torch.float32, device=dev)
        at = pos_dev
        for b0 in range(0, R, B):
            n = min(B, R - b0)
            if b0:
                at = pos_dev + b0
            sparse._scores[(n, (np_max + 64 * sparse.SCORE_LOOP - 1) // (64 * sparse.SCORE_LOOP))](
                qi[b0:b0 + n], ikr[b0:b0 + n, D:], D + H, pk, scores, at, n, np_max, scale, wscale,
                H=H, HP=32, D=D, BP=64, L=sparse.SCORE_LOOP, num_warps=4)
            e["blocks"].append(f"{tag}_sc{b0}.bin")
            _save(out, f"{tag}_sc{b0}", scores[:n])

        # -- selection on those scores: tokens and counts, the kernel path (np_max >= 512) --------------------------
        tokens, counts = sparse.select_tokens(qi, ikr[:, D:], pk, pos, R, np_max, pos_dev)
        e["tokens"], e["counts"] = f"{tag}_tokens.bin", f"{tag}_counts.bin"
        _save(out, f"{tag}_tokens", tokens)
        _save(out, f"{tag}_counts", counts)
        e["sha256"] = {"tokens": exl3_oracle_layer.sha_of(tokens), "counts": exl3_oracle_layer.sha_of(counts)}

        # -- sparse latent attention on the fp8 cache over the window's own selection -------------------------------
        lat = _rand(g, cap, latent.L, dtype=torch.bfloat16, scale=0.5)
        cache = torch.empty((cap, latent.L), dtype=torch.float8_e4m3fn, device=dev)
        scales = torch.empty((cap,), dtype=torch.float32, device=dev)
        for w0 in range(0, cap, 2048):
            pw = torch.tensor([w0], dtype=torch.int32, device=dev)
            latent.latent_write(lat[w0:w0 + 2048], cache, pw, scales)
        qa = _rand(g, R, a.heads, latent.L, dtype=torch.bfloat16)
        out_in = _rand(g, R, a.heads, latent.L, dtype=torch.bfloat16, scale=0.3)
        out_out = out_in.clone()
        latent.sparse_attention(qa, cache, tokens, counts, out_out, scale, scales)
        e["heads"] = a.heads
        for name, t in (("qa", qa), ("cache", cache), ("scales", scales), ("out_in", out_in), ("out", out_out)):
            e[name] = f"{tag}_{name}.bin"
            _save(out, f"{tag}_{name}", t)
        e["sha256"]["out"] = exl3_oracle_layer.sha_of(out_out)
        fam["cases"][tag] = e
    return fam
