# GLM-5.3-Flash latent-cache oracle: fp8/bf16 cache write, kv_b absorb/expand, dense 512-key chunks + merge.

from __future__ import annotations

import os

os.environ.setdefault("TF_GLM_LATENT", "1")

import json
from pathlib import Path

import torch

import exl3_oracle_layer  # save, sha_of

SEED = 9311
CAP = 4096
# dense (pos, R): contexts 1, 37, 511, 512, 2051; mid-chunk starts; the 512-row prompt block over 4K.
DENSE = ((0, 1), (29, 8), (1000, 8), (2040, 8), (504, 8), (2043, 8), (0, 512), (1536, 512), (2048, 512))
ABSORB_ROWS = (1, 8, 64, 512)
WRITE_ROWS = ((0, 1), (2040, 8), (2048, 512))


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


def _seeded_cache(g, cap: int, kind, lw: int):
    """A cache whose rows hold seeded values; the window's rows are garbage the kernel must overwrite."""
    cache = (torch.randn(cap, lw, generator=g, dtype=torch.float32) * 0.5).to(kind).to("cuda")
    scales = torch.rand(cap, generator=g, dtype=torch.float32).to("cuda") + 0.01
    return cache, scales


def write(out: Path, model_dir: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    from tensorfold.families.glm5_next.cuda import latent, weights

    dev = "cuda"
    li = _pick_dsa_layer(model_dir)
    w = weights.load(model_dir, rank=0, device=dev)
    a = w.layers[li].dsa
    ab = a.absorb
    H, D, LW, V = ab.heads, ab.qk_dim, ab.lw, ab.v_dim
    scale = D ** -0.5
    _save(out, "wk", ab.wk)
    _save(out, "wv", ab.wv)
    fam: dict = {"params": {"layer": li, "H": H, "D": D, "LW": LW, "V": V, "heads": a.heads,
                            "scale": scale, "world": w.world, "rank": w.rank},
                 "write": {}, "absorb": {}, "expand": {}, "dense": {}}

    # -- the cache write: fp8 codes with a per-token amax / 448 scale, and a plain bf16 copy ---------------------
    for n, (pos, R) in enumerate(WRITE_ROWS):
        g = torch.Generator().manual_seed(SEED + n)
        lat = _rand(g, R, LW, dtype=torch.bfloat16, scale=0.7)
        for kind in ("fp8", "bf16"):
            fp8 = kind == "fp8"
            cache8, scales = _seeded_cache(torch.Generator().manual_seed(SEED + 40 + n), CAP, torch.bfloat16, LW)
            if fp8:
                cache8 = cache8.to(torch.float8_e4m3fn)
            cache_pre, scales_pre = cache8.clone(), scales.clone()
            pd = torch.tensor([pos], dtype=torch.int32, device=dev)
            latent.latent_write(lat, cache8, pd, scales if fp8 else None)
            torch.cuda.synchronize()
            tag = f"w{pos}r{R}_{kind}"
            e = {"pos": pos, "R": R, "fp8": fp8, "lat": f"{tag}_lat.bin", "cache_pre": f"{tag}_cpre.bin",
                 "cache": f"{tag}_cache.bin"}
            if fp8:
                e["scales_pre"] = f"{tag}_spre.bin"
                e["scales"] = f"{tag}_scales.bin"
            for name, t in (("lat", lat), ("cache_pre", cache_pre), ("cache", cache8),
                            ("scales_pre", scales_pre), ("scales", scales)):
                if name in e:
                    _save(out, e[name][:-4], t)
            fam["write"][tag] = e

    # -- absorb and expand on the layer's kv_b (bf16 AbsorbW: the recipe's TF_GLM_KVB default) -------------------
    for n, R in enumerate(ABSORB_ROWS):
        g = torch.Generator().manual_seed(SEED + 300 + n)
        q = _rand(g, R, H, D, dtype=torch.bfloat16)
        qa = torch.empty(R, H, LW, dtype=torch.bfloat16, device=dev)
        latent.absorb_q(q, ab, qa)
        ol = _rand(g, R, H, LW, dtype=torch.bfloat16, scale=0.4)
        ov = torch.empty(R, H, V, dtype=torch.bfloat16, device=dev)
        latent.expand_v(ol, ab, ov)
        torch.cuda.synchronize()
        e = {"R": R, "q": f"a{R}_q.bin", "qa": f"a{R}_qa.bin", "ol": f"a{R}_ol.bin", "out": f"a{R}_out.bin"}
        for name, t in (("q", q), ("qa", qa), ("ol", ol), ("out", ov)):
            _save(out, f"a{R}_{name}", t)
        fam["absorb"][str(R)] = e
        fam["expand"][str(R)] = e

    # -- dense chunks and the merge: every row over its own causal keys in 512-key chunks -------------------------
    for n, (pos, R) in enumerate(DENSE):
        for kind in ("fp8", "bf16"):
            fp8 = kind == "fp8"
            tag = f"d{pos}r{R}_{kind}"
            g = torch.Generator().manual_seed(SEED + 600 + n)
            cache, scales = _seeded_cache(g, CAP, torch.bfloat16, LW)
            if fp8:
                cache = cache.to(torch.float8_e4m3fn)
            qa = _rand(g, R, H, LW, dtype=torch.bfloat16)
            nch = -(-(pos + R) // latent.CHUNK)
            s = latent.LatentScratch(R, H, nch, dev, lw=LW)
            outb = torch.empty(R, H, LW, dtype=torch.bfloat16, device=dev)
            pd = torch.tensor([pos], dtype=torch.int32, device=dev)
            latent.attention(qa, cache, pd, s, scale=scale, nch=nch, out=outb,
                             scales=scales if fp8 else None)
            torch.cuda.synchronize()
            e = {"pos": pos, "R": R, "nch": nch, "fp8": fp8, "heads": H,
                 "qa": f"{tag}_qa.bin", "cache_pre": f"{tag}_cpre.bin", "cache": f"{tag}_cache.bin",
                 "out": f"{tag}_out.bin"}
            if fp8:
                e["scales"] = f"{tag}_scales.bin"
                e["scales_pre"] = f"{tag}_spre.bin"
            for name, t in (("qa", qa), ("cache_pre", cache.clone()), ("cache", cache),
                            ("scales_pre", scales.clone()), ("scales", scales)):
                if name in e:
                    _save(out, e[name][:-4], t)
            _save(out, f"{tag}_out", outb)
            e["sha256"] = {"out": exl3_oracle_layer.sha_of(outb)}
            fam["dense"][tag] = e
    return fam
