"""Dev-time oracle: replay fixtures for `tf-dsv41-test all` from the reference CUDA extensions on seeded inputs."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts as x3
from tensorfold.families.deepseek_v41.cuda import expert_loads, expert_prefill
from tensorfold.families.deepseek_v41.cuda import x3gm as gm

from oracle_glue import DEV, save, plan_case, topk_case, pointwise_case, glue_case

CB = 2  # mul1


def trellis(g: torch.Generator, k: int, n: int, k2: int) -> torch.Tensor:
    """A random stored trellis int16 [K/16, N/16, 16 K2 / 2 ... ] (every word value is a valid mul1 state)."""
    return torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), generator=g, dtype=torch.int32).to(torch.int16)


def scales(g: torch.Generator, e: int, n: int, mag: float) -> torch.Tensor:
    sign = torch.randint(0, 2, (e, n), generator=g) * 2 - 1
    return (sign * (0.5 + torch.rand(e, n, generator=g)) * mag).to(torch.float16)


class Layer:
    """E experts' gate / up / down trellises back to back in one int16 buffer (ragged: a width per expert)."""

    def __init__(self, g, E, D, I, k2g, k2d, shx=False):
        self.E, self.D, self.I = E, D, I
        parts, offs = [], {"g": [], "u": [], "d": []}
        at = 0
        for kind, K, N, ws in (("g", D, I, k2g), ("u", D, I, k2g), ("d", I, D, k2d)):
            for e in range(E):
                t = trellis(g, K, N, ws[e])
                offs[kind].append(at * 2)
                parts.append(t.reshape(-1))
                at += t.numel()
        self.words = torch.cat(parts).to(DEV)
        self.off = {k: torch.tensor(v, dtype=torch.int64) for k, v in offs.items()}
        self.k2 = {"g": list(k2g), "u": list(k2g), "d": list(k2d)}
        base = self.words.data_ptr()
        self.tp = {k: (v + base).to(DEV) for k, v in self.off.items()}
        self.k2t = {k: torch.tensor(v, dtype=torch.int32, device=DEV) for k, v in self.k2.items()}
        self.suh_g = scales(g, E, D, 1.0).to(DEV)
        self.suh_u = self.suh_g.clone() if shx else scales(g, E, D, 1.0).to(DEV)
        self.svh_g = scales(g, E, I, 0.02).to(DEV)
        self.svh_u = scales(g, E, I, 0.02).to(DEV)
        self.suh_d = scales(g, E, I, 1.0).to(DEV)
        self.svh_d = scales(g, E, D, 0.02).to(DEV)

    def arrays(self) -> dict:
        return {"words": self.words, "off_g": self.off["g"], "off_u": self.off["u"], "off_d": self.off["d"],
                "k2_g": self.k2t["g"], "k2_u": self.k2t["u"], "k2_d": self.k2t["d"], "suh_g": self.suh_g,
                "suh_u": self.suh_u, "svh_g": self.svh_g, "svh_u": self.svh_u, "suh_d": self.suh_d,
                "svh_d": self.svh_d}


def experts_case(out: Path, g, *, R, slots, E, D, I, k2g, k2d, input_dtype, limit, skip_every=0, prefill=False):
    """The decode chain (exl3.experts.routed's launches) with every x3ld / x3pf setting's Z beside upstream's."""
    L = Layer(g, E, D, I, k2g, k2d)
    up, ld, pf = x3._ext(), expert_loads._ext(), expert_prefill._ext()
    P = R * slots
    maxu = min(P, E)
    x = torch.randn(R, D, generator=g).to(input_dtype).to(DEV)
    pick = torch.stack([torch.randperm(E - 1, generator=g)[:slots - 1] for _ in range(R)])
    pick = torch.cat([pick, torch.full((R, 1), E - 1)], 1).to(torch.int32)        # the last slot: the shared expert
    if skip_every:
        pick[::skip_every, 0] = E                                                    # a slot past the table: skipped
    pick = pick.to(DEV)
    wts = torch.rand(R, slots, generator=g).to(DEV)
    ids = torch.zeros(maxu, dtype=torch.int32, device=DEV)
    count = torch.zeros(1, dtype=torch.int32, device=DEV)
    members = torch.full((maxu, R), -1, dtype=torch.int32, device=DEV)
    up.group(pick, ids, count, members, R, slots, E)
    xg = torch.zeros(P, D, dtype=torch.float16, device=DEV)
    xu = torch.zeros_like(xg)
    up.rot_in(x, x.stride(0), pick, L.suh_g, L.suh_u, xg, xu, R, D, slots, E)
    ntg, wg, skg, pfg = x3.default_config(D, I, True)
    ntd, wd, skd, pfd = x3.default_config(I, D, False)
    z = torch.zeros(max(2 * skg * I, skd * D) * P, dtype=torch.float32, device=DEV)
    k2gu = L.k2["g"] + L.k2["u"]
    lo_gu, hi_gu, lo_d, hi_d = min(k2gu), max(k2gu), min(L.k2["d"]), max(L.k2["d"])
    arrays = {"x": x, "pick": pick, "wts": wts, **L.arrays(), "uids": ids, "ucount": count, "members": members,
              "xg": xg, "xu": xu}

    def zrun(tag, mats, x0, x1, tp, k2t, K, N, sk, lo, hi, nt, w, pfn):
        z.zero_()
        up.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk, slots, CB, nt, w,
                   pfn, lo, hi)
        arrays[f"z_{tag}"] = z.clone()
        for nt_, pd in expert_loads.CFGS:
            if expert_loads.k2_range(lo, hi) and expert_loads.fits(K, N, sk, w, (nt_, pd)):
                z.zero_()
                ld.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk, slots,
                           CB, nt_, pd, 0, *expert_loads.k2_range(lo, hi), False)
                arrays[f"z_{tag}_ld_{nt_}_{pd}"] = z.clone()
        if prefill:
            for nt_, mtl in ((4, 4), (8, 2)):
                if expert_prefill.k2_range(lo, hi) and expert_prefill.fits(K, N, sk, w, (nt_, mtl)):
                    z.zero_()
                    pf.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk,
                               slots, CB, nt_, mtl, *expert_prefill.k2_range(lo, hi))
                    arrays[f"z_{tag}_pf_{nt_}_{mtl}"] = z.clone()
        z.zero_()
        up.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk, slots, CB, nt, w,
                   pfn, lo, hi)

    zrun("gu", 2, xg, xu, (L.tp["g"], L.tp["u"]), (L.k2t["g"], L.k2t["u"]), D, I, skg, lo_gu, hi_gu, ntg, wg, pfg)
    xd = torch.zeros(P, I, dtype=torch.float16, device=DEV)
    up.gateup_epilogue(z, pick, L.svh_g, L.svh_u, L.suh_d, xd, R, P, I, skg, slots, E, float(limit), x3.ACT_F32)
    zrun("d", 1, xd, xd, (L.tp["d"], L.tp["d"]), (L.k2t["d"], L.k2t["d"]), I, D, skd, lo_d, hi_d, ntd, wd, pfd)
    y = torch.zeros(P, D, dtype=torch.float32, device=DEV)
    up.down_epilogue(z, pick, L.svh_d, y, R, P, D, skd, slots, E)
    outc = torch.zeros(R, D, dtype=torch.float32, device=DEV)
    up.combine(y, wts, outc, R, D, slots)
    y2 = torch.zeros_like(y)
    outd = torch.zeros_like(outc)
    up.down_combine(z, pick, L.svh_d, y2, wts, outd, R, P, D, skd, slots, E)
    torch.cuda.synchronize()
    assert torch.equal(outc, outd), "down_combine != down_epilogue + combine"
    for k, v in list(arrays.items()):
        if k.startswith("z_") and "_ld_" in k or "_pf_" in k:
            base = "z_gu" if k.startswith("z_gu") else "z_d"
            assert torch.equal(v, arrays[base]), f"{k} != upstream"           # the reference's own claim
    arrays.update(xd=xd, y=y, out=outc)
    save(out, {"kind": "experts", "R": R, "slots": slots, "E": E, "D": D, "I": I, "input": "bf16"
               if input_dtype == torch.bfloat16 else "f16", "limit": limit if math.isfinite(limit) else 0.0,
               "limit_inf": int(not math.isfinite(limit)),
               "act_mode": x3.ACT_F32, "lo_gu": lo_gu, "hi_gu": hi_gu, "lo_d": lo_d, "hi_d": hi_d, "sk_gu": skg,
               "sk_d": skd, "nt_gu": ntg, "w_gu": wg, "pf_gu": pfg, "nt_d": ntd, "w_d": wd, "pf_d": pfd}, arrays)
    return {"ld_settings": sum("_ld_" in k for k in arrays), "pf_settings": sum("_pf_" in k for k in arrays)}


def x3gm_case(out: Path, g, *, R, slots, E, D, I, k2g, k2d, shx, input_dtype, limit, runs, dequant=None):
    """x3gm.run / _run_ragged's launches for each (gu cfg, dn cfg, ticket) in ``runs`` ("tuned": the table's)."""
    ragged = len(set(k2g)) > 1 or len(set(k2d)) > 1
    L = Layer(g, E, D, I, k2g, k2d, shx=shx)
    ext, up = gm._ext(), x3._ext()
    P = R * slots
    mats_x = 1 if shx else 2
    x = torch.randn(R, D, generator=g).to(input_dtype).to(DEV)
    pick = torch.stack([torch.randperm(E, generator=g)[:slots] for _ in range(R)]).to(torch.int32).to(DEV)
    wts = torch.rand(R, slots, generator=g).to(DEV)
    xs = torch.zeros(mats_x, P, D, dtype=torch.float16, device=DEV)
    xg, xu = xs[0], xs[mats_x - 1]
    ext.rot(x, x.stride(0), pick, L.suh_g, L.suh_u, xg, xu, D, slots, P, mats_x)
    arrays = {"x": x, "pick": pick, "wts": wts, **L.arrays(), "xs": xs}
    params = {"kind": "x3gm", "R": R, "slots": slots, "E": E, "D": D, "I": I, "shx": int(shx), "ragged": int(ragged),
              "input": "bf16" if input_dtype == torch.bfloat16 else "f16", "limit": limit}
    if ragged:
        tg, tu, td = L.tp["g"], L.tp["u"], L.tp["d"]
        k2w = torch.tensor(L.k2["g"] + [0], dtype=torch.int32, device=DEV)
        k2dw = torch.tensor(L.k2["d"] + [0], dtype=torch.int32, device=DEV)
    else:
        n_gate = E * D * I // 256 * 16 * k2g[0] // 2      # int16 elements of the gate stack
        tg = L.words[:n_gate].view(torch.int32).view(E, D // 16, I // 16, 4 * k2g[0])
        tu = L.words[n_gate:2 * n_gate].view(torch.int32).view(E, D // 16, I // 16, 4 * k2g[0])
        td = L.words[2 * n_gate:].view(torch.int32).view(E, I // 16, D // 16, 4 * k2d[0])
        params.update(base_u=n_gate * 2, base_d=4 * n_gate)
    ticket = torch.zeros(1, dtype=torch.int32, device=DEV)
    base_cfg = gm.config({})
    for r, run in enumerate(runs):
        xd = torch.zeros(P, I, dtype=torch.float16, device=DEV)
        y = torch.zeros(P, D, dtype=torch.float32, device=DEV)
        tk = True
        cg_all = cd_all = None
        if run != "tuned":
            cg_all, cd_all, tk = run
        for kind, widths in (("gu", sorted(set(k2g))), ("dn", sorted(set(k2d)))):
            for k2 in widths:
                if cg_all is None:
                    cg, cd = gm.tuned(base_cfg, R, slots, E, shx, k2gu=k2, k2d=k2)
                else:
                    cg, cd = cg_all, cd_all
                c = cg if kind == "gu" else cd
                if not gm.fits(kind, c, k2, 1 if shx else 2):
                    raise SystemExit(f"run {run}: {kind} cfg {c} does not fit K2 {k2}")
                pk = pick
                if ragged:
                    w = (k2dw if kind == "dn" else k2w)[pick.long()]
                    pk = torch.where(w == k2, pick, torch.full_like(pick, E))
                order, pe, poff, pcnt, npass = gm.plan(pk, E, gm.bm_of(kind, c))
                arrays[f"run{r}_{kind}_{k2}_pk"] = pk.contiguous()          # x3gm_plan.cu's input
                for part, t in zip(("order", "pe", "poff", "pcnt", "npass"), (order, pe, poff, pcnt, npass)):
                    arrays[f"run{r}_{kind}_{k2}_{part}"] = t
                ticket.zero_()
                if kind == "gu":
                    ext.gateup(xg, xu, tg, tu, order, pe, poff, pcnt, npass, ticket, L.svh_g, L.svh_u, L.suh_d, xd,
                               D, I, k2, shx, c, tk, float(limit), ragged)
                else:
                    ext.down(xd, td, order, pe, poff, pcnt, npass, ticket, L.svh_d, y, I, D, k2, c, tk, ragged)
                params[f"run{r}_{kind}_{k2}"] = c
        params[f"run{r}_ticket"] = int(tk)
        o = torch.zeros(R, D, dtype=torch.float32, device=DEV)
        up.combine(y, wts, o, R, D, slots)
        arrays.update({f"run{r}_xd": xd, f"run{r}_y": y, f"run{r}_out": o})
    params["runs"] = len(runs)
    if hasattr(ext, "gateup2"):                        # x3gm v2 (gm2_kernel): _run_v2's "one" mode, both tickets
        k2g_t = torch.tensor(L.k2["g"] + [k2g[0] if not ragged else 0], dtype=torch.int32, device=DEV)
        k2d_t = torch.tensor(L.k2["d"] + [k2d[0] if not ragged else 0], dtype=torch.int32, device=DEV)
        cg, cd = gm.tuned2({"gu": None, "dn": None}, shx)
        order, pe, poff, pcnt, npass = gm.plan(pick, E, gm.bm_of("gu", cg))
        arrays.update(v2_k2g=k2g_t, v2_k2d=k2d_t, v2_order=order, v2_pe=pe, v2_poff=poff, v2_pcnt=pcnt, v2_npass=npass)
        for tk in (True, False):
            xd = torch.zeros(P, I, dtype=torch.float16, device=DEV)
            y = torch.zeros(P, D, dtype=torch.float32, device=DEV)
            ticket.zero_()
            ext.gateup2(xg, xu, tg, tu, k2g_t, order, pe, poff, pcnt, npass, ticket, L.svh_g, L.svh_u, L.suh_d, xd, D,
                        I, shx, cg, tk, float(limit), ragged)
            ticket.zero_()
            ext.down2(xd, td, k2d_t, order, pe, poff, pcnt, npass, ticket, L.svh_d, y, I, D, cd, tk, ragged)
            o = torch.zeros(R, D, dtype=torch.float32, device=DEV)
            up.combine(y, wts, o, R, D, slots)
            torch.cuda.synchronize()
            if tk:                                     # v2 == gm_kernel, bit for bit (dsq-x3gm-1's claim, rechecked)
                assert torch.equal(xd, arrays["run0_xd"]) and torch.equal(y, arrays["run0_y"]), "gm2 != gm"
            arrays.update({f"v2_t{int(tk)}_xd": xd, f"v2_t{int(tk)}_y": y, f"v2_t{int(tk)}_out": o})
        params.update(v2=1, v2_gu=cg, v2_dn=cd)
    if dequant:
        K, N, k2 = dequant
        t = trellis(g, K, N, k2).to(DEV)
        a = torch.empty(K, N, dtype=torch.float16, device=DEV)
        ext.dequant(t.view(torch.int32), a, k2)
        b = torch.empty_like(a)
        up.dequant(t, b, k2, CB)
        arrays.update(dequant_in=t, dequant_gm=a, dequant_up=b)
        params.update(dequant_k=K, dequant_n=N, dequant_k2=k2)
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"runs": len(runs), "ragged": ragged}



def want(name: str) -> bool:
    """TF_DSV41_ORACLE_ONLY: a regex of the case names to write (unset: every case)."""
    pat = os.environ.get("TF_DSV41_ORACLE_ONLY")
    return pat is None or re.match(pat, name) is not None


def main() -> None:
    out = Path(sys.argv[1]) / "fixtures"
    g = torch.Generator().manual_seed(4141)
    # Independent generators leave every existing fixture's random sequence unchanged.
    report = {}
    D, I, E = 1024, 512, 13
    # decode / verify (x3ld): routed 2-3.5 bits + 6-bit shared (E1 (2, 12)), 5-bit shared (2, 10), DSpark (8, 8)
    mixed_g = [4, 5, 6, 7, 4, 5, 6, 7, 4, 6, 5, 7, 12]
    mixed_d = [5, 4, 7, 6, 5, 4, 6, 7, 5, 4, 6, 7, 12]
    # Wide decode glue: runs the actual Zig groupRotIn ABI at its served input width.
    if want("experts_group_rot_r24"):
        report["experts_group_rot_r24"] = experts_case(
            out / "experts_group_rot_r24", torch.Generator().manual_seed(4142), R=24, slots=7, E=E, D=5120, I=I,
            k2g=mixed_g, k2d=mixed_d, input_dtype=torch.bfloat16, limit=10.0, skip_every=2)
    for R in (1, 3, 16):
        report[f"experts_e1_r{R}"] = experts_case(out / f"experts_e1_r{R}", g, R=R, slots=7, E=E, D=D, I=I,
                                                  k2g=mixed_g, k2d=mixed_d, input_dtype=torch.bfloat16, limit=10.0,
                                                  skip_every=2 if R > 1 else 0)
    if want("experts_k210_r8"): report["experts_k210_r8"] = experts_case(out / "experts_k210_r8", g, R=8, slots=7, E=E, D=D, I=I,
                                             k2g=[4] * 12 + [10], k2d=[6] * 12 + [8], input_dtype=torch.float16,
                                             limit=math.inf)
    if want("experts_dspark_r4"): report["experts_dspark_r4"] = experts_case(out / "experts_dspark_r4", g, R=4, slots=7, E=E, D=D, I=I,
                                               k2g=[8] * E, k2d=[8] * E, input_dtype=torch.bfloat16, limit=10.0)
    # prefill windows (x3pf beside x3ld / upstream)
    if want("experts_pf_r64"): report["experts_pf_r64"] = experts_case(out / "experts_pf_r64", g, R=64, slots=7, E=E, D=D, I=I, k2g=mixed_g,
                                            k2d=mixed_d, input_dtype=torch.bfloat16, limit=10.0, prefill=True)
    # x3gm: uniform (one launch a projection) and ragged (one a width), every configuration that fits
    E2 = 16
    cfgs = ["tuned", (0, 0, True), (1, 1, False), (2, 2, True), (3, 3, True), (4, 4, True), (5, 0, False)]
    if want("x3gm_k6_shx"): report["x3gm_k6_shx"] = x3gm_case(out / "x3gm_k6_shx", g, R=128, slots=6, E=E2, D=D, I=I, k2g=[6] * E2,
                                      k2d=[6] * E2, shx=True, input_dtype=torch.bfloat16, limit=10.0, runs=cfgs,
                                      dequant=(256, 256, 6))
    if want("x3gm_k4_two"): report["x3gm_k4_two"] = x3gm_case(out / "x3gm_k4_two", g, R=128, slots=6, E=E2, D=D, I=I, k2g=[4] * E2,
                                      k2d=[4] * E2, shx=False, input_dtype=torch.bfloat16, limit=10.0,
                                      runs=["tuned", (0, 0, True), (1, 1, True), (2, 2, False), (4, 4, True),
                                            (5, 3, True)], dequant=(128, 256, 4))
    widths_g = [3, 4, 5, 6, 7, 8, 10, 4, 3, 5, 6, 7, 8, 10, 4, 6]
    widths_d = [10, 8, 7, 6, 5, 4, 3, 6, 4, 4, 3, 5, 7, 8, 10, 6]
    if want("x3gm_ragged_shx"): report["x3gm_ragged_shx"] = x3gm_case(out / "x3gm_ragged_shx", g, R=256, slots=6, E=E2, D=D, I=I, k2g=widths_g,
                                          k2d=widths_d, shx=True, input_dtype=torch.bfloat16, limit=10.0,
                                          runs=["tuned", (0, 0, True), (2, 2, True), (5, 1, False)],
                                          dequant=(256, 128, 3))
    if want("x3gm_ragged_two"): report["x3gm_ragged_two"] = x3gm_case(out / "x3gm_ragged_two", g, R=200, slots=6, E=E2, D=D, I=I,
                                          k2g=widths_g, k2d=widths_d, shx=False, input_dtype=torch.float16,
                                          limit=10.0, runs=["tuned", (1, 0, True), (2, 4, True)],
                                          dequant=(128, 128, 10))
    # prefill chunks (2,048 / 1,024 / 512 rows, a short tail): the tuned v1 run, 2,048 rows over 384 experts
    widths_g384 = [widths_g[i % len(widths_g)] for i in range(384)]
    widths_d384 = [widths_d[(i * 7) % len(widths_d)] for i in range(384)]
    if want("x3gm_pf2048_ragged"): report["x3gm_pf2048_ragged"] = x3gm_case(out / "x3gm_pf2048_ragged", g, R=2048, slots=6, E=384, D=D, I=I,
                                             k2g=widths_g384, k2d=widths_d384, shx=True, input_dtype=torch.bfloat16,
                                             limit=10.0, runs=["tuned"])
    # the served geometry (D 5,120, I 1,152, widths 4 / 6 / 8 / 10) over 16 experts so the fixture stays near 1 GB
    if want("x3gm_pf2048_reference"): report["x3gm_pf2048_reference"] = x3gm_case(out / "x3gm_pf2048_reference", g, R=2048, slots=6, E=16, D=5120, I=1152,
                                           k2g=[4, 6, 8, 10] * 4, k2d=[6, 4, 10, 8] * 4, shx=False,
                                           input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_pf1024_k4"): report["x3gm_pf1024_k4"] = x3gm_case(out / "x3gm_pf1024_k4", g, R=1024, slots=6, E=64, D=D, I=I, k2g=[4] * 64,
                                         k2d=[4] * 64, shx=False, input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_pf512_shx"): report["x3gm_pf512_shx"] = x3gm_case(out / "x3gm_pf512_shx", g, R=512, slots=6, E=64, D=D, I=I, k2g=[6] * 64,
                                         k2d=[6] * 64, shx=True, input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_pf_tail1337"): report["x3gm_pf_tail1337"] = x3gm_case(out / "x3gm_pf_tail1337", g, R=1337, slots=6, E=64, D=D, I=I,
                                           k2g=widths_g384[:64], k2d=widths_d384[:64], shx=False,
                                           input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_plan"): report["x3gm_plan"] = plan_case(out / "x3gm_plan", g)
    if want("topk_keys"): report["topk_keys"] = topk_case(out / "topk_keys", g)
    if want("pointwise"): report["pointwise"] = pointwise_case(out / "pointwise", g)
    if want("glue"): report["glue"] = glue_case(out / "glue", g)
    (out.parent / "oracle.json").write_text(json.dumps({"device": torch.cuda.get_device_name(0), **report}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
