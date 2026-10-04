"""DeepSeek-V4.1's MoE on one rank: routing, EXL3 routed experts split by intermediate dim, the shared expert's
single fp32 add, DSpark's NVFP4 experts; real layers against the fp32 reference (with the checkpoint).

Random layers keep the real rank shapes (D 5120, I 1152 a rank, 384 experts, 2-bit mcg trellises); a few trellis
sets are shared between experts, each expert with its own suh/svh.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytest.importorskip("safetensors")

from dsv41_layouts import exl3_dim1_half, mx8_from_block
from dsv41_ref_weights import hadamard

from tensorfold.cuda.exl3 import experts as exl3
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import moe
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.convert import make_experts4
from tensorfold.families.deepseek_v41.cuda.weights import MoEW

CFG = Config.read(Path(__file__).parents[1] / "fixtures" / "deepseek_v41")
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
D, I, E, K = CFG.hidden_size, CFG.moe_intermediate_size, CFG.n_routed_experts, CFG.num_experts_per_tok
LIMIT, CAP, SETS, HAD = CFG.swiglu_limit, 512, 8, 128
T2_REL, T2_COS = 4 * 2.0**-8, 0.9999
F64 = torch.float64
_CACHE: dict = {}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _x(rows: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn((rows, D), generator=_gen(seed), device="cuda") * scale).to(torch.bfloat16)


def _sign_scale(shape, mag: float, g) -> torch.Tensor:
    sign = torch.randint(0, 2, shape, generator=g, device="cuda").float() * 2 - 1
    return (sign * (torch.rand(shape, generator=g, device="cuda") + 0.5) * mag).half()


def _fp8(n: int, k: int, g) -> tuple[torch.Tensor, torch.Tensor]:
    """Random e4m3 [n, k] (finite, below 2^7) with 32x32 E8M0 block scales 2^-9 .. 2^-6."""

    w = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8, device="cuda")
    w[(w & 0x7F) >= 0x70] = 0x30
    return w.view(torch.float8_e4m3fn), torch.randint(118, 122, (n // 32, k // 32), generator=g, dtype=torch.uint8,
                                                       device="cuda")


def _fp8_dense(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    pow2 = torch.ldexp(torch.ones_like(s, dtype=F64), s.to(torch.int64) - 127)
    return w.to(F64) * pow2.repeat_interleave(32, 0).repeat_interleave(32, 1)[:w.shape[0], :w.shape[1]]


def _shared(rank: int, world: int, mats) -> tuple:
    """A rank's shared expert from full (w1, s1, w3, s3, w2, s2): stacked [w1 | w3] rows, w2 columns."""

    w1, s1, w3, s3, w2, s2 = mats
    h, hb = I // world, I // world // 32
    rows, brows = slice(rank * h, (rank + 1) * h), slice(rank * hb, (rank + 1) * hb)
    gu = mx8_from_block(torch.cat([w1[rows], w3[rows]]).contiguous(), torch.cat([s1[brows], s3[brows]]).contiguous())
    d = mx8_from_block(w2[:, rows].contiguous(), s2[:, brows].contiguous())
    return gu, d, _fp8_dense(w2[:, rows], s2[:, brows])


def _random_layer(rank: int = 0, world: int = 2) -> tuple[MoEW, dict]:
    """A random rank part, cached; the dict keeps what the fp64 references need."""

    key = ("layer", rank, world)
    if key not in _CACHE:
        g = _gen(7)
        k2 = 4
        sets = [[torch.randint(-32768, 32768, shape, generator=g, device="cuda", dtype=torch.int32).to(torch.int16)
                 for shape in ((D // 16, I // 16, 8 * k2), (D // 16, I // 16, 8 * k2), (I // 16, D // 16, 8 * k2))]
                for _ in range(SETS)]
        suh_g, suh_u, svh_d = (_sign_scale((E, D), 0.02, g) for _ in range(3))
        svh_g, svh_u, suh_d = (_sign_scale((E, I), 1.0, g) for _ in range(3))
        h = I // world
        cols = slice(rank * h, (rank + 1) * h)
        halves = [(exl3_dim1_half(a, rank, world), exl3_dim1_half(b, rank, world), c[rank * h // 16:(rank + 1) * h // 16])
                  for a, b, c in sets]
        ex = exl3.prepare([(halves[e % SETS][0], suh_g[e], svh_g[e, cols]) for e in range(E)],
                          [(halves[e % SETS][1], suh_u[e], svh_u[e, cols]) for e in range(E)],
                          [(halves[e % SETS][2], suh_d[e, cols], svh_d[e]) for e in range(E)], "mcg")
        gs = _gen(8)
        mats = (*_fp8(I, D, gs), *_fp8(I, D, gs), *_fp8(D, I, gs))
        gu, sd, w2 = _shared(rank, world, mats)
        gate = (torch.randn((E, D), generator=gs, device="cuda") * 0.02).to(torch.bfloat16)
        bias = torch.randn((E,), generator=gs, device="cuda") * 0.1
        wq = [[exl3.dequant(t, "mcg").to(F64) for t in s] for s in halves]
        _CACHE[key] = (MoEW(gate, bias, bias, ex, gu, sd), {"wq": wq, "w2": w2})
    return _CACHE[key]


def _buffers(world: int = 2, rows: int = 6, prefill: bool = False) -> Buffers:
    return Buffers(CFG, world, rows, CAP, prefill)


def _half_ulp16(v: torch.Tensor) -> torch.Tensor:
    """Half the fp16 spacing at ``v`` (subnormals included): the most a rounding to fp16 moves a value."""

    v = v.to(F64).abs()
    return torch.ldexp(torch.ones_like(v), torch.frexp(v).exponent - 12).clamp_min(2.0**-25)


def _ulps_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    top = torch.maximum(a.abs(), b.abs()).to(F64)
    return (a.to(F64) - b.to(F64)).abs() / torch.ldexp(torch.ones_like(top), torch.frexp(top).exponent - 8).clamp_min(
        2.0**-133)


def _abs_had(v: torch.Tensor) -> torch.Tensor:
    """The magnitude bound of a 128-point Hadamard's outputs: each block's sum of |v|."""

    return v.abs().unflatten(-1, (-1, HAD)).sum(-1, keepdim=True).expand(*v.shape[:-1], -1, HAD).flatten(-2)


# -- routing ------------------------------------------------------------------------------------------------------

def _torch_route(mlog: torch.Tensor, bias: torch.Tensor, k: int, scale: float):
    """Routing in torch fp32 on the device from the same logits: same scores, stable sort for lower-id ties."""

    s = torch.nn.functional.softplus(mlog).sqrt()
    idx = torch.sort(s + bias, dim=-1, descending=True, stable=True).indices[:, :k]
    w = s.gather(1, idx)
    return idx.int(), w / (w.sum(-1, keepdim=True) + 1e-20) * scale


@pytest.mark.parametrize("slots", [K, K + 1])
def test_route_picks_by_biased_sqrt_softplus_with_lower_ids_on_ties(slots):
    m, _ = _random_layer()
    gate = m.gate.clone()
    gate[11], gate[300] = gate[3], gate[3]                    # three experts with the same logits and bias
    bias = m.bias.clone()
    bias[[11, 300]] = bias[3] = 2.0                           # and the highest choice: ids 3, 11, 300 in that order
    x = torch.cat([_x(30, 1), _x(2, 2, 40.0)])                # the last rows push logits past softplus's threshold
    mlog = torch.empty((32, E), device="cuda")
    pick = torch.empty((32, slots), dtype=torch.int32, device="cuda")
    wts = torch.empty((32, slots), device="cuda")
    moe.route(x, gate, bias, mlog, pick, wts, K, CFG.routed_scaling_factor)
    assert float(mlog[30:].abs().max()) > 20
    idx, w = _torch_route(mlog, bias, K, CFG.routed_scaling_factor)
    assert torch.equal(pick[:, :K], idx)
    assert torch.equal(pick[:30, :3], torch.tensor([[3, 11, 300]] * 30, dtype=torch.int32, device="cuda"))
    assert float(((wts[:, :K] - w).abs() / w.abs()).max()) <= 4 * 2.0**-24
    if slots > K:                                             # the shared expert last, weight 1
        assert bool((pick[:, K] == E).all()) and bool((wts[:, K] == 1).all())


_REF: dict = {}


def _ref():
    if "w" not in _REF:
        from dsv41_ref_weights import RefWeights
        _REF["w"] = RefWeights(MODEL, experts=4)
    return _REF["w"]


def _near_ties(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, k: int) -> torch.Tensor:
    """Rows whose k-th and (k+1)-th fp64 biased scores are closer than 4x the fp32 bound on either."""

    x, w = x.to(F64), gate.to(F64)
    lg = x @ w.T
    s = torch.nn.functional.softplus(lg).sqrt()
    bound = 8 * math.sqrt(x.shape[1]) * 2.0**-24 * (x.abs() @ w.abs().T) + 4 * 2.0**-24 * (s + bias.abs())
    top = torch.sort(s + bias.to(F64), dim=-1, descending=True, stable=True)
    margin = top.values[:, k - 1] - top.values[:, k]
    b = torch.maximum(bound.gather(1, top.indices[:, k - 1:k]), bound.gather(1, top.indices[:, k:k + 1]))[:, 0]
    return margin < 4 * b


@needs_model
@pytest.mark.parametrize("layer", [0, 20, 39])
def test_routed_sets_equal_the_reference_gate_on_real_layers(layer):
    from dsv41_reference import gate as ref_gate

    rw = _ref()
    p = f"layers.{layer}.ffn"
    gate, bias = rw(f"{p}.gate.weight"), rw(f"{p}.gate.bias")
    x = _x(256, 30 + layer).cpu().float() * rw(f"layers.{layer}.ffn_norm.weight")
    xb = x.to(torch.bfloat16).cuda()
    mlog = torch.empty((256, E), device="cuda")
    pick = torch.empty((256, K), dtype=torch.int32, device="cuda")
    wts = torch.empty((256, K), device="cuda")
    moe.route(xb, gate.to(torch.bfloat16).cuda(), bias.cuda(), mlog, pick, wts, K, CFG.routed_scaling_factor)
    rw_, ridx = ref_gate(xb.cpu().float(), gate, bias, K, CFG)
    same = (pick.cpu().sort(-1).values == ridx.int().sort(-1).values).all(-1)
    ties = _near_ties(xb.cpu().float(), gate, bias.double(), K)
    print(f"layer {layer}: {int((~same).sum())} of 256 rows differ, {int(ties.sum())} near-ties")
    assert bool((same | ties).all())
    rows = same & (pick.cpu() == ridx.int()).all(-1)
    assert float(((wts.cpu() - rw_).abs() / rw_.abs())[rows].max()) <= 1e-5


# -- the backbone's MoE -------------------------------------------------------------------------------------------

def test_shared_add_equals_a_seventh_slot_of_weight_one():
    m, _ = _random_layer()
    b = _buffers()
    s7 = exl3.Scratch(m.experts, 6, K + 1)
    for R in range(1, 7):
        x = _x(R, 50 + R)
        part = moe.backbone(CFG, m, x, b).clone()
        pick7 = torch.cat([b.pick[:R], torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
        wts7 = torch.cat([b.wts[:R], torch.ones((R, 1), device="cuda")], 1).contiguous()
        s7.y[:R * (K + 1)].view(R, K + 1, D)[:, K].copy_(b.sd[:R])
        out = exl3.routed(x, pick7, wts7, m.experts, s7, None, R, LIMIT, exl3.ACT_F32)
        assert torch.equal(out, part), R


def _projected(x: torch.Tensor, wq: torch.Tensor, svh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An EXL3 projection of the kernels' fp16 input in fp64, (value, magnitude of its terms through H and svh)."""

    x = x.to(F64)
    v = hadamard(x @ wq, -1) / math.sqrt(HAD) * svh.to(F64)
    return v, _abs_had(x.abs() @ wq.abs()) / math.sqrt(HAD) * svh.to(F64).abs()


def _rank_ref(m: MoEW, keep: dict, b: Buffers, R: int) -> tuple:
    """One rank in fp64 from the kernels' own fp16 inputs of each stage: the share from its down inputs and bf16
    shared activations with the magnitude of its terms; the down inputs from its gate/up inputs, with their rounding
    budget (fp32 sums over K and the Hadamard, SwiGLU's slopes, fp16 rounding not included)."""

    ex, s = m.experts, b.exl3
    pick, wts = b.pick[:R].reshape(-1).tolist(), b.wts[:R].to(F64)
    share = (b.sact[:R].to(F64) @ keep["w2"].T)
    absum = b.sact[:R].to(F64).abs() @ keep["w2"].abs().T
    xd64, budget = torch.empty_like(s.xd[:R * K], dtype=F64), torch.empty_like(s.xd[:R * K], dtype=F64)
    fp32 = 8 * math.sqrt(D + HAD) * 2.0**-24
    for p, e in enumerate(pick):
        wg, wu, wd = keep["wq"][e % SETS]
        (gv, ag), (uv, au) = _projected(s.xg[p], wg, ex.svh_g[e]), _projected(s.xu[p], wu, ex.svh_u[e])
        gg, uu = gv.clamp(max=LIMIT), uv.clamp(-LIMIT, LIMIT)
        silu = gg / (1 + torch.exp(-gg))
        v = silu * uu * ex.suh_d[e].to(F64)
        dact = 1.1 * uu.abs() * fp32 * ag + silu.abs() * fp32 * au + 4 * 2.0**-24 * (silu * uu).abs()
        xd64[p] = hadamard(v, -1) / math.sqrt(HAD)
        budget[p] = (_abs_had(dact * ex.suh_d[e].to(F64).abs()) + fp32 * _abs_had(v)) / math.sqrt(HAD)
        z = s.xd[p].to(F64) @ wd
        w = wts[p // K, p % K]
        share[p // K] += w * hadamard(z, -1) / math.sqrt(HAD) * ex.svh_d[e].to(F64)
        absum[p // K] += w.abs() * _abs_had(s.xd[p].to(F64).abs() @ wd.abs()) / math.sqrt(HAD) * ex.svh_d[e].abs()
    return share, absum, xd64, budget


def test_rank_halves_add_in_rank_order_to_the_fp64_reference_within_t1():
    R = 6
    x = _x(R, 60)
    parts, ref, absum = [], 0, 0
    for rank in (0, 1):
        m, keep = _random_layer(rank)
        b = _buffers()
        parts.append(moe.backbone(CFG, m, x, b).clone())
        share, a, xd64, budget = _rank_ref(m, keep, b, R)
        xd = b.exl3.xd[:R * K]                                # the fp16 down inputs: the op-level bound on gate/up
        used = float(((xd.to(F64) - xd64).abs() / (budget + _half_ulp16(xd))).max())
        off = float((xd != xd64.half()).double().mean())
        print(f"rank {rank}: xd {used:.3f} of the rounding budget, {off:.2e} off the rounded fp64 value")
        assert used <= 1 and off <= 0.005
        ref, absum = ref + share, absum + a
    total = parts[0] + parts[1]
    bound = 8 * math.sqrt(I + HAD + 2 * K) * 2.0**-24 * absum
    ratio = float(((total.to(F64) - ref).abs() / bound.clamp_min(1e-300)).max())
    print(f"rank-order sum: {ratio:.3f} of the op-level bound")
    assert ratio <= 1


def test_rows_do_not_depend_on_the_row_count_or_the_prompt_window():
    m, _ = _random_layer()
    b = _buffers()
    x = _x(6, 70)
    full = moe.backbone(CFG, m, x, b).clone()
    for r in range(1, 7):
        assert torch.equal(moe.backbone(CFG, m, x[:r], b), full[:r]), r
        assert torch.equal(moe.backbone(CFG, m, x[6 - r:], b), full[6 - r:]), r
    bp = _buffers(rows=2048, prefill=True)
    assert bp.exl3.rows == 1024
    xp = _x(2048, 71)
    full = moe.backbone(CFG, m, xp, bp, prompt=True).clone()
    for a, n in ((0, 1), (1000, 30), (1020, 9), (1023, 2), (2047, 1), (5, 1024), (1024, 1024)):
        assert torch.equal(moe.backbone(CFG, m, xp[a:a + n], bp, prompt=True), full[a:a + n]), (a, n)


def test_fp16_overflow_raises_on_the_first_forwards_only():
    m, _ = _random_layer()
    b = _buffers()
    x = torch.full((2, D), 1e7, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(FloatingPointError):
        moe.backbone(CFG, m, x, b)
    b.exl3.guard_left = 0                                     # past the first forwards nothing is read back
    moe.backbone(CFG, m, x, b)
    assert bool(torch.isinf(b.exl3.xg[:2 * K]).any())


def test_an_overflow_on_one_rank_raises_on_both():
    from dsv41_pair import run_pair

    layers = [_random_layer(rank)[0] for rank in (0, 1)]
    xs = [torch.full((2, D), 1e7, dtype=torch.bfloat16, device="cuda"), _x(2, 72)]

    def rank(r):
        def body(comm):
            try:
                moe.backbone(CFG, layers[r], xs[r], _buffers(), comm=comm)
            except FloatingPointError:
                return "raised"
            return "went on"
        return body

    assert run_pair(rank(0), rank(1)) == ("raised", "raised"), "rank 1 would wait in the next gather"


def _capture(step) -> torch.cuda.CUDAGraph:
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def test_graph_replays_equal_eager():
    m, _ = _random_layer()
    (ds, _), b = _dspark_layer(), _buffers()
    for R in range(1, 7):
        x = _x(R, 80 + R)
        graph = _capture(lambda x=x: moe.backbone(CFG, m, x, b))
        for seed in (90, 91):
            x.copy_(_x(R, seed + R))
            graph.replay()
            got = b.part[:R].clone()
            assert torch.equal(got, moe.backbone(CFG, m, x, b).clone()), R
    xd = _x(CFG.dspark_block_size, 92)
    graph = _capture(lambda: moe.dspark_moe(CFG, ds, xd, b))
    xd.copy_(_x(CFG.dspark_block_size, 93))
    graph.replay()
    got = b.part[:CFG.dspark_block_size].clone()
    assert torch.equal(got, moe.dspark_moe(CFG, ds, xd, b))


# -- real layers against the fp32 reference ------------------------------------------------------------------------

def _real_moe(layer: int, rank: int, need: set[int]) -> MoEW:
    """Rank part of a backbone layer's MoE from the pack, routed experts outside ``need`` left as zero trellises."""

    rw = _ref()
    pk, p, h = rw.pack, f"layers.{layer}.ffn", I // 2
    zg = torch.zeros((D // 16, h // 16, 32), dtype=torch.int16, device="cuda")
    zd = torch.zeros((h // 16, D // 16, 32), dtype=torch.int16, device="cuda")
    one_d, one_i = torch.ones(D, dtype=torch.half, device="cuda"), torch.ones(h, dtype=torch.half, device="cuda")
    mats = {"w1": [], "w3": [], "w2": []}
    for e in range(E):
        for w, out in mats.items():
            if e not in need:
                out.append((zd, one_i, one_d) if w == "w2" else (zg, one_d, one_i))
                continue
            t, suh, svh = (pk.tensor(f"{p}.experts.{e}.{w}.{q}").cuda() for q in ("trellis", "suh", "svh"))
            if w == "w2":
                out.append((t[rank * h // 16:(rank + 1) * h // 16].contiguous(), suh[rank * h:(rank + 1) * h], svh))
            else:
                out.append((exl3_dim1_half(t, rank, 2), suh, svh[rank * h:(rank + 1) * h]))
    ex = exl3.prepare(mats["w1"], mats["w3"], mats["w2"], "mcg")
    full = []
    for w in ("w1", "w3", "w2"):
        name = f"{p}.shared_experts.{w}"
        full += [pk.tensor(f"{name}.weight").cuda(), pk.tensor(f"{name}.scale").view(torch.uint8).cuda()]
    gu, sd, _ = _shared(rank, 2, (full[0], full[1], full[2], full[3], full[4], full[5]))
    return MoEW(pk.tensor(f"{p}.gate.weight").cuda(), pk.tensor(f"{p}.gate.bias").cuda(), None, ex, gu, sd)


@needs_model
@pytest.mark.parametrize("layer", [0, 20, 39])
def test_layer_output_tracks_the_mirror_reference_on_real_layers(layer):
    from dsv41_reference import MIRROR
    from dsv41_reference import gate as ref_gate

    rw = _ref()
    x = (_x(6, 100 + layer).cpu().float() * rw(f"layers.{layer}.ffn_norm.weight")).to(torch.bfloat16).cuda()
    b, p = _buffers(), f"layers.{layer}.ffn"
    moe.route(x, rw.pack.tensor(f"{p}.gate.weight").cuda(), rw.pack.tensor(f"{p}.gate.bias").cuda(), b.mlog[:6],
              b.pick[:6], b.wts[:6], K, CFG.routed_scaling_factor)
    need = set(b.pick[:6].reshape(-1).tolist())
    ranks, outs = [_real_moe(layer, r, need) for r in (0, 1)], {}
    for prompt in (False, True):
        parts = [moe.backbone(CFG, m, x, b, prompt=prompt).clone() for m in ranks]
        outs[prompt] = (parts[0] + parts[1]).to(torch.bfloat16).float().cpu()
    ref = rw.reference(MIRROR).moe(layer, x.cpu().float())
    _, ridx = ref_gate(x.cpu().float(), rw(f"layers.{layer}.ffn.gate.weight"), rw(f"layers.{layer}.ffn.gate.bias"),
                       K, CFG)
    rows = (b.pick[:6].cpu().sort(-1).values == ridx.int().sort(-1).values).all(-1)
    print(f"layer {layer}: {int((~rows).sum())} rows routed differently (near-ties) left out")
    for prompt, got in outs.items():
        got, want = got[rows].double(), ref[rows].double()
        rel = float((got - want).norm() / want.norm())
        cos = float(torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0))
        u = _ulps_bf16(got, want)
        print(f"layer {layer} {'prompt' if prompt else 'decode'}: rel-L2 {rel:.2e}, cos {cos:.6f}, "
              f"max {float(u.max()):.0f} ulp, {float((u > 1).double().mean()):.2e} over 1 ulp")
        assert rel <= T2_REL and cos >= T2_COS


# -- DSpark's Experts4 ---------------------------------------------------------------------------------------------

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=F64)


def _mxfp4_dense(words: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 (e2m1 x2 low nibble first, E8M0 per 32) -> fp64 [N, K], as model.py's dequant."""

    b = words.view(torch.uint8).to(torch.int64)
    codes = torch.stack([b & 0xF, b >> 4], -1).flatten(-2)
    pow2 = torch.ldexp(torch.ones(scale.shape, dtype=F64, device=b.device), scale.view(torch.uint8).long() - 127)
    return E2M1.to(b.device)[codes] * pow2.repeat_interleave(32, -1)


def _dspark_layer(rank: int = 0) -> tuple:
    """A random DSpark stage's MoE rank part (MXFP4 experts through the exact NVFP4 rewrite) and its fp64 experts."""

    key = ("dspark", rank)
    if key not in _CACHE:
        g, h, n = _gen(11), I // 2, CFG.dspark_n_routed_experts
        def mxfp4(r: int, c: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (torch.randint(-128, 128, (r, c // 2), generator=g, dtype=torch.int8, device="cuda"),
                    torch.randint(114, 123, (r, c // 32), generator=g, dtype=torch.uint8, device="cuda"))

        mats = [[mxfp4(h, D) for _ in range(n)], [mxfp4(h, D) for _ in range(n)], [mxfp4(D, h) for _ in range(n)]]
        sh = _random_layer(rank)[0]
        gate = (torch.randn((n, D), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        bias = torch.randn((n,), generator=g, device="cuda") * 0.1
        m = MoEW(gate, bias, bias, make_experts4(*mats, limit=LIMIT), sh.shared_gu, sh.shared_d)
        _CACHE[key] = (m, lambda e, w: _mxfp4_dense(*mats[("w1", "w3", "w2").index(w)][e]))
    return _CACHE[key]


def _bf16(v: torch.Tensor) -> torch.Tensor:
    return v.to(torch.bfloat16).to(F64)


def _check_dspark(m: MoEW, x: torch.Tensor, b: Buffers, dense) -> None:
    """Experts4's SwiGLU within the budget of its roundings (fp32 sums over K, then bf16 g, u, silu and product)
    of the fp64 MXFP4 expert, and equal to those roundings of the fp64 sums except where g, u or silu lies within its
    fp32 error of a bf16 rounding boundary; its fp32 down outputs within the fp32 bound of the fp64 product of its
    own activations; the combine and the shared slot."""

    R, k, NI = x.shape[0], CFG.dspark_num_experts_per_tok, m.experts.width
    part = moe.dspark_moe(CFG, m, x, b).clone()
    pick, xf = b.dpick[:R].tolist(), x.to(F64)
    used, off, unexplained, bound_used, bf = 0.0, 0.0, 0, 0.0, 2.0**-8
    for r in range(R):
        for j in range(k):
            w1, w3, w2 = (dense(pick[r][j], w) for w in ("w1", "w3", "w2"))
            g, u = xf[r] @ w1.T, xf[r] @ w3.T
            dg = 8 * math.sqrt(D) * 2.0**-24 * (xf[r].abs() @ w1.abs().T)
            du = 8 * math.sqrt(D) * 2.0**-24 * (xf[r].abs() @ w3.abs().T)
            near = (_bf16(g - dg) != _bf16(g + dg)) | (_bf16(u - du) != _bf16(u + du))
            dg, du = dg + bf * (g.abs() + dg), du + bf * (u.abs() + du)
            gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
            silu = gc / (1 + torch.exp(-gc))
            ds = 1.1 * dg + bf * (silu.abs() + 1.1 * dg) + 4 * 2.0**-24 * silu.abs()
            dact = uc.abs() * ds + (silu.abs() + ds) * du
            dact = dact + bf * ((silu * uc).abs() + dact)
            act = b.dact[r, j].to(F64)
            used = max(used, float(((act - silu * uc).abs() / dact.clamp_min(1e-300)).max()))
            gb, ub = _bf16(g).clamp(max=LIMIT), _bf16(u).clamp(-LIMIT, LIMIT)
            sb = (gb.float() / (1 + torch.exp(-gb.float()))).to(F64)          # fp32's range, as the kernel
            near |= _bf16(sb * (1 - 4 * 2.0**-24)) != _bf16(sb * (1 + 4 * 2.0**-24))
            differs = act != _bf16(_bf16(sb) * ub)
            off, unexplained = off + float(differs.double().sum()), unexplained + int((differs & ~near).sum())
            y = act @ w2.T
            bound = 8 * math.sqrt(NI) * 2.0**-24 * (act.abs() @ w2.abs().T)
            bound_used = max(bound_used, float(((b.dey[r, j].to(F64) - y).abs() / bound.clamp_min(1e-300)).max()))
    off /= R * k * NI
    print(f"dspark: act {used:.3f} of the rounding budget, {off:.2e} off the rounded fp64 sums ({unexplained} not at"
          f" a boundary); down {bound_used:.3f} of the fp32 bound")
    assert used <= 1 and unexplained == 0 and bound_used <= 1
    assert torch.equal(b.dey[:R, k], b.sd[:R]) and bool((b.dpick[:R, k] == CFG.dspark_n_routed_experts).all())
    want = (b.dey[:R].to(F64) * b.dwts[:R, :, None].to(F64)).sum(1)
    absum = (b.dey[:R].to(F64) * b.dwts[:R, :, None].to(F64)).abs().sum(1)
    assert bool(((part.to(F64) - want).abs() <= 8 * 2 * 2.0**-24 * absum).all())


def test_dspark_experts4_track_the_mxfp4_expert_within_t1():
    (m, dense), b = _dspark_layer(), _buffers()
    _check_dspark(m, _x(CFG.dspark_block_size, 110), b, dense)


@needs_model
def test_dspark_experts4_track_the_real_mxfp4_experts_within_t1():
    rw = _ref()
    pk, p, h, n = rw.pack, "mtp.0.ffn", I // 2, CFG.dspark_n_routed_experts
    x = (_x(CFG.dspark_block_size, 120).cpu().float() * rw("mtp.0.ffn_norm.weight")).to(torch.bfloat16).cuda()
    gate, bias = pk.tensor(f"{p}.gate.weight").cuda(), pk.tensor(f"{p}.gate.bias").cuda()
    b = _buffers()
    moe.route(x, gate, bias, b.dmlog[:5], b.dpick[:5], b.dwts[:5], 3, CFG.routed_scaling_factor)
    need = set(b.dpick[:5, :3].reshape(-1).tolist())
    zero = {w: (torch.zeros(s, dtype=torch.int8, device="cuda"), torch.full(z, 127, dtype=torch.uint8, device="cuda"))
            for w, s, z in (("w1", (h, D // 2), (h, D // 32)), ("w2", (D, h // 2), (D, h // 32)))}
    zero["w3"] = zero["w1"]
    for rank in (0, 1):
        rows = slice(rank * h, (rank + 1) * h)
        mats, dense = {w: [] for w in ("w1", "w3", "w2")}, {}
        for e in range(n):
            for w, out in mats.items():
                if e not in need:
                    out.append(zero[w])
                    continue
                name = f"{p}.experts.{e}.{w}"
                wt, sc = pk.tensor(f"{name}.weight"), pk.tensor(f"{name}.scale").view(torch.uint8)
                if w == "w2":
                    wt, sc = wt[:, rank * h // 2:(rank + 1) * h // 2], sc[:, rank * h // 32:(rank + 1) * h // 32]
                else:
                    wt, sc = wt[rows], sc[rows]
                out.append((wt.contiguous().cuda(), sc.contiguous().cuda()))
                dense[e, w] = rw.pack.dense_fp32(name, None if w == "w2" else rows).to(F64).cuda()
                if w == "w2":
                    dense[e, w] = dense[e, w][:, rows]
        m = _random_layer(rank)[0]
        real = MoEW(gate, bias, None, make_experts4(mats["w1"], mats["w3"], mats["w2"], limit=LIMIT), m.shared_gu,
                    m.shared_d)
        _check_dspark(real, x, b, lambda e, w, dense=dense: dense[e, w])
