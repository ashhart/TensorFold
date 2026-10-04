"""DeepSeek-V4.1 RoPE and quantize-dequantize kernels against the fp32 reference port, row by row and in graphs."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_reference as ref

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, quant, rope

DEV = "cuda"
CFG = Config.read(Path(__file__).parents[1] / "fixtures" / "deepseek_v41")
CAPACITY = (1 << 20) + MAX_ROWS                 # a 1M context plus a verify window
EDGES = [0, 1, 2, 127, 128, 129, 1023, 1024, 1025, 16383, 16384, 16385, 65535, 65536, 65537, 1 << 19,
         (1 << 20) - 1, 1 << 20, CAPACITY - 1]
QDQ = {     # kernel, reference, group
    "fp8_1x32": (quant.fp8_qdq_1x32, ref.act_quant, 32),
    "fp4_1x32_e8m0": (quant.fp4_qdq_1x32_e8m0, lambda x: ref.fp4_act_quant(x, 32), 32),
    "fp4_1x16_e4m3": (quant.fp4_qdq_1x16_e4m3, lambda x: ref.fp4_act_quant(x, 16), 16),
}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _positions(n: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.tensor(EDGES), torch.randint(0, CAPACITY, (n,), generator=g)])


def _reference_table(kind: str, device: str) -> torch.Tensor:
    """The reference port's ``freqs_cis`` for the kind, built with torch on ``device``, as (cos, sin) pairs."""

    c = CFG
    original, base = (c.original_max_position_embeddings, c.compress_rope_theta) if kind == "yarn" else \
        (0, c.rope_theta)
    with torch.device(device):
        cis = ref.freqs_cis(c.qk_rope_head_dim, CAPACITY, original, base, c.rope_factor, c.beta_fast, c.beta_slow)
    return torch.view_as_real(cis)


@pytest.fixture(scope="module", params=rope.KINDS)
def table(request) -> tuple[str, torch.Tensor]:
    return request.param, rope.tables(CFG, request.param, CAPACITY, DEV)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32).cpu()


def _bf16_values(limit: float) -> torch.Tensor:
    """Every finite bf16 value with magnitude <= ``limit``, both signs, -0 included."""

    v = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    v = v[v.float() <= limit]
    return torch.cat([v, -v])


def _grouped(values: torch.Tensor, head: float, group: int) -> torch.Tensor:
    """Groups of ``group``: ``head`` first (it sets the scale), then ``values`` in order, padded with zeros."""

    per = group - 1
    body = torch.cat([values, values.new_zeros((-values.numel()) % per)]).reshape(-1, per)
    return torch.cat([torch.full((body.shape[0], 1), head, dtype=values.dtype), body], 1)


# -- RoPE ---------------------------------------------------------------------------------------------------

def test_tables_bit_equal_the_reference_built_on_the_device(table):
    kind, t = table
    assert t.shape == (CAPACITY, CFG.qk_rope_head_dim // 2, 2) and t.dtype == torch.float32 and t.is_contiguous()
    assert torch.equal(_bits(t), _bits(_reference_table(kind, DEV)))


def test_tables_t1_against_a_cpu_build(table):
    """CUDA's ``pow`` and scalar divide (YaRN's ramp, amplified by the blend) move a frequency by <= 8 ulps."""

    kind, t = table
    c = CFG
    p = _positions()
    got, want = t[p.to(DEV)].cpu().double(), _reference_table(kind, "cpu")[p].double()
    dim = c.qk_rope_head_dim
    f = 1 / (c.compress_rope_theta if kind == "yarn" else c.rope_theta) ** (torch.arange(0, dim, 2).double() / dim)
    if kind == "yarn":
        low, high = c.yarn
        smooth = 1 - ((torch.arange(dim // 2).double() - low) / max(high - low, 1e-3)).clamp(0, 1)
        f = f / c.rope_factor * (1 - smooth) + f * smooth
    # the angles differ by those 8 ulps plus their own rounding, times the position; sin/cos by a few ulps
    bound = 10 * p.double()[:, None, None] * 2.0 ** (torch.floor(torch.log2(f)) - 23)[None, :, None] + 2.0**-21
    assert ((got - want).abs() <= bound).all()


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape", [(32, 512), (512,), (64, 128)])
def test_apply_t1_against_the_reference_mirror(table, inverse, shape):
    _, t = table
    p = _positions(seed=1)
    g = torch.Generator(device=DEV).manual_seed(2)
    x = (torch.randn((len(p), *shape), generator=g, device=DEV) * 4).to(torch.bfloat16)
    got = rope.apply(x.clone(), p.to(DEV), t, inverse)
    want = ref.rope(x, torch.view_as_complex(t[p.to(DEV)]), ref.MIRROR, inverse).to(torch.bfloat16)
    rd = 2 * t.shape[1]
    assert torch.equal(_bits(got[..., :-rd]), _bits(x[..., :-rd]))
    a, b = got[..., -rd:].float(), want[..., -rd:].float()
    ulp = torch.where(b == 0, torch.zeros_like(b), 2.0 ** (torch.floor(torch.log2(b.abs())) - 7))
    assert ((a - b).abs() <= ulp).all()
    assert (a != b).float().mean().item() <= 0.005


def test_apply_rows_do_not_depend_on_the_window(table):
    """Row r alone at its position equals row r inside windows of 2..6 rows, by base position and by row list."""

    _, t = table
    g = torch.Generator(device=DEV).manual_seed(3)
    for base in (0, 125, 1022, 65533, CAPACITY - MAX_ROWS):
        x = torch.randn((MAX_ROWS, 32, 512), generator=g, device=DEV).to(torch.bfloat16)
        alone = [rope.apply(x[r:r + 1].clone(), torch.tensor([base + r], device=DEV), t) for r in range(MAX_ROWS)]
        for rows in range(2, MAX_ROWS + 1):
            by_base = rope.apply(x[:rows].clone(), torch.tensor([base], device=DEV, dtype=torch.int32), t)
            by_list = rope.apply(x[:rows].clone(), torch.arange(base, base + rows, device=DEV), t)
            for r in range(rows):
                assert torch.equal(_bits(by_base[r]), _bits(alone[r][0]))
                assert torch.equal(_bits(by_list[r]), _bits(alone[r][0]))


def test_apply_inverse_undoes_the_rotation_within_rounding(table):
    _, t = table
    p = _positions(seed=4).to(DEV)
    x = torch.randn((len(p), 4, 512), device=DEV).to(torch.bfloat16)
    back = rope.apply(rope.apply(x.clone(), p, t), p, t, inverse=True)
    assert (back.float() - x.float()).abs().max().item() <= 2.0**-6 * x.float().abs().max().item()


def test_apply_refuses_bad_arguments(table):
    _, t = table
    x = torch.zeros((3, 512), device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        rope.apply(x.float(), torch.zeros(3, device=DEV, dtype=torch.int64), t)
    with pytest.raises(ValueError):
        rope.apply(x, torch.zeros(2, device=DEV, dtype=torch.int64), t)
    with pytest.raises(ValueError):
        rope.apply(x, torch.zeros(3, device=DEV), t)
    with pytest.raises(ValueError):
        rope.tables(CFG, "ntk", 8, DEV)


# -- QDQ ------------------------------------------------------------------------------------------------

def _check(name: str, x: torch.Tensor) -> None:
    """The kernel, in place and into another buffer, bit-equals the reference port run on the CPU."""

    kernel, reference, _ = QDQ[name]
    want = _bits(reference(x.cpu()))
    xd = x.to(DEV)
    out = torch.empty_like(xd)
    assert kernel(xd, out) is out
    assert torch.equal(_bits(xd), _bits(x))
    assert torch.equal(_bits(out), want)
    assert torch.equal(_bits(kernel(xd)), want) and torch.equal(_bits(xd), want)


@pytest.mark.parametrize("name", QDQ)
def test_qdq_random(name):
    g = torch.Generator().manual_seed(5)
    for scale in (1e-30, 1e-3, 1.0, 37.0, 1e4, 1e30):
        _check(name, (torch.randn((MAX_ROWS, 512), generator=g) * scale).to(torch.bfloat16))
    _check(name, (torch.randn((5, 64, 128), generator=g) * torch.rand((5, 64, 1), generator=g) * 50)
           .to(torch.bfloat16))


@pytest.mark.parametrize("name", QDQ)
def test_qdq_extremes(name):
    big = torch.finfo(torch.bfloat16).max
    tiny = torch.finfo(torch.bfloat16).smallest_normal
    sub = 2.0**-133                                    # the smallest bf16 subnormal
    rows = [
        [0.0] * 32, [-0.0] * 32, [big] + [-big] * 31, [big, sub] * 16, [tiny] * 32, [sub, -sub] * 16,
        [1e-4, -1e-5] * 16, [6 * 2.0**-126, 0.0] * 16, [6 * 2.0**-9, -0.0] * 16, [448.0, 1e-30] * 16,
        [2688.0, -2687.0] * 16, [1e30, 1.0] * 16, [-big, -0.0] * 16, [5e4, -3e-3] * 16,
    ]
    _check(name, torch.tensor(rows, dtype=torch.float32).to(torch.bfloat16))


@pytest.mark.parametrize("name", QDQ)
def test_qdq_every_bf16_amax(name):
    """Every positive finite bf16 value sets one group's scale: all power-of-two and e4m3 rounding boundaries."""

    group = QDQ[name][2]
    amax = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    g = torch.Generator().manual_seed(6)
    u = torch.rand((len(amax), group - 1), generator=g) * 2 - 1
    _check(name, torch.cat([amax[:, None], amax[:, None] * u], 1).to(torch.bfloat16))


def test_fp8_every_value_and_tie_at_unit_scale():
    """amax 400 gives scale 1, so every bf16 value up to 400 (each e4m3 midpoint among them) rounds alone."""

    _check("fp8_1x32", _grouped(_bf16_values(400.0), 400.0, 32))


def test_fp4_e8m0_every_value_and_tie_at_power_of_two_scales():
    for k in (-20, -1, 0, 3, 40):
        _check("fp4_1x32_e8m0", _grouped(_bf16_values(6.0 * 2.0**k), 6.0 * 2.0**k, 32))


def test_fp4_e4m3_exact_ties_of_both_divides():
    """x = s * m for every e4m3 scale s and every e2m1 point and midpoint m; amax = 6 * a midpoint of e4m3."""

    e4m3 = torch.arange(1, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    m = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    for s in e4m3[e4m3 >= 2.0**-9]:
        _check("fp4_1x16_e4m3", _grouped((s * torch.cat([m, -m])).to(torch.bfloat16), 6 * s.item(), 16))
    mid = (e4m3[:-1] + e4m3[1:]) / 2
    heads = 6 * mid[6 * mid <= 2688]
    assert torch.equal(heads.to(torch.bfloat16).float(), heads)
    x = torch.cat([heads[:, None], heads[:, None] * torch.linspace(-1, 1, 15)[None, :]], 1)
    _check("fp4_1x16_e4m3", x.to(torch.bfloat16))


def test_e2m1_rounds_to_nearest_even():
    """FP4 QDQ at scale 1 against the nearest e2m1 point, ties to the even code, computed in fp64 here."""

    v = _bf16_values(6.0)
    got = quant.fp4_qdq_1x32_e8m0(_grouped(v, 6.0, 32).to(DEV))[:, 1:].reshape(-1)[:len(v)].cpu()
    grid = torch.tensor(E2M1, dtype=torch.float64)
    a = v.double().abs()
    d = (a[:, None] - grid[None, :]).abs()
    best = d.min(1, keepdim=True).values
    code = torch.where(d == best, torch.arange(8) + 16 * (torch.arange(8) % 2), 99).argmin(1)
    want = torch.copysign(grid[code], v.double())
    assert torch.equal(_bits(got), _bits(want.to(torch.bfloat16)))


def test_e2m1_matches_torch_float4_when_it_converts():
    v = _bf16_values(6.0).float()
    try:
        back = v.to(DEV).to(torch.float4_e2m1fn_x2).to(torch.float32).cpu()
    except (AttributeError, RuntimeError, TypeError, NotImplementedError) as e:
        pytest.skip(f"this torch does not convert to float4_e2m1fn_x2: {e}")
    if back.shape != v.shape:
        pytest.skip("this torch packs float4_e2m1fn_x2 two to an element")
    got = quant.fp4_qdq_1x32_e8m0(_grouped(v.to(torch.bfloat16), 6.0, 32).to(DEV))
    got = got[:, 1:].reshape(-1)[:len(v)].float().cpu()
    assert torch.equal(got.abs(), back.abs())


@pytest.mark.parametrize("name", QDQ)
def test_qdq_rows_do_not_depend_on_the_window(name):
    kernel = QDQ[name][0]
    x = (torch.randn((MAX_ROWS, 512), device=DEV) * 3).to(torch.bfloat16)
    alone = [_bits(kernel(x[r:r + 1].clone())) for r in range(MAX_ROWS)]
    for rows in range(2, MAX_ROWS + 1):
        y = kernel(x[:rows].clone())
        assert all(torch.equal(_bits(y[r:r + 1]), alone[r]) for r in range(rows))


def test_qdq_refuses_bad_arguments():
    x = torch.zeros((2, 48), device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        quant.fp8_qdq_1x32(x)
    with pytest.raises(ValueError):
        quant.fp4_qdq_1x16_e4m3(x[:, :32].float())
    with pytest.raises(ValueError):
        quant.fp4_qdq_1x16_e4m3(x[:, :32], torch.empty((2, 16), device=DEV, dtype=torch.bfloat16))


# -- graphs --------------------------------------------------------------------------------------------------

def test_eager_equals_graph(table):
    """The decode window's chain (RoPE at pos_dev + r, then each QDQ into a buffer) replays the eager bits."""

    _, t = table
    rows = MAX_ROWS
    pos = torch.zeros(1, device=DEV, dtype=torch.int32)
    x = torch.zeros((rows, 512), device=DEV, dtype=torch.bfloat16)
    work = torch.empty_like(x)
    outs = [torch.empty_like(x) for _ in QDQ]

    def chain():
        work.copy_(x)
        rope.apply(work, pos, t)
        for (kernel, _, _), out in zip(QDQ.values(), outs):
            kernel(work, out)

    chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    g = torch.Generator(device=DEV).manual_seed(7)
    for base in (0, 127, 4093, CAPACITY - rows):
        x.copy_(torch.randn((rows, 512), generator=g, device=DEV) * 10)
        pos.fill_(base)
        graph.replay()
        torch.cuda.synchronize()
        replayed = [_bits(o) for o in outs]
        chain()
        assert all(torch.equal(_bits(o), r) for o, r in zip(outs, replayed))
