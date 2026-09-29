"""Packed FP8 attention rows (``kv8``, the 27B's ``--kv-dtype fp8``) without a GPU: the torch reference's exponent,
bytes and values, packed caches through growth and commit, and the engine's admission and refusals."""

import importlib
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.test_cuda_capacity import HEAD, Loaded, checkpoint, fake_runtime, small_config  # noqa: E402,F401

Q = "tensorfold.families.qwen3_5.cuda."


@pytest.fixture
def cuda(monkeypatch):
    """``kv8`` and the 27B's modules, with an import-only Triton stand-in where Triton is missing; modules first
    imported here are dropped afterwards, as in ``test_qwen27_prompt_end_cache_host``."""

    before = set(sys.modules)
    if importlib.util.find_spec("triton") is None:
        lang = ModuleType("triton.language")
        lang.constexpr = object
        triton = ModuleType("triton")
        triton.language = lang
        triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
        triton.cdiv = lambda a, b: (a + b - 1) // b
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        monkeypatch.setitem(sys.modules, "triton", triton)
        monkeypatch.setitem(sys.modules, "triton.language", lang)
    names = ("tensorfold.cuda.kernels.kv8", Q + "glue", Q + "forward", Q + "prefill")
    try:
        yield SimpleNamespace(**{name.rpartition(".")[2]: importlib.import_module(name) for name in names})
    finally:
        added = [(name, module) for name, module in list(sys.modules.items())
                 if name not in before and name.startswith("tensorfold.")
                 and (".cuda" in name or name.startswith("tensorfold.cuda"))]
        for name, module in sorted(added, key=lambda item: len(item[0]), reverse=True):
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child, None) is module:
                delattr(parent, child)
            sys.modules.pop(name, None)


def _bits(x):
    return x.contiguous().view(torch.int16)


def _row(values, scale=1.0):
    """One (1, 1, 256) bf16 row: ``values`` then zeros, times ``scale``."""

    row = torch.zeros(256, dtype=torch.float64)
    row[:len(values)] = torch.tensor(values, dtype=torch.float64)
    return (row * scale).to(torch.bfloat16).view(1, 1, 256)


def test_the_exponent_is_the_least_that_puts_every_bf16_amax_at_or_under_448(cuda):
    amax = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)   # every finite a >= 0
    got = cuda.kv8.exponent(amax).tolist()
    for a, e in zip(amax.double().tolist(), got):
        if a < 2.0 ** -126:                          # zero and subnormal rows: the clamp
            assert e == -127
            continue
        want = math.ceil(math.log2(a / 448.0))
        assert e == max(-127, want), a
        if e > -127:                                 # and it is the least such exponent (exact in float64)
            assert a * 2.0 ** -e <= 448.0 < a * 2.0 ** -(e - 1)


@pytest.mark.parametrize("k", [-100, -20, 0, 7, 100])
def test_a_mantissa_of_1_75_is_the_boundary(cuda, k):
    """448 * 2^k needs e = k; the next bf16 up (450 * 2^k, mantissa 0x61) needs k + 1; the one below stays at k."""

    for amax, e in ((448.0, k), (450.0, k + 1), (446.0, k), (256.0, k), (511.0, k + 1)):
        packed = cuda.kv8.reference_pack(_row([amax], 2.0 ** k))
        assert packed[0, 0, 256].view(torch.int8).item() == e, (amax, k)


@pytest.mark.parametrize("scale,e", [(1.0, 0), (2.0 ** 20, 20), (2.0 ** -40, -40)])
def test_codes_round_to_nearest_even_after_an_exact_scale(cuda, scale, e):
    kv8 = cuda.kv8
    # 432, 400 and 17 sit halfway between e4m3 neighbours (416|448, 384|416, 16|18), 2^-10 and 3 * 2^-10 between
    # subnormals: each goes to the even code
    values = [448.0, 432.0, 400.0, 2.0 ** -10, 3 * 2.0 ** -10, -0.0, -2.0 ** -11, 1.0, -448.0, 17.0]
    packed = kv8.reference_pack(_row(values, scale))
    assert packed.shape == (1, 1, kv8.ROW8) and packed.dtype == torch.uint8
    assert packed[0, 0, :10].tolist() == [0x7E, 0x7E, 0x7C, 0x00, 0x02, 0x80, 0x80, 0x38, 0xFE, 0x58]
    assert packed[0, 0, 10:256].eq(0).all()
    assert packed[0, 0, 256].item() == e & 0xFF and packed[0, 0, 257:].eq(0).all()
    want = [448.0, 448.0, 384.0, 0.0, 2.0 ** -8, -0.0, -0.0, 1.0, -448.0, 16.0]
    assert _bits(kv8.unpack(packed)).equal(_bits(_row(want, scale)))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_zero_rows_keep_their_zeros(cuda, sign):
    kv8 = cuda.kv8
    x = _row([], 1.0) * sign                                                  # +0.0 or -0.0 everywhere
    packed = kv8.reference_pack(x)
    assert packed[0, 0, 256].view(torch.int8).item() == -127
    assert packed[0, 0, :256].eq(0x80 if sign < 0 else 0).all() and packed[0, 0, 257:].eq(0).all()
    assert _bits(kv8.unpack(packed)).equal(_bits(x))


def _random(rows=64, heads=4, low=-60, high=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    shift = torch.randint(low, high, (rows, heads, 1), generator=g).double()
    return (torch.randn(rows, heads, 256, generator=g, dtype=torch.float64) * torch.exp2(shift)).bfloat16()


def test_values_are_the_codes_times_2e_as_the_simulation_rounded_them(cuda):
    kv8 = cuda.kv8
    x = _random()
    packed = kv8.reference_pack(x)
    xf = x.float()
    scale = torch.exp2(torch.ceil(torch.log2(xf.abs().amax(-1, keepdim=True) / 448.0)))
    simulated = ((xf / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)      # the first TF_KV_FP8
    assert _bits(kv8.unpack(packed)).equal(_bits(simulated))
    assert packed[..., 257:].eq(0).all() and packed.shape == (64, 4, kv8.ROW8)
    assert _bits(cuda.glue.fp8_round(x)).equal(_bits(simulated))               # one quantizer for the simulation too


def test_a_rounded_row_packs_to_the_same_values(cuda):
    """Values are a fixed point; bytes too unless rounding carried amax down to 448 * 2^(e - 1), where e drops by one
    and the codes double."""

    kv8 = cuda.kv8
    x = torch.cat([_random(heads=1, seed=1), _row([454.0, 3.0])])            # 454 rounds to 448 at e = 1
    first = kv8.reference_pack(x)
    values = kv8.unpack(first)
    again = kv8.reference_pack(values)
    assert _bits(kv8.unpack(again)).equal(_bits(values))
    dropped = first[..., 256].view(torch.int8) - again[..., 256].view(torch.int8)
    assert set(dropped.unique().tolist()) <= {0, 1} and dropped[-1, 0].item() == 1
    same = dropped == 0
    assert again[same].equal(first[same])


def test_strided_rows_pack_as_their_copies(cuda):
    kv8 = cuda.kv8
    both = _random(rows=8, heads=8)
    value = both[:, 4:]                                                        # a row-strided view, as [k | v] gives
    assert kv8.reference_pack(value).equal(kv8.reference_pack(value.contiguous()))


def test_the_startup_estimate_counts_the_packed_row(cuda):
    from tensorfold.cuda.geometry import kv8_bytes

    assert kv8_bytes(cuda.kv8.D8) == cuda.kv8.ROW8


def _weights(kv_fp8, attention=2):
    config = SimpleNamespace(k_heads=1, dk=2, v_heads=1, dv=2, conv_kernel=4, kv_heads=2, head_dim=256)
    return SimpleNamespace(config=config, layers=[SimpleNamespace(linear=False)] * attention,
                           norm=torch.ones(1, dtype=torch.bfloat16), kv_fp8=kv_fp8)


def test_packed_caches_grow_and_take_a_commits_packed_rows(cuda):
    kv8, forward, prefill = cuda.kv8, cuda.forward, cuda.prefill
    st = forward.State(_weights(True))
    assert all(kv[0].dtype == torch.uint8 and kv[0].shape == (0, 2, kv8.ROW8) for kv in st.kv)
    k0, v0 = prefill._grow(st, 0, 5)
    assert k0.dtype == v0.dtype == torch.uint8 and k0.shape == (1024, 2, kv8.ROW8)
    forward.reserve(st, 2048)
    assert all(kv[0].shape == (2048, 2, kv8.ROW8) and kv[1].dtype == torch.uint8 for kv in st.kv)
    keys, values = _random(rows=5, heads=2, seed=2), _random(rows=5, heads=2, seed=3)
    (rk, pk), (rv, pv) = [(kv8.unpack(kv8.reference_pack(t)), kv8.reference_pack(t)) for t in (keys, values)]
    record = [forward.AttentionRecord(rk, rv, pk, pv) for _ in st.kv]
    path = [0, 2, 3]
    rows = torch.tensor(path + [0, 0], dtype=torch.int32)
    forward.commit(st, record, path, (rows, torch.tensor([3], dtype=torch.int32), rows[:3].long()))
    assert st.pos == 3
    for kbuf, vbuf in st.kv:
        assert kbuf[:3].equal(pk[path]) and vbuf[:3].equal(pv[path])
        assert _bits(kv8.unpack(kbuf[:3])).equal(_bits(rk[path]))             # what the window's own path saw


def test_one_forward_reads_one_kind_of_cache(cuda):
    forward = cuda.forward
    packed, plain = forward.State(_weights(True)), forward.State(_weights(False))
    assert forward._packed([packed, packed]) and not forward._packed([plain])
    with pytest.raises(ValueError, match="not both"):
        forward._packed([packed, plain])


# ------------------------------------------------------------------------------------------------- the engine
@pytest.fixture
def rocm(monkeypatch):
    from tensorfold.cuda import build

    monkeypatch.setattr(torch.version, "hip", "7.14")
    build.hip.cache_clear()
    yield
    build.hip.cache_clear()


def _start(path, monkeypatch, kv_fp8, tp=1, streams=1):
    """Qwen27Engine's startup up to its first weight load (``fake_runtime`` raises ``Loaded`` there)."""

    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    monkeypatch.setitem(sys.modules, Q + "qmm_fast", SimpleNamespace(prepare=None))    # ROCm's fused layout
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", lambda *a: None)  # streams' growable segments
    engine = Qwen27Engine.__new__(Qwen27Engine)
    with pytest.raises(Loaded):
        engine.__init__(path, None, tp=tp, master="example", streams=streams, kv_fp8=kv_fp8)
    return engine.capacity_plan


@pytest.mark.parametrize("streams", [1, 4])
def test_fp8_on_rocm_admits_a_longer_window_on_the_same_budget(tmp_path, monkeypatch, fake_runtime, rocm, streams):  # noqa: F811
    from tensorfold.cuda.geometry import gdn_geometry, stream_geometry

    calls, capacity = fake_runtime
    text = dict(small_config(), head_dim=256)
    checkpoint(tmp_path, text, HEAD)
    geometry = gdn_geometry(text, 1, 12) if streams == 1 else stream_geometry(text, 1, streams, 3)
    budget = geometry.needed(12000) + 32768
    monkeypatch.setattr(capacity, "available_bytes", lambda t: budget)
    bf16, fp8 = _start(tmp_path, monkeypatch, False, streams=streams), _start(tmp_path, monkeypatch, True,
                                                                              streams=streams)
    assert len(calls) == 2
    assert 0 < bf16["context_window"] < fp8["context_window"] <= 65536
    assert fp8["total_bytes_estimate"] <= budget


@pytest.mark.parametrize("hip,head_dim,tp,env,message", [
    (False, 256, 1, None, "NVIDIA serves bf16"),
    (True, 256, 2, None, "drop --tp 2"),
    (True, 64, 1, None, "not head size 64"),
    (True, 256, 1, "TF_ROCM_ATTN_KERNEL", "TF_ROCM_ATTN_KERNEL=triton"),
    (True, 256, 1, "TF_ROCM_TREE_KERNEL", "TF_ROCM_TREE_KERNEL=triton"),
])
def test_fp8_is_refused_before_loading_where_no_kernel_reads_it(tmp_path, monkeypatch, fake_runtime, hip, head_dim,  # noqa: F811
                                                               tp, env, message):
    from tensorfold.cuda import build
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    calls, _ = fake_runtime
    checkpoint(tmp_path, dict(small_config(), head_dim=head_dim), HEAD)
    monkeypatch.setattr(torch.version, "hip", "7.14" if hip else None)
    build.hip.cache_clear()
    if env:
        monkeypatch.setenv(env, "triton")
    try:
        with pytest.raises(ValueError, match=message):
            Qwen27Engine(tmp_path, None, tp=tp, master="example", kv_fp8=True)
    finally:
        build.hip.cache_clear()
    assert not calls


def test_the_family_passes_fp8_to_its_engine_and_refuses_other_caches(tmp_path, monkeypatch):
    from tensorfold.families import qwen3_5
    from tensorfold.families.qwen3_5.cuda import engine

    made = []
    monkeypatch.setattr(engine, "Qwen27Engine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    assert qwen3_5.CUDA_KV_DTYPES == ("bf16", "fp8")
    assert qwen3_5.cuda_engine(tmp_path, no_drafts=True, kv_dtype="fp8").kv_fp8 is True
    assert qwen3_5.cuda_engine(tmp_path, no_drafts=True).kv_fp8 is False
    with pytest.raises(ValueError, match="kv-dtype 'int8'"):
        qwen3_5.cuda_engine(tmp_path, no_drafts=True, kv_dtype="int8")
    assert len(made) == 2
