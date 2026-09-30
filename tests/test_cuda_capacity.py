"""CUDA startup capacity checks without models or devices."""

import json
import math
import struct
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.families.glm5_next.cuda import LATENT


def checkpoint(path, config, tensors):
    (path / "config.json").write_text(json.dumps(config))
    entries = {}
    offset = 0
    for name, dtype, shape, size in tensors:
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_omitted_cuda_context_reaches_engine_as_native(tmp_path, monkeypatch):
    from tensorfold.cuda import server

    observed = []
    def engine(path, **options):
        observed.append(options)
        return SimpleNamespace(context_window=options.get("context", 8192))
    family = SimpleNamespace(title="Test", model_type="test", package=SimpleNamespace(cuda_engine=engine))
    monkeypatch.setattr(server, "App", lambda *a, **kw: SimpleNamespace(effective_context_window=a[0].context_window))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    cli._serve_cuda(args, family, tmp_path, 262144)
    assert observed[0]["context"] == 262144
    assert observed[0]["context_explicit"] is False


def test_glm_nonfit_refuses_before_weight_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.cuda import capacity
    from tensorfold.families.glm5_next.cuda import engine
    import sys

    glm = SimpleNamespace(dense_limit=2051, mtp_layers=1, layers=4)
    weights = SimpleNamespace(Config=SimpleNamespace(read=lambda *a: glm), load=None)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.weights", weights)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.decode", SimpleNamespace(Engine=None))

    config = {"max_position_embeddings": 131072, "num_hidden_layers": 2,
              "layer_types": ["linear_attention", "full_attention"], "num_attention_heads": 64,
              "linear_num_heads": 64, "linear_head_dim": 128, "linear_conv_kernel_dim": 4,
              "qk_nope_head_dim": 256, "v_head_dim": 256, "index_head_dim": 128,
              "hidden_size": 4096, "vocab_size": 65536, "num_nextn_predict_layers": 1,
              "moe_intermediate_size": 1024, "num_experts_per_tok": 8, "n_routed_experts": 288}
    checkpoint(tmp_path, config, [("model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                                  "U32", [65536, 32768], 8 * 1024**3)])
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (6 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)      # a Spark's MemAvailable would admit it
    monkeypatch.setattr(weights.Config, "read", lambda *a: glm)
    monkeypatch.setattr(engine.GlmEngine, "_gather_ints", lambda self, x: [x, x])
    def load(*a, **kw):
        pytest.fail("weight allocation was reached before startup admission")
    monkeypatch.setattr(weights, "load", load)
    comm = SimpleNamespace(barrier=lambda: None, ready=lambda *a, **k: None)
    with pytest.raises(ValueError, match="fit|memory|budget"):
        engine.GlmEngine(tmp_path, rank=0, master="example", port=29551, context=65536, comm=comm)


@pytest.mark.parametrize("override", [None, 0, 12345])
def test_cli_preserves_default_vs_explicit_context(tmp_path, monkeypatch, override):
    from tensorfold.cuda import server
    seen = []
    package = SimpleNamespace(cuda_engine=lambda *a, **kw: seen.append(kw) or SimpleNamespace(context_window=99))
    family = SimpleNamespace(title="Test", model_type="test", package=package)
    monkeypatch.setattr(server, "App", lambda *a, **kw: SimpleNamespace(effective_context_window=99))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"]
    if override is not None:
        command += ["--context", str(override)]
    args = cli.build_parser().parse_args(command)
    cli._serve_cuda(args, family, tmp_path, 262144 if override is None else override)
    assert seen[0]["context"] == (262144 if override is None else override)
    assert seen[0]["context_explicit"] is (override is not None)


def small_config():
    return {"max_position_embeddings": 65536, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512, "head_dim": 64,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 64,
            "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
            "linear_num_heads": 8, "linear_head_dim": 128, "qk_nope_head_dim": 256,
            "v_head_dim": 256, "vocab_size": 1024, "num_nextn_predict_layers": 1,
            "moe_intermediate_size": 512, "intermediate_size": 1024, "num_experts_per_tok": 2,
            "n_routed_experts": 8, "index_topk": 2048, "index_head_dim": 128,
            "quantization": {"group_size": 64, "bits": 4}}


# a 4-bit head as MLX packs it: the words, then a bf16 scale and bias for each group of 64 inputs
HEAD = [("lm_head.weight", "U32", [64, 8], 2048), ("lm_head.scales", "BF16", [64, 1], 128),
        ("lm_head.biases", "BF16", [64, 1], 128)]


class Loaded(Exception):
    pass


@pytest.fixture
def fake_runtime(monkeypatch):
    import sys
    import torch
    from tensorfold.cuda import capacity

    original_tensor, original_empty = torch.tensor, torch.empty
    def cpu(fn):
        def call(*a, **kw):
            if kw.get("device") == "cuda":
                kw["device"] = "cpu"
            return fn(*a, **kw)
        return call
    monkeypatch.setattr(torch, "tensor", cpu(original_tensor))
    monkeypatch.setattr(torch, "empty", cpu(original_empty))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(capacity, "available_bytes", lambda t: 16 * capacity.GIB)
    monkeypatch.setattr(capacity, "total_bytes", lambda t: 128 * capacity.GIB)       # a GB10: 4096-row prompt chunks
    calls = []
    def load(*a, **kw):
        calls.append(kw)
        raise Loaded
    def both(send, recv):
        recv.view(-1).copy_(torch.cat([send.view(-1), send.view(-1)]))
    comm = SimpleNamespace(barrier=lambda: None, ready=lambda *a, **k: None, all_gather=both)
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.comm", SimpleNamespace(NCCL=lambda *a: comm))
    for family in ("qwen3_5", "qwen4_exp", "glm5_next"):
        prefix = f"tensorfold.families.{family}.cuda"
        weights = SimpleNamespace(load=load, draft_token_ids=lambda *a: None,
                                  Config=SimpleNamespace(read=lambda *a: SimpleNamespace(dense_limit=2051, mtp_layers=1,
                                                                                         layers=4)))
        monkeypatch.setitem(sys.modules, prefix + ".weights", weights)
        monkeypatch.setitem(sys.modules, prefix + ".decode", SimpleNamespace(Engine=None))
    import torch.distributed as dist
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **kw: None)
    monkeypatch.setattr(dist, "all_gather_into_tensor", lambda recv, send: both(send, recv))
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.distributed", SimpleNamespace(split_weights=None))
    return calls, capacity


def construct(family, path, requested, explicit, world, rank=0):
    if family == "linear":
        from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
        obj = Qwen27Engine.__new__(Qwen27Engine)
        start = lambda: obj.__init__(path, None, tp=world, rank=rank, master="example", context=requested, context_explicit=explicit)
    elif family == "indexed":
        from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
        obj = FlashNextEngine.__new__(FlashNextEngine)
        start = lambda: obj.__init__(path, tp=world, rank=rank, master="example", max_len=requested, context_explicit=explicit, depth=0)
    else:
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine
        obj = GlmEngine.__new__(GlmEngine)
        start = lambda: obj.__init__(path, rank=rank, master="example", port=29551, context=requested, context_explicit=explicit)
    return obj, start


@pytest.mark.parametrize("family,world", [("linear", 1), ("linear", 2), ("indexed", 1), ("indexed", 2), ("mla", 2)])
@pytest.mark.parametrize("requested,explicit,window", [(None, None, 65536), (65536, False, 65536),
                                                        (0, True, 65536), (1024, True, 1024)])
@pytest.mark.torch
def test_real_constructors_choose_native_or_explicit_before_loading(tmp_path, fake_runtime, family, world,
                                                                    requested, explicit, window):
    checkpoint(tmp_path, small_config(), HEAD)
    calls, _ = fake_runtime
    obj, start = construct(family, tmp_path, requested, explicit, world)
    with pytest.raises(Loaded):
        start()
    assert len(calls) == 1
    window = 2051 if family == "mla" and not explicit else window          # GLM stays dense unless asked
    assert obj.capacity_plan["native_window"] == 65536
    assert obj.capacity_plan["context_window"] == window
    assert obj.capacity_plan["cache_slots"] >= window + (1 if family == "indexed" else 8)
    assert getattr(obj, "context_window", getattr(obj, "limit", window)) == window


@pytest.mark.torch
@pytest.mark.parametrize("family,world", [("linear", 1), ("linear", 2), ("indexed", 1), ("indexed", 2), ("mla", 2)])
def test_real_constructors_shrink_default_and_refuse_explicit_before_loading(tmp_path, monkeypatch, fake_runtime,
                                                                         family, world):
    checkpoint(tmp_path, small_config(), HEAD)
    calls, capacity = fake_runtime
    from tensorfold.cuda.geometry import gdn_geometry, mla_geometry
    text = small_config()
    geometry = (mla_geometry(text, world, 8, latent=LATENT) if family == "mla" else
                gdn_geometry(text, world, 1 if family == "indexed" else 12, indexed=family == "indexed"))
    budget = geometry.needed(12000) + 32768
    monkeypatch.setattr(capacity, "available_bytes", lambda t: budget)
    obj, start = construct(family, tmp_path, None, None, world)
    with pytest.raises(Loaded):
        start()
    fitting = obj.capacity_plan["largest_window"]
    assert 0 < fitting < 65536
    assert obj.capacity_plan["context_window"] == (2051 if family == "mla" else fitting)
    assert obj.capacity_plan["total_bytes_estimate"] <= budget
    calls.clear()
    _, reject = construct(family, tmp_path, fitting + 1, True, world)
    with pytest.raises(ValueError, match=f"largest fitting.*{fitting}"):
        reject()
    assert not calls
    _, retry = construct(family, tmp_path, fitting, True, world)
    with pytest.raises(Loaded):
        retry()
    assert len(calls) == 1


@pytest.mark.torch
@pytest.mark.parametrize("room,wanted", [(8, None), (1.25, None), (8, "0.5")])
def test_glm_keeps_other_conversations_in_what_the_window_leaves(tmp_path, monkeypatch, fake_runtime, room, wanted):
    """GLM's kept conversations get min(TF_GLM_CACHE_GIB, budget - estimate) in whole MiB, inside the reported estimate,
    so an optional cache never shrinks the window and never pushes the engine past its budget."""
    checkpoint(tmp_path, small_config(), [("lm_head.weight", "U32", [64, 8], 2048)])
    calls, capacity = fake_runtime
    if wanted is not None:
        monkeypatch.setenv("TF_GLM_CACHE_GIB", wanted)
    obj, start = construct("mla", tmp_path, None, None, 2)
    with pytest.raises(Loaded):
        start()                                    # 16 GiB: the geometry's own estimate
    total = obj.capacity_plan["total_bytes_estimate"] - obj.cache_bytes
    monkeypatch.setattr(capacity, "available_bytes", lambda t: total + int(room * capacity.GIB))
    obj, start = construct("mla", tmp_path, None, None, 2)
    with pytest.raises(Loaded):
        start()
    plan = obj.capacity_plan
    grant = min(int(float(wanted or 3) * capacity.GIB), int(room * capacity.GIB)) >> 20 << 20
    assert obj.cache_bytes == plan["kept_bytes"] == grant
    assert plan["total_bytes_estimate"] == total + grant <= plan["budget_bytes"]


@pytest.mark.parametrize("peer_fit", [0, 9000])
def test_rank_minimum_agrees_and_refuses_every_rank(peer_fit):
    from tensorfold.cuda.capacity import Geometry, Weights, make_plan, choose
    geometry = Geometry(lambda slots: slots * 32, 8)
    plans = [make_plan(20000, None, False, fit * 32 + 256, Weights(0, 0), geometry)
             for fit in (12000, peer_fit)]
    peers = [p.settings + [p.fitting] for p in plans]
    if peer_fit:
        assert [choose(p, peers) for p in plans] == [peer_fit, peer_fit]
    else:
        for plan in plans:
            with pytest.raises(ValueError, match="0 tokens"):
                choose(plan, peers)
    explicit = [make_plan(20000, 10000, True, fit * 32 + 256, Weights(0, 0), geometry)
                for fit in (12000, peer_fit)]
    peers = [p.settings + [p.fitting] for p in explicit]
    for plan in explicit:
        with pytest.raises(ValueError, match="largest fitting"):
            choose(plan, peers)


def test_rank_mismatched_flags_refuse_before_allocation():
    from tensorfold.cuda.capacity import Geometry, Weights, make_plan, choose
    plan = make_plan(20000, None, False, 2**30, Weights(0, 0), Geometry(lambda slots: slots, 8))
    peer = [20000, 0, 1, 20000]
    with pytest.raises(ValueError, match="same flags"):
        choose(plan, [plan.settings + [plan.fitting], peer])


def test_loading_peak_is_separate_from_serving_peak():
    from tensorfold.cuda.capacity import Geometry, Weights, make_plan, choose
    geometry = Geometry(lambda slots: slots, 8)
    plan = make_plan(1000, None, False, 2100, Weights(1000, 1000), geometry)
    assert choose(plan) == 1000
    receipt = plan.receipt(1000)
    assert receipt["startup_peak_bytes_estimate"] == 2000
    assert receipt["serving_peak_bytes_estimate"] == 2008
    assert receipt["total_bytes_estimate"] == 2008
    with pytest.raises(ValueError, match="0 tokens"):
        choose(make_plan(1000, None, False, 1999, Weights(1000, 1000), geometry))


@pytest.mark.torch
@pytest.mark.parametrize("family", ["linear", "indexed", "mla"])
@pytest.mark.parametrize("explicit", [False, True])
def test_actual_distributed_startup_agrees_on_smaller_rank_before_loading(tmp_path, monkeypatch, fake_runtime,
                                                                        family, explicit):
    from tensorfold.cuda.geometry import gdn_geometry, mla_geometry, linear_weights, indexed_weights, split_weights
    from tensorfold.families.glm5_next.cuda.split import rule
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    checkpoint(tmp_path, small_config(), HEAD)
    calls, capacity = fake_runtime
    geom = (mla_geometry(small_config(), 2, 8, latent=LATENT) if family == "mla" else
            gdn_geometry(small_config(), 2, 1, indexed=True) if family == "indexed" else
            gdn_geometry(small_config(), 2, 12, rows=12, prompt=4096))     # the 27B engine's, prompt chunks on a GB10
    transform = split_weights(rule) if family == "mla" else indexed_weights(2, False) if family == "indexed" else linear_weights
    weights = capacity.estimate_weights(tmp_path, transform)
    if family == "linear":
        weights = capacity.Weights(weights.resident, weights.staging + weights.resident)
    budgets = [weights.resident + geom.needed(n) for n in (20000, 9000)]
    requested = 12000 if explicit else 2051 if family == "mla" else None
    plans = [capacity.make_plan(65536, requested, explicit, budget, weights, geom) for budget in budgets]
    statuses = [[0, *plan.settings, plan.fitting, plan.largest] for plan in plans]
    def gather(*args):
        values = list(args[-1])
        return statuses if values in statuses else [values, values]      # admission's status rows, else agreement
    monkeypatch.setattr(capacity, "gather_ints", gather)
    monkeypatch.setattr(GlmEngine, "_gather_ints", lambda self, values: gather(values))
    for rank in (0, 1):
        calls.clear()
        monkeypatch.setattr(capacity, "available_bytes", lambda t, r=rank: budgets[r])
        obj, start = construct(family, tmp_path, requested, explicit, 2, rank=rank)
        if explicit:
            with pytest.raises(ValueError, match="largest fitting"):
                start()
            assert not calls
        else:
            with pytest.raises(Loaded):
                start()
            assert obj.capacity_plan["context_window"] == min(plan.fitting for plan in plans)
            assert len(calls) == 1


@pytest.mark.torch
def test_admission_propagates_peer_header_failure_before_loading(tmp_path, fake_runtime):
    calls, capacity = fake_runtime
    checkpoint(tmp_path, small_config(), HEAD)
    from tensorfold.cuda.geometry import gdn_geometry, linear_weights
    with pytest.raises(ValueError, match="another rank could not read"):
        capacity.admit(tmp_path, None, False, None, lambda t: gdn_geometry(t, 2, 12), linear_weights,
                       rank=0, world=2, gather=lambda mine: [mine, [1, 0, -1, 0, 0]])
    assert not calls


def test_speculative_reserve_does_not_double_committed_dynamic_kv():
    from tensorfold.cuda.geometry import gdn_geometry
    geometry = gdn_geometry(small_config(), 1, 12)
    # The real decoder clips the kept path before commit; speculative keys live outside committed KV.
    a, b = geometry.needed(32768), geometry.needed(32769)
    assert b > a
    assert b - a > 16 * 1024**2


def test_fp8_block_scales_are_sized(tmp_path):
    """The FP4 checkpoints store their block scales as ``F8_E4M3`` (the published Swift revision carries
    73,728 of them: the one dtype the startup estimate could not size), one byte a value like every fp8."""

    from tensorfold.cuda import capacity

    checkpoint(tmp_path, small_config(), [("model.layers.0.mlp.experts.0.down_proj.weight_scale", "F8_E4M3",
                                           [2560, 40], 2560 * 40)])
    weights = capacity.estimate_weights(
        tmp_path, lambda name, info: (math.prod(info["shape"]) * capacity.itemsize(info, name), 0))
    assert weights.resident == 2560 * 40
    assert capacity.SIZES["F8_E4M3"] == capacity.SIZES["F8_E5M2"] == 1


def test_an_unsizable_dtype_names_its_tensor_in_the_operators_message(tmp_path, monkeypatch):
    """A dtype the estimate cannot size must say which tensor carried it, and that has to survive ``admit``,
    whose message is what the operator reads when a rank cannot read its checkpoint."""

    from tensorfold.cuda import capacity

    checkpoint(tmp_path, small_config(), [("model.layers.0.w", "F8_E9M9", [4, 4], 16)])
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: 16 * capacity.GIB)
    monkeypatch.setattr(capacity, "page_room", lambda torch: None)
    geometry = capacity.Geometry(lambda slots: slots * 1024, 8)
    with pytest.raises(ValueError, match=r"F8_E9M9.*model\.layers\.0\.w|model\.layers\.0\.w.*F8_E9M9"):
        capacity.admit(tmp_path, None, None, object(), geometry, lambda name, info: (1, 0))


def test_a_4bit_drafter_is_admitted_at_its_packed_bytes(tmp_path):
    from tensorfold.cuda.capacity import estimate_weights
    from tensorfold.families.qwen3_5.cuda.affine_memory import draft_bytes, packed_draft

    checkpoint(tmp_path, {}, [("layers.0.mlp.gate_proj.weight", "BF16", [5120, 5120], 2 * 5120 * 5120),
                              ("layers.0.self_attn.k_proj.weight", "BF16", [1024, 5120], 2 * 1024 * 5120),
                              ("layers.0.input_layernorm.weight", "BF16", [5120], 2 * 5120),
                              ("layers.0.narrow.weight", "BF16", [5120, 64], 2 * 5120 * 64)])
    q4 = lambda n: n // 2 + n // 64 * 4
    assert estimate_weights(tmp_path, draft_bytes).resident == (q4(5120 * 5120) + 2 * q4(1024 * 5120) + 2 * 5120
                                                                + 2 * 5120 * 64)
    assert packed_draft("a.weight", [5120, 5120]) and not packed_draft("a.weight", [5120, 64])
    assert not packed_draft("a.bias", [5120, 5120]) and not packed_draft("a.weight", [5120])
