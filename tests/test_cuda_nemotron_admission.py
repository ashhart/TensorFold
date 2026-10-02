"""Nemotron on CUDA is admitted before any weight loads: its engine, serial twin, MTP head and kept prompt ends."""

import importlib
import json
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_cuda_capacity import Loaded, checkpoint, fake_runtime  # noqa: F401
from tests.test_cuda_geometry import Allocation, bytes_in

CONFIG = {"hidden_size": 512, "vocab_size": 131072, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
          "mamba_num_heads": 8, "mamba_head_dim": 64, "n_groups": 2, "ssm_state_size": 64, "conv_kernel": 4,
          "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 128,
          "moe_shared_expert_intermediate_size": 256, "max_position_embeddings": 65536,
          "layers_block_type": ["mamba", "attention", "moe", "mamba", "attention", "moe"], "num_hidden_layers": 6}
WEIGHTS = [("backbone.layers.0.mixer.in_proj.weight", "U32", [1088, 64], 1088 * 64 * 4),
           ("backbone.layers.2.mixer.switch_mlp.fc1.weight", "U32", [8, 128, 64], 8 * 128 * 64 * 4),
           ("lm_head.weight", "U32", [1024, 64], 1024 * 64 * 4)]


def mtp_file(path):
    raw = json.dumps({"layers.0.mixer.q_proj.weight": {"dtype": "U32", "shape": [512, 64],
                                                        "data_offsets": [0, 512 * 64 * 4]}}).encode()
    (path / "mtp-4bit.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)


@pytest.fixture
def nemotron(tmp_path, monkeypatch, fake_runtime):  # noqa: F811
    calls, capacity = fake_runtime
    checkpoint(tmp_path, CONFIG, WEIGHTS)
    mtp_file(tmp_path)
    prefix = "tensorfold.families.nemotron_h.cuda"

    def load(*a, **kw):
        calls.append(kw)
        raise Loaded

    monkeypatch.setitem(sys.modules, prefix + ".engine", SimpleNamespace(ROWS=16, Engine=None))
    monkeypatch.setitem(sys.modules, prefix + ".mtp", SimpleNamespace(MTPHead=None))
    monkeypatch.setitem(sys.modules, prefix + ".attention", SimpleNamespace(CHUNK=512))
    monkeypatch.setitem(sys.modules, prefix + ".weights", SimpleNamespace(MTP_FILE="mtp-4bit.safetensors", load=load))
    from tensorfold.families import nemotron_h
    return tmp_path, calls, capacity, nemotron_h


@pytest.mark.torch
@pytest.mark.parametrize("world", [1, 2])
def test_nemotron_context_that_cannot_fit_is_refused_before_loading(monkeypatch, nemotron, world):
    from tensorfold.cuda.geometry import hybrid_geometry
    path, calls, capacity, family = nemotron
    geometry = hybrid_geometry(CONFIG, world, 16, rows=16, chunk=512, drafts=True, draft=32768)
    monkeypatch.setattr(capacity, "available_bytes", lambda t: geometry.needed(12000) + 2**24)
    kw = dict(tp=world, rank=0, master="example")
    with pytest.raises(ValueError, match="largest fitting"):
        family.cuda_engine(path, context=60000, context_explicit=True, **kw)
    assert not calls
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
    obj = NemotronEngine.__new__(NemotronEngine)
    with pytest.raises(Loaded):
        obj.__init__(path, context=16384, context_explicit=False, **kw)
    window = obj.capacity_plan["context_window"]
    assert 12000 <= window < 16384 and obj.max_len % 512 == 0 and obj.max_len >= window + 16
    assert obj.capacity_plan["total_bytes_estimate"] <= obj.capacity_plan["budget_bytes"]


@pytest.mark.torch
def test_nemotron_default_window_is_the_family_default(nemotron):
    path, calls, _, family = nemotron
    from tensorfold.families.nemotron_h.cuda import CONTEXT
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
    obj = NemotronEngine.__new__(NemotronEngine)
    with pytest.raises(Loaded):
        obj.__init__(path, context=CONTEXT, context_explicit=False)
    assert obj.capacity_plan["context_window"] == CONTEXT and len(calls) == 1


class Tensor(Allocation):
    def __init__(self, shape, dtype, device):
        super().__init__((shape,) if isinstance(shape, int) else shape, dtype, device)

    def element_size(self):
        return 8 if self.dtype in ("int64", "fp64") else super().element_size()

    def pin_memory(self):
        return self

    def clone(self):
        return Allocation(self.shape, self.dtype, self.device)


@pytest.fixture
def fake_torch(monkeypatch):
    lang = ModuleType("triton.language")
    lang.constexpr = object
    triton = ModuleType("triton")
    triton.language = lang
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    before = set(sys.modules)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    recorded = []

    def allocate(shape, **kw):
        tensor = Tensor(shape, kw.get("dtype", "fp32"), kw.get("device", "cpu"))
        recorded.append(tensor)
        return tensor

    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    fake = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32", float64="fp64", int16="int16",
                           int32="int32", int64="int64", zeros=allocate, empty=allocate,
                           full=lambda shape, fill, **kw: allocate(shape, **kw),
                           zeros_like=lambda x, **kw: allocate(x.shape, dtype=x.dtype, device=x.device),
                           empty_like=lambda x, **kw: allocate(x.shape, dtype=x.dtype, device=x.device),
                           arange=lambda n, **kw: allocate((n,), **kw),
                           cuda=SimpleNamespace(is_available=lambda: False, Event=lambda **kw: event))
    try:
        yield recorded, fake
    finally:
        for name in [n for n in sys.modules if n not in before and n.startswith("tensorfold.families.nemotron_h.cuda")]:
            sys.modules.pop(name, None)


@pytest.mark.torch
def test_nemotron_geometry_bounds_the_engine_twin_head_and_snapshots(monkeypatch, fake_torch):
    recorded, fake = fake_torch
    engine = importlib.import_module("tensorfold.families.nemotron_h.cuda.engine")
    mtp = importlib.import_module("tensorfold.families.nemotron_h.cuda.mtp")
    for mod in (engine, mtp, engine.grouped, engine.S):
        monkeypatch.setattr(mod, "torch", fake)
    from tensorfold.cuda.geometry import hybrid_geometry
    from tensorfold.families.nemotron_h.cuda.weights import Config
    c = Config(hidden=512, vocab=1024, pattern="M*EM*E", heads=8, kv_heads=2, head_dim=64, m_heads=8,
               m_head_dim=64, m_groups=2, m_state=64, conv_kernel=4, experts=8, top_k=2, moe_width=128,
               shared_width=256, scaling=1.0, norm_topk=True, eps=1e-5)
    w = SimpleNamespace(config=c, norm_f=SimpleNamespace(device="cpu"), mtp=SimpleNamespace(), extra={},
                        head=SimpleNamespace())
    length = 65536
    main = engine.Engine(w, max_len=length, graphs=False)
    head = mtp.MTPHead(main, draft_ids=None)
    live = bytes_in(recorded)
    engine.Engine(w, max_len=length, graphs=False)                   # the serial twin
    state = bytes_in(getattr(main, name) for name in main.STATE) + bytes_in([head.k_cache, head.v_cache])
    used = bytes_in(recorded) + 3 * state                            # three prompt-end snapshots
    kv = 2 * 2 * length * c.kv_heads * c.head_dim * 2
    assert kv <= live and used <= hybrid_geometry(CONFIG, 1, 16, rows=16, chunk=512, drafts=True,
                                                  draft=0).bytes_at(length - 16)


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("drafts", [False, True])
def test_long_window_budgets_serial_cache_and_snapshot_state(world: int, drafts: bool) -> None:
    """Bound persistent cache allocations, including the lazy serial engine."""

    from tensorfold.cuda.geometry import hybrid_geometry

    text = CONFIG | {"vocab_size": 1024}
    length, rows = 262144, 16
    kv = text["num_key_value_heads"] // world
    heads = text["mamba_num_heads"] // world
    conv_dim = heads * text["mamba_head_dim"] + 2 * (text["n_groups"] // world) * text["ssm_state_size"]
    # Engine.STATE includes ssm, conv_base and the raw/xc/dt rollback rows; snapshot() clones all of them.
    state = text["layers_block_type"].count("mamba") * (
        heads * text["mamba_head_dim"] * text["ssm_state_size"] * 4
        + (text["conv_kernel"] - 1) * conv_dim * 2
        + 2 * rows * (2 * conv_dim * 2 + heads * 4)
    )
    one_kv = 2 * length * kv * text["head_dim"] * 2
    # Live engine + serial twin + three snapshots; MTP has no serial twin.
    persistent = 5 * (state + text["layers_block_type"].count("attention") * one_kv)
    persistent += 4 * one_kv if drafts else 0
    estimate = hybrid_geometry(text, world, rows, rows=rows, chunk=512, drafts=drafts, draft=0)
    assert estimate.bytes_at(length - rows) >= persistent


def test_nemotron_weights_split_by_rank_and_keep_the_mtp_head_whole():
    from tensorfold.cuda.geometry import hybrid_weights
    one, two = hybrid_weights(1), hybrid_weights(2)
    fc1 = {"dtype": "U32", "shape": [8, 1856, 64]}
    assert two("backbone.layers.2.mixer.switch_mlp.fc1.weight", fc1)[0] == 8 * 960 * 64 * 4    # 15 of 29 tiles
    conv = {"dtype": "BF16", "shape": [1024, 4, 1]}
    assert one("backbone.layers.0.mixer.conv1d.weight", conv)[0] == 1024 * 4 * 4                # fp32 at load
    q = {"dtype": "U32", "shape": [512, 64]}
    assert two("layers.0.mixer.q_proj.weight", q)[0] == one("layers.0.mixer.q_proj.weight", q)[0] * 3 // 2


@pytest.mark.torch
@pytest.mark.parametrize("drafts", [0, 3])
def test_an_unindexed_checkpoint_counts_the_mtp_file_once_and_only_when_drafting(nemotron, drafts):
    from tensorfold.cuda import capacity
    from tensorfold.cuda.geometry import hybrid_weights
    path, calls, _, _ = nemotron
    main = capacity.estimate_weights(path, hybrid_weights(1), files=[path / "model.safetensors"]).resident
    head = capacity.estimate_weights(path, hybrid_weights(1), files=[path / "mtp-4bit.safetensors"]).resident
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
    obj = NemotronEngine.__new__(NemotronEngine)
    with pytest.raises(Loaded):
        obj.__init__(path, drafts=drafts, context=4096, context_explicit=True)
    assert obj.capacity_plan["weight_bytes_estimate"] == main + (head if drafts else 0)


@pytest.mark.torch
@pytest.mark.parametrize("ids", [[1, 1, 2], [1, 2, 131072]])
def test_draft_ids_must_be_distinct_tokens_of_the_vocabulary(nemotron, ids):
    path, calls, _, _ = nemotron
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
    with pytest.raises(ValueError, match="distinct token ids"):
        NemotronEngine(path, draft_ids=ids, context=4096, context_explicit=True)
    assert not calls


@pytest.mark.torch
def test_two_ranks_with_the_same_draft_ids_in_another_order_refuse_to_start(fake_runtime):  # noqa: F811
    import torch
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
    first = list(range(63)) + [1023] + list(range(63, 127))
    second = list(first)
    second[63], second[64] = second[64], second[63]           # same length and sum, another order

    def settings(ids, peer=None):
        sent = []

        def gather(mine, both):
            sent.append(mine.clone())
            both.copy_(torch.cat([mine, peer if peer is not None else mine]))
        obj = SimpleNamespace(drafts=3, confidence=0.2, max_len=1024, comm=SimpleNamespace(all_gather=gather))
        NemotronEngine._same_settings(obj, torch, ids)
        return sent[0]

    theirs = settings(second)
    with pytest.raises(RuntimeError, match="different settings"):
        settings(first, theirs)
    settings(first, settings(first))                          # the same list in the same order starts


@pytest.mark.torch
def test_two_ranks_with_different_prompt_precision_refuse_to_start(fake_runtime):  # noqa: F811
    import torch
    from tensorfold.cuda import prompt_precision
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine

    def settings(peer=None):
        sent = []

        def gather(mine, both):
            sent.append(mine.clone())
            both.copy_(torch.cat([mine, peer if peer is not None else mine]))
        obj = SimpleNamespace(drafts=3, confidence=0.2, max_len=1024, comm=SimpleNamespace(all_gather=gather))
        NemotronEngine._same_settings(obj, torch, [1, 2, 3])
        return sent[0]

    with prompt_precision.using(True):                        # rank 1 started with --prefill-fp8
        theirs = settings()
    with prompt_precision.using(False), pytest.raises(RuntimeError, match="prompt precision.*--prefill-fp8"):
        settings(theirs)
