"""CUDA image admission, checkpoint and transport contracts without accelerator runtimes."""

import ast
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.qwen_cuda import (EncodedVision, broadcast_encoded, capacity_geometry,
                                       checkpoint_vision, validate_encoded, weight_transform)


def _checkpoint(path):
    vision = {"model_type": "qwen3_5", "hidden_size": 8, "out_hidden_size": 8, "depth": 1,
              "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
              "intermediate_size": 12, "num_heads": 2, "num_position_embeddings": 4}
    config = {"model_type": "qwen3_5", "vision_config": vision, "text_config": {
        "hidden_size": 8, "head_dim": 8, "rope_parameters": {"mrope_interleaved": True,
        "mrope_section": [2, 1, 1], "partial_rotary_factor": 1}}}
    (path / "config.json").write_text(json.dumps(config))
    shapes = {"patch_embed.proj.weight": [8, 2, 2, 2, 3], "patch_embed.proj.bias": [8],
              "pos_embed.weight": [4, 8], "merger.norm.weight": [8], "merger.norm.bias": [8],
              "merger.linear_fc1.weight": [32, 32], "merger.linear_fc1.bias": [32],
              "merger.linear_fc2.weight": [8, 32], "merger.linear_fc2.bias": [8]}
    for part, shape in {"norm1": [8], "norm2": [8], "attn.qkv": [24, 8], "attn.proj": [8, 8],
                        "mlp.linear_fc1": [12, 8], "mlp.linear_fc2": [8, 12]}.items():
        shapes[f"blocks.0.{part}.weight"] = shape
        shapes[f"blocks.0.{part}.bias"] = [shape[0]]
    offset, entries = 0, {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 2
        entries["vision_tower." + name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    _write_tensors(path, entries, offset)
    return entries, offset


def _write_tensors(path, entries, size):
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(size))


def test_vision_headers_do_not_load_tensor_payloads(tmp_path):
    _, size = _checkpoint(tmp_path)
    config, resident = checkpoint_vision(tmp_path)
    assert config["out_hidden_size"] == 8
    assert resident == size


@pytest.mark.parametrize("damage", ["missing", "quantized", "range", "shape"])
def test_incomplete_or_incompatible_towers_refuse_before_loading(tmp_path, damage):
    entries, size = _checkpoint(tmp_path)
    key = "vision_tower.blocks.0.attn.qkv.weight"
    if damage == "missing":
        del entries[key]
    elif damage == "quantized":
        entries[key]["dtype"] = "U32"
    elif damage == "range":
        entries[key]["data_offsets"] = [size, size + 24 * 8 * 2]
    else:
        entries[key]["shape"] = [12, 16]
    _write_tensors(tmp_path, entries, size)
    with pytest.raises(ValueError):
        checkpoint_vision(tmp_path)


def test_every_placeholder_has_exactly_one_feature_and_position():
    prompt = [10, 99, 99, 99, 99, 11, 12]
    positions = [[0, 1, 1, 1, 1, 3, 4], [0, 1, 1, 2, 2, 3, 4], [0, 1, 2, 1, 2, 3, 4]]
    validate_encoded((1, 2, 3, 4), positions, -2, prompt, 99, (4, 8), 8)
    for rows, pos, delta, shape in [((0, 1, 2, 3), positions, -2, (4, 8)),
                                   ((1, 2, 3, 4), positions, 0, (4, 8)),
                                   ((1, 2, 3, 4), [positions[0]] * 2, -2, (4, 8)),
                                   ((1, 2, 3, 4), positions, -2, (3, 8))]:
        with pytest.raises(ValueError):
            validate_encoded(rows, pos, delta, prompt, 99, shape, 8)


def test_vision_memory_is_reserved_only_on_the_tower_rank(tmp_path):
    from tensorfold.cuda.capacity import Geometry

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    zero = capacity_geometry(base, tmp_path, True, 0)({})
    one = capacity_geometry(base, tmp_path, True, 1)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert zero.needed(32) > one.needed(32) > plain.needed(32)
    original = lambda *args: (0, 0)
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(original, True, 0)("vision_tower.x", info) == (128, 0)
    assert weight_transform(original, True, 1)("vision_tower.x", info) == (0, 0)


def test_offloaded_tower_leaves_the_gpu_budget_but_keeps_a_smaller_workspace(tmp_path):
    from tensorfold.cuda.capacity import Geometry
    from tensorfold.vision.qwen_cuda import OFFLOAD_WORKSPACE_BYTES, WORKSPACE_BYTES

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    resident = capacity_geometry(base, tmp_path, True, 0)({})
    offloaded = capacity_geometry(base, tmp_path, True, 0, offload=True)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert resident.needed(32) > offloaded.needed(32) > plain.needed(32)
    assert offloaded.needed(32) - plain.needed(32) == OFFLOAD_WORKSPACE_BYTES < WORKSPACE_BYTES
    original = lambda *args: (0, 0)
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(original, True, 0, True)("vision_tower.x", info) == (0, 0)
    assert weight_transform(original, True, 0, False)("vision_tower.x", info) == (128, 0)


def test_tp_transports_features_and_negative_offset_bit_for_bit(monkeypatch):
    records, arrays = [], []
    rank = [0]
    def share(values, r, device):
        if r == 0:
            records.append(list(values))
            return list(values)
        return records.pop(0)
    def broadcast(value, source):
        if rank[0] == 0:
            arrays.append(value.copy())
        else:
            value[:] = arrays.pop(0)
    torch = SimpleNamespace(bfloat16=np.uint16, int32=np.int32,
                            empty=lambda shape, dtype, device: np.empty(shape, dtype=dtype))
    distributed = SimpleNamespace(broadcast=broadcast)
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode_tp", SimpleNamespace(_share=share))
    features = np.array([[0, 65535], [1, 32768]], dtype=np.uint16)
    positions = np.array([[0, 1, 1, 2], [0, 1, 2, 3], [0, 1, 1, 2]], dtype=np.int32)
    outgoing = EncodedVision((1, 2), features, positions, -1)
    broadcast_encoded(outgoing, 0, "cpu", hidden=2, prompt_length=4)
    rank[0] = 1
    received = broadcast_encoded(None, 1, "cpu", hidden=2, prompt_length=4)
    assert received.rows == (1, 2) and received.rope_delta == -1
    np.testing.assert_array_equal(received.features, features)
    np.testing.assert_array_equal(received.positions, positions)


def _engine():
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    engine = object.__new__(Qwen27Engine)
    engine.vision = SimpleNamespace(encode=lambda prepared, prompt: prepared)
    engine.context_window, engine.tp, engine.scheduler = 100, 1, None
    engine.w = object()
    engine.draft, engine.max_rows, engine.allow_copy = None, 12, True
    engine.cache = [([1, 2], SimpleNamespace(pos=2), None)]
    return engine


def test_images_never_reuse_or_pollute_text_prefix_cache(monkeypatch):
    calls = []
    fake = SimpleNamespace(prefill=lambda *args, **kw: (calls.append(kw) or SimpleNamespace(pos=3), 4),
                           draft_decode=lambda *args, **kw: None)
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode", fake)
    engine, payload = _engine(), object()
    cache = engine.cache.copy()
    result = engine.generate([1, 2, 3], 1, None, lambda tokens: True, vision=payload)
    assert calls[0]["state"] is None and calls[0]["vision"] is payload
    assert result["cached"] == 0 and engine.cache == cache


def test_image_encoding_is_deferred_to_scheduler_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.decode",
                        SimpleNamespace(prefill=None, draft_decode=None))
    engine, payload, calls = _engine(), object(), []
    engine.vision.encode = lambda *args: pytest.fail("HTTP thread invoked the image tower")
    engine.scheduler = SimpleNamespace(submit=lambda *args, **kw: calls.append((args, kw)) or {})
    engine.generate([1, 2, 3], 1, None, lambda tokens: True, vision=payload)
    assert calls[0][1] == {"stop_eos": True, "vision": payload}


def test_state_clone_preserves_image_offset_without_importing_cuda():
    path = Path(__file__).parents[1] / "src/tensorfold/families/qwen3_5/cuda/decode.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "clone_state")
    function.returns = None
    function.args.args[0].annotation = None
    state_type = type("State", (), {})
    namespace = {"State": state_type}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    state = state_type()
    state.pos, state.limit, state.rope_delta, state.room = 257, 1024, -192, object()
    state.conv, state.rec, state.kv = [1], [2], [3]
    cloned = namespace["clone_state"](state)
    assert (cloned.pos, cloned.rope_delta, cloned.limit, cloned.room) == (257, -192, 1024, state.room)
    assert cloned.conv == state.conv and cloned.conv is not state.conv
    assert cloned.kv is state.kv                   # one attention list: a grow reaches every clone


def test_a_meta_built_tower_matches_a_normally_built_one_in_the_installed_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from tensorfold.vision.qwen_cuda import rotary_frequencies

    raw = {"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64, "patch_size": 4,
           "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3, "out_hidden_size": 32,
           "num_position_embeddings": 16, "deepstack_visual_indexes": []}
    config = Qwen3_5VisionConfig(**raw)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    built = Qwen3_5VisionModel(config).eval()
    with torch.device("meta"):
        meta = Qwen3_5VisionModel(config)
    meta.load_state_dict(built.state_dict(), strict=True, assign=True)
    rotary_frequencies(meta.rotary_pos_emb, raw, "cpu")
    for name, buffer in built.rotary_pos_emb.named_buffers():
        assert torch.equal(dict(meta.rotary_pos_emb.named_buffers())[name], buffer)
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    grid = torch.tensor([[1, 4, 4]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = meta.eval()(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert torch.equal(got, want)
