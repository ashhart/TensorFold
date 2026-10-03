"""TENSORFOLD_TIGHT_STAGING: an EXL3 pack's startup loading estimate as the loader holds it, for 16 GB cards."""

import json
import math
import struct

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.geometry import exl3_staging, exl3_weights, tight_staging

SIZES = {"U8": 1, "BF16": 2, "F16": 2, "U32": 4}
GIB = capacity.GIB


def _write(folder, tensors):
    """A header-only safetensors file: the startup estimates read headers, not data."""

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5", "quantization_config": {"quant_method": "exl3", "bits": 2.5},
         "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "max_position_embeddings": 8192}))
    entries, offset = {}, 0
    for name, dtype, shape in tensors:
        size = math.prod(shape) * SIZES[dtype]
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return folder


@pytest.fixture
def exl3_pack(tmp_path):
    """An EXL3-shaped header set: a head group bigger than layer 0 though each of its parts is smaller, a big embed."""

    p = "model.language_model."
    return _write(tmp_path / "exl3", [
        (p + "embed_tokens.weight", "BF16", [4096, 32]),
        (p + "norm.weight", "BF16", [32]),
        ("lm_head.trellis", "U8", [2000]), ("lm_head.suh", "F16", [16]), ("lm_head.svh", "F16", [512]),
        (p + "layers.0.self_attn.q_proj.trellis", "U8", [800]),
        (p + "layers.0.self_attn.q_proj.suh", "F16", [16]),
        (p + "layers.0.self_attn.q_proj.svh", "F16", [256]),
        (p + "layers.0.self_attn.q_norm.weight", "BF16", [16]),
        (p + "layers.0.mlp.gate_proj.trellis", "U8", [1200]),
        (p + "layers.0.mlp.gate_proj.suh", "F16", [16]),
        (p + "layers.0.mlp.gate_proj.svh", "F16", [256]),
        (p + "layers.0.input_layernorm.weight", "BF16", [32]),
    ])


@pytest.fixture
def isolated(monkeypatch):
    """Admission without a GPU check or this machine's memory, as test_cuda_host_staging's startup fixture."""

    from tensorfold.cuda import build

    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(build, "refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "unified", lambda torch: False)
    monkeypatch.setattr(capacity, "page_room", lambda torch: None)
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    return monkeypatch


EMBED = 4096 * 32 * 2
LM_HEAD = 2000 * 7 // 5 + 16 * 2 + 1024 * 7 // 5           # trellis and svh carry the drafter slice's extra
LAYER0 = 800 + 16 * 2 + 256 * 2 + 1200 + 16 * 2 + 256 * 2


def test_exl3_staging_counts_group_parts_only():
    info = {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]}
    assert exl3_staging("model.language_model.layers.0.mlp.gate_proj.trellis", info, 10) == 10
    assert exl3_staging("lm_head.svh", info, 10) == 10
    assert exl3_staging("model.language_model.embed_tokens.weight", info, 10) == 0
    assert exl3_staging("model.language_model.layers.0.input_layernorm.weight", info, 10) == 0


def test_default_staging_is_three_times_the_largest_layer_or_tensor(exl3_pack):
    plain = capacity.estimate_weights(exl3_pack, exl3_weights)
    assert plain.staging == 3 * max(LAYER0, EMBED, 2000 * 7 // 5) == 3 * EMBED
    assert plain.host_staging == 0


def test_tight_staging_bills_one_load_unit(exl3_pack):
    tight = capacity.estimate_weights(exl3_pack, exl3_weights, transient=exl3_staging)
    loose = capacity.estimate_weights(exl3_pack, exl3_weights)
    assert (tight.resident, tight.mapped) == (loose.resident, loose.mapped)
    assert LM_HEAD > LAYER0 > 2000 * 7 // 5
    assert tight.staging == LM_HEAD                         # the head's three parts as one unit
    assert tight.host_staging == 3 * EMBED                  # the host check keeps the default bill


def test_a_negative_staging_estimate_refuses(exl3_pack):
    with pytest.raises(ValueError, match="negative"):
        capacity.estimate_weights(exl3_pack, exl3_weights, transient=lambda n, i, s: -1)
    assert capacity.estimate_weights(exl3_pack, exl3_weights, transient=lambda n, i, s: 0).staging == 0


def test_admission_bills_the_tight_staging(exl3_pack, isolated):
    isolated.setattr(capacity, "available_bytes", lambda torch: 2 * GIB)
    geometry = capacity.Geometry(lambda slots: slots * 64, 0)
    tight = capacity.admit(exl3_pack, 4096, True, None, geometry, exl3_weights, transient=exl3_staging)
    loose = capacity.admit(exl3_pack, 4096, True, None, geometry, exl3_weights)
    assert (tight["loading_bytes_estimate"], loose["loading_bytes_estimate"]) == (LM_HEAD, 3 * EMBED)


def test_tight_staging_with_a_drafter_keeps_the_default_host_check(exl3_pack, isolated):
    isolated.setattr(capacity, "available_bytes", lambda torch: 2 * GIB)
    args = (exl3_pack, 4096, True, None, capacity.Geometry(lambda slots: slots * 64, 0), exl3_weights)
    drafter = {"draft_dir": exl3_pack, "draft_weights": lambda folder: capacity.Weights(EMBED, EMBED)}
    receipt = capacity.admit(*args, transient=exl3_staging, **drafter)
    assert receipt["loading_bytes_estimate"] == max(LM_HEAD - EMBED, EMBED) == EMBED   # the drafter's load is larger
    isolated.setattr(capacity, "_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 2 * GIB + 3 * EMBED - 1})
    with pytest.raises(ValueError, match="host staging"):     # a drafter's tensors wait on the host until they pack
        capacity.admit(*args, transient=exl3_staging, **drafter)
    isolated.setattr(capacity, "_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 2 * GIB + 3 * EMBED})
    assert capacity.admit(*args, transient=exl3_staging, **drafter)["context_window"] == 4096


def test_tight_staging_needs_the_variable_and_a_discrete_gpu(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_TIGHT_STAGING", raising=False)
    monkeypatch.setattr(capacity, "unified", lambda torch: False)
    assert tight_staging(None) is None                      # unset: three times the largest layer or tensor
    monkeypatch.setenv("TENSORFOLD_TIGHT_STAGING", "1")
    assert tight_staging(None) is exl3_staging
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    assert tight_staging(None) is None                      # one pool holds the raw read and the copy
