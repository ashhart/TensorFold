"""Admission uses donor estimates and refuses before the native session starts."""

import json
from unittest.mock import patch

import pytest

from tensorfold.families.deepseek_v4.cuda import capacity
from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine

GIB = 1 << 30


def model(tmp_path):
    config = {
        "model_type": "deepseek_v4",
        "quantization_config": {"quant_method": "gguf"},
        "gguf_file": "/mock.gguf",
        "native_library": "/mock.so",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "descriptor.json").write_text(json.dumps({"reserve_gib": 2}))
    return {
        "aligned_artifact_extra_bytes": GIB,
        "source_size": 10 * GIB,
        "arch": {"deepseek4.context_length": 262144},
        "header_sha256": "header",
        "source_identity": {},
    }


def test_admitted_context_counts_growth_once(tmp_path):
    report = model(tmp_path)
    with (
        patch.object(capacity, "inspect_inputs", return_value=report),
        patch.object(capacity, "estimate", return_value={"graph_bytes": 6 * GIB}),
        patch.object(capacity, "available", return_value=32 * GIB),
    ):
        plan = capacity.admit(tmp_path, 262144, True)
    assert plan["context_window"] == plan["cache_slots"] == 262144
    assert plan["required_bytes"] == 28 * GIB
    assert plan["companion_growth_bytes"] == 2 * GIB


def test_refusal_never_starts_session_and_preserves_explicit_context(tmp_path):
    report = model(tmp_path)
    with (
        patch.object(capacity, "inspect_inputs", return_value=report),
        patch.object(capacity, "estimate", return_value={"graph_bytes": 6 * GIB}),
        patch.object(capacity, "available", return_value=20 * GIB),
        pytest.raises(ValueError, match="cannot fit.*262144"),
    ):
        DeepSeekEngine(tmp_path, context=262144, _session_factory=lambda **_: pytest.fail("model load reached"))
    with (
        patch.object(capacity, "available", side_effect=ValueError("memory unavailable")),
        patch.object(capacity, "estimate", side_effect=AssertionError("native reached")),
        pytest.raises(ValueError, match="memory unavailable"),
    ):
        capacity.admit(tmp_path, 262144, True)
