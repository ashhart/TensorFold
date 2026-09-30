"""Tensor-schema / storage-descriptor tests for DeepSeek-V4-Flash 0731 (T03.01).

The 0731 GGUF is a donor layout. This scope maps that audited donor tensor schema
to a versioned storage descriptor (src/tensorfold/families/deepseek_v4/gguf.py)
and validates a parsed tensor inventory against it:

  * missing tensor
  * wrong dimension (rank / shape)
  * unsupported quant
  * contradictory architecture metadata

The validator only inspects stored descriptors: it never expands or rewrites the
actual IQ2_XXS/Q2_K/Q8_0/F16/F32 payloads, so the donor data is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tensorfold.families.deepseek_v4.gguf import (
    DeepSeekV4GGUFSchema,
    SchemaReport,
    schema_v1,
    validate_deepseek_v4_gguf,
)

# ---------------------------------------------------------------------------
# oracle helpers: a tiny, independent inventory object the validator reads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeTensor:
    name: str
    type_id: int = 0
    type_name: str = "F32"
    shape: tuple[int, ...] = ()


@dataclass
class FakeInventory:
    tensors: list[FakeTensor] = field(default_factory=list)

    def by_name(self, name: str) -> FakeTensor | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def count_prefix(self, prefix: str) -> int:
        return sum(1 for t in self.tensors if t.name.startswith(prefix))


def make_arch(**overrides) -> dict[str, object]:
    """Architecture metadata; *overrides* use full gguf keys (e.g. ``deepseek4.block_count``)."""
    base = {
        "deepseek4.block_count": 1,
        "deepseek4.embedding_length": 4096,
        "deepseek4.attention.head_count": 64,
        "deepseek4.attention.head_count_kv": 8,
        "deepseek4.attention.key_length": 512,
        "deepseek4.attention.q_lora_rank": 1536,
        "deepseek4.attention.output_lora_rank": 1024,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.expert_feed_forward_length": 2048,
        "deepseek4.attention.sliding_window": 128,
    }
    base.update(overrides)
    return base


def _base_tensors() -> list[FakeTensor]:
    """A valid donor inventory for one layer (block_count=1)."""
    L = 0
    return [
        FakeTensor("token_embd.weight", type_name="F16", shape=(4096, 129280)),
        FakeTensor("output_norm.weight", type_name="F32", shape=(4096,)),
        FakeTensor("output.weight", type_name="Q8_0", shape=(4096, 129280)),
        FakeTensor(f"blk.{L}.attn_norm.weight", type_name="F32", shape=(4096,)),
        FakeTensor(f"blk.{L}.attn_q_a.weight", type_name="Q8_0", shape=(4096, 1536)),
        FakeTensor(f"blk.{L}.attn_q_b.weight", type_name="Q8_0", shape=(1536, 32768)),
        FakeTensor(f"blk.{L}.attn_kv.weight", type_name="Q8_0", shape=(4096, 512)),
        FakeTensor(f"blk.{L}.attn_output_a.weight", type_name="Q8_0", shape=(512, 1024)),
        FakeTensor(f"blk.{L}.attn_output_b.weight", type_name="Q8_0", shape=(1024, 4096)),
        FakeTensor(f"blk.{L}.ffn_norm.weight", type_name="F32", shape=(4096,)),
        FakeTensor(f"blk.{L}.ffn_gate_inp.weight", type_name="F16", shape=(4096, 256)),
        FakeTensor(f"blk.{L}.ffn_gate_shexp.weight", type_name="Q8_0", shape=(4096, 2048)),
        FakeTensor(f"blk.{L}.ffn_up_shexp.weight", type_name="Q8_0", shape=(4096, 2048)),
        FakeTensor(f"blk.{L}.ffn_down_shexp.weight", type_name="Q8_0", shape=(2048, 4096)),
        FakeTensor(f"blk.{L}.ffn_gate_exps.weight", type_name="IQ2_XXS", shape=(4096, 2048, 256)),
        FakeTensor(f"blk.{L}.ffn_up_exps.weight", type_name="Q2_K", shape=(4096, 2048, 256)),
        FakeTensor(f"blk.{L}.ffn_down_exps.weight", type_name="Q8_0", shape=(2048, 4096, 256)),
    ]


def _replace(base: list[FakeTensor], name: str, type_name: str,
             shape: tuple[int, ...]) -> list[FakeTensor]:
    """Return a copy of *base* with the named tensor replaced."""
    out = []
    for t in base:
        if t.name == name:
            out.append(FakeTensor(name, type_name=type_name, shape=shape))
        else:
            out.append(t)
    assert any(t.name == name for t in out), f"{name} not present"
    return out


def _report(schema: DeepSeekV4GGUFSchema, inventory, arch: dict) -> SchemaReport:
    return validate_deepseek_v4_gguf(inventory, arch, schema=schema)


# ---------------------------------------------------------------------------
# schema factory
# ---------------------------------------------------------------------------

def test_schema_v1_is_versioned_and_complete():
    schema = schema_v1()
    assert schema.version == "1"
    assert schema.required_arch
    assert schema.global_tensors
    assert schema.layer_tensors
    names = {t.name for t in schema.global_tensors}
    assert "token_embd.weight" in names
    layer_names = {t.name for t in schema.layer_tensors}
    assert "blk.{layer}.attn_q_a.weight" in layer_names


# ---------------------------------------------------------------------------
# valid inventory -> no errors
# ---------------------------------------------------------------------------

def test_valid_inventory_reports_no_errors():
    schema = schema_v1()
    inv = FakeInventory(_base_tensors())
    report = _report(schema, inv, make_arch())
    assert report.errors == []
    assert report.missing == []
    assert report.arch_errors == []


# ---------------------------------------------------------------------------
# missing tensor
# ---------------------------------------------------------------------------

def test_missing_layer_tensor_is_reported():
    schema = schema_v1()
    tensors = [t for t in _base_tensors() if t.name != "blk.0.ffn_gate_exps.weight"]
    report = _report(schema, FakeInventory(tensors), make_arch())
    assert any("ffn_gate_exps.weight" in e and "blk.0" in e for e in report.missing)


def test_missing_global_tensor_is_reported():
    schema = schema_v1()
    tensors = [t for t in _base_tensors() if t.name != "token_embd.weight"]
    report = _report(schema, FakeInventory(tensors), make_arch())
    assert any("token_embd.weight" in e for e in report.missing)


# ---------------------------------------------------------------------------
# wrong dimension (rank / shape)
# ---------------------------------------------------------------------------

def test_wrong_rank_is_reported_as_dimension_error():
    schema = schema_v1()
    inv = FakeInventory(_replace(_base_tensors(), "blk.0.attn_q_a.weight",
                                 "Q8_0", shape=(4096, 1536, 1)))
    report = _report(schema, inv, make_arch())
    assert any("dimension" in e.lower() and "attn_q_a" in e for e in report.errors)


def test_wrong_embedding_dim_is_contradictory_architecture():
    schema = schema_v1()
    # token_embd first dim disagrees with deepseek4.embedding_length
    inv = FakeInventory([
        FakeTensor("token_embd.weight", type_name="F16", shape=(2048, 129280)),
        *[t for t in _base_tensors() if t.name != "token_embd.weight"],
    ])
    report = _report(schema, inv, make_arch(embedding_length=4096))
    assert any("embedding_length" in e or "embedding" in e for e in report.arch_errors)


# ---------------------------------------------------------------------------
# unsupported quant
# ---------------------------------------------------------------------------

def test_unsupported_quant_is_reported():
    schema = schema_v1()
    # attn_q_a is expected Q8_0/F16/F32; store Q4_0
    inv = FakeInventory(_replace(_base_tensors(), "blk.0.attn_q_a.weight",
                                 "Q4_0", shape=(4096, 1536)))
    report = _report(schema, inv, make_arch())
    assert any("quant" in e.lower() and "attn_q_a" in e for e in report.errors)


def test_expert_quant_allowlist_includes_donor_types():
    schema = schema_v1()
    inv = FakeInventory(_base_tensors())
    report = _report(schema, inv, make_arch())
    assert report.errors == []          # IQ2_XXS, Q2_K, Q8_0 are donor-allowed


# ---------------------------------------------------------------------------
# contradictory architecture metadata
# ---------------------------------------------------------------------------

def test_block_count_mismatch_is_contradictory_architecture():
    schema = schema_v1()
    # one layer's tensors present, but block_count claims 2 -> missing layer 1
    inv = FakeInventory(_base_tensors())
    report = _report(schema, inv, make_arch(**{"deepseek4.block_count": 2}))
    assert any("blk.1" in e for e in report.missing)


def test_missing_required_arch_key_is_reported():
    schema = schema_v1()
    arch = make_arch()
    del arch["deepseek4.attention.q_lora_rank"]
    report = _report(schema, FakeInventory(_base_tensors()), arch)
    assert any("q_lora_rank" in e for e in report.arch_errors)


def test_non_integer_arch_value_is_reported():
    schema = schema_v1()
    report = _report(schema, FakeInventory(_base_tensors()),
                     make_arch(**{"deepseek4.block_count": "two"}))
    assert any("block_count" in e for e in report.arch_errors)


# ---------------------------------------------------------------------------
# integration: real tensorfold.gguf.GGUFTensorInventory through the adapter
# ---------------------------------------------------------------------------

import struct

from tensorfold.gguf import (GGUF_TYPE_UINT32, GGUF_TYPE_UINT64, GGUF_MAGIC,
                             parse_gguf_tensors)

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 6


def _s(value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def _scalar(type_, value) -> bytes:
    if type_ == GGUF_TYPE_UINT32:
        return struct.pack("<I", value)
    if type_ == GGUF_TYPE_UINT64:
        return struct.pack("<Q", value)
    raise AssertionError


def _tensor_desc(name: str, dims: list, type_id: int, offset: int) -> bytes:
    return (_s(name)
            + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<I", type_id)
            + struct.pack("<Q", offset))


def test_real_gguf_inventory_is_adapted_and_validated():
    # Build a real GGUF with global tensors token_embd + output_norm and the
    # deepseek4.* metadata, then validate the parsed inventory through the
    # adapter. Layer tensors are absent, so they must be reported missing.
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, 2, 10)
    meta = b"".join([
        _s("deepseek4.block_count") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 1),
        _s("deepseek4.embedding_length") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 4096),
        _s("deepseek4.attention.head_count") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 64),
        _s("deepseek4.attention.head_count_kv") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 8),
        _s("deepseek4.attention.key_length") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 512),
        _s("deepseek4.attention.q_lora_rank") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 1536),
        _s("deepseek4.attention.output_lora_rank") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 1024),
        _s("deepseek4.expert_count") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 256),
        _s("deepseek4.expert_used_count") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 6),
        _s("deepseek4.expert_feed_forward_length") + bytes([GGUF_TYPE_UINT32]) + _scalar(GGUF_TYPE_UINT32, 2048),
    ])
    descs = [
        _tensor_desc("token_embd.weight", [8, 4], GGML_F16, 0),
        _tensor_desc("output_norm.weight", [8], GGML_F32, 64),
    ]
    data = bytes(((-len(header + meta + b"".join(descs))) % 32)) + bytes(64 + 32)
    inv = parse_gguf_tensors(header + meta + b"".join(descs) + data)

    arch = {"deepseek4.block_count": 1, "deepseek4.embedding_length": 4096,
            "deepseek4.attention.head_count": 64,
            "deepseek4.attention.head_count_kv": 8,
            "deepseek4.attention.key_length": 512,
            "deepseek4.attention.q_lora_rank": 1536,
            "deepseek4.attention.output_lora_rank": 1024,
            "deepseek4.expert_count": 256, "deepseek4.expert_used_count": 6,
            "deepseek4.expert_feed_forward_length": 2048}
    report = validate_deepseek_v4_gguf(inv, arch)
    assert "token_embd.weight" not in [e for e in report.errors]
    # token_embd and output_norm present; every layer tensor missing
    assert any("blk.0" in e for e in report.missing) or any("blk.1" in e for e in report.missing)


# ---------------------------------------------------------------------------
# T03.02: pinned tokenizer/config provenance validation
# ---------------------------------------------------------------------------

from tensorfold.families.deepseek_v4.gguf import (
    DeepSeekV4TokenizerProvenance,
    TokenizerReport,
    tokenizer_provenance_v1,
    validate_tokenizer_provenance,
    validate_tokenizer_provenance_or_raise,
)

PINNED = tokenizer_provenance_v1()

def _candidate(**overrides) -> dict:
    """A candidate tokenizer/config metadata dict (the kind preparation generates)."""
    base = {
        "checkpoint": PINNED.checkpoint,
        "vocab_size": PINNED.vocab_size,
        "tokenizer_type": PINNED.tokenizer_type,
        "eos_tokens": list(PINNED.eos_tokens),
        "provenance": {"source": "/home/josh/gguf/pinned-0731.gguf", "sha256": "a" * 64},
    }
    base.update(overrides)
    return base


def test_provenance_pins_audited_0731_facts():
    assert PINNED.version == "1"
    assert PINNED.checkpoint == "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
    assert PINNED.vocab_size == 129280
    assert PINNED.tokenizer_type == "joyai-llm"
    assert PINNED.eos_tokens == ("",)
    assert PINNED.provenance_keys


def test_valid_candidate_reports_no_errors():
    report = validate_tokenizer_provenance(_candidate())
    assert report.errors == []


def test_wrong_vocabulary_is_rejected():
    report = validate_tokenizer_provenance(_candidate(vocab_size=129281))
    assert any("vocab" in e.lower() for e in report.errors)


def test_wrong_eos_tokens_are_rejected():
    report = validate_tokenizer_provenance(_candidate(eos_tokens=["<|EOS|>"]))
    assert any("eos" in e.lower() for e in report.errors)


def test_tokenizer_type_mismatch_is_rejected():
    report = validate_tokenizer_provenance(_candidate(tokenizer_type="gpt2"))
    assert any("tokenizer" in e.lower() for e in report.errors)


def test_missing_provenance_is_rejected():
    report = validate_tokenizer_provenance(_candidate(provenance=None))
    assert any("provenance" in e.lower() for e in report.errors)
    report2 = validate_tokenizer_provenance(_candidate(provenance={}))
    assert any("provenance" in e.lower() for e in report2.errors)


def test_changed_checkpoint_identity_is_rejected():
    report = validate_tokenizer_provenance(_candidate(checkpoint="Some-Other-Model.gguf"))
    assert any("checkpoint" in e.lower() for e in report.errors)


def test_raise_variant_raises_on_incompatible_input():
    with pytest.raises(ValueError):
        validate_tokenizer_provenance_or_raise(_candidate(vocab_size=1))
    # valid input returns the report
    ok = validate_tokenizer_provenance_or_raise(_candidate())
    assert ok.errors == []
