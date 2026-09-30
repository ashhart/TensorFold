"""Versioned storage descriptor for the DeepSeek-V4-Flash 0731 donor GGUF.

This module maps the *audited* 0731 donor tensor schema to a versioned
descriptor (``DeepSeekV4GGUFSchema``) and validates a parsed tensor inventory
against it without touching the stored payloads.

Design notes
============
* The donor GGUF stores tensors under ggml-style names (``token_embd.weight``,
  ``blk.{layer}.attn_q_a.weight``, ...) and per-tensor quant types such as
  IQ2_XXS, Q2_K, Q8_0, F16, F32.
* Validation is descriptor-only: it inspects tensor *names*, *ranks* and
  *quant type names* and cross-checks the ``deepseek4.*`` architecture metadata.
  It never dequantizes, expands, or rewrites payloads, so the actual
  IQ2_XXS/Q2_K/Q8_0/F16/F32 data is preserved exactly.
* A schema is versioned so a later donor layout can be mapped to a new version
  instead of silently changing this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# architecture metadata key map (canonical -> gguf key)
# ---------------------------------------------------------------------------

ARCH_KEYS: dict[str, str] = {
    "block_count": "deepseek4.block_count",
    "embedding_length": "deepseek4.embedding_length",
    "head_count": "deepseek4.attention.head_count",
    "head_count_kv": "deepseek4.attention.head_count_kv",
    "key_length": "deepseek4.attention.key_length",
    "q_lora_rank": "deepseek4.attention.q_lora_rank",
    "output_lora_rank": "deepseek4.attention.output_lora_rank",
    "expert_count": "deepseek4.expert_count",
    "expert_used_count": "deepseek4.expert_used_count",
    "expert_feed_forward_length": "deepseek4.expert_feed_forward_length",
    "sliding_window": "deepseek4.attention.sliding_window",
}

REQUIRED_ARCH = (
    "block_count",
    "embedding_length",
    "head_count",
    "head_count_kv",
    "key_length",
    "q_lora_rank",
    "output_lora_rank",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
)

# Donor quant types the 0731 layout stores. Validation only checks the tensor's
# own type name is in the spec's allowlist; the payload is left untouched.
DONOR_QUANTS = frozenset({"IQ2_XXS", "Q2_K", "Q8_0", "F16", "F32", "I32"})


class DeepSeekV4SchemaError(ValueError):
    """A tensor inventory does not match the versioned 0731 donor schema."""


@dataclass(frozen=True)
class TensorSpec:
    """Expected donor tensor, named with a ``{layer}`` placeholder when per-layer."""

    name: str                       # donor gguf tensor name
    canonical: str                  # TensorFold canonical alias
    ndims: int                      # expected rank
    quant: frozenset[str]           # allowed type_name values
    shape: tuple[int, ...] | None = None   # exact shape if fully known

    def layer_name(self, layer: int) -> str:
        return self.name.format(layer=layer)


@dataclass(frozen=True)
class DeepSeekV4GGUFSchema:
    version: str
    arch: dict[str, str]
    required_arch: tuple[str, ...]
    global_tensors: tuple[TensorSpec, ...]
    layer_tensors: tuple[TensorSpec, ...]


# ---------------------------------------------------------------------------
# the v1 donor descriptor (0731)
# ---------------------------------------------------------------------------

def schema_v1() -> DeepSeekV4GGUFSchema:
    """The audited 0731 donor schema (version 1)."""
    G = TensorSpec
    return DeepSeekV4GGUFSchema(
        version="1",
        arch=dict(ARCH_KEYS),
        required_arch=tuple(REQUIRED_ARCH),
        global_tensors=(
            G("token_embd.weight", "embed", 2, frozenset({"F32", "F16"})),
            G("output_norm.weight", "norm", 1, frozenset({"F32", "F16"})),
            G("output.weight", "output", 2, frozenset({"F32", "Q8_0", "F16"})),
        ),
        layer_tensors=(
            G("blk.{layer}.attn_norm.weight", "attn_norm", 1, frozenset({"F32", "F16"})),
            G("blk.{layer}.attn_q_a.weight", "attn_q_a", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.attn_q_b.weight", "attn_q_b", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.attn_kv.weight", "attn_kv", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.attn_output_a.weight", "attn_output_a", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.attn_output_b.weight", "attn_output_b", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.ffn_norm.weight", "ffn_norm", 1, frozenset({"F32", "F16"})),
            G("blk.{layer}.ffn_gate_inp.weight", "router", 2,
              frozenset({"F16", "F32", "Q8_0"})),
            G("blk.{layer}.ffn_gate_shexp.weight", "ffn_gate", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.ffn_up_shexp.weight", "ffn_up", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.ffn_down_shexp.weight", "ffn_down", 2,
              frozenset({"Q8_0", "F16", "F32"})),
            G("blk.{layer}.ffn_gate_exps.weight", "exps_gate", 3,
              frozenset(DONOR_QUANTS)),
            G("blk.{layer}.ffn_up_exps.weight", "exps_up", 3,
              frozenset(DONOR_QUANTS)),
            G("blk.{layer}.ffn_down_exps.weight", "exps_down", 3,
              frozenset(DONOR_QUANTS)),
        ),
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class TensorInventory(Protocol):
    """The subset of a parsed GGUF inventory the validator reads."""

    def by_name(self, name: str) -> Any | None: ...      # tensor with .type_name/.shape
    def count_prefix(self, prefix: str) -> int: ...      # layer tensors with prefix


class _AdaptedInventory:
    """Wrap a real ``GGUFTensorInventory`` (from tensorfold.gguf) for the validator."""

    def __init__(self, tensors) -> None:
        self._by_name = {t.name: t for t in tensors}

    def by_name(self, name: str) -> Any | None:
        return self._by_name.get(name)

    def count_prefix(self, prefix: str) -> int:
        return sum(1 for n in self._by_name if n.startswith(prefix))


def adapt_inventory(inventory: TensorInventory | Any) -> TensorInventory:
    """Return an object exposing ``by_name``/``count_prefix``.

    ``tensorfold.gguf.GGUFTensorInventory`` has a ``.tensors`` tuple; anything
    already exposing ``by_name`` is passed through unchanged.
    """
    if hasattr(inventory, "by_name"):
        return inventory
    tensors = getattr(inventory, "tensors", None)
    if tensors is None:
        raise TypeError(
            "inventory must expose by_name/count_prefix or a .tensors collection")
    return _AdaptedInventory(tensors)


@dataclass
class SchemaReport:
    errors: list[str] = None  # type: ignore[assignment]
    missing: list[str] = None  # type: ignore[assignment]
    wrong_dimension: list[str] = None  # type: ignore[assignment]
    wrong_quant: list[str] = None  # type: ignore[assignment]
    arch_errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.missing is None:
            self.missing = []
        if self.wrong_dimension is None:
            self.wrong_dimension = []
        if self.wrong_quant is None:
            self.wrong_quant = []
        if self.arch_errors is None:
            self.arch_errors = []


def _read_arch(schema: DeepSeekV4GGUFSchema, arch: dict[str, Any],
               report: SchemaReport) -> dict[str, int]:
    params: dict[str, int] = {}
    for canon in schema.required_arch:
        key = schema.arch[canon]
        value = arch.get(key)
        if value is None:
            report.arch_errors.append(f"missing architecture metadata {key!r}")
            continue
        try:
            params[canon] = int(value)
        except (TypeError, ValueError):
            report.arch_errors.append(
                f"non-integer architecture metadata {key!r}: {value!r}")
    return params


def _check_tensor(spec: TensorSpec, layer: int | None, tensor: Any, name: str,
                  report: SchemaReport) -> None:
    if tensor is None:
        report.missing.append(name)
        return
    rank = len(getattr(tensor, "shape", ()))
    if rank != spec.ndims:
        report.wrong_dimension.append(
            f"{name}: wrong dimension: expected rank {spec.ndims}, got {rank}")
        return
    type_name = getattr(tensor, "type_name", None) or "?"
    if type_name not in spec.quant:
        report.wrong_quant.append(
            f"{name}: quant {type_name!r} not in allowed {sorted(spec.quant)!r}")


def _validate_layer(schema: DeepSeekV4GGUFSchema, inventory: TensorInventory,
                    layer: int, report: SchemaReport) -> None:
    for spec in schema.layer_tensors:
        name = spec.layer_name(layer)
        _check_tensor(spec, layer, inventory.by_name(name), name, report)


def _validate_arch_cross(schema: DeepSeekV4GGUFSchema, inventory: TensorInventory,
                         params: dict[str, int], report: SchemaReport) -> None:
    emb = inventory.by_name("token_embd.weight")
    if emb is not None and "embedding_length" in params:
        shape = getattr(emb, "shape", ())
        if shape and shape[0] != params["embedding_length"]:
            report.arch_errors.append(
                f"token_embd.weight dim0 {shape[0]} != "
                f"deepseek4.embedding_length {params['embedding_length']}")
    if "block_count" in params:
        n = params["block_count"]
        # every required layer tensor should appear once per layer; the layer
        # loop below already reports missing per-layer names. Here we only
        # flag a contradictory *count* when the first layer tensor's occurrence
        # count disagrees with block_count.
        first = schema.layer_tensors[0].name.format(layer=0)
        prefix = f"blk."
        seen = inventory.count_prefix(prefix)
        expected = len(schema.layer_tensors) * n
        if seen < expected:
            report.arch_errors.append(
                f"layer tensor count {seen} < expected {expected} "
                f"(block_count={n})")


def validate_tensor_schema(schema: DeepSeekV4GGUFSchema, inventory: TensorInventory,
                           arch: dict[str, Any]) -> SchemaReport:
    report = SchemaReport()
    params = _read_arch(schema, arch, report)

    for spec in schema.global_tensors:
        _check_tensor(spec, None, inventory.by_name(spec.name), spec.name, report)

    if "block_count" in params:
        for layer in range(params["block_count"]):
            _validate_layer(schema, inventory, layer, report)
        _validate_arch_cross(schema, inventory, params, report)

    report.errors = list(report.missing) + list(report.wrong_dimension) \
        + list(report.wrong_quant) + list(report.arch_errors)
    return report


def validate_deepseek_v4_gguf(inventory: TensorInventory | Any, arch: dict[str, Any],
                              schema: DeepSeekV4GGUFSchema | None = None,
                              ) -> SchemaReport:
    """Validate a parsed 0731 donor inventory against the versioned schema.

    Descriptor-only: reads names, ranks and quant type names; never mutates the
    stored IQ2_XXS/Q2_K/Q8_0/F16/F32 payloads. Accepts either an object already
    exposing ``by_name``/``count_prefix`` or a real
    ``tensorfold.gguf.GGUFTensorInventory``.
    """
    schema = schema or schema_v1()
    return validate_tensor_schema(schema, adapt_inventory(inventory), dict(arch))


def validate_deepseek_v4_gguf_or_raise(inventory: TensorInventory,
                                       arch: dict[str, Any],
                                       schema: DeepSeekV4GGUFSchema | None = None,
                                       ) -> SchemaReport:
    report = validate_deepseek_v4_gguf(inventory, arch, schema=schema)
    if report.errors:
        raise DeepSeekV4SchemaError("; ".join(report.errors))
    return report
