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

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
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


# ---------------------------------------------------------------------------
# pinned tokenizer/config provenance validation (T03.02)
# ---------------------------------------------------------------------------

# Pinned facts are grounded in the audited 0731 donor: tokenizer.ggml.pre is
# "joyai-llm", the vocabulary is 129280 entries, and the EOS token is the empty
# string (ds4's vocab_load looks up `""` for eos_id).

CHECKPOINT_0731 = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
VOCAB_SIZE_0731 = 129280
TOKENIZER_TYPE_0731 = "joyai-llm"
EOS_TOKENS_0731 = ("",)

PROVENANCE_KEYS = ("source", "sha256")


@dataclass(frozen=True)
class DeepSeekV4TokenizerProvenance:
    """The pinned provenance facts a candidate tokenizer/config must match."""

    version: str
    checkpoint: str
    vocab_size: int
    tokenizer_type: str
    eos_tokens: tuple[str, ...]
    provenance_keys: tuple[str, ...]


def tokenizer_provenance_v1() -> DeepSeekV4TokenizerProvenance:
    return DeepSeekV4TokenizerProvenance(
        version="1",
        checkpoint=CHECKPOINT_0731,
        vocab_size=VOCAB_SIZE_0731,
        tokenizer_type=TOKENIZER_TYPE_0731,
        eos_tokens=EOS_TOKENS_0731,
        provenance_keys=tuple(PROVENANCE_KEYS),
    )


@dataclass
class TokenizerReport:
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def validate_tokenizer_provenance(tokenizer: dict[str, Any],
                                  prov: DeepSeekV4TokenizerProvenance | None = None,
                                  ) -> TokenizerReport:
    """Reject a candidate tokenizer/config metadata dict that is incompatible.

    ``tokenizer`` carries the fields preparation records: checkpoint identity,
    vocab_size, tokenizer_type, eos_tokens, and a provenance record (source
    path + sha256 + checkpoint). A missing provenance record or a mismatch
    against the pinned 0731 facts is reported as an error.
    """
    prov = prov or tokenizer_provenance_v1()
    report = TokenizerReport()

    provenance = tokenizer.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        report.errors.append("missing provenance record for tokenizer/config")
    else:
        for key in prov.provenance_keys:
            if not provenance.get(key):
                report.errors.append(f"missing provenance field {key!r}")

    checkpoint = tokenizer.get("checkpoint")
    if checkpoint != prov.checkpoint:
        report.errors.append(
            f"changed checkpoint identity: {checkpoint!r} != {prov.checkpoint!r}")

    vocab_size = tokenizer.get("vocab_size")
    if vocab_size != prov.vocab_size:
        report.errors.append(
            f"wrong vocabulary size: {vocab_size!r} != {prov.vocab_size!r}")

    eos_tokens = tokenizer.get("eos_tokens")
    if tuple(eos_tokens or ()) != prov.eos_tokens:
        report.errors.append(
            f"wrong EOS tokens: {eos_tokens!r} != {prov.eos_tokens!r}")

    tokenizer_type = tokenizer.get("tokenizer_type")
    if tokenizer_type != prov.tokenizer_type:
        report.errors.append(
            f"tokenizer mismatch: {tokenizer_type!r} != {prov.tokenizer_type!r}")

    return report


def validate_tokenizer_provenance_or_raise(tokenizer: dict[str, Any],
                                           prov: DeepSeekV4TokenizerProvenance | None = None,
                                           ) -> TokenizerReport:
    report = validate_tokenizer_provenance(tokenizer, prov=prov)
    if report.errors:
        raise DeepSeekV4SchemaError("; ".join(report.errors))
    return report


# ---------------------------------------------------------------------------
# T03.03: atomic sidecars and family discovery
# ---------------------------------------------------------------------------

MODEL_TYPE = "deepseek_v4"
SIDECAR_VERSION = "1"


class PrepareConflictError(ValueError):
    """Existing candidate sidecars conflict with the requested preparation."""


@dataclass
class PrepareReport:
    config_path: Path
    descriptor_path: Path
    changed: bool
    replaced: bool
    descriptor_digest: str


def _canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes so identical inputs yield identical sidecars."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to *path* through a same-directory temp file + rename.

    Any stale temp left by an earlier interrupted write is removed first, so a
    partial temp never becomes the published file. On failure the temp is
    unlinked; the final path is untouched (atomic publish).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)          # clean a temp from an interrupted write
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_existing(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _descriptor_digest(desc: dict[str, Any]) -> str:
    import hashlib
    # digest over the canonical descriptor without its own digest field
    body = {k: v for k, v in desc.items() if k != "descriptor_digest"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def prepare_candidate(model_dir: str | Path, *,
                      arch: dict[str, Any],
                      tokenizer: dict[str, Any],
                      source: str | Path,
                      source_size: int,
                      source_sha256: str,
                      reserve_gib: float,
                      replace: bool = False,
                      ) -> PrepareReport:
    """Generate candidate ``config.json`` + ``descriptor.json`` atomically and idempotently.

    ``arch`` is the architecture settings reproduced from the checkpoint's GGUF
    metadata; ``tokenizer`` is the tokenizer/config provenance record (validated
    against the pinned 0731 facts). ``source_size``/``source_sha256``/``reserve_gib``
    are measured facts the caller records. Existing sidecars that exactly match the
    desired output are left untouched (idempotent); incompatible existing output
    requires ``replace=True``.
    """
    model_dir = Path(model_dir)
    validate_tokenizer_provenance_or_raise(tokenizer)
    reserve_gib = float(reserve_gib)
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError(f"companion reserve must be a finite non-negative Gib, got {reserve_gib!r}")

    config = {"model_type": MODEL_TYPE, "text_config": dict(arch),
              "quantization_config": {"quant_method": "gguf"}, "gguf_file": str(source)}
    config_path = model_dir / "config.json"
    descriptor_path = model_dir / "descriptor.json"

    desc: dict[str, Any] = {
        "version": SIDECAR_VERSION,
        "source": str(source),
        "size": int(source_size),
        "sha256": str(source_sha256),
        "provenance": dict(tokenizer),
        "reserve_gib": reserve_gib,
    }
    digest = _descriptor_digest(desc)
    desc["descriptor_digest"] = digest

    config_bytes = _canonical_json(config)
    desc_bytes = _canonical_json(desc)

    cur_cfg = _read_existing(config_path)
    cur_desc = _read_existing(descriptor_path)

    if cur_cfg == config_bytes and cur_desc == desc_bytes:
        return PrepareReport(config_path, descriptor_path, changed=False, replaced=False,
                             descriptor_digest=digest)

    if (cur_cfg is not None and cur_cfg != config_bytes) or \
       (cur_desc is not None and cur_desc != desc_bytes):
        if not replace:
            raise PrepareConflictError(
                f"candidate sidecars in {model_dir} conflict with requested preparation; "
                "pass replace=True to overwrite")
        replaced = True
    else:
        replaced = False

    _atomic_write(config_path, config_bytes)
    _atomic_write(descriptor_path, desc_bytes)
    return PrepareReport(config_path, descriptor_path, changed=True, replaced=replaced,
                         descriptor_digest=digest)
