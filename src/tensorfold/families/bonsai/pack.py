"""Read and check a Prism Hadamard pack (config.json, hadamard.json) and build its Qwen3.8 dense model."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

MODEL_TYPE = "prism_hadamard_qwen35"
PREFIX = "language_model."
BLOCK = 1024
PACKED = ("weight", "scales", "biases", "signs")
# the pack's unquantized recurrent-layer gates: fp32 row-exact dense projections, never rotated
GATES = ("in_proj_a", "in_proj_b")
# how projections hold codes: M5 2-bit lanes, widened 4-bit, or the pack's own 2-bit; "widened:N" widens N layers
FORMS = ("lanes", "widened", "packed")
ROOM = 12 << 30           # the drafter, one prompt chunk and caches beside the weights (the admission's measure)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def contract(model_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(config, hadamard) of a pack whose format this family reads exactly; anything else is refused by name."""

    folder = Path(model_dir)
    config = _read(folder / "config.json")
    why = refusal(config)
    if why:
        raise ValueError(f"Ternary Bonsai cannot run this checkpoint: {why}")
    if not (folder / "hadamard.json").is_file():
        return config, {}                          # a config-only cache: checked again once the files arrive
    hadamard = _read(folder / "hadamard.json")
    why = transform_refusal(hadamard)
    if why:
        raise ValueError(f"Ternary Bonsai cannot run this checkpoint: {why}")
    return config, hadamard


def refusal(config: dict[str, Any]) -> str | None:
    """Why config.json is outside the Prism contract this family implements, else None."""

    quant = config.get("quantization") or {}
    text = config.get("text_config") or {}
    if config.get("model_type") != MODEL_TYPE or config.get("base_model_type") != "qwen3_5":
        return f"model_type {config.get('model_type')!r} over {config.get('base_model_type')!r} is not {MODEL_TYPE} over qwen3_5"
    if config.get("schema_version") not in (1, 2) or config.get("hadamard_config") not in (None, "hadamard.json"):
        return f"pack schema {config.get('schema_version')!r} is not the version-1 Prism contract"
    if (quant.get("bits"), quant.get("group_size"), quant.get("mode", "affine")) != (2, 128, "affine"):
        return f"weights {quant} are not affine 2-bit in groups of 128"
    if config.get("gdn_activation_layout", "grouped") != "grouped" or (config.get("components") or {}).get("mtp"):
        return "the recurrent layers' activations must be grouped and the pack must have no MTP head"
    if text.get("tie_word_embeddings") or text.get("num_experts") or text.get("model_type") != "qwen3_5_text":
        return "only the untied dense Qwen3.8 text model is supported"
    for record in config.get("modules") or [None]:
        if not isinstance(record, dict) or record.get("dtype") != "float16" or record.get("block") not in (0, BLOCK):
            return f"packed module {record!r} is not an fp16 module with a {BLOCK}-block or no transform"
    return None


def transform_refusal(hadamard: dict[str, Any]) -> str | None:
    """Why hadamard.json is outside the version-1 contract (normalized Sylvester blocks, explicit signs), else None."""

    want = {"version": 1, "block_size": BLOCK, "transform": "normalized-sylvester-walsh-hadamard",
            "axis": "input-last-dimension", "sign_mode": "explicit", "gdn_v_grouped": True}
    for key, value in want.items():
        if hadamard.get(f"prism.hadamard.{key}") != value:
            return f"prism.hadamard.{key} is {hadamard.get(f'prism.hadamard.{key}')!r}, not {value!r}"
    widths, values = hadamard.get("prism.hadamard.sign_widths"), hadamard.get("prism.hadamard.sign_values")
    if not isinstance(widths, list) or not isinstance(values, list) or sum(widths) != len(values):
        return "the sign vectors do not match their widths"
    if any(w % BLOCK for w in widths) or any(v not in (-1, 1) for v in values):
        return f"sign widths must be whole {BLOCK}-blocks and signs must be +1 or -1"
    if hadamard.get("prism.hadamard.inverse_weight_names") != [PREFIX + "model.embed_tokens.weight"]:
        return "only the embedding may take the inverse transform"
    return None


def signs_by_width(hadamard: dict[str, Any]) -> dict[int, list[float]]:
    out, at = {}, 0
    for width in hadamard["prism.hadamard.sign_widths"]:
        out[int(width)] = [float(v) for v in hadamard["prism.hadamard.sign_values"][at:at + width]]
        at += width
    return out


def fingerprint(model_dir: str | Path) -> str:
    """The transform and layout identity a prefix snapshot's bits depend on."""

    folder = Path(model_dir)
    digest = hashlib.sha256()
    for name in ("config.json", "hadamard.json"):
        digest.update(json.dumps(_read(folder / name), sort_keys=True).encode())
    return digest.hexdigest()[:12]


def _layer(path: str) -> int | None:
    parts = path.split(".")
    return int(parts[2]) if parts[:2] == ["model", "layers"] and parts[2].isdigit() else None


def widening(model_dir: str | Path) -> tuple[int, list[int], int]:
    """(the language model's bytes, the widening bytes, the rest's), from the header alone."""

    config, _ = contract(model_dir)
    with open(Path(model_dir) / "model.safetensors", "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    size = {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"}
    layers: dict[int, int] = {}
    rest = 0
    for record in config["modules"]:
        if record["embedding"]:
            continue
        added = sum(size[f"{PREFIX}{record['path']}.{part}"] for part in PACKED[:3])
        index = _layer(record["path"])
        if index is None:
            rest += added
        else:
            layers[index] = layers.get(index, 0) + added
    return sum(v for k, v in size.items() if k.startswith(PREFIX)), [layers[i] for i in sorted(layers)], rest


def sizes(model_dir: str | Path) -> tuple[int, int]:
    """(the language model's bytes, the bytes widening its projections to 4 bits adds), from the header alone."""

    model, layers, rest = widening(model_dir)
    return model, sum(layers) + rest


def pre_m5_form(model_dir: str | Path, budget: int) -> str:
    """Before M5: widen codes to 4 bits where the budget fits: all projections, N layers, or the pack's own."""

    model, layers, rest = widening(model_dir)
    room = budget - model - ROOM
    if sum(layers) + rest <= room:
        return "widened"
    count = 0
    while count < len(layers) and layers[count] <= room:
        room -= layers[count]
        count += 1
    return f"widened:{count}" if count else "packed"


def module_form(form: str, path: str) -> str:
    """A projection's own form: "widened:N" widens the first N decoder layers, the rest stay packed."""

    if not form.startswith("widened:"):
        return form
    index = _layer(path)
    return "widened" if index is not None and index < int(form.split(":")[1]) else "packed"


def build(model_dir: str | Path, *, form: str) -> Any:
    """The mlx_lm Qwen3.8 model with the pack's weights, its projections in ``form`` (FORMS)."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from tensorfold.families.bonsai.modules import RotatedEmbedding, RotatedLinear, RowDense

    if form not in FORMS and not (form.startswith("widened:") and form.split(":")[1].isdigit()):
        raise ValueError(f"Ternary Bonsai: projections are {', '.join(FORMS)} or widened:N, not {form!r}")
    config, hadamard = contract(model_dir)
    if not hadamard:
        raise ValueError("Ternary Bonsai needs hadamard.json beside the weights")
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": dict(config["text_config"])}))
    lm = model.language_model
    tensors = mx.load(str(Path(model_dir) / "model.safetensors"))
    text = {k[len(PREFIX):]: v for k, v in tensors.items() if k.startswith(PREFIX)}
    signs = {w: mx.array(v, dtype=mx.float32) for w, v in signs_by_width(hadamard).items()}
    used: set[str] = set()
    records = {r["path"]: r for r in config["modules"]}
    for path, record in records.items():
        weight, scales, biases, stored = (text[f"{path}.{part}"] for part in PACKED)
        used.update(f"{path}.{part}" for part in PACKED)
        width = int(weight.shape[1]) * 16
        if record["block"] != BLOCK or width not in signs or not mx.array_equal(stored, signs[width]).item():
            raise ValueError(f"{path}: its signs are not hadamard.json's {width}-wide signs")
        parent, leaf = _parent(lm, path)
        if record["embedding"]:
            module: Any = RotatedEmbedding(weight, scales, biases, signs[width], 128)
        else:
            module = RotatedLinear(_linear(weight, scales, biases, module_form(form, path)), signs[width])
        setattr(parent, leaf, module)
    for index, layer in enumerate(lm.model.layers):
        if layer.is_linear:
            for name in GATES:
                path = f"model.layers.{index}.linear_attn.{name}"
                setattr(layer.linear_attn, name, RowDense(text[path + ".weight"].astype(mx.float32)))
                used.add(path + ".weight")
    rest = [(k, v.astype(mx.bfloat16)) for k, v in text.items() if k not in used]
    lm.load_weights(rest, strict=False)
    missing = {k for k, _ in _flat(lm.parameters())} - used - {k for k, _ in rest} - _module_keys(lm)
    if missing or len(rest) + len(used) != len(text):
        raise ValueError(f"Ternary Bonsai: unmatched tensors {sorted(missing)[:4]}")
    mx.eval(lm.parameters())
    model.eval()                  # inference mode: mlx_lm's recurrent layers then run their Metal kernel, not the loop
    return model


def _linear(weight: Any, scales: Any, biases: Any, form: str) -> Any:
    """The same weight values as g64 bf16 groups of the 2-bit or widened 4-bit codes, or the pack's g128 fp16 as is."""

    import mlx.core as mx
    import mlx.nn as nn

    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(layer)
    layer.mode = "affine"
    if form == "packed":
        layer.bits, layer.group_size, layer.weight, layer.scales, layer.biases = 2, 128, weight, scales, biases
    else:
        layer.bits, layer.group_size = (2 if form == "lanes" else 4), 64
        layer.weight = weight if form == "lanes" else widen(weight)
        layer.scales = mx.repeat(scales, 2, axis=1).astype(mx.bfloat16)
        layer.biases = mx.repeat(biases, 2, axis=1).astype(mx.bfloat16)
    layer.freeze()
    return layer


def widen(weight: Any, chunk: int = 8192) -> Any:
    """2-bit codes [N, K/16] as MLX 4-bit codes [N, K/8]: element k stays code k, only its field grows."""

    import mlx.core as mx

    n, words = (int(d) for d in weight.shape)
    shifts = mx.arange(16, dtype=mx.uint32) * 2
    parts = []
    for start in range(0, n, chunk):
        codes = (weight[start:start + chunk, :, None] >> shifts) & 3        # [rows, words, 16]
        codes = codes.reshape(-1, words * 2, 8)                             # 8 codes a 4-bit word
        word = codes[..., 0]
        for j in range(1, 8):
            word = word | (codes[..., j] << (4 * j))
        mx.eval(word)
        parts.append(word)
    return mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _parent(root: Any, path: str) -> tuple[Any, str]:
    *parts, leaf = path.split(".")
    node = root
    for part in parts:
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node, leaf


def _flat(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return tree_flatten(tree, prefix=prefix)


def _module_keys(lm: Any) -> set[str]:
    """Parameter names our modules add (the rotated layers' inner matmuls and shared signs)."""

    from tensorfold.families.bonsai.modules import RotatedEmbedding, RotatedLinear, RowDense

    keys = set()
    for name, module in lm.named_modules():
        if isinstance(module, (RotatedLinear, RotatedEmbedding, RowDense)):
            keys.update(f"{name}.{k}" for k, _ in _flat(module.parameters()))
    return keys


__all__ = ["BLOCK", "FORMS", "MODEL_TYPE", "ROOM", "build", "contract", "fingerprint", "module_form", "pre_m5_form",
           "refusal", "sizes", "transform_refusal", "widen", "widening"]
