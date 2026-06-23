#!/usr/bin/env python3
"""Synthetic Nemotron-H gather_qmm table-size probe.

This does not load model weights. It creates random MLX quantized switch
linear tables with Nemotron-like dimensions and measures whether keeping a
larger resident expert table slows `mx.gather_qmm` even when only top-k rows
are indexed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class QmmTableCase:
    table_rows: int
    selected_rows: int
    input_dims: int
    hidden_dims: int
    group_size: int
    bits: int
    table_bytes: int


def _projection_quantized_bytes(
    *,
    rows: int,
    input_dims: int,
    output_dims: int,
    group_size: int,
    bits: int,
) -> int:
    if input_dims % group_size:
        raise ValueError("input_dims must be divisible by group_size")
    if (input_dims * bits) % 32:
        raise ValueError("input_dims * bits must be divisible by 32")
    packed_words = input_dims * bits // 32
    weight_bytes = rows * output_dims * packed_words * 4
    scale_slots = input_dims // group_size
    # MLX affine quantization stores scales and biases as BF16 side tables.
    scales_bytes = rows * output_dims * scale_slots * 2
    biases_bytes = scales_bytes
    return weight_bytes + scales_bytes + biases_bytes


def nemotron_expert_row_bytes(
    *,
    input_dims: int = 1024,
    hidden_dims: int = 2688,
    group_size: int = 64,
    bits: int = 4,
) -> int:
    return _projection_quantized_bytes(
        rows=1,
        input_dims=input_dims,
        output_dims=hidden_dims,
        group_size=group_size,
        bits=bits,
    ) + _projection_quantized_bytes(
        rows=1,
        input_dims=hidden_dims,
        output_dims=input_dims,
        group_size=group_size,
        bits=bits,
    )


def nemotron_switch_tensor_names(layer_index: int) -> tuple[str, ...]:
    prefix = f"backbone.layers.{int(layer_index)}.mixer.switch_mlp"
    return tuple(
        sorted(
            f"{prefix}.{projection}.{field}"
            for projection in ("fc1", "fc2")
            for field in ("weight", "scales", "biases")
        )
    )


def local_indices_for_table(
    *,
    selected_experts: Sequence[int],
    table_order: Sequence[int],
) -> list[int]:
    slot_by_expert = {int(expert): slot for slot, expert in enumerate(table_order)}
    missing = [int(expert) for expert in selected_experts if int(expert) not in slot_by_expert]
    if missing:
        raise ValueError(f"selected experts missing from table_order: {missing}")
    return [slot_by_expert[int(expert)] for expert in selected_experts]


def parse_byte_budget(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?i?B|B)?", text, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte budget: {raw}")
    value = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    multipliers = {
        "b": 1,
        "kb": 1_000,
        "mb": 1_000_000,
        "gb": 1_000_000_000,
        "tb": 1_000_000_000_000,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    return int(value * multipliers[unit])


def enforce_real_weight_probe_byte_ceiling(
    *,
    table_bytes: int,
    max_table_bytes: int | None,
) -> None:
    if max_table_bytes is not None and int(table_bytes) > int(max_table_bytes):
        raise ValueError(
            "real-weight table bytes exceed ceiling: "
            f"{table_bytes} > {max_table_bytes}"
        )


def real_weight_probe_dry_run(
    *,
    model_dir: str,
    layer_index: int,
    table_experts: Sequence[int],
    selected_experts: Sequence[int],
    table_bytes: int,
) -> dict[str, Any]:
    table_order = [int(expert) for expert in table_experts]
    selected = [int(expert) for expert in selected_experts]
    return {
        "mode": "real-weight-probe-dry-run",
        "model_dir": str(model_dir),
        "layer_index": int(layer_index),
        "table_experts": table_order,
        "selected_experts": selected,
        "local_indices": local_indices_for_table(
            selected_experts=selected,
            table_order=table_order,
        ),
        "table_rows": len(table_order),
        "selected_rows": len(selected),
        "table_bytes": int(table_bytes),
    }


def estimate_real_weight_probe_table_bytes(
    *,
    model_dir: str,
    layer_index: int,
    table_experts: Sequence[int],
) -> int:
    from smarttensor.manifest import SmartTensorManifest

    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {model_dir}")
    manifest = SmartTensorManifest.from_safetensors(files)
    table_rows = len(table_experts)
    total = 0
    for name in nemotron_switch_tensor_names(layer_index):
        record = manifest.tensors[name]
        total += record.nbytes * table_rows // record.shape[0]
    return total


def build_cases(
    *,
    table_rows: Sequence[int],
    selected_rows: int,
    input_dims: int,
    hidden_dims: int,
    group_size: int,
    bits: int,
) -> list[QmmTableCase]:
    if selected_rows < 1:
        raise ValueError("selected_rows must be positive")
    row_bytes = nemotron_expert_row_bytes(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        group_size=group_size,
        bits=bits,
    )
    cases: list[QmmTableCase] = []
    for rows in table_rows:
        if rows < selected_rows:
            raise ValueError("table_rows must be >= selected_rows")
        cases.append(
            QmmTableCase(
                table_rows=int(rows),
                selected_rows=selected_rows,
                input_dims=input_dims,
                hidden_dims=hidden_dims,
                group_size=group_size,
                bits=bits,
                table_bytes=int(rows) * row_bytes,
            )
        )
    return cases


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    baseline = min(results, key=lambda item: int(item["table_rows"]))
    baseline_ms = float(baseline["median_ms"])
    enriched = []
    for item in results:
        copy = dict(item)
        copy["vs_baseline"] = float(item["median_ms"]) / baseline_ms if baseline_ms else 0.0
        enriched.append(copy)
    slowest = max(enriched, key=lambda item: float(item["median_ms"]))
    return {
        "baseline_table_rows": int(baseline["table_rows"]),
        "slowest_table_rows": int(slowest["table_rows"]),
        "slowest_vs_baseline": float(slowest["vs_baseline"]),
        "results": enriched,
    }


def summarize_component_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    by_name = {str(item["component"]): dict(item) for item in results}
    slowest = max(results, key=lambda item: float(item["median_ms"]))
    return {
        "slowest_component": str(slowest["component"]),
        "slowest_median_ms": float(slowest["median_ms"]),
        "results": by_name,
    }


def summarize_first_use_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    first = [float(item["first_ms"]) for item in results]
    repeat = [float(item["repeat_ms"]) for item in results]
    first_median = statistics.median(first)
    repeat_median = statistics.median(repeat)
    return {
        "block_count": len(results),
        "first_median_ms": first_median,
        "repeat_median_ms": repeat_median,
        "first_vs_repeat": first_median / repeat_median if repeat_median else 0.0,
        "first_total_ms": sum(first),
        "repeat_total_ms": sum(repeat),
        "first_max_ms": max(first),
        "repeat_max_ms": max(repeat),
        "results": results,
    }


def _spread_indices(table_rows: int, selected_rows: int) -> list[int]:
    if selected_rows == 1:
        return [0]
    return sorted(
        {
            round(i * (table_rows - 1) / (selected_rows - 1))
            for i in range(selected_rows)
        }
    )


def _parse_side_table_dtype(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in {"float32", "bf16", "float16"}:
        raise argparse.ArgumentTypeError(
            "--side-table-dtype must be float32, bf16, or float16"
        )
    return normalized


def _mlx_side_table_dtype(name: str) -> Any:
    import mlx.core as mx

    if name == "float32":
        return mx.float32
    if name == "bf16":
        return mx.bfloat16
    if name == "float16":
        return mx.float16
    raise ValueError(f"unsupported side table dtype: {name}")


def _cast_quant_side_tables(module: Any, dtype_name: str) -> None:
    if dtype_name == "float32":
        return
    dtype = _mlx_side_table_dtype(dtype_name)
    module.scales = module.scales.astype(dtype)
    if getattr(module, "biases", None) is not None:
        module.biases = module.biases.astype(dtype)


def run_case(
    case: QmmTableCase,
    *,
    iterations: int,
    warmup: int,
    spread_indices: bool,
    side_table_dtype: str,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    fc1 = QuantizedSwitchLinear(
        case.input_dims,
        case.hidden_dims,
        case.table_rows,
        False,
        case.group_size,
        case.bits,
    )
    fc2 = QuantizedSwitchLinear(
        case.hidden_dims,
        case.input_dims,
        case.table_rows,
        False,
        case.group_size,
        case.bits,
    )
    _cast_quant_side_tables(fc1, side_table_dtype)
    _cast_quant_side_tables(fc2, side_table_dtype)
    x = mx.random.normal((1, 1, case.input_dims))
    selected = (
        _spread_indices(case.table_rows, case.selected_rows)
        if spread_indices
        else list(range(case.selected_rows))
    )
    indices = mx.array([[selected]])
    mx.eval(x, indices, fc1.weight, fc1.scales, fc1.biases, fc2.weight, fc2.scales, fc2.biases)

    def step() -> Any:
        hidden = mx.maximum(fc1(x, indices), 0)
        return fc2(hidden, indices)

    for _ in range(warmup):
        mx.eval(step())

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        mx.eval(step())
        samples.append(time.perf_counter() - started)

    median = statistics.median(samples)
    return {
        **asdict(case),
        "indices": selected,
        "iterations": iterations,
        "warmup": warmup,
        "side_table_dtype": side_table_dtype,
        "median_ms": median * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
    }


def run_component_probe(
    *,
    table_rows: int,
    selected_rows: int,
    model_dims: int,
    latent_dims: int,
    routed_hidden_dims: int,
    shared_hidden_dims: int,
    routed_experts: int,
    group_size: int,
    bits: int,
    side_table_dtype: str,
    iterations: int,
    warmup: int,
    spread_indices: bool,
) -> dict[str, Any]:
    """Benchmark synthetic Nemotron E-block components without loading weights.

    The measured native-hotset slowdown localized to `E` blocks. This probe
    keeps the same rough shapes and quantized linear primitives but times the
    routed switch core, dense/shared side path, gate, and full synthetic E-block
    separately.
    """

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    if table_rows < selected_rows:
        raise ValueError("table_rows must be >= selected_rows")
    if routed_experts < selected_rows:
        raise ValueError("routed_experts must be >= selected_rows")

    x_model = mx.random.normal((1, 1, model_dims))
    x_latent = mx.random.normal((1, 1, latent_dims))
    selected = (
        _spread_indices(table_rows, selected_rows)
        if spread_indices
        else list(range(selected_rows))
    )
    indices = mx.array([[selected]])
    scores = mx.ones((1, 1, selected_rows)) / float(selected_rows)

    gate_weight = mx.random.normal((routed_experts, model_dims))
    gate_bias = mx.zeros((routed_experts,))

    latent_in = nn.QuantizedLinear(
        model_dims, latent_dims, bias=False, group_size=group_size, bits=bits
    )
    latent_out = nn.QuantizedLinear(
        latent_dims, model_dims, bias=False, group_size=group_size, bits=bits
    )
    shared_up = nn.QuantizedLinear(
        model_dims, shared_hidden_dims, bias=False, group_size=group_size, bits=bits
    )
    shared_down = nn.QuantizedLinear(
        shared_hidden_dims, model_dims, bias=False, group_size=group_size, bits=bits
    )
    routed_fc1 = QuantizedSwitchLinear(
        latent_dims,
        routed_hidden_dims,
        table_rows,
        False,
        group_size,
        bits,
    )
    routed_fc2 = QuantizedSwitchLinear(
        routed_hidden_dims,
        latent_dims,
        table_rows,
        False,
        group_size,
        bits,
    )
    for module in (
        latent_in,
        latent_out,
        shared_up,
        shared_down,
        routed_fc1,
        routed_fc2,
    ):
        _cast_quant_side_tables(module, side_table_dtype)

    mx.eval(
        x_model,
        x_latent,
        indices,
        scores,
        gate_weight,
        gate_bias,
        latent_in.weight,
        latent_in.scales,
        latent_in.biases,
        latent_out.weight,
        latent_out.scales,
        latent_out.biases,
        shared_up.weight,
        shared_up.scales,
        shared_up.biases,
        shared_down.weight,
        shared_down.scales,
        shared_down.biases,
        routed_fc1.weight,
        routed_fc1.scales,
        routed_fc1.biases,
        routed_fc2.weight,
        routed_fc2.scales,
        routed_fc2.biases,
    )

    def gate_step() -> Any:
        gates = x_model @ gate_weight.T
        raw = mx.sigmoid(gates.astype(mx.float32))
        biased = raw + gate_bias
        picked = mx.argpartition(-biased, kth=selected_rows - 1, axis=-1)[
            ..., :selected_rows
        ]
        picked_scores = mx.take_along_axis(raw, picked, axis=-1)
        return picked, picked_scores / (picked_scores.sum(axis=-1, keepdims=True) + 1e-20)

    def routed_step(latent: Any = x_latent, idx: Any = indices, weight: Any = scores) -> Any:
        y = routed_fc1(latent, idx)
        y = nn.relu2(y)
        y = routed_fc2(y, idx)
        return (y * weight[..., None]).sum(axis=-2).astype(y.dtype)

    def latent_routed_step() -> Any:
        latent = latent_in(x_model)
        y = routed_step(latent, indices, scores)
        return latent_out(y)

    def shared_step() -> Any:
        return shared_down(nn.relu2(shared_up(x_model)))

    def full_e_step() -> Any:
        normed = mx.fast.rms_norm(x_model, mx.ones((model_dims,)), 1e-5)
        idx, weight = gate_step()
        latent = latent_in(normed)
        y = routed_step(latent, idx, weight)
        y = latent_out(y)
        y = y + shared_down(nn.relu2(shared_up(normed)))
        return x_model + y

    components = {
        "gate_topk": gate_step,
        "routed_qmm": routed_step,
        "latent_plus_routed_qmm": latent_routed_step,
        "shared_expert_mlp": shared_step,
        "full_e_block": full_e_step,
    }

    results: list[dict[str, Any]] = []
    for name, step in components.items():
        for _ in range(warmup):
            mx.eval(step())
        samples: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            mx.eval(step())
            samples.append(time.perf_counter() - started)
        results.append(
            {
                "component": name,
                "iterations": iterations,
                "warmup": warmup,
                "median_ms": statistics.median(samples) * 1000.0,
                "min_ms": min(samples) * 1000.0,
                "max_ms": max(samples) * 1000.0,
            }
        )

    summary = summarize_component_results(results)
    summary.update(
        {
            "table_rows": table_rows,
            "selected_rows": selected_rows,
            "model_dims": model_dims,
            "latent_dims": latent_dims,
            "routed_hidden_dims": routed_hidden_dims,
            "shared_hidden_dims": shared_hidden_dims,
            "routed_experts": routed_experts,
            "group_size": group_size,
            "bits": bits,
            "side_table_dtype": side_table_dtype,
            "spread_indices": spread_indices,
            "expert_row_bytes": nemotron_expert_row_bytes(
                input_dims=latent_dims,
                hidden_dims=routed_hidden_dims,
                group_size=group_size,
                bits=bits,
            ),
        }
    )
    return summary


def run_first_use_probe(
    *,
    block_count: int,
    table_rows: int,
    selected_rows: int,
    model_dims: int,
    latent_dims: int,
    routed_hidden_dims: int,
    group_size: int,
    bits: int,
    side_table_dtype: str,
    spread_indices: bool,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    if block_count < 1:
        raise ValueError("block_count must be positive")
    if table_rows < selected_rows:
        raise ValueError("table_rows must be >= selected_rows")

    selected = (
        _spread_indices(table_rows, selected_rows)
        if spread_indices
        else list(range(selected_rows))
    )
    indices = mx.array([[selected]])
    scores = mx.ones((1, 1, selected_rows)) / float(selected_rows)
    x = mx.random.normal((1, 1, model_dims))
    mx.eval(indices, scores, x)

    results: list[dict[str, Any]] = []
    for block_index in range(block_count):
        latent_in = nn.QuantizedLinear(
            model_dims, latent_dims, bias=False, group_size=group_size, bits=bits
        )
        latent_out = nn.QuantizedLinear(
            latent_dims, model_dims, bias=False, group_size=group_size, bits=bits
        )
        fc1 = QuantizedSwitchLinear(
            latent_dims,
            routed_hidden_dims,
            table_rows,
            False,
            group_size,
            bits,
        )
        fc2 = QuantizedSwitchLinear(
            routed_hidden_dims,
            latent_dims,
            table_rows,
            False,
            group_size,
            bits,
        )
        for module in (latent_in, latent_out, fc1, fc2):
            _cast_quant_side_tables(module, side_table_dtype)
        mx.eval(
            latent_in.weight,
            latent_in.scales,
            latent_in.biases,
            latent_out.weight,
            latent_out.scales,
            latent_out.biases,
            fc1.weight,
            fc1.scales,
            fc1.biases,
            fc2.weight,
            fc2.scales,
            fc2.biases,
        )

        def step() -> Any:
            latent = latent_in(x)
            y = fc1(mx.expand_dims(latent, (-2, -3)), indices)
            y = nn.relu2(y)
            y = fc2(y, indices).squeeze(-2)
            y = (y * scores[..., None]).sum(axis=-2)
            return latent_out(y)

        started = time.perf_counter()
        mx.eval(step())
        first_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        mx.eval(step())
        repeat_ms = (time.perf_counter() - started) * 1000.0

        results.append(
            {
                "block": block_index,
                "first_ms": first_ms,
                "repeat_ms": repeat_ms,
            }
        )

    summary = summarize_first_use_results(results)
    summary.update(
        {
            "table_rows": table_rows,
            "selected_rows": selected_rows,
            "model_dims": model_dims,
            "latent_dims": latent_dims,
            "routed_hidden_dims": routed_hidden_dims,
            "group_size": group_size,
            "bits": bits,
            "side_table_dtype": side_table_dtype,
            "spread_indices": spread_indices,
            "expert_row_bytes": nemotron_expert_row_bytes(
                input_dims=latent_dims,
                hidden_dims=routed_hidden_dims,
                group_size=group_size,
                bits=bits,
            ),
        }
    )
    return summary


def run_real_weight_probe(
    *,
    model_dir: str,
    layer_index: int,
    table_experts: Sequence[int],
    selected_experts: Sequence[int],
    iterations: int,
    warmup: int,
    max_table_bytes: int | None = None,
) -> dict[str, Any]:
    """Load one real Nemotron routed expert table and benchmark actual QMM.

    This intentionally does not construct the full model. It loads only the
    requested first-dimension expert rows for one layer's `switch_mlp`.
    """

    import mlx.core as mx
    from smarttensor.adapters.mlx import MlxSelectiveLoader

    table_order = [int(expert) for expert in table_experts]
    selected = [int(expert) for expert in selected_experts]
    local_indices = local_indices_for_table(
        selected_experts=selected,
        table_order=table_order,
    )
    names = nemotron_switch_tensor_names(layer_index)
    table_bytes = estimate_real_weight_probe_table_bytes(
        model_dir=model_dir,
        layer_index=layer_index,
        table_experts=table_order,
    )
    enforce_real_weight_probe_byte_ceiling(
        table_bytes=table_bytes,
        max_table_bytes=max_table_bytes,
    )
    load_started = time.perf_counter()
    loader = MlxSelectiveLoader.from_model_dir(
        model_dir,
        backend="native",
        cache_file_handles=False,
    )
    batch = loader.load_first_dim_slices(names, table_order, evaluate=True)
    load_seconds = time.perf_counter() - load_started

    prefix = f"backbone.layers.{int(layer_index)}.mixer.switch_mlp"
    fc1_weight = batch.arrays[f"{prefix}.fc1.weight"]
    fc1_scales = batch.arrays[f"{prefix}.fc1.scales"]
    fc1_biases = batch.arrays[f"{prefix}.fc1.biases"]
    fc2_weight = batch.arrays[f"{prefix}.fc2.weight"]
    fc2_scales = batch.arrays[f"{prefix}.fc2.scales"]
    fc2_biases = batch.arrays[f"{prefix}.fc2.biases"]

    x = mx.random.normal((1, 1, 1024))
    indices = mx.array([[local_indices]], dtype=mx.int32)
    mx.eval(
        x,
        indices,
        fc1_weight,
        fc1_scales,
        fc1_biases,
        fc2_weight,
        fc2_scales,
        fc2_biases,
    )

    def step() -> Any:
        hidden = mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)),
            fc1_weight,
            fc1_scales,
            fc1_biases,
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=4,
            mode="affine",
            sorted_indices=False,
        )
        hidden = mx.maximum(hidden, 0)
        return mx.gather_qmm(
            hidden,
            fc2_weight,
            fc2_scales,
            fc2_biases,
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=4,
            mode="affine",
            sorted_indices=False,
        )

    for _ in range(warmup):
        mx.eval(step())

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        mx.eval(step())
        samples.append(time.perf_counter() - started)

    return {
        "model_dir": str(model_dir),
        "layer_index": int(layer_index),
        "table_experts": table_order,
        "selected_experts": selected,
        "local_indices": local_indices,
        "table_rows": len(table_order),
        "selected_rows": len(selected),
        "load_seconds": load_seconds,
        "table_bytes": batch.nbytes,
        "iterations": iterations,
        "warmup": warmup,
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
    }


def run_probe(
    *,
    table_rows: Sequence[int],
    selected_rows: int,
    input_dims: int,
    hidden_dims: int,
    group_size: int,
    bits: int,
    side_table_dtype: str,
    iterations: int,
    warmup: int,
    spread_indices: bool,
) -> dict[str, Any]:
    cases = build_cases(
        table_rows=table_rows,
        selected_rows=selected_rows,
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        group_size=group_size,
        bits=bits,
    )
    results = [
        run_case(
            case,
            iterations=iterations,
            warmup=warmup,
            spread_indices=spread_indices,
            side_table_dtype=side_table_dtype,
        )
        for case in cases
    ]
    summary = summarize_results(results)
    summary.update(
        {
            "selected_rows": selected_rows,
            "input_dims": input_dims,
            "hidden_dims": hidden_dims,
            "group_size": group_size,
            "bits": bits,
            "side_table_dtype": side_table_dtype,
            "spread_indices": spread_indices,
            "expert_row_bytes": nemotron_expert_row_bytes(
                input_dims=input_dims,
                hidden_dims=hidden_dims,
                group_size=group_size,
                bits=bits,
            ),
        }
    )
    return summary


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-probe",
        action="store_true",
        help="Benchmark synthetic full Nemotron E-block components instead of only routed QMM table size.",
    )
    parser.add_argument(
        "--first-use-probe",
        action="store_true",
        help="Benchmark first vs repeat eval for many fresh synthetic Nemotron routed blocks.",
    )
    parser.add_argument(
        "--real-weight-probe",
        action="store_true",
        help=(
            "Load one real Nemotron layer's selected switch_mlp expert rows "
            "from safetensors and benchmark actual gather_qmm. This touches "
            "model weights and should obey the shared inference-window lock."
        ),
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--real-weight-dry-run", action="store_true")
    parser.add_argument(
        "--max-table-bytes",
        type=parse_byte_budget,
        help="Abort --real-weight-probe if the selected real table exceeds this byte ceiling.",
    )
    parser.add_argument(
        "--table-experts",
        help="Comma-separated expert ids to load as the real table. Defaults to --selected-experts.",
    )
    parser.add_argument(
        "--selected-experts",
        help="Comma-separated selected expert ids to route through the real table.",
    )
    parser.add_argument("--block-count", type=int, default=40)
    parser.add_argument("--table-rows", default="22,44,88,128")
    parser.add_argument("--selected-rows", type=int, default=22)
    parser.add_argument("--model-dims", type=int, default=4096)
    parser.add_argument("--input-dims", type=int, default=1024)
    parser.add_argument("--hidden-dims", type=int, default=2688)
    parser.add_argument("--shared-hidden-dims", type=int, default=5376)
    parser.add_argument("--routed-experts", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--side-table-dtype",
        type=_parse_side_table_dtype,
        default="float32",
        help=(
            "Synthetic quantization scales/biases dtype. Real Nemotron-H "
            "artifacts store quant side tables as BF16; MLX random quantize "
            "defaults to float32."
        ),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--spread-indices", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    table_rows = _parse_ints(args.table_rows)
    probe_modes = [
        bool(args.component_probe),
        bool(args.first_use_probe),
        bool(args.real_weight_probe),
    ]
    if sum(probe_modes) > 1:
        raise ValueError(
            "--component-probe, --first-use-probe, and --real-weight-probe are mutually exclusive"
        )
    if args.component_probe:
        if len(table_rows) != 1:
            raise ValueError("--component-probe expects exactly one --table-rows value")
        result = run_component_probe(
            table_rows=table_rows[0],
            selected_rows=args.selected_rows,
            model_dims=args.model_dims,
            latent_dims=args.input_dims,
            routed_hidden_dims=args.hidden_dims,
            shared_hidden_dims=args.shared_hidden_dims,
            routed_experts=args.routed_experts,
            group_size=args.group_size,
            bits=args.bits,
            side_table_dtype=args.side_table_dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            spread_indices=args.spread_indices,
        )
    elif args.first_use_probe:
        if len(table_rows) != 1:
            raise ValueError("--first-use-probe expects exactly one --table-rows value")
        result = run_first_use_probe(
            block_count=args.block_count,
            table_rows=table_rows[0],
            selected_rows=args.selected_rows,
            model_dims=args.model_dims,
            latent_dims=args.input_dims,
            routed_hidden_dims=args.hidden_dims,
            group_size=args.group_size,
            bits=args.bits,
            side_table_dtype=args.side_table_dtype,
            spread_indices=args.spread_indices,
        )
    elif args.real_weight_probe:
        if not args.model_dir:
            raise ValueError("--real-weight-probe requires --model-dir")
        if not args.selected_experts:
            raise ValueError("--real-weight-probe requires --selected-experts")
        selected_experts = _parse_ints(args.selected_experts)
        table_experts = (
            _parse_ints(args.table_experts)
            if args.table_experts
            else list(selected_experts)
        )
        if args.real_weight_dry_run:
            table_bytes = estimate_real_weight_probe_table_bytes(
                model_dir=args.model_dir,
                layer_index=args.layer_index,
                table_experts=table_experts,
            )
            enforce_real_weight_probe_byte_ceiling(
                table_bytes=table_bytes,
                max_table_bytes=args.max_table_bytes,
            )
            result = real_weight_probe_dry_run(
                model_dir=args.model_dir,
                layer_index=args.layer_index,
                table_experts=table_experts,
                selected_experts=selected_experts,
                table_bytes=table_bytes,
            )
        else:
            result = run_real_weight_probe(
                model_dir=args.model_dir,
                layer_index=args.layer_index,
                table_experts=table_experts,
                selected_experts=selected_experts,
                iterations=args.iterations,
                warmup=args.warmup,
                max_table_bytes=args.max_table_bytes,
            )
    else:
        result = run_probe(
            table_rows=table_rows,
            selected_rows=args.selected_rows,
            input_dims=args.input_dims,
            hidden_dims=args.hidden_dims,
            group_size=args.group_size,
            bits=args.bits,
            side_table_dtype=args.side_table_dtype,
            iterations=args.iterations,
            warmup=args.warmup,
            spread_indices=args.spread_indices,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
