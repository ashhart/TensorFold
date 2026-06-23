"""Output-head primitives for low-memory greedy decoding."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def _shape(array: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(array, "shape", ()))


@lru_cache(maxsize=16)
def _affine4_lm_head_logits_kernel(
    *,
    input_dims: int,
    vocab: int,
    packed_cols: int,
    groups: int,
    group_size: int,
) -> Any:
    import mlx.core as mx

    packs_per_group = group_size // 8
    source = f"""
        uint row = thread_position_in_grid.x;
        if (row >= {int(vocab)}) {{
            return;
        }}

        float acc = 0.0f;
        for (uint group = 0; group < {int(groups)}; ++group) {{
            uint scale_offset = row * {int(groups)} + group;
            float scale = float(scales[scale_offset]);
            float bias = float(biases[scale_offset]);
            float group_sum = 0.0f;
            float group_accum = 0.0f;

            for (uint pack = 0; pack < {int(packs_per_group)}; ++pack) {{
                uint weight_offset = row * {int(packed_cols)}
                    + group * {int(packs_per_group)} + pack;
                uint packed = weight[weight_offset];
                uint in_col = group * {int(group_size)} + pack * 8;
                float x0 = float(x[in_col]);
                float x1 = float(x[in_col + 1]);
                float x2 = float(x[in_col + 2]);
                float x3 = float(x[in_col + 3]);
                float x4 = float(x[in_col + 4]);
                float x5 = float(x[in_col + 5]);
                float x6 = float(x[in_col + 6]);
                float x7 = float(x[in_col + 7]);

                group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
                group_accum +=
                    x0 * float(packed & 0x0000000fu) +
                    (x1 / 16.0f) * float(packed & 0x000000f0u) +
                    (x2 / 256.0f) * float(packed & 0x00000f00u) +
                    (x3 / 4096.0f) * float(packed & 0x0000f000u) +
                    x4 * float((packed >> 16) & 0x000fu) +
                    (x5 / 16.0f) * float((packed >> 16) & 0x00f0u) +
                    (x6 / 256.0f) * float((packed >> 16) & 0x0f00u) +
                    (x7 / 4096.0f) * float((packed >> 16) & 0xf000u);
            }}
            acc += scale * group_accum + group_sum * bias;
        }}
        logits[row] = acc;
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_affine4_lm_head_logits_"
            f"v{vocab}_i{input_dims}_g{group_size}"
        ),
        input_names=["x", "weight", "scales", "biases"],
        output_names=["logits"],
        source=source,
    )


def affine4_lm_head_logits(
    hidden: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    *,
    group_size: int = 64,
) -> Any:
    """Compute affine4 output-head logits without MLX's full qmatmul workspace."""

    import mlx.core as mx

    hidden_shape = _shape(hidden)
    weight_shape = _shape(weight)
    scales_shape = _shape(scales)
    if not hidden_shape:
        raise ValueError("hidden must have at least one dimension")
    if len(weight_shape) != 2:
        raise ValueError("weight must have shape (vocab, packed_cols)")
    if len(scales_shape) != 2 or _shape(biases) != scales_shape:
        raise ValueError("scales and biases must have shape (vocab, groups)")
    input_dims = int(hidden_shape[-1])
    vocab, packed_cols = weight_shape
    if scales_shape[0] != vocab:
        raise ValueError("scale rows must match vocab rows")
    groups = int(scales_shape[1])
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if groups * group_size != input_dims:
        raise ValueError("hidden size must equal scales groups * group_size")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal hidden size / 8 for affine4")
    x = hidden.reshape((input_dims,))
    kernel = _affine4_lm_head_logits_kernel(
        input_dims=input_dims,
        vocab=int(vocab),
        packed_cols=int(packed_cols),
        groups=groups,
        group_size=group_size,
    )
    return kernel(
        inputs=[x, weight, scales, biases],
        grid=(int(vocab), 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(int(vocab),)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=32)
def _affine4_lm_head_candidate_logits_kernel(
    *,
    candidate_count: int,
    input_dims: int,
    vocab: int,
    packed_cols: int,
    groups: int,
    group_size: int,
) -> Any:
    import mlx.core as mx

    packs_per_group = group_size // 8
    source = f"""
        uint candidate_offset = thread_position_in_grid.x;
        if (candidate_offset >= {int(candidate_count)}) {{
            return;
        }}
        int row_value = candidate_ids[candidate_offset];
        if (row_value < 0 || row_value >= {int(vocab)}) {{
            logits[candidate_offset] = -INFINITY;
            return;
        }}
        uint row = uint(row_value);

        float acc = 0.0f;
        for (uint group = 0; group < {int(groups)}; ++group) {{
            uint scale_offset = row * {int(groups)} + group;
            float scale = float(scales[scale_offset]);
            float bias = float(biases[scale_offset]);
            float group_sum = 0.0f;
            float group_accum = 0.0f;

            for (uint pack = 0; pack < {int(packs_per_group)}; ++pack) {{
                uint weight_offset = row * {int(packed_cols)}
                    + group * {int(packs_per_group)} + pack;
                uint packed = weight[weight_offset];
                uint in_col = group * {int(group_size)} + pack * 8;
                float x0 = float(x[in_col]);
                float x1 = float(x[in_col + 1]);
                float x2 = float(x[in_col + 2]);
                float x3 = float(x[in_col + 3]);
                float x4 = float(x[in_col + 4]);
                float x5 = float(x[in_col + 5]);
                float x6 = float(x[in_col + 6]);
                float x7 = float(x[in_col + 7]);

                group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
                group_accum +=
                    x0 * float(packed & 0x0000000fu) +
                    (x1 / 16.0f) * float(packed & 0x000000f0u) +
                    (x2 / 256.0f) * float(packed & 0x00000f00u) +
                    (x3 / 4096.0f) * float(packed & 0x0000f000u) +
                    x4 * float((packed >> 16) & 0x000fu) +
                    (x5 / 16.0f) * float((packed >> 16) & 0x00f0u) +
                    (x6 / 256.0f) * float((packed >> 16) & 0x0f00u) +
                    (x7 / 4096.0f) * float((packed >> 16) & 0xf000u);
            }}
            acc += scale * group_accum + group_sum * bias;
        }}
        logits[candidate_offset] = acc;
    """
    return mx.fast.metal_kernel(
        name=(
            "smarttensor_affine4_lm_head_candidate_logits_"
            f"c{candidate_count}_v{vocab}_i{input_dims}_g{group_size}"
        ),
        input_names=["x", "weight", "scales", "biases", "candidate_ids"],
        output_names=["logits"],
        source=source,
    )


def affine4_lm_head_candidate_logits(
    hidden: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    candidate_ids: Any,
    *,
    group_size: int = 64,
) -> Any:
    """Score selected affine4 output-head rows without a full-vocab projection."""

    import mlx.core as mx

    hidden_shape = _shape(hidden)
    weight_shape = _shape(weight)
    scales_shape = _shape(scales)
    candidate_shape = _shape(candidate_ids)
    if not hidden_shape:
        raise ValueError("hidden must have at least one dimension")
    if len(weight_shape) != 2:
        raise ValueError("weight must have shape (vocab, packed_cols)")
    if len(scales_shape) != 2 or _shape(biases) != scales_shape:
        raise ValueError("scales and biases must have shape (vocab, groups)")
    if len(candidate_shape) != 1:
        raise ValueError("candidate_ids must be a rank-1 array")
    input_dims = int(hidden_shape[-1])
    vocab, packed_cols = weight_shape
    if scales_shape[0] != vocab:
        raise ValueError("scale rows must match vocab rows")
    groups = int(scales_shape[1])
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if groups * group_size != input_dims:
        raise ValueError("hidden size must equal scales groups * group_size")
    if packed_cols * 8 != input_dims:
        raise ValueError("packed_cols must equal hidden size / 8 for affine4")
    x = hidden.reshape((input_dims,))
    candidate_count = int(candidate_shape[0])
    kernel = _affine4_lm_head_candidate_logits_kernel(
        candidate_count=candidate_count,
        input_dims=input_dims,
        vocab=int(vocab),
        packed_cols=int(packed_cols),
        groups=groups,
        group_size=group_size,
    )
    return kernel(
        inputs=[x, weight, scales, biases, candidate_ids],
        grid=(candidate_count, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(candidate_count,)],
        output_dtypes=[mx.float32],
    )[0]
