"""Fused SSM glue primitives for low-dispatch NemotronH decode.

The Mamba2 mixer's gated RMSNorm (mlx_lm ``MambaRMSNormGated.__call__``) is a
chain of tiny "glue" ops -- silu/swiglu gate, reshape, ``mx.fast.rms_norm``,
flatten, weight-scale -- each a separate Metal dispatch whose cost at decode is
pure host-side launch overhead. :func:`fused_mamba_rmsnorm_gated` collapses the
gated branch to ONE ``mx.fast.metal_kernel`` dispatch that matches the stock
math to within ~1 fp32 ULP.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def _shape(array: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(array, "shape", ()))


# Threads per (row, group) threadgroup. Fixed power-of-two so the SIMD-group
# count (TG_WIDTH / 32) is exact; threads stride over ``group_size`` so this is
# independent of group_size (handles group_size both smaller and larger).
_TG_WIDTH = 128


@lru_cache(maxsize=64)
def _rmsgate_kernel(*, group_size: int, n_groups: int, eps: float, dtype: Any) -> Any:
    import mlx.core as mx

    # One THREADGROUP per (row, group). ``threadgroup_position_in_grid.x`` ==
    # row*n_groups + group; the group's ``group_size`` channels live contiguously
    # at ``base = that_index * group_size``. Each thread strides over the group
    # computing ``silu(gate)*x`` and a partial fp32 sum-of-squares, the threadgroup
    # reduces the partials (simd_sum within each 32-lane SIMD group, then combine
    # across SIMD groups via shared memory), and every thread rewrites its strided
    # outputs scaled by ``rsqrt(meansq + eps)`` and the per-channel weight.
    n_warps = (_TG_WIDTH + 31) // 32
    source = f"""
        const uint GROUP_SIZE = {int(group_size)};
        const uint N_GROUPS = {int(n_groups)};
        const uint TG_WIDTH = {int(_TG_WIDTH)};
        const uint N_WARPS = {int(n_warps)};

        uint grp_idx = threadgroup_position_in_grid.x;   // row * N_GROUPS + group
        uint tid = thread_position_in_threadgroup.x;     // 0 .. TG_WIDTH-1
        uint lane = tid % 32u;                            // lane within SIMD group
        uint warp = tid / 32u;                            // SIMD-group index
        uint base = grp_idx * GROUP_SIZE;                 // first channel of group
        uint grp = grp_idx % N_GROUPS;                    // group within the row
        uint wbase = grp * GROUP_SIZE;                    // weight column offset

        // Partial sum-of-squares over this thread's strided slice. Every thread
        // (even when GROUP_SIZE < TG_WIDTH) participates in the SIMD reduction; a
        // thread with no elements contributes 0.
        float sq = 0.0f;
        for (uint i = tid; i < GROUP_SIZE; i += TG_WIDTH) {{
            float gv = float(gate[base + i]);
            float xv = float(x[base + i]);
            float s = gv / (1.0f + fast::exp(-gv));   // silu(gate)
            float val = s * xv;
            sq += val * val;
        }}

        // Reduce partials across the threadgroup: simd_sum collapses each 32-lane
        // SIMD group, lane-0 of each writes to shared memory, then SIMD group 0
        // reduces those and broadcasts the inverse-RMS through shared memory.
        threadgroup float tg_partial[N_WARPS];
        threadgroup float tg_inv;
        float warp_sum = simd_sum(sq);
        if (lane == 0u) {{
            tg_partial[warp] = warp_sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (warp == 0u) {{
            float pv = (lane < N_WARPS) ? tg_partial[lane] : 0.0f;
            float total = simd_sum(pv);
            if (lane == 0u) {{
                float meansq = total / float(GROUP_SIZE);
                tg_inv = 1.0f / fast::sqrt(meansq + {float(eps)}f);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = tg_inv;

        // Recompute val and write the normalized, per-channel-weighted output for
        // this thread's strided slice (recompute avoids a GROUP_SIZE shared array).
        for (uint i = tid; i < GROUP_SIZE; i += TG_WIDTH) {{
            float gv = float(gate[base + i]);
            float xv = float(x[base + i]);
            float s = gv / (1.0f + fast::exp(-gv));
            float val = s * xv;
            out[base + i] = T(float(weight[wbase + i]) * (val * inv));
        }}
    """
    return mx.fast.metal_kernel(
        name=f"smarttensor_rmsgate_g{group_size}_n{n_groups}",
        input_names=["x", "gate", "weight"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def fused_mamba_rmsnorm_gated(
    hidden: Any,
    gate: Any,
    weight: Any,
    *,
    n_groups: int,
    eps: float,
) -> Any:
    """Fused swiglu-gate + per-group RMS-norm + per-channel weight in one dispatch.

    Reproduces mlx_lm ``MambaRMSNormGated.__call__``:
    ``x = silu(gate) * x`` (when ``gate`` is not None), reshape the last axis to
    ``(n_groups, group_size)``, ``x / sqrt(mean(x^2) + eps)`` per group, flatten,
    then scale by per-channel ``weight``.

    Args:
        hidden: Input array; the last axis is the per-channel feature dim.
        gate: Same shape as ``hidden`` for the silu gate, or ``None`` to skip the
            gate (then the result equals plain per-group RMS-norm * weight).
        weight: One-dimensional per-channel scale with size ``hidden.shape[-1]``.
        n_groups: Number of RMS groups the last axis splits into; the per-group
            size is ``hidden.shape[-1] // n_groups``.
        eps: Additive epsilon inside the RMS denominator (``... + eps`` under the
            sqrt), matching ``mx.fast.rms_norm``.

    Returns:
        Array with the same shape and dtype as ``hidden``.
    """

    import mlx.core as mx

    hidden_shape = _shape(hidden)
    if not hidden_shape:
        raise ValueError("hidden must have at least one dimension")
    dim = int(hidden_shape[-1])
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if dim % n_groups != 0:
        raise ValueError("hidden last dim must be divisible by n_groups")
    group_size = dim // n_groups
    weight_shape = _shape(weight)
    if len(weight_shape) != 1 or weight_shape[0] != dim:
        raise ValueError("weight must be 1-D with size hidden.shape[-1]")

    if gate is None:
        x = mx.unflatten(hidden, axis=-1, shape=(-1, group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=eps)
        return weight * x.flatten(-2)

    if _shape(gate) != hidden_shape:
        raise ValueError("gate must match hidden shape")

    dtype = hidden.dtype
    rows = 1
    for d in hidden_shape[:-1]:
        rows *= int(d)
    total_groups = rows * n_groups

    kernel = _rmsgate_kernel(
        group_size=group_size, n_groups=n_groups, eps=eps, dtype=dtype
    )
    # One threadgroup of _TG_WIDTH threads per (row, group). MLX uses
    # dispatchThreads semantics, so ``grid`` is the TOTAL thread count
    # (= num_threadgroups * threadgroup_width), not the threadgroup count.
    out = kernel(
        inputs=[hidden, gate, weight],
        template=[("T", dtype)],
        grid=(total_groups * _TG_WIDTH, 1, 1),
        threadgroup=(_TG_WIDTH, 1, 1),
        output_shapes=[hidden_shape],
        output_dtypes=[dtype],
    )[0]
    return out
