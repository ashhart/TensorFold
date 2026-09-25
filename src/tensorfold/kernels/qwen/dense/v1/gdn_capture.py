"""Roll a verify window back to its accepted prefix without re-feeding it.

The lane engine's default rollback restores a row's recurrent state to where
it stood before the window and re-feeds the accepted tokens next round. For
one stream this module does what serial decoding would have done instead:
while a window is verified it records each Gated DeltaNet layer's inputs (the
same ``mlx_lm`` call, op for op, plus references), and on rollback it re-runs
the recurrence over only the kept prefix from the recorded initial state and
keeps the matching convolution tail; attention layers are trimmed. The rows
kept are bit-identical to that many serial steps (the recurrence kernel's
per-step arithmetic does not depend on how many steps a call covers), which
the lane engine's byte-exactness tests check end to end.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

_captures: list[tuple[Any, ...]] | None = None   # recording while not None
_ORIG: Any = None


def _gdn_call(self_layer: Any, inputs: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
    # mlx_lm 0.31.3 qwen3_5.GatedDeltaNet.__call__, op for op, recording its recurrence inputs
    from mlx_lm.models.gated_delta import gated_delta_update

    B, S, _ = inputs.shape
    if self_layer.sharding_group is not None:
        from mlx_lm.models.qwen3_5 import sum_gradients

        inputs = sum_gradients(self_layer.sharding_group)(inputs)
    qkv = self_layer.in_proj_qkv(inputs)
    z = self_layer.in_proj_z(inputs).reshape(B, S, self_layer.num_v_heads, self_layer.head_v_dim)
    b = self_layer.in_proj_b(inputs)
    a = self_layer.in_proj_a(inputs)
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, self_layer.conv_kernel_size - 1, self_layer.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = self_layer.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self_layer.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
            [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
            [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
        )
    ]
    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    if _captures is not None:
        _captures.append((q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask, conv_input,
                          int(self_layer.conv_kernel_size)))
    out, state = gated_delta_update(q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask,
                                    use_kernel=not self_layer.training)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = self_layer.norm(out, z)
    out = self_layer.out_proj(out.reshape(B, S, -1))
    if self_layer.sharding_group is not None:
        out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
    return out


def install() -> None:
    """Swap in the recording call (identical arithmetic). Idempotent."""

    global _ORIG
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    if _ORIG is None:
        _ORIG = GatedDeltaNet.__call__
    GatedDeltaNet.__call__ = _gdn_call


def begin() -> None:
    global _captures
    _captures = []


def end() -> list[tuple[Any, ...]]:
    global _captures
    got, _captures = (_captures or []), None
    return got


def rollback(cache: list[Any], captured: list[tuple[Any, ...]], window: int, keep: int) -> None:
    """Leave ``cache`` (one row) holding only the first ``keep`` of the ``window`` tokens just fed."""

    from mlx_lm.models.gated_delta import gated_delta_update

    drop = int(window) - int(keep)
    if drop <= 0:
        return
    j = 0
    for item in cache:
        if hasattr(item, "keys") and hasattr(item, "values"):
            item.trim(drop)
            continue
        if j >= len(captured):
            raise RuntimeError("recurrent layer without recorded inputs")
        q, k, v, a, b, A_log, dt_bias, state0, mask, conv_input, kernel = captured[j]
        j += 1
        if keep > 0:
            _, state = gated_delta_update(
                q[:, :keep], k[:, :keep], v[:, :keep], a[:, :keep], b[:, :keep], A_log, dt_bias, state0,
                None if mask is None else mask[:, :keep], use_kernel=True,
            )
        else:
            state = state0
        item[1] = state
        item[0] = mx.contiguous(conv_input[:, keep: keep + kernel - 1, :])
    if j != len(captured):
        raise RuntimeError(f"recorded {len(captured)} recurrent layers, cache has {j}")


__all__ = ["begin", "end", "install", "rollback"]
