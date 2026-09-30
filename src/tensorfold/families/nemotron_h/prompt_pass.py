"""Nemotron's prompt as a pass of several chunks through mlx_lm's backbone, each chunk with its own forward's bits."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _runs(sizes: tuple[int, ...], joins: Any) -> list[tuple[int, list[int]]]:
    """Consecutive chunks grouped: a run of chunks for which ``joins(n)`` holds, every other chunk alone."""

    out: list[tuple[int, list[int]]] = []
    group: list[int] = []
    start = at = 0
    for n in sizes:
        if joins(n):
            group.append(n)
            at += n
            continue
        if group:
            out.append((start, group))
        out.append((at, [n]))
        at += n
        group, start = [], at
    if group:
        out.append((start, group))
    return out


def linear(layer: Any, x: mx.array, sizes: tuple[int, ...]) -> mx.array:
    """``layer(x)`` for x [1, R, K], each chunk its own call."""

    parts = [layer(x[:, a:a + n]) for a, n in zip(_starts(0, list(sizes)), sizes)]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def _aligned(switch: Any, rows: int, top: int) -> bool:
    """Whether a chunk's own expert call is MLX's sorted gather at 4+ pairs an expert (the aligned gather's bits)."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as pm

    fc = switch.fc1
    experts = int(fc["weight"].shape[0])
    return (rows * top >= 64 and rows * top // experts >= 4 and pm.fast_prefill() and pm.tiles()
            and all(int(p.bits) == 4 and int(p.group_size) % 32 == 0 for p in (switch.fc1, switch.fc2)))


def _experts(switch: Any, x: mx.array, idx: mx.array) -> mx.array:
    """SwitchMLP on x [R, D] for idx [R, k] through the aligned gather: [R, k, D] bf16."""

    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as pm

    h = mx.expand_dims(x, (-2, -3))
    h, ids, order = _gather_sort(h, idx)
    flat = h.reshape(-1, h.shape[-1])
    fc1, fc2 = switch.fc1, switch.fc2
    y = pm.gather_sorted(flat, fc1["weight"], fc1["scales"], fc1["biases"], ids)
    y = switch.activation(y)
    y = pm.gather_sorted(y, fc2["weight"], fc2["scales"], fc2["biases"], ids)
    y = y.reshape(*h.shape[:-1], y.shape[-1])
    return _scatter_unsort(y, order, idx.shape).squeeze(-2)


def moe(block: Any, x: mx.array, sizes: tuple[int, ...]) -> mx.array:
    """NemotronHMoE on x [1, R, D] for a pass (no latent projection): each chunk's own forward's bits."""

    top = int(block.num_experts_per_tok)
    outs = []
    for a, group in _runs(sizes, lambda n: _aligned(block.switch_mlp, n, top)):
        b = a + sum(group)
        part = x[:, a:b]
        if len(group) == 1:
            outs.append(block(part))                              # the chunk's own call
            continue
        routes = [block.gate(x[:, s:s + n]) for s, n in zip(_starts(a, group), group)]
        inds = mx.concatenate([r[0] for r in routes], axis=1)
        scores = mx.concatenate([r[1] for r in routes], axis=1)
        y = _experts(block.switch_mlp, part.reshape(-1, part.shape[-1]), inds.reshape(-1, top))
        y = y.reshape(*part.shape[:-1], top, -1)
        sums, at = [], 0
        for n in group:                                   # each chunk's own reduction over its experts
            sums.append((y[:, at:at + n] * scores[:, at:at + n, :, None]).sum(axis=-2).astype(y.dtype))
            at += n
        y = mx.concatenate(sums, axis=1)
        if block.config.n_shared_experts is not None:
            mlp = block.shared_experts                           # NemotronHMLP: down(relu2(up(x)))
            y = y + linear(mlp.down_proj, nn.relu2(linear(mlp.up_proj, part, tuple(group))), tuple(group))
        outs.append(y)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


def _starts(a: int, sizes: list[int]) -> list[int]:
    out = []
    for n in sizes:
        out.append(a)
        a += n
    return out


def hidden(backbone: Any, tokens: mx.array, cache: list[Any], sizes: tuple[int, ...]) -> mx.array:
    """mlx_lm's NemotronHModel on consecutive prompt chunks in one forward: the final-normed rows [1, R, D]."""

    from mlx_lm.models.base import create_ssm_mask

    starts = _starts(0, list(sizes))
    h = backbone.embeddings(tokens)
    counter = 0
    for layer in backbone.layers:
        x = layer.norm(h)
        kind = layer.block_type
        if kind in "M*":
            c = cache[counter]
            counter += 1
            outs = []
            for a, n in zip(starts, sizes):
                part = x[:, a:a + n]
                mask = ("causal" if n > 1 else None) if kind == "*" else create_ssm_mask(part, cache[backbone.ssm_idx])
                outs.append(layer.mixer(part, mask=mask, cache=c))
            y = mx.concatenate(outs, axis=1)
        elif kind == "E" and getattr(layer.mixer, "moe_latent_size", None) is None:
            y = moe(layer.mixer, x, sizes)
        else:
            y = mx.concatenate([layer.mixer(x[:, a:a + n]) for a, n in zip(starts, sizes)], axis=1)
        h = h + y
    return backbone.norm_f(h)


__all__ = ["hidden", "linear", "moe"]
