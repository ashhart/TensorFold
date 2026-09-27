"""Row-exact 4-bit matmul and stacked projections for targets; draft projections may use kernels without row-exactness."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import mlx.core as mx

# rows a verify window takes at most
WINDOW_ROWS = 16
# Maximum rows whose calls read prepared input fragments.
FRAGMENT_ROWS = 8


class Backend:
    """Provide row-exact qmm for up to max_rows rows and prepare each weight, scales, biases and group size once at install."""

    def __init__(self, name: str, qmm: Callable[..., mx.array], max_rows: int, fits: Callable[[Any], bool],
                 prepare: Callable[[list[tuple[mx.array, mx.array, mx.array, int]]], None] | None = None) -> None:
        self.name, self.qmm, self.max_rows, self.fits, self.prepare = name, qmm, int(max_rows), fits, prepare

    def __call__(self, x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
        return self.qmm(x, weight, scales, biases, group_size)


def simd_qmm_backend() -> Backend:
    """``simd_qmm``, its one-row calls through the MMA kernel for any shape whose scalar kernel's bits differ here."""

    from tensorfold.kernels.qwen.dense.v1 import simd_qmm

    def prepare(weights: list[tuple[mx.array, mx.array, mx.array, int]]) -> None:
        seen: set[tuple[int, int, int]] = set()
        for w, s, b, gs in weights:
            shape = (int(w.shape[0]), int(w.shape[1]) * 8, int(gs))
            if shape not in seen:
                seen.add(shape)
                if not simd_qmm.check(w, s, b, group_size=gs):
                    simd_qmm.mma_one_row.add(shape)

    def qmm(x: mx.array, w: mx.array, s: mx.array, b: mx.array, gs: int) -> mx.array:
        rows = x.size // int(x.shape[-1])
        if 2 <= rows <= FRAGMENT_ROWS and gs == simd_qmm.GROUP:   # same bits as simd_qmm.qmm(x), less input work
            return simd_qmm.qmm_fragments(simd_qmm.fragments(x), w, s, b).reshape(*x.shape[:-1], int(w.shape[0]))
        return simd_qmm.qmm(x, w, s, b, gs)

    return Backend("simd_qmm", qmm, int(simd_qmm.MAX_ROWS), simd_qmm.fits, prepare=prepare)


BACKEND: Backend | None = None


GROUPS: dict[str, tuple[str, ...]] = {
    "in": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),     # recurrent layers
    "qkv": ("q_proj", "k_proj", "v_proj"),                             # attention layers
    "gu": ("gate_proj", "up_proj"),                                     # MLPs
}
_ATTR = "_row_forward_stacks"


class Stack:
    """Weights of projections that read the same rows, concatenated along the outputs; the members hold views."""

    __slots__ = ("weight", "scales", "biases", "group_size", "sizes", "members", "held")

    def __init__(self, members: Sequence[Any]) -> None:
        self.members = tuple(members)
        self.group_size = int(members[0].group_size)
        self.sizes = tuple(int(m["weight"].shape[0]) for m in members)
        self.weight = mx.concatenate([m["weight"] for m in members], axis=0)
        self.scales = mx.concatenate([m["scales"] for m in members], axis=0)
        self.biases = mx.concatenate([m["biases"] for m in members], axis=0)
        mx.eval(self.weight, self.scales, self.biases)
        offset = 0
        for m, n in zip(members, self.sizes):
            m.weight = self.weight[offset:offset + n]
            m.scales = self.scales[offset:offset + n]
            m.biases = self.biases[offset:offset + n]
            offset += n
        mx.eval([a for m in members for a in (m["weight"], m["scales"], m["biases"])])
        self.held = tuple(m["weight"] for m in members)

    def valid(self) -> bool:
        return all(m["weight"] is w for m, w in zip(self.members, self.held))


def _stackable(members: Sequence[Any], backend: Backend) -> bool:
    import mlx.nn as nn

    if not all(isinstance(m, nn.QuantizedLinear) and backend.fits(m) and "bias" not in m for m in members):
        return False
    k8 = {int(m["weight"].shape[1]) for m in members}
    gs = {int(m.group_size) for m in members}
    return len(k8) == 1 and len(gs) == 1


def stack_of(parent: Any, kind: str) -> Stack | None:
    stacks = parent.__dict__.get(_ATTR)
    if stacks is None:
        return None
    stack = stacks.get(kind)
    return stack if stack is not None and stack.valid() else None


def build(model: Any, backend: Backend) -> dict[str, int]:
    """Stack every group now: {kind: groups stacked}. No weight is stored twice."""

    counts = {kind: 0 for kind in GROUPS}
    for _, module in model.named_modules():
        for kind, names in GROUPS.items():
            members = [getattr(module, name, None) for name in names]
            if any(m is None for m in members) or stack_of(module, kind) is not None:
                continue
            if not _stackable(members, backend):
                continue
            stacks = module.__dict__.setdefault(_ATTR, {})
            stacks[kind] = Stack(members)
            counts[kind] += 1
    mx.clear_cache()          # the replaced arrays' buffers would otherwise sit in MLX's buffer cache
    return counts


def project(module: Any, x: mx.array) -> mx.array:
    """One projection through the backend (any bias added after)."""

    y = BACKEND(x, module["weight"], module["scales"], module["biases"], module.group_size)
    if "bias" in module:
        y = y + module["bias"]
    return y


def project_stack(stack: Stack, x: mx.array) -> mx.array:
    return BACKEND(x, stack.weight, stack.scales, stack.biases, stack.group_size)


def logits(head: Any, x: mx.array) -> mx.array:
    """The head over rows ``x`` (final-normed hidden states [1, R, D]) through the row-exact matmul."""

    return BACKEND(x, head["weight"], head["scales"], head["biases"], head.group_size)


_DRAFT_ATTR = "_row_forward_draft"
_draft_orig: Any = None


def draft_matmul(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int,
                 bits: int = 4) -> mx.array:
    """Use the backend's multi-row matmul when available and MLX otherwise; draft logits need no row-exactness."""

    rows = x.size // int(x.shape[-1])
    if (BACKEND is not None and BACKEND.name == "simd_qmm" and 2 <= rows <= WINDOW_ROWS and bits == 4
            and group_size == 64):
        return BACKEND.qmm(x, weight, scales, biases, group_size)
    return mx.quantized_matmul(x, weight, scales, biases, transpose=True, group_size=group_size, bits=bits)


def _draft_call(self: Any, x: mx.array) -> mx.array:
    if not getattr(self, _DRAFT_ATTR, False):
        return _draft_orig(self, x)
    y = draft_matmul(x, self["weight"], self["scales"], self["biases"], self.group_size, self.bits)
    if "bias" in self:
        y = y + self["bias"]
    return y


def route_drafter(model: Any) -> int:
    """Route a draft model's 4-bit linears through ``draft_matmul``; returns how many. Idempotent."""

    global _draft_orig
    import mlx.nn as nn

    if BACKEND is None or BACKEND.name != "simd_qmm":
        return 0
    if _draft_orig is None:
        _draft_orig = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _draft_call
    count = 0
    warm, seen = [], set()
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and BACKEND.fits(module):
            object.__setattr__(module, _DRAFT_ATTR, True)
            count += 1
            shape = (int(module["weight"].shape[0]), int(module["weight"].shape[1]) * 8)
            if shape not in seen:                   # compiled now, not inside the first request
                seen.add(shape)
                for rows in (2, WINDOW_ROWS):
                    warm.append(draft_matmul(mx.zeros((rows, shape[1]), dtype=mx.bfloat16), module["weight"],
                                             module["scales"], module["biases"], module.group_size, module.bits))
    mx.eval(warm)
    return count


def fits(model: Any, backend: Backend) -> bool:
    """Whether every projection and the head take the backend's layout."""

    import mlx.nn as nn

    language_model = getattr(model, "language_model", model)
    head = getattr(language_model, "lm_head", None)
    if not isinstance(head, nn.QuantizedLinear) or not backend.fits(head):
        return False
    for layer in language_model.model.layers:
        inner = layer.linear_attn if getattr(layer, "is_linear", False) else layer.self_attn
        for _, module in list(inner.named_modules()) + list(layer.mlp.named_modules()):
            if isinstance(module, nn.QuantizedLinear) and not backend.fits(module):
                return False
    return True


def install(model: Any, backend: Backend | None = None) -> dict[str, int]:
    """Use ``backend`` (default ``simd_qmm``) and stack the model's projection groups. Idempotent."""

    global BACKEND
    BACKEND = backend or simd_qmm_backend()
    stacked = build(model, BACKEND)
    if BACKEND.prepare is not None:
        import mlx.nn as nn

        weights = [(m["weight"], m["scales"], m["biases"], int(m.group_size)) for _, m in model.named_modules()
                   if isinstance(m, nn.QuantizedLinear)]
        for _, module in model.named_modules():
            for stack in module.__dict__.get(_ATTR, {}).values():
                weights.append((stack.weight, stack.scales, stack.biases, stack.group_size))
        BACKEND.prepare(weights)
    return stacked


__all__ = ["BACKEND", "Backend", "GROUPS", "Stack", "WINDOW_ROWS", "build", "draft_matmul", "fits", "install", "logits",
           "project", "project_stack", "route_drafter", "simd_qmm_backend", "stack_of"]
