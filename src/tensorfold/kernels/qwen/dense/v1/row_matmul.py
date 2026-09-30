"""Row-exact affine matmul and compatible stacked projections for targets; draft projections may use stock kernels."""

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
                 prepare: Callable[[list[tuple[mx.array, mx.array, mx.array, int, int]]], None] | None = None) -> None:
        self.name, self.qmm, self.max_rows, self.fits, self.prepare = name, qmm, int(max_rows), fits, prepare

    def __call__(self, x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int,
                 bits: int = 4) -> mx.array:
        if bits == 4:
            return self.qmm(x, weight, scales, biases, group_size)
        return self.qmm(x, weight, scales, biases, group_size, bits)


def simd_qmm_backend() -> Backend:
    """``simd_qmm`` (4-bit) and ``simd_qmm_bits`` (5/6/8-bit, groups of 64), else affine_rows; checked per shape."""

    from tensorfold.kernels.qwen.dense.v1 import affine_rows, simd_qmm, simd_qmm_bits

    def prepare(weights: list[tuple[mx.array, mx.array, mx.array, int, int]]) -> None:
        seen: set[tuple[Any, ...]] = set()
        for w, s, b, gs, bits in weights:
            shape = (int(w.shape[0]), int(w.shape[1]) * 32 // bits, int(gs))
            key = (*shape, bits, s.dtype)
            if key not in seen:
                seen.add(key)
                if fast(w, s, b, gs, bits):
                    if not simd_qmm.check(w, s, b, group_size=gs):
                        simd_qmm.mma_one_row.add(shape)
                elif simd_qmm_bits.fits(w, s, b, gs, bits):
                    if not simd_qmm_bits.check(w, s, b, bits):
                        simd_qmm_bits.fallback.add((shape[0], shape[1], bits))
                else:
                    mx.eval(affine_rows.qmm(mx.zeros((1, shape[1]), dtype=mx.bfloat16), w, s, b, gs, bits))

    def fast(w: mx.array, s: mx.array, b: mx.array, gs: int, bits: int) -> bool:
        return (bits == 4 and gs in (32, 64) and s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16
                and int(w.shape[0]) % 8 == 0)

    def qmm(x: mx.array, w: mx.array, s: mx.array, b: mx.array, gs: int, bits: int = 4) -> mx.array:
        if not fast(w, s, b, gs, bits):
            if simd_qmm_bits.fits(w, s, b, gs, bits):
                return simd_qmm_bits.qmm(x, w, s, b, bits)
            return affine_rows.qmm(x, w, s, b, gs, bits)
        rows = x.size // int(x.shape[-1])
        if 2 <= rows <= FRAGMENT_ROWS and gs == simd_qmm.GROUP:   # same bits as simd_qmm.qmm(x), less input work
            return simd_qmm.qmm_fragments(simd_qmm.fragments(x), w, s, b).reshape(*x.shape[:-1], int(w.shape[0]))
        return simd_qmm.qmm(x, w, s, b, gs)

    return Backend("simd_qmm", qmm, int(simd_qmm.MAX_ROWS), affine_rows.fits, prepare=prepare)


BACKEND: Backend | None = None


GROUPS: dict[str, tuple[str, ...]] = {
    "in": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),     # recurrent layers
    "qkv": ("q_proj", "k_proj", "v_proj"),                             # attention layers
    "gu": ("gate_proj", "up_proj"),                                     # MLPs
}
_ATTR = "_row_forward_stacks"


class Stack:
    """Weights of projections that read the same rows, concatenated along the outputs; the members hold views."""

    __slots__ = ("weight", "scales", "biases", "group_size", "bits", "sizes", "members", "held")

    def __init__(self, members: Sequence[Any]) -> None:
        formats = {(int(m.bits), int(m.group_size), getattr(m, "mode", "affine"), int(m["weight"].shape[1]),
                    m["scales"].dtype, m["biases"].dtype) for m in members}
        if len(formats) != 1:
            raise ValueError("Stacked projections must share their bit width, group size, layout and scale dtype")
        self.members = tuple(members)
        self.group_size = int(members[0].group_size)
        self.bits = int(members[0].bits)
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
        return all(m["weight"] is w and int(m.bits) == self.bits and int(m.group_size) == self.group_size
                   and getattr(m, "mode", "affine") == "affine" for m, w in zip(self.members, self.held))


def _stackable(members: Sequence[Any], backend: Backend) -> bool:
    import mlx.nn as nn

    if not all(isinstance(m, nn.QuantizedLinear) and backend.fits(m) and "bias" not in m for m in members):
        return False
    k8 = {int(m["weight"].shape[1]) for m in members}
    formats = {(int(m.bits), int(m.group_size), getattr(m, "mode", "affine"), m["scales"].dtype) for m in members}
    return len(k8) == 1 and len(formats) == 1


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
    """One projection through the backend (any bias added after); a module with ``project_rows`` runs its own."""

    own = getattr(module, "project_rows", None)
    if own is not None:
        return own(x)
    y = BACKEND(x, module["weight"], module["scales"], module["biases"], module.group_size, module.bits)
    if "bias" in module:
        y = y + module["bias"]
    return y


def project_stack(stack: Stack, x: mx.array) -> mx.array:
    return BACKEND(x, stack.weight, stack.scales, stack.biases, stack.group_size, stack.bits)


def logits(head: Any, x: mx.array) -> mx.array:
    """The head over rows ``x`` (final-normed hidden states [1, R, D]) through the row-exact matmul."""

    own = getattr(head, "project_rows", None)
    if own is not None:
        return own(x)
    return BACKEND(x, head["weight"], head["scales"], head["biases"], head.group_size, head.bits)


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
            shape = (int(module["weight"].shape[0]), int(module["weight"].shape[1]) * 32 // module.bits,
                     int(module.bits), int(module.group_size))
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
    head = getattr(head, "inner", head)              # a head that transforms its rows first wraps its matmul
    if not isinstance(head, nn.QuantizedLinear) or not backend.fits(head):
        return False
    for layer in language_model.model.layers:
        inner = layer.linear_attn if getattr(layer, "is_linear", False) else layer.self_attn
        # MoE: router and experts run MLX's kernels (row_forward.moe); only the shared expert uses the backend
        mlp = layer.mlp.shared_expert if hasattr(layer.mlp, "switch_mlp") else layer.mlp
        for _, module in list(inner.named_modules()) + list(mlp.named_modules()):
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

        weights = [(m["weight"], m["scales"], m["biases"], int(m.group_size), int(m.bits)) for _, m in model.named_modules()
                   if isinstance(m, nn.QuantizedLinear)]
        for _, module in model.named_modules():
            for stack in module.__dict__.get(_ATTR, {}).values():
                weights.append((stack.weight, stack.scales, stack.biases, stack.group_size, stack.bits))
        BACKEND.prepare(weights)
    return stacked


__all__ = ["BACKEND", "Backend", "GROUPS", "Stack", "WINDOW_ROWS", "build", "draft_matmul", "fits", "install", "logits",
           "project", "project_stack", "route_drafter", "simd_qmm_backend", "stack_of"]
