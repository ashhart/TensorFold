"""RoPE re-rotation of stored keys, and the splice primitive over a ``State``.

TensorFold commits keys post-RoPE: ``glue.attn_prep`` (a Triton kernel) rotates
each key at its absolute position with the rotate-half pairing — dim ``d`` pairs
with ``d + HALF`` (``HALF = inv_freq.numel()``),
``x'_d = x_d cos - x_{d+H} sin`` / ``x'_{d+H} = x_{d+H} cos + x_d sin``,
``angle = pos * inv_freq[d % HALF]``. Rotating a stored key from position ``p`` to
``p + delta`` is therefore the same rotation applied at ``angle = delta *
inv_freq`` — the composition ``R(p+delta) = R(delta) R(p)`` — which is exactly
SCR's ``_rotate_keys`` math, rebuilt from ``w.inv_freq`` instead of SGLang's
``cos_sin_cache``. Re-rotation runs once per splice, off the decode hot path.
"""

import torch


def rotate_keys_inplace(k: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> None:
    """Re-rotate committed keys ``k`` (``(n, kv_heads, head_dim)``, bf16) stored at
    position ``p`` to position ``p + delta``. A no-op at ``delta == 0``.

    Qwen3.8 uses partial rotary: only the first ``2 * inv_freq.numel()`` dims of
    each head rotate (pairs ``(d, d + half)`` for ``d < half``), the rest pass
    through — the same regions ``glue.attn_prep``'s kernel handles."""
    if delta == 0 or k.shape[0] == 0:
        return
    d = k.shape[-1]
    half = inv_freq.numel()
    rot = 2 * half
    if d < rot:
        raise ValueError(f"head_dim {d} < rotary region {rot}; unexpected rotary layout")
    f32 = k.float()
    ang = float(delta) * inv_freq.float()          # (half,)
    cos, sin = ang.cos(), ang.sin()
    first, second = f32[..., :half], f32[..., half:rot]
    out = torch.cat([first * cos - second * sin,
                     second * cos + first * sin,
                     f32[..., rot:]], dim=-1)
    k.copy_(out.to(k.dtype))


def _grow(st, i: int, need: int, grow_fn=None):
    """The engine's ``grow`` for layer ``i``'s caches to ``need`` rows (frees kept
    buffers through ``st.room`` when memory is short). ``grow_fn`` overrides it in
    tests (a fake ``State`` has no engine ``grow``)."""
    if grow_fn is not None:
        return grow_fn(st, i, need)
    from tensorfold.families.qwen3_5.cuda.forward import grow

    return grow(st, i, need)


def execute_splice(st, sp, snap, prompt_ids, inv_freq, log=None, rid="?", blocks="",
                   grow_fn=None) -> bool:
    """Relocate one span: copy ``snap`` rows ``[old, old+n)`` into ``st`` at
    ``[at, at+n)``, re-rotating by the position shift. Returns False on any guard
    failure — the caller then falls back to plain prefill from ``st.pos``."""
    at, old, n = sp.at, sp.old, sp.n
    if st.pos != at:
        if log:
            log(f"splice declined: st.pos {st.pos} != at {at}")
        return False
    if at + n > len(prompt_ids) or old + n > snap.pos:
        if log:
            log(f"splice declined: range out of bounds at={at} n={n} old={old} snap.pos={snap.pos}")
        return False
    if list(prompt_ids[at:at + n]) != list(snap.ids[old:old + n]):
        if log:
            log(f"splice declined: token guard failed (snapshot rows stale) rid={rid}")
        return False
    delta = at - old
    for i, pair in enumerate(st.kv):
        if pair is None:
            continue                                # linear-attention layer: no rows to move
        k, v = pair
        if k.shape[0] < at + n:
            _grow(st, i, at + n, grow_fn=grow_fn)
            k, v = st.kv[i]
        k[at:at + n] = snap.kv[i][0][old:old + n]
        v[at:at + n] = snap.kv[i][1][old:old + n]
        if delta:
            rotate_keys_inplace(k[at:at + n], delta, inv_freq)
    st.pos = at + n
    if log:
        log(f"splice OK: rid={rid} at={at} old={old} rows={n} delta={delta} {blocks}")
    return True