"""How many requests share a round: the process budget, less what the rest of the machine holds."""

from __future__ import annotations

import os
from typing import Any


def _envelope_enabled() -> bool:
    return os.environ.get("TF_ADMISSION_ENVELOPE") == "1"


class EnvelopeStreamMemory:
    """The measured StreamMemory with the prefill workspace priced by a monotonic
    envelope over observed peaks instead of the two-point linear fit.

    Everything but ``prefill_bytes`` is the base memory verbatim (attribute
    delegation): stream cache growth, per-token growth and the shared-round
    working set are measured models with honest receipts and stay untouched.
    Within the envelope's observed range, workspace prices at margin * hull(L).
    Beyond it, a family that declared its regime switch extends the last hull
    segment; an undeclared family keeps the shipped fit — extension from
    possibly wrong-regime seeds under-prices silently (mission P1, Q3/D8).
    """

    # attributes the admission path reads on StreamMemory, forwarded explicitly
    # (no __getattr__: special-method and type checks must not silently proxy)
    _FORWARD = ("short", "short_tokens", "long", "long_tokens", "per_token",
                "round_bytes", "chunk", "prefill_a", "prefill_b", "stream_bytes")

    def __init__(self, base: Any, envelope: Any, extend_beyond: bool) -> None:
        self._base = base
        self._envelope = envelope
        self._extend = extend_beyond
        for name in self._FORWARD:
            setattr(self, name, getattr(base, name))

    def prefill_bytes(self, tokens: int) -> int:
        L = int(tokens)
        hull = self._envelope.snapshot()
        if hull and (L <= hull[-1][0] or self._extend):
            return int(self._envelope.price(L))
        return int(self._base.prefill_bytes(L))


def _build_envelope_stream(engine: Any, base: Any, seeds: list) -> Any:
    """Wrap the measured StreamMemory with the ladder pricing (flag ON only).

    TF_ADMISSION_ENVELOPE=1: prefill workspace prices from the measured ladder
    (prefix-max staircase over raw probe peaks). Flag OFF calls measure(engine)
    exactly as shipped. Learning (observe/rung insertion) is v1.1: this PR is
    pricing-only; the counters exist and the hook design is reviewed separately.
    """

    from tensorfold.engine.ladder import LadderWorkspace

    model = getattr(engine, "model", None)
    # the DECLARATION is the #245 attribute (family asserts its prefill steps there),
    # not the mere existence of index_topk: every sparse model carries the config
    # field; only a family that measured its switch may extend past the last point
    switch = getattr(model, "prefill_regime_switch", None)
    if switch is None and getattr(model, "declares_prefill_regime_switch", False):
        switch = getattr(getattr(model, "args", None), "index_topk", None)
    switch_val: int | None = None
    if isinstance(switch, int) and switch > 0:
        switch_val = switch
    elif isinstance(switch, str) and switch.startswith("args."):
        switch_val = getattr(getattr(model, "args", None), switch[5:], None)
        switch_val = int(switch_val) if isinstance(switch_val, int) and switch_val > 0 else None
    inner = LadderWorkspace([(float(n), float(t)) for n, t in seeds], switch=switch_val)
    seeds_out = ", ".join(f"{n:,}" for n, _ in sorted(seeds))
    print(f"[tensorfold] admission workspace ladder: rungs at {seeds_out} tokens "
          f"(raw prefill peaks)", flush=True)
    # the ladder NEVER defers beyond its frontier (global max, by design):
    # the shipped fit's fallback for undeclared families is gone, not gated
    return EnvelopeStreamMemory(base, inner, extend_beyond=True)


def concurrency(engine: Any, prompt_memory: Any, fraction: float, lanes: int, reply_tokens: int) -> Any:
    """Admit within the default RAM share or a larger process budget, accounting for memory held elsewhere."""

    from tensorfold.engine import memory

    if _envelope_enabled():
        envelope_seeds: list | None = []
        stream = memory.measure(engine, envelope_seeds=envelope_seeds)
        if envelope_seeds:
            stream = _build_envelope_stream(engine, stream, envelope_seeds)
    else:
        stream = memory.measure(engine)          # the shipped call, unchanged
    getattr(engine, "release_rounds", lambda: None)()     # the probe round's rollback rows: no stream's
    used = memory._mlx_used()
    ram = memory.ram_bytes()
    elsewhere = memory.used_elsewhere(used)
    allowance = int(fraction * ram)
    share = prompt_memory.budget if prompt_memory is not None else allowance
    if prompt_memory is not None:
        allowance = max(allowance, prompt_memory.process_budget)
    admission = memory.Admission(min(allowance - elsewhere, share), stream,
                                 used=None if prompt_memory is None else prompt_memory.held, lanes=lanes)
    tokens = reply_tokens + 4096
    gib, mib = 1024**3, 1024**2
    print(f"[tensorfold] concurrency: up to {lanes} requests share each round; memory budget "
          f"{admission.budget / gib:.1f} GB (MLX's share {share / gib:.1f} GB, or {allowance / ram:.0%} of "
          f"{ram / gib:.0f} GB less {elsewhere / gib:.1f} GB in use elsewhere); a stream "
          f"{stream.short / mib:.0f} MB at {stream.short_tokens} tokens, "
          f"{stream.long / mib:.0f} MB at {stream.long_tokens:,}, then {stream.per_token / 1024:.1f} KB a token; "
          f"a shared round up to {stream.round_bytes / gib:.2f} GB at {lanes} streams; {admission.fitting(tokens)} "
          f"streams of "
          f"{tokens:,} tokens fit now (more wait their turn)", flush=True)
    return admission


__all__ = ["concurrency"]
