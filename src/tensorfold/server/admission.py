"""How many requests share a round: the process budget, less what the rest of the machine holds."""

from __future__ import annotations

from typing import Any


def concurrency(engine: Any, prompt_memory: Any, fraction: float, lanes: int, reply_tokens: int) -> Any:
    """Admit within the default RAM share or a larger process budget, accounting for memory held elsewhere."""

    from tensorfold.engine import memory

    stream = memory.measure(engine)
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
