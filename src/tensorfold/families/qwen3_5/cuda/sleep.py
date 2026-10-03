"""Dense-Qwen sleep hooks: independent State objects and optional DFlash context."""

from __future__ import annotations

from typing import Any

FAMILY = "qwen3_5"


def settings(engine: Any) -> dict[str, Any]:
    """Only scalar settings survive the old runtime; no bound method or device array does."""

    weights = getattr(engine, "w", None)
    settings = {name: getattr(engine, name, None) for name in (
        "context_window", "tp", "rank", "max_rows", "tree_rows", "allow_copy", "concurrent", "vision_enabled")}
    settings.update({name: getattr(weights, name, None) for name in ("precision", "quant", "fast_prefill")})
    settings["own"] = tuple(sorted((getattr(weights, "own", None) or {}).items()))
    settings["draft"] = getattr(engine, "draft", None) is not None
    settings["vision"] = getattr(engine, "vision", None) is not None
    settings["streams"] = getattr(getattr(engine, "scheduler", None), "max_streams", 1)
    return settings



def prefix_codec():
    from . import prefix_snapshot

    return prefix_snapshot


def cache(engine):
    return engine.multi.cache if engine.concurrent else engine.cache


def close(engine):
    engine.close()


def attach(engine, store):
    owner = engine.multi if engine.concurrent else engine
    room = None if engine.concurrent else engine.room
    owner.cache = store.attach(owner.cache, device=engine.w.norm.device, room=room,
                               admit=lambda size, prompt: prefix_room(engine, size, prompt))
    if room is not None:
        room.cache = owner.cache


def prefix_room(engine, size: int, prompt: list[int]) -> bool:
    """Reserve space for both the saved prefix and the request's working state."""

    if engine.concurrent:
        decoder = engine.multi
        if decoder.memory_gate is None:
            return True
        from tensorfold.cuda.streams import Stream

        rows = decoder._first(Stream(prompt, engine.context_window - len(prompt)))
        return decoder.memory_gate.fits(size + rows * decoder.row_bytes)
    from tensorfold.cuda.prefix_store import prefix_room as fits

    return fits(size, prompt)
