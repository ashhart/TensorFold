"""The two-rank DeepSeek-V4.1-Flash engine (not yet implemented: Phase 2 of TODO.md)."""

from __future__ import annotations

from pathlib import Path


class Dsv41Engine:
    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, engram: Path, drafts: bool,
                 context: int | None = None, context_explicit: bool | None = None) -> None:
        raise NotImplementedError("DeepSeek-V4.1-Flash's CUDA engine is under construction (TODO.md, Phase 2)")
