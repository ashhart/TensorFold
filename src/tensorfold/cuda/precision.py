"""CUDA math for NVFP4 checkpoints: their own (FP4 x FP4, FP8 x FP8 where a GPU has each mma) or full (bf16 rows)."""

from __future__ import annotations

from contextlib import contextmanager

CHECKPOINT, FULL = "checkpoint", "full"
CHOICES = (CHECKPOINT, FULL)
_mode = CHECKPOINT
_asked = False


def own_math(capability: tuple[int, int]) -> dict[str, bool]:
    """Formats in the checkpoint's own math at ``capability``: NVFP4 on SM 12.x, FP8 from SM 8.9; the rest at full."""

    major, minor = (int(v) for v in capability)
    return {"nvfp4": major == 12, "fp8": (major, minor) >= (8, 9)}


def mode() -> str:
    """The math NVFP4 checkpoints run (set once at startup, before any weight loads)."""

    return _mode


def asked() -> bool:
    """Whether the mode was named (``--precision``); either way no supported GPU is refused."""

    return _asked


def set_mode(value: str, asked: bool = False) -> None:
    global _mode, _asked
    if value not in CHOICES:
        raise ValueError(f"--precision {value}: choose one of {', '.join(CHOICES)}")
    _mode, _asked = value, bool(asked)


@contextmanager
def using(value: str, asked: bool = False):
    """A mode inside the block (tests), the previous one after."""

    was = (_mode, _asked)
    set_mode(value, asked)
    try:
        yield
    finally:
        set_mode(*was)
