"""CUDA prompt precision: bf16 activations, or FP8 (e4m3, a scale a row) where a prompt matmul has an FP8 kernel."""

from __future__ import annotations

from contextlib import contextmanager

FP8_BY_DEFAULT = False      # the default: False = bf16 prompts (--prefill-fp8 asks for FP8), True = FP8 prompts
_fp8 = FP8_BY_DEFAULT


def fp8() -> bool:
    """Whether prompt matmuls that have an FP8 kernel run it (set once at startup, before any weight loads)."""

    return _fp8


def set_fp8(on: bool) -> None:
    global _fp8
    _fp8 = bool(on)


def same_on_ranks(rank0: int, rank1: int) -> None:
    """Refuse two ranks started with different prompt precision: their partial sums would mix FP8 and bf16 prompts."""

    if int(rank0) != int(rank1):
        names = ("bf16", "FP8")
        raise RuntimeError(f"the two ranks were started with different prompt precision (rank 0 {names[int(rank0)]}, "
                           f"rank 1 {names[int(rank1)]} prompts); give both the same --prefill-fp8 or --no-prefill-fp8")


@contextmanager
def using(on: bool):
    """FP8 prompts on or off inside the block (tests), the previous choice after."""

    was = _fp8
    set_fp8(on)
    try:
        yield
    finally:
        set_fp8(was)
