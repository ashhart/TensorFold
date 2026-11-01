"""The memory verdict: TensorFold's budget rules applied to a checkpoint's size, before anything downloads.

Calibrated on Apple Silicon: the public 32 GB report (issue #50) and the Mac mini M6 32 GB measured in this project
(Gemma 4 admitted 43,008 tokens at the default budget; Qwen3.8-27B with its drafter 1,024 tokens at the 25 GiB GPU
ceiling and 17,408 at 28 GiB). CUDA machines use TensorFold's own rule for discrete cards (free memory less a tenth,
at least 4 GiB) but nothing has been measured there yet, so their verdicts say so.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

GIB = 1024**3
DEFAULT_FRACTION = 0.70       # TensorFold's default process allowance on Macs
RAISED_FRACTION = 0.85        # what a Mac used mostly for TensorFold can give it (TENSORFOLD_MEMORY_LIMIT_GB)
PROCESS_RESERVE = 3 * GIB     # kept outside the MLX allocator
DEFAULT_WORKSPACE = 3 * GIB   # one prompt chunk, caches and draft rows when a model does not say otherwise
COMFORT_TOKENS = 16_384       # context at the default budget that counts as "fits"
MIN_TOKENS = 2_048            # context at the raised budget that still counts as "fits, tight"
RETAINED_COPIES = 2           # TensorFold sizes the window so a request's prompt can also be kept for the next turn
# TensorFold 1.0's native engine sizes its own memory up to what macOS lets the GPU hold, with no budget setting;
# measured on the M6 (10 Oct 2026): Nemotron idles 4.2 GiB above `info`'s floor (lanes, caches), then about 13 KB a
# token of prompt. Its window is not doubled for a kept prompt as the Python line's is.
NATIVE_OVERHEAD = int(4.2 * GIB)
NATIVE_COPIES = 1
RAM_CLASSES_GB = (16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512)


@dataclass
class Capacity:
    """The memory a backend can give the engine, and the rules that shape its budget."""

    bytes: int                 # RAM on unified machines, VRAM on a discrete card
    fraction: float            # default share of it
    raised: float              # the most the user can ask for
    reserve_bytes: int         # kept outside the model's allocator
    ceiling_bytes: int | None  # a platform limit below the fractions (macOS's GPU working set)
    basis: str                 # "mlx-calibrated", "gb10-floor-measured", "cuda-unverified", "native-measured",
                               # "native-unverified" (macOS's GPU limit unread, guessed), "none"


def mlx_capacity(ram_bytes: int, ceiling_bytes: int | None) -> Capacity:
    return Capacity(ram_bytes, DEFAULT_FRACTION, RAISED_FRACTION, PROCESS_RESERVE, ceiling_bytes, "mlx-calibrated")


def native_capacity(ram_bytes: int, ceiling_bytes: int | None) -> Capacity:
    """The native engine on a Mac: macOS's GPU working-set limit is the budget; nothing is kept back for it."""

    if ceiling_bytes and ram_bytes:
        share = ceiling_bytes / ram_bytes
        return Capacity(ram_bytes, share, share, 0, ceiling_bytes, "native-measured")
    # the limit could not be read (no MLX to ask): a guess, and the verdicts say so
    return Capacity(ram_bytes, RAISED_FRACTION, RAISED_FRACTION, 0, None, "native-unverified")


def cuda_capacity(memory_bytes: int, *, unified: bool) -> Capacity:
    """TensorFold grants a card its memory less a floor of a tenth, at least 4 GiB; unified hosts the same on RAM."""

    floor = max(4 * GIB, memory_bytes // 10)
    share = max(0.1, (memory_bytes - floor) / memory_bytes) if memory_bytes else 0.1
    # checked on a GB10 (GX10, 10 Oct 2026): TensorFold 1.0.4 kept a 12.2 GiB reserve of 122 GB, this very rule.
    # Only that floor was measured there: the process reserve and the room kept for a conversation are the Mac's
    # figures, hence "floor-measured". Discrete cards are not measured yet
    return Capacity(memory_bytes, share, share, 1 * GIB if not unified else PROCESS_RESERVE, None,
                    "gb10-floor-measured" if unified else "cuda-unverified")


@dataclass
class Verdict:
    level: str                 # "fits", "tight", "no"
    label: str                 # "Fits", "Fits, tight", "Needs 64 GB"
    budget_bytes: int          # the budget the level assumes
    budget_fraction: float
    headroom_bytes: int        # what is left for context at that budget
    context_tokens: int        # estimate at that budget
    left_for_apps_bytes: int   # capacity - budget
    needs_ram_gb: int | None   # for "no": the smallest class that would fit
    reason: str
    weights_bytes: int = 0     # what the estimate counted as resident: model plus the drafter's resident share
    basis: str = "mlx-calibrated"

    def to_dict(self) -> dict:
        return asdict(self)


def budget(capacity_bytes: int, fraction: float, ceiling: int | None = None) -> int:
    b = int(capacity_bytes * fraction)
    return min(b, int(ceiling)) if ceiling else b


def context_estimate(head: int, bytes_per_token: int, max_context: int, copies: int = RETAINED_COPIES) -> int:
    if head <= 0 or bytes_per_token <= 0:
        return 0
    return max(0, min(int(head // (copies * bytes_per_token)), max_context)) // 1024 * 1024


def smallest_class(weights: int, workspace: int, bytes_per_token: int, reserve: int, fraction: float,
                   copies: int = RETAINED_COPIES) -> int | None:
    need = (weights + reserve + workspace + copies * COMFORT_TOKENS * bytes_per_token) / fraction
    return next((gb for gb in RAM_CLASSES_GB if gb * GIB >= need), None)


def verdict(cap: Capacity, weights_bytes: int, *, bytes_per_token: int, max_context: int,
            workspace_bytes: int = DEFAULT_WORKSPACE, model_fraction: float | None = None,
            requested_fraction: float | None = None, strict: bool = False, copies: int = RETAINED_COPIES) -> Verdict:
    """Three levels for ``cap``. ``requested_fraction`` is the budget the user set.

    Without ``strict`` the verdict may answer for a higher budget than the one requested when the requested one
    cannot run the model, and says which budget it used. With ``strict`` it answers for the requested budget only.
    """

    if weights_bytes <= 0 or cap.bytes <= 0:
        raise ValueError("weights and capacity must be positive")
    default = model_fraction or cap.fraction
    raised = max(default, cap.raised)

    def at(f: float) -> tuple[int, int]:
        head = budget(cap.bytes, f, cap.ceiling_bytes) - cap.reserve_bytes - weights_bytes - workspace_bytes
        return head, context_estimate(head, bytes_per_token, max_context, copies)

    def pack(level: str, label: str, f: float, reason: str) -> Verdict:
        head, tokens = at(f)
        b = budget(cap.bytes, f, cap.ceiling_bytes)
        need = (smallest_class(weights_bytes, workspace_bytes, bytes_per_token, cap.reserve_bytes, default, copies)
                if level == "no" else None)
        if cap.basis == "native-unverified" and level != "no":
            label += ", probably"                 # sized on a guessed GPU limit
            reason += " macOS's GPU memory limit could not be read, so this assumes 85% of memory."
        return Verdict(level, label, b, round(f, 2), head, tokens, cap.bytes - b, need, reason, weights_bytes,
                       cap.basis)

    if strict and requested_fraction:
        head, tokens = at(requested_fraction)
        if head >= 0 and tokens >= COMFORT_TOKENS:
            return pack("fits", "Fits", requested_fraction, "Fits at this budget with room for context.")
        if head >= 0 and tokens >= MIN_TOKENS:
            return pack("tight", "Fits, tight", requested_fraction, "Runs at this budget with a short context.")
        return pack("no", "Does not fit at this budget", requested_fraction,
                    "At this budget the weights and one prompt chunk leave no usable context.")

    chosen = requested_fraction or default
    head_default, tokens_default = at(default)
    if head_default >= 0 and tokens_default >= COMFORT_TOKENS:
        if chosen != default and MIN_TOKENS <= at(chosen)[1] < COMFORT_TOKENS:
            # the budget set in Settings is the one a start uses: there it is tight, whatever the default would give
            return pack("tight", "Fits, tight", chosen, "At the memory budget set in Settings it runs with a short "
                        "context; the default budget gives it room.")
        f = chosen if at(chosen)[1] >= MIN_TOKENS else default
        return pack("fits", "Fits", f, "Fits at the default memory budget, with room for context and for your apps.")
    head_raised, tokens_raised = at(raised)
    if head_raised >= 0 and tokens_raised >= MIN_TOKENS:
        f = chosen if at(chosen)[1] >= MIN_TOKENS else raised
        gb = math.ceil(budget(cap.bytes, f, cap.ceiling_bytes) / GIB)
        return pack("tight", "Fits, tight", f,
                    f"Fits when the engine may use {gb} GB; context is limited and other apps should stay light.")
    need = smallest_class(weights_bytes, workspace_bytes, bytes_per_token, cap.reserve_bytes, default, copies)
    return pack("no", f"Needs {need} GB" if need else "Needs more memory", raised,
                f"Its weights are {weights_bytes / GIB:.0f} GB; this machine cannot hold them beside a prompt.")
