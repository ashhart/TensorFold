"""Estimate local checkpoint weights against MLX budgets without a model or workspace probe."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys

from tensorfold.server import memory_budget


def _positive(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _checkpoint_bytes(directory: Path) -> int:
    """Require complete local safetensors before treating file sizes as weight evidence."""

    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = json.loads(index.read_text())["weight_map"]
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("local weight index is invalid; no weight estimate is available") from None
        if not isinstance(names, dict) or not names:
            raise ValueError("local weight index is empty; no weight estimate is available")
        files = []
        for name in names.values():
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("local weight index contains an unsafe shard name")
            path = directory / name
            if path.suffix != ".safetensors" or not path.is_file():
                raise FileNotFoundError("local weight shard is missing; complete the checkpoint before planning")
            files.append(path)
        size = sum(path.stat().st_size for path in set(files))
    else:
        from tensorfold import hub

        files = [path for path in directory.glob("*.safetensors") if path.is_file()]
        if files and any(directory.glob("model-*-of-*.safetensors")) and not hub._cached_weights_complete(directory):
            raise FileNotFoundError("local weight shards are incomplete; no weight estimate is available")
        size = sum(path.stat().st_size for path in files)
    if size <= 0 or any(path.stat().st_size <= 0 for path in files):
        raise FileNotFoundError("local safetensors weights are missing or empty; no weight estimate is available")
    return size


def _resident_weights(directory: Path, package, checkpoint: int) -> tuple[int, str]:
    estimate = getattr(package, "weight_bytes", None)
    if estimate is None:
        return checkpoint, "local safetensors file sizes"
    if not callable(estimate):
        raise ValueError("family weight estimate is unavailable")
    defaults = {"ple_on_ssd": False, "ssd_experts": None}
    signature = inspect.signature(estimate)
    kwargs = {name: value for name, value in defaults.items() if name in signature.parameters}
    value = estimate(directory, **kwargs)
    number = _positive(value, "family weight estimate")
    if number > sys.maxsize:
        raise ValueError("family weight estimate is too large to size")
    weights = int(number)
    if weights <= 0:
        raise ValueError("family weight estimate is zero; no weight estimate is available")
    return weights, "family resident-weight estimate from local files"


def _budget(mx, package, ram: int, environ) -> tuple[int, float, int]:
    fraction = _positive(memory_budget.model_fraction(package, ram), "model memory allowance")
    if fraction > 1:
        raise ValueError("model memory allowance must not exceed physical RAM")
    limit = memory_budget.memory_limit_bytes(mx, fraction=fraction, physical_bytes=ram, environ=environ)
    return limit, fraction, memory_budget.budget_ceiling(mx, ram)


def cmd_plan(args: argparse.Namespace) -> int:
    """Report a weights-only local estimate with the same budget resolver as serve."""

    if sys.platform != "darwin":
        raise ValueError("plan estimates the Mac MLX path; CUDA capacity is reported by its own startup")
    explicit = None if args.memory_gb is None else _positive(args.memory_gb, "--memory-gb")
    classes = list(args.ram or ())
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in classes):
        raise ValueError("--ram must be a positive integer number of GiB")
    from tensorfold import families, hub
    from tensorfold.cli import _note_untested

    directory = hub.resolve(args.model, download=False)
    if not (directory / "config.json").is_file():
        raise FileNotFoundError("plan needs a local or already cached config.json; no download is attempted")
    family = families.detect(directory)
    families.require_readable(family, families.read_config(directory), "mlx")
    _note_untested(family, args.model)
    checkpoint = _checkpoint_bytes(directory)
    weights, provenance = _resident_weights(directory, family.package, checkpoint)
    import mlx.core as mx

    ram = memory_budget.physical_memory_bytes()
    if ram <= 0:
        raise ValueError("physical memory must be positive")
    environ = dict(os.environ)
    budget, fraction, ceiling = _budget(mx, family.package, ram, environ)
    budgets = [("this Mac's current budget", budget)]
    if explicit is not None:
        selected = {**environ, memory_budget.LIMIT_ENV: str(explicit)}
        budgets.append(("--memory-gb", _budget(mx, family.package, ram, selected)[0]))
    for value in classes:
        class_ram = value * memory_budget.GIB
        budgets.append((f"--ram {value} GiB class", _budget(mx, family.package, class_ram, environ)[0]))
    gib, process = memory_budget.GIB, memory_budget.PROCESS_BYTES
    print(f"[tensorfold] plan for {family.title} ({args.model})")
    print(f"[tensorfold] RAM {ram / gib:.0f} GiB, default allowance {fraction:.0%}, ceiling {ceiling / gib:.1f} GiB")
    print(f"[tensorfold] weights {weights / gib:.1f} GiB ({provenance}; local files {checkpoint / gib:.1f} GiB)")
    print(f"[tensorfold] scope: checkpoint weights plus {process / gib:.0f} GiB process reserve; serve still measures "
          "prompt workspace, caches and streams, and loads any drafter")
    if memory_budget.LIMIT_ENV in environ:
        print(f"[tensorfold] current budgets honor {memory_budget.LIMIT_ENV}={environ[memory_budget.LIMIT_ENV]}")
    if classes:
        print("[tensorfold] RAM classes use each class's allowance and RAM, capped by this Mac's reported GPU ceiling")
    verdict = 0
    for name, amount in budgets:
        headroom = amount - process - weights
        if headroom <= 0:
            print(f"[tensorfold] {name}: {amount / gib:.1f} GiB budget; weights exceed the allowance; need more than "
                  f"{(weights + process) / gib:.1f} GiB for weights and process reserve")
            verdict = 1
        else:
            print(f"[tensorfold] {name}: {amount / gib:.1f} GiB budget; weights within the allowance; "
                  f"{headroom / gib:.1f} GiB remains before workspace, caches and drafter")
    return verdict
