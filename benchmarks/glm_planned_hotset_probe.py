#!/usr/bin/env python3
"""Replay a GLM route trace with a planned slot-arena hotset.

This is intentionally a probe, not product code. It is used to answer one
question: does prewarming a route-derived expert hotset make GLM generation fast
enough to justify a high-memory TensorFold tier?
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx

from smarttensor.adapters.mlx import DeepSeekV3StreamingForwardRunner


def _rss_kib(pid: int) -> int:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        ).strip()
        return int(output or "0")
    except Exception:
        return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _route_hotsets(route_path: Path, k: int) -> dict[int, list[int]]:
    route_data = json.loads(route_path.read_text())["routes"]
    counters: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for record in route_data:
        counters[int(record["layer"])].update(int(e) for e in record["experts"])
    return {
        layer: [expert for expert, _count in counter.most_common(k)]
        for layer, counter in counters.items()
    }


def _route_hotsets_global(route_path: Path, rows: int) -> dict[int, list[int]]:
    route_data = json.loads(route_path.read_text())["routes"]
    counters: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for record in route_data:
        counters[int(record["layer"])].update(int(e) for e in record["experts"])

    ranked: list[tuple[int, int, int]] = []
    for layer, counter in counters.items():
        for expert, count in counter.items():
            ranked.append((int(count), int(layer), int(expert)))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected: dict[int, list[int]] = collections.defaultdict(list)
    for _count, layer, expert in ranked[: max(0, int(rows))]:
        selected[layer].append(expert)
    return dict(selected)


def _install_forward_pass_timer(
    runner: Any,
    timings: dict[str, dict[str, float | int]],
) -> None:
    """Record total wall time by runner ``pass_kind`` without changing output."""

    original = runner._stream_forward_tokens

    def timed_stream_forward_tokens(*args: Any, **kwargs: Any) -> Any:
        pass_kind = str(kwargs.get("pass_kind") or "unknown")
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            bucket = timings.setdefault(pass_kind, {"seconds": 0.0, "calls": 0})
            bucket["seconds"] = float(bucket["seconds"]) + elapsed
            bucket["calls"] = int(bucket["calls"]) + 1

    runner._stream_forward_tokens = timed_stream_forward_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--global-rows", type=int)
    parser.add_argument("--prompt", default="Write a short greeting.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--resident-budget-gib", type=float, default=220.0)
    parser.add_argument("--soft-rss-gib", type=float, default=225.0)
    parser.add_argument("--hard-rss-gib", type=float, default=238.0)
    parser.add_argument("--pack-dir")
    parser.add_argument("--pack-read-workers", type=int, default=1)
    parser.add_argument(
        "--preserve-prewarm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compute-mode",
        choices=(
            "slot_arena_direct_qmm",
            "slot_arena_guarded_direct_qmm",
            "slot_arena_mixed_direct_qmm",
            "slot_arena_compact_direct_qmm",
            "slot_arena_compact_defer_eval",
            "slot_arena_static_direct_defer",
            "slot_arena_hotcold_qmv",
        ),
        default="slot_arena_direct_qmm",
    )
    parser.add_argument(
        "--base-retain",
        choices=("auto", "none"),
        default="auto",
        help="Whether to let the runner retain non-expert base layers.",
    )
    parser.add_argument(
        "--drop-mmap-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help="Override GLM mmap-cache dropping; auto preserves runner defaults.",
    )
    parser.add_argument(
        "--profile-forward-passes",
        action="store_true",
        help="Record wall time by _stream_forward_tokens pass_kind.",
    )
    args = parser.parse_args()
    if (args.k is None) == (args.global_rows is None):
        parser.error("exactly one of --k or --global-rows is required")

    out_path = Path(args.out)
    pid = os.getpid()
    soft_limit_kib = int(args.soft_rss_gib * 1024 * 1024)
    hard_limit_kib = int(args.hard_rss_gib * 1024 * 1024)
    samples: list[dict[str, Any]] = []
    stop = False
    killed = False
    phase = "start"
    runner = None
    forward_pass_timings: dict[str, dict[str, float | int]] = {}
    result: dict[str, Any] = {
        "ok": False,
        "phase": phase,
        "k": args.k,
        "soft_rss_gib": args.soft_rss_gib,
        "hard_rss_gib": args.hard_rss_gib,
    }

    def monitor() -> None:
        nonlocal killed
        sent_soft = False
        while not stop:
            rss = _rss_kib(pid)
            samples.append({"t": time.perf_counter(), "rss_kib": rss, "phase": phase})
            if rss > soft_limit_kib and not sent_soft:
                killed = True
                sent_soft = True
                try:
                    os.kill(pid, signal.SIGINT)
                except Exception:
                    pass
            if rss > hard_limit_kib:
                killed = True
                monitor_payload = dict(result)
                monitor_payload.update(
                    {
                        "ok": False,
                        "phase": phase,
                        "error": "hard RSS guard exceeded in monitor",
                        "peak_rss_kib": max((s["rss_kib"] for s in samples), default=rss),
                        "samples": samples[
                            :: max(1, len(samples) // 250)
                        ],
                        "killed": True,
                    }
                )
                _write_json(out_path.with_suffix(out_path.suffix + ".monitor"), monitor_payload)
                os._exit(137)
            time.sleep(0.5)

    threading.Thread(target=monitor, daemon=True).start()

    try:
        if args.global_rows is not None:
            hotsets = _route_hotsets_global(Path(args.routes), args.global_rows)
            plan_mode = "global_rows"
        else:
            hotsets = _route_hotsets(Path(args.routes), args.k)
            plan_mode = "uniform_k"
        result.update(
            {
                "plan_mode": plan_mode,
                "hotset_layers": len(hotsets),
                "hotset_rows": sum(len(v) for v in hotsets.values()),
                "base_retain": args.base_retain,
                "pack_dir": args.pack_dir,
                "pack_read_workers": args.pack_read_workers,
                "drop_mmap_cache": args.drop_mmap_cache,
            }
        )

        phase = "construct"
        runner = DeepSeekV3StreamingForwardRunner(
            args.model,
            retain_layers=(set() if args.base_retain == "none" else None),
            resident_budget_bytes=int(args.resident_budget_gib * 1024**3),
            pin_policy="phase",
            backend="native",
            clear_on_evict=False,
            warm_embeddings=False,
            weight_page_budget_bytes=None,
            expert_prefetch="off",
            expert_compute_mode=args.compute_mode,
            expert_slot_capacity=None,
            pack_dir=args.pack_dir,
            pack_read_workers=args.pack_read_workers,
            drop_mmap_cache_after_read=(
                None if args.drop_mmap_cache == "auto" else args.drop_mmap_cache == "on"
            ),
            trace=False,
        )
        if args.profile_forward_passes:
            _install_forward_pass_timer(runner, forward_pass_timings)
        runner.preserve_slot_arenas_on_reset = bool(args.preserve_prewarm)

        phase = "prewarm_hotset"
        prewarm_timing = {
            "seconds": 0.0,
            "slice_s": 0.0,
            "slice_calls": 0,
            "slice_bytes": 0,
            "layers": 0,
            "rows": 0,
        }
        loader = runner.session.loader
        orig_slices = loader.load_first_dim_slices
        orig_tensors = loader.load_tensors
        timing = {
            "slice_s": 0.0,
            "slice_calls": 0,
            "slice_bytes": 0,
            "tensor_s": 0.0,
            "tensor_calls": 0,
            "tensor_bytes": 0,
        }

        def timed_slices(*slice_args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            batch = orig_slices(*slice_args, **kwargs)
            elapsed = time.perf_counter() - start
            if phase == "prewarm_hotset":
                prewarm_timing["slice_s"] += elapsed
                prewarm_timing["slice_calls"] += 1
                prewarm_timing["slice_bytes"] += int(getattr(batch, "nbytes", 0) or 0)
            else:
                timing["slice_s"] += elapsed
                timing["slice_calls"] += 1
                timing["slice_bytes"] += int(getattr(batch, "nbytes", 0) or 0)
            return batch

        def timed_tensors(*tensor_args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            batch = orig_tensors(*tensor_args, **kwargs)
            elapsed = time.perf_counter() - start
            timing["tensor_s"] += elapsed
            timing["tensor_calls"] += 1
            timing["tensor_bytes"] += int(getattr(batch, "nbytes", 0) or 0)
            return batch

        loader.load_first_dim_slices = timed_slices
        loader.load_tensors = timed_tensors

        start = time.perf_counter()
        for layer in sorted(hotsets):
            experts = hotsets[layer]
            arena = runner._create_deepseek_slot_arena(layer, experts, len(experts))
            if arena is None:
                raise RuntimeError(
                    f"failed to prewarm layer {layer} rows={len(experts)}"
                )
            prewarm_timing["layers"] += 1
            prewarm_timing["rows"] += len(experts)
        prewarm_timing["seconds"] = time.perf_counter() - start
        result["prewarm"] = prewarm_timing
        result["timing"] = timing

        phase = "generate"
        start = time.perf_counter()
        generated = runner.generate_batch(
            [args.prompt], max_tokens=args.max_tokens, temperature=0.0
        )
        generate_seconds = time.perf_counter() - start
        result.update(
            {
                "ok": True,
                "phase": phase,
                "prewarm": prewarm_timing,
                "generate_seconds": generate_seconds,
                "rss_after_generate_kib": _rss_kib(pid),
                "peak_rss_kib": max(
                    (s["rss_kib"] for s in samples), default=_rss_kib(pid)
                ),
                "base_retain_layer_count": len(
                    getattr(runner, "base_retain_layers", [])
                ),
                "timing": timing,
                "forward_pass_timings": forward_pass_timings,
                "expert_telemetry": runner._expert_telemetry(),
                "result": generated.to_dict(),
                "metal_after_generate": {
                    "active": int(mx.get_active_memory()),
                    "cache": int(mx.get_cache_memory()),
                    "peak": int(mx.get_peak_memory()),
                },
            }
        )
    except BaseException as exc:
        result.update(
            {
                "ok": False,
                "phase": phase,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "peak_rss_kib": max(
                    (s["rss_kib"] for s in samples), default=_rss_kib(pid)
                ),
            }
        )
    finally:
        phase = "close"
        try:
            if runner is not None:
                runner.close()
        except Exception as exc:
            result["close_error"] = repr(exc)
        stop = True
        time.sleep(0.5)
        result["rss_after_close_kib"] = _rss_kib(pid)
        result["samples"] = samples[:: max(1, len(samples) // 250)]
        result["killed"] = killed
        if args.profile_forward_passes:
            result["forward_pass_timings"] = forward_pass_timings
        try:
            result["metal_after_close"] = {
                "active": int(mx.get_active_memory()),
                "cache": int(mx.get_cache_memory()),
                "peak": int(mx.get_peak_memory()),
            }
        except Exception:
            pass
        _write_json(out_path, result)

    summary_keys = [
        "ok",
        "phase",
        "error",
        "prewarm",
        "generate_seconds",
        "hotset_rows",
        "peak_rss_kib",
        "rss_after_generate_kib",
        "rss_after_close_kib",
        "timing",
        "expert_telemetry",
        "killed",
    ]
    print(json.dumps({k: result.get(k) for k in summary_keys}, indent=2)[:12000])
    generated = result.get("result", {})
    print(
        "generated",
        generated.get("generated_texts"),
        "tokens",
        generated.get("generated_token_count"),
        "tps",
        generated.get("tokens_per_second"),
    )
    print("WROTE", out_path)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
