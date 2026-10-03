"""Measure real single-device CUDA sleep/wake; use --synthetic for a tiny local plumbing check.

Run with PYTHONPATH=src and a CUDA-enabled TensorFold Python. Real checkpoints are local
paths only. The report distinguishes random fixtures from pretrained checkpoints.
HTTP authorization, stream draining and Responses continuation have separate host tests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import weakref


def qualify(args, model_dir, cache_dir=None):
    import torch
    import tensorfold
    from tensorfold.cuda import precision, prompt_precision
    from tensorfold.cuda.server import App
    from tensorfold.cuda.sleep import CudaSleep
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5 import cuda_engine
    from tensorfold.server.lifecycle import Lifecycle

    precision.set_mode(args.precision, asked=True)
    prompt_precision.set_fp8(args.prefill_fp8)
    options = dict(drafter=str(args.draft.resolve()) if args.draft else "", no_drafts=not args.draft,
                   parallel=args.parallel, context=args.context, context_explicit=True)
    if args.preserve_cache and args.parallel > 1:
        options["checkpoint_slots"] = max(3, args.parallel)
    app = App(cuda_engine(model_dir, **options), model_dir, "sleep-qualification")
    frontend = (id(app), id(app.tok), id(app.template))
    adapter = CudaSleep(app, cuda_engine, model_dir, options, cache_dir=cache_dir)
    lifecycle = Lifecycle(release=adapter.release, restore=adapter.restore, cleanup=adapter.cleanup,
                          preflight=adapter.prepare)

    def memory():
        torch.cuda.synchronize()
        result = adapter.memory_snapshot()
        status = Path("/proc/self/status")
        if status.exists():
            result["rss_bytes"] = next(int(line.split()[1]) * 1024 for line in status.read_text().splitlines()
                                       if line.startswith("VmRSS:"))
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            fields = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
            for key in ("MemAvailable", "MemFree", "Cached", "AnonPages"):
                result[key + "_bytes"] = int(fields[key].split()[0]) * 1024
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_free_bytes=free, cuda_total_bytes=total)
        return result

    def generate(draft, offset=0):
        tokens = []
        first_token_s = None
        sampling = Sampling(args.seed + offset, 1.0, 20, .95) if args.seed is not None else None

        def received(ids):
            nonlocal first_token_s
            if ids and first_token_s is None:
                first_token_s = time.perf_counter() - started
            tokens.extend(ids)

        with lifecycle.admit():
            started = time.perf_counter()
            stats = app.engine.generate([1 + (i + offset) % 128 for i in range(args.prompt_tokens)],
                                        args.tokens, sampling, received,
                                        draft=draft, stop_eos=False)
            elapsed_s = time.perf_counter() - started
        assert len(tokens) == args.tokens, "generation did not reach the requested reply length"
        return dict(tokens=tokens, stats=stats, first_token_s=first_token_s, elapsed_s=elapsed_s)

    def runs():
        serial = [generate(False, offset) for offset in range(args.parallel)]
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            concurrent = list(pool.map(lambda offset: generate(bool(args.draft) or args.preserve_cache, offset),
                                       range(args.parallel)))
        assert [r["tokens"] for r in serial] == [r["tokens"] for r in concurrent], "serial/concurrent tokens differ"
        return dict(serial=serial, concurrent=concurrent)

    report = dict(synthetic=args.synthetic, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, tensorfold=tensorfold.__version__,
                  precision=args.precision, prefill_fp8=args.prefill_fp8,
                  seed=args.seed, preserve_cache=args.preserve_cache,
                  options={**options, "drafter": bool(args.draft)}, cycles=[])
    report["checkpoint_files"] = [dict(source=adapter.identity.paths.index(Path(root)), file=name, sha256=digest)
                                  for (root, name), digest in adapter.identity.manifest.items()]
    try:
        generate(False)  # compile/warm kernels before measuring cold requests (serial bypasses prefix reuse)
        before = runs()
        report["before"] = before
        if args.preserve_cache:
            report["warm"] = runs()
            expected_cached = [r["stats"]["cached"] for r in report["warm"]["concurrent"]]
            assert all(expected_cached), "warm requests did not reuse a prefix"
        for _ in range(args.cycles):
            old = weakref.ref(app.engine)
            loaded = memory()
            start = time.perf_counter()
            lifecycle.sleep()
            sleep_s = time.perf_counter() - start
            asleep = memory()
            saved_cache = adapter.cache_snapshot()
            assert old() is None, "old engine is retained"
            assert asleep["allocated_bytes"] < loaded["allocated_bytes"], "CUDA allocations were not released"
            start = time.perf_counter()
            lifecycle.wake_up()
            wake_s = time.perf_counter() - start
            if args.preserve_cache:
                engine = app.engine
                assert not (engine.multi.cache if engine.concurrent else engine.cache).entries, "eager cache load"
                del engine
            after = runs()
            assert [r["tokens"] for r in after["serial"]] == [r["tokens"] for r in before["serial"]], \
                "tokens changed across wake"
            assert frontend == (id(app), id(app.tok), id(app.template)), "frontend was replaced"
            cycle = dict(loaded=loaded, asleep=asleep, awake=memory(), sleep_s=sleep_s, wake_s=wake_s, after=after)
            if args.preserve_cache:
                assert asleep["allocated_bytes"] == asleep["reserved_bytes"] == 0, "cache tensors survived sleep"
                assert [r["stats"]["cached"] for r in after["concurrent"]] == expected_cached, "prefix reuse lost"
                assert adapter.cache_snapshot()["loaded_prefixes"] == args.parallel, "prefix was not restored"
                assert adapter.cache_snapshot()["load_failures"] == 0, "prefix restore failed"
                cycle.update(saved_cache=saved_cache, restored_cache=adapter.cache_snapshot())
            report["cycles"].append(cycle)
            print(json.dumps({"cycle": len(report["cycles"]), **{k: v for k, v in cycle.items() if k != "after"}}),
                  flush=True)
        assert len({c["asleep"]["allocated_bytes"] for c in report["cycles"]}) == 1, "sleep allocations grew"
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        try:
            if lifecycle.snapshot()["state"] == "awake":
                lifecycle.sleep()
        finally:
            adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-draft", action="store_true", help="use 64 tiny target layers and a random drafter")
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, help="sample at temperature 1, top-k 20, top-p .95 (default: greedy)")
    parser.add_argument("--precision", choices=("checkpoint", "full"), default="checkpoint")
    parser.add_argument("--prefill-fp8", action="store_true")
    parser.add_argument("--preserve-cache", action="store_true", help="save and lazily restore reusable prefixes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.parallel, args.context, args.prompt_tokens, args.tokens, args.cycles) < 1:
        parser.error("counts must be positive")
    if args.prompt_tokens + args.tokens > args.context:
        parser.error("context must cover prompt and reply")
    if args.synthetic and args.draft:
        parser.error("use --synthetic-draft for the synthetic model's matching drafter")
    if args.synthetic_draft and not args.synthetic:
        parser.error("--synthetic-draft requires --synthetic")
    with tempfile.TemporaryDirectory(prefix="tensorfold-sleep-") as directory:
        if args.synthetic:
            from sleep_fixture import create, create_draft
            model = create(Path(directory) / "model", layers=64 if args.synthetic_draft else 2,
                           hidden=2048 if args.synthetic_draft else 128)
            if args.synthetic_draft:
                args.draft = create_draft(Path(directory) / "draft")
        else:
            model = args.model.resolve()
        qualify(args, model, Path(directory) / "cache" if args.preserve_cache else None)


if __name__ == "__main__":
    main()
