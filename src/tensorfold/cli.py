"""Load local or Hugging Face models and serve them through their family kernels at an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from tensorfold import __version__

COMMANDS = ("serve", "pull", "models", "info", "update")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorfold",
        description="Fast, exact LLM decoding on Apple Silicon and NVIDIA GPUs behind an OpenAI-compatible endpoint.",
    )
    parser.add_argument("--version", action="version", version=f"tensorfold {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve a model at an OpenAI-compatible endpoint",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    serve.add_argument("model", help="a Hugging Face repo id (downloaded on first use) or a model directory")
    endpoint = serve.add_argument_group("endpoint")
    endpoint.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0: every interface)")
    endpoint.add_argument("--port", type=int, default=8080)
    endpoint.add_argument("--name", default="", help="model id clients ask for (default: the model's name)")
    endpoint.add_argument("--alias", action="append", default=[], help="another model id to answer to")

    generation = serve.add_argument_group("generation (requests can override each of these)")
    generation.add_argument("--context", type=int, default=None,
                            help="prompt plus reply window (default: model config; CUDA default/0: affordable native capacity; Metal 0: remove metadata cap)")
    generation.add_argument("--max-tokens", type=int, default=4096,
                            help="reply tokens when a request does not say")
    generation.add_argument("--temperature", type=float, default=None,
                            help="0 decodes greedily (default: the model's generation_config.json, else 0)")
    generation.add_argument("--top-p", type=float, default=None, help="(default: the model's generation config)")
    generation.add_argument("--top-k", type=int, default=None, help="(default: the model's generation config)")
    generation.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                            help="open a think block when the chat template supports it")
    generation.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium",
                            help="for chat templates that take one (Qwen3.8); medium adds no system-prompt text")
    generation.add_argument("--thinking-budget", type=int, default=0,
                            help="most thinking tokens before the server closes the think block (0: no limit)")

    speed = serve.add_argument_group("drafting and caches")
    speed.add_argument("--no-drafts", action="store_true",
                       help="one token a round: the serial reference (same output, slower)")
    speed.add_argument("--drafter", default="auto",
                       help="a draft model (repo id or directory); auto: the family's draft model when it has been "
                            "pulled; none: no draft model")
    speed.add_argument("--drafter-bits", type=int, default=4, help="quantize the draft model's linears (0: bf16)")
    speed.add_argument("--mtp-drafts", type=int, default=None,
                       help="most MTP drafts a round (Qwen3.8 Flash Next: 3 on Mac; on CUDA 6, stopping under 30%% "
                            "confidence); 0: no MTP drafts (any family)")
    speed.add_argument("--lane-kernels", choices=("auto", "on", "off"), default="auto",
                       help="lane kernels for Qwen3.8 dense (auto: on GPUs with tensor units)")
    speed.add_argument("--prompt-cache-gib", type=float, default=None,
                       help="memory for cached conversation prefixes (0: off; default: an eighth of RAM, at most 16)")
    speed.add_argument("--snapshot-dir", default=str(Path.home() / ".cache" / "tensorfold" / "prefix-snapshots"),
                       help="where system-block and conversation snapshots are kept ('none': in memory only)")
    speed.add_argument("--max-snapshots", type=int, default=3, help="system-block snapshots loaded at start")
    speed.add_argument("--parallel", default="auto",
                       help="requests decoded together, their windows sharing each round's forward: a number, or "
                            "auto (Mac: up to 8, each started only while the projected memory fits 70%% of RAM; "
                            "CUDA: one at a time, the others waiting their turn)")
    speed.add_argument("--mlx-cache-gib", type=float, default=8.0, help="MLX's cache of freed buffers")
    speed.add_argument("--ssd-experts", type=float, default=None, metavar="GIB",
                       help="stream routed experts from the checkpoint into a GPU pool of this many GiB, for models "
                            "past the memory budget (the rest stays resident; output is the resident model's)")
    speed.add_argument("--ple-on-ssd", action="store_true",
                       help="Flash Next: read the n-gram (PLE) tables from the checkpoint on SSD at each lookup "
                            "instead of holding them in memory. A trade: a few percent of decode speed for about "
                            "40 GiB less at peak (the tables are 29.8 GiB); a 128 GB Mac needs it")

    speed.add_argument("--no-update-check", action="store_true",
                       help="don't ask GitHub whether a newer release exists (also TENSORFOLD_NO_UPDATE_CHECK=1)")

    cuda = serve.add_argument_group("NVIDIA GPUs (DGX Spark)")
    cuda.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto",
                      help="auto: MLX on macOS, CUDA elsewhere")
    cuda.add_argument("--tp", type=int, choices=(1, 2), default=1,
                      help="GPUs (one per machine) the model is split over; run the same command on each")
    cuda.add_argument("--rank", type=int, choices=(0, 1), default=0,
                      help="with --tp 2: this machine's rank; rank 0 serves HTTP, rank 1 follows it")
    cuda.add_argument("--master", default="", help="with --tp 2: rank 0's address on the link between the machines")
    cuda.add_argument("--master-port", type=int, default=29551, help="with --tp 2: rank 0's rendezvous port")
    serve.set_defaults(func=cmd_serve)

    pull = commands.add_parser("pull", help="download models (or draft models) from Hugging Face")
    pull.add_argument("repos", nargs="+", help="repo ids, e.g. Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    pull.set_defaults(func=cmd_pull)

    models = commands.add_parser("models", help="list the model families and the checkpoints they are tested with")
    models.set_defaults(func=cmd_models)

    update = commands.add_parser("update", help="install the newest TensorFold release from GitHub")
    update.add_argument("--check", action="store_true", help="only say whether a newer release exists")
    update.add_argument("--force", action="store_true", help="reinstall the newest release even when it is current")
    update.set_defaults(func=cmd_update)

    info = commands.add_parser("info", help="show which family serves a model (reads its config.json only)")
    info.add_argument("model", help="a Hugging Face repo id or a model directory")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``tensorfold MODEL ...`` is ``tensorfold serve MODEL ...``
    if argv and not argv[0].startswith("-") and argv[0] not in COMMANDS:
        from tensorfold import hub

        if Path(argv[0]).expanduser().is_dir() or hub.is_repo_id(argv[0]):
            argv = ["serve", *argv]
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError) as exc:
        print(f"tensorfold: {exc}", file=sys.stderr)
        return 1


def _config_dir(model: str) -> Path:
    """Resolve config.json without downloading weights so family compatibility checks run first."""

    from tensorfold import hub

    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if not hub.is_repo_id(model):
        raise FileNotFoundError(f"{model} is neither a directory nor a Hugging Face repo id (owner/name)")
    found = hub.cached(model)
    if found is not None and (found / "config.json").is_file():
        return found
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, "config.json")).parent


def cmd_pull(args: argparse.Namespace) -> int:
    from tensorfold import families, hub

    for repo in args.repos:
        if not hub.is_repo_id(repo):
            raise ValueError(f"{repo} is not a Hugging Face repo id (owner/name)")
        config = _config_dir(repo)
        try:
            family = families.detect(config)
        except ValueError:
            family = None          # a draft model, for example
        if family is not None:
            settings = families.read_config(config)
            readable = [b for b in families.backends_of(family)
                        if families.quant_method(settings) in families.readable_quants(family, b)]
            if not readable:
                families.require_readable(family, settings, families.backends_of(family)[0])
            _note_untested(family, repo)
            check = getattr(family.package, "check", None)
            if check is not None:
                check(config)
        path = hub.pull(repo)
        required_files = getattr(family.package, "REQUIRED_FILES", {}).get(repo, ()) if family is not None else ()
        if required_files and not hub._cached_weights_complete(path, required_files=required_files):
            raise FileNotFoundError(f"{repo} is missing required files: {', '.join(required_files)}")
        what = f"{family.title} ({family.model_type})" if family is not None else "no model family (a draft model?)"
        print(f"{repo}: {hub.size_of(path) / 1e9:.1f} GB in {path} [{what}]")
        if required_files:
            print(f"[tensorfold] required model files ready: {', '.join(required_files)}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    from tensorfold import update

    return update.update(check_only=bool(args.check), force=bool(args.force))


def cmd_models(args: argparse.Namespace) -> int:
    from tensorfold import families

    for kind, family in sorted(families.families().items()):
        package = family.package
        print(f"{family.title} ({kind}; {_engines(family)})")
        kernel_package = getattr(package, "KERNEL_PACKAGE", "")
        if kernel_package:
            print(f"  kernels  {kernel_package.removeprefix('tensorfold.kernels.').replace('.', '/')}")
        for repo in getattr(package, "MODELS", ()):
            print(f"  model    {repo}")
        drafter = getattr(package, "DRAFTER", "")
        if drafter:
            print(f"  drafter  {drafter}")
    return 0


def _engines(family: Any) -> str:
    """Which backends serve a family: MLX (its lane or serial engine) and CUDA."""

    found = []
    if hasattr(family.package, "load"):
        found.append(f"MLX {'lane engine' if family.lanes else 'serial engine'}")
    if hasattr(family.package, "cuda_engine"):
        found.append("CUDA engine")
    return ", ".join(found) or "no engine"


def cmd_info(args: argparse.Namespace) -> int:
    from tensorfold import families

    directory = _config_dir(args.model)
    config = families.read_config(directory)
    text = config.get("text_config", config)
    family = families.detect(directory)
    print(f"model_type   {family.model_type}")
    print(f"family       {family.title} ({family.module})")
    print(f"engine       {_engines(family)}")
    kernel_package = getattr(family.package, "KERNEL_PACKAGE", "")
    if kernel_package:
        print(f"kernels      {kernel_package.removeprefix('tensorfold.kernels.').replace('.', '/')}")
    for key in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok", "n_routed_experts",
                "vocab_size", "max_position_embeddings"):
        if key in text:
            print(f"{key:12s} {text[key]}" if len(key) <= 12 else f"{key} {text[key]}")
    print(f"quantization {families.describe_quantization(config)}")
    readers = [b for b in families.backends_of(family)
               if families.quant_method(config) in families.readable_quants(family, b)]
    if readers:
        print(f"runs on      {', '.join('NVIDIA GPUs (CUDA)' if b == 'cuda' else 'Apple Silicon (MLX)' for b in readers)}")
    else:
        print(f"runs on      not yet: no {family.title} engine reads these weights. {families.OWN_MODEL_HELP}")
    generation = _generation_config(directory)
    if generation:
        print(f"sampling     {generation}")
    check = getattr(family.package, "check", None)
    if check is not None:
        check(directory)
    return 0


def _generation_config(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "generation_config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    sampling = {k: config[k] for k in ("temperature", "top_k", "top_p") if k in config}
    if config.get("do_sample") is False:
        sampling["temperature"] = 0.0
    elif config.get("do_sample") is True and "temperature" not in sampling:
        sampling["temperature"] = 1.0
    return sampling


def _model_context(model_dir: Path) -> int:
    from tensorfold.families import read_config

    config = read_config(model_dir)
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


def _drafter(family: Any, choice: str) -> str:
    """The draft model directory for ``--drafter`` (auto: the family's draft model if it has been pulled)."""

    from tensorfold import hub

    if choice in ("", "none"):
        return ""
    if choice != "auto":
        return str(hub.resolve(choice))
    repo = getattr(family.package, "DRAFTER", "")
    if not repo:
        return ""
    found = hub.cached(repo)
    if found is None or not hub._cached_weights_complete(found):
        print(f"[tensorfold] no draft model: `tensorfold pull {repo}` once to draft with it", flush=True)
        return ""
    return str(found)


def _note_untested(family: Any, model: str) -> None:
    """Explain that an unlisted Hugging Face checkpoint runs when its storage format matches the family kernels."""

    from tensorfold import families, hub

    tested = tuple(getattr(family.package, "MODELS", ())) + tuple(filter(None, [getattr(family.package, "DRAFTER", "")]))
    if hub.is_repo_id(model) and model not in tested:
        print(f"[tensorfold] note: {model} is not a checkpoint TensorFold is tested with ({', '.join(tested) or 'none'}). "
              f"It runs when its format matches what the {family.title} kernels read: replies stay exact to serial "
              f"decoding, speed and quality are unmeasured. {families.OWN_MODEL_HELP}", flush=True)


def _backend(choice: str, family: Any) -> str:
    """mlx or cuda: auto picks MLX on macOS and CUDA elsewhere; a family serves only the backends it has."""

    backend = choice if choice != "auto" else ("mlx" if sys.platform == "darwin" else "cuda")
    if backend == "cuda" and not hasattr(family.package, "cuda_engine"):
        raise ValueError(f"{family.title} has no CUDA engine yet: serve it on Apple Silicon")
    if backend == "mlx" and not hasattr(family.package, "load"):
        raise ValueError(f"{family.title} runs on NVIDIA GPUs only (see docs/recipes)")
    return backend


def _serve_cuda(args: argparse.Namespace, family: Any, model_dir: Path, context: int | None = None) -> int:
    """Serve with the family's CUDA engine (``cuda_engine``) behind ``tensorfold.cuda.server``."""

    from tensorfold import hub

    if args.tp == 2 and not args.master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    if args.tp == 1 and args.rank != 0:
        raise ValueError("--rank 1 needs --tp 2")
    started = time.perf_counter()
    drafter = "" if args.no_drafts else _drafter(family, args.drafter)
    options: dict[str, Any] = {"drafter": drafter, "tp": int(args.tp), "rank": int(args.rank), "master": args.master,
                               "master_port": int(args.master_port), "no_drafts": bool(args.no_drafts)}
    if args.mtp_drafts is not None:
        options["mtp_drafts"] = int(args.mtp_drafts)
    if args.ple_on_ssd:
        options["ple_on_ssd"] = True
    options["context"] = context if context is not None else args.context
    options["context_explicit"] = args.context is not None
    streams = 1 if str(args.parallel).strip().lower() == "auto" else _parallel(args.parallel)
    if streams > 1:
        options["parallel"] = streams
    served = args.name or (args.model.rstrip("/").split("/")[-1] if hub.is_repo_id(args.model) else model_dir.name)
    where = f", rank {args.rank} of 2" if args.tp == 2 else ""
    print(f"[tensorfold] loading {served}: {family.title} ({family.model_type}) on CUDA{where}", flush=True)
    engine = family.package.cuda_engine(model_dir, **options)
    if args.tp == 2 and args.rank == 1:
        print(f"[tensorfold] rank 1 ready in {time.perf_counter() - started:.1f}s, following rank 0", flush=True)
        engine.follow()
        return 0
    from tensorfold.cuda.server import App, serve

    sampling = _generation_config(model_dir)
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k)):
        if value is not None:
            sampling[key] = value
    app_class = getattr(family.package, "CUDA_APP", None) or App
    app = app_class(engine, model_dir, served, default_thinking=bool(args.thinking), sampling=sampling,
                    max_tokens=int(args.max_tokens), context_window=context if context is not None else args.context)
    shown = "greedy" if float(sampling.get("temperature", 1.0)) <= 0 else ", ".join(
        f"{k} {v}" for k, v in sampling.items())
    effective_context = app.effective_context_window
    print(f"[tensorfold] serving {served} at http://{args.host}:{args.port}/v1 on CUDA{where} "
          f"(sampling: {shown}; drafts: {'off' if args.no_drafts else 'on'}; "
          f"context: {'unlimited' if effective_context is None else effective_context}; "
          f"loaded in {time.perf_counter() - started:.1f}s)", flush=True)
    serve(app, args.host, int(args.port))
    return 0


# Admit concurrent requests within this RAM fraction, or the family's, after the rest of the machine.
MEMORY_FRACTION = 0.70
# a resume point begins a prompt chunk when at least this many tokens follow the last chunk start
MIN_CHUNK = 256


def _parallel(value: Any) -> int:
    """``--parallel``: "auto" is up to 8 requests at once; a number caps it."""

    if str(value).strip().lower() == "auto":
        return 8
    try:
        return max(1, int(value))
    except ValueError:
        raise SystemExit(f"--parallel takes a number or auto, not {value!r}") from None


def cmd_serve(args: argparse.Namespace) -> int:
    from tensorfold import families, hub

    if not args.no_update_check:
        from tensorfold import update

        update.check_in_background()
        news = update.first_run_notice()
        if news:
            print(news, flush=True)
    config_dir = _config_dir(args.model)
    family = families.detect(config_dir)
    if args.ssd_experts is not None and (args.ssd_experts <= 0 or not hasattr(family.package, "expert_bytes")):
        raise ValueError(f"--ssd-experts takes a positive GiB count for a family that streams experts; "
                         f"{family.title} does not")
    if args.ple_on_ssd and not hasattr(family.package, "ple_bytes"):
        raise ValueError(f"--ple-on-ssd: {family.title} has no n-gram (PLE) tables to read from SSD")
    backend = _backend(args.backend, family)
    families.require_readable(family, families.read_config(config_dir), backend)
    _note_untested(family, args.model)
    required_files = getattr(family.package, "REQUIRED_FILES", {}).get(args.model, ())
    native_context = _model_context(config_dir)
    context = native_context if args.context is None else int(args.context)
    if context < 0:
        raise ValueError("--context must be 0 or a positive token count")
    if native_context and context > native_context:
        raise ValueError(f"--context {context} exceeds this model's {native_context}-token window")
    check = getattr(family.package, "check", None)
    if check is not None:
        check(config_dir)                        # refuse an unsupported checkpoint before downloading its weights
    needs_full_snapshot = hub.is_repo_id(args.model) and not hub._cached_weights_complete(
        config_dir, required_files=required_files)
    model_dir = hub.resolve(args.model, required_files=required_files)
    if needs_full_snapshot and check is not None:
        check(model_dir)                         # checks that need the complete index, such as an MTP head
    if backend == "cuda":
        return _serve_cuda(args, family, model_dir, context)
    for key, value in getattr(family.package, "MLX_ENV", {}).items():
        os.environ.setdefault(key, value)       # before MLX starts: it reads them once

    # `kill -USR1 <pid>` prints every thread's Python stack: the way to see where a silent server waits
    import faulthandler

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    import mlx.core as mx

    from tensorfold.server.memory_budget import PROCESS_BYTES, configure_mlx, model_fraction

    fraction = model_fraction(family.package)
    memory_limit = configure_mlx(mx, int(float(args.mlx_cache_gib) * 1024**3), fraction=fraction)
    gib = 1024**3
    note = f" ({fraction:.0%} of RAM, this model's allowance)" if fraction > MEMORY_FRACTION else ""
    print(f"[tensorfold] memory budget {memory_limit / gib:.1f} GiB{note}: MLX's buffers up to "
          f"{(memory_limit - PROCESS_BYTES) / gib:.1f} GiB, {PROCESS_BYTES / gib:.0f} GiB for the rest of the process",
          flush=True)
    checkpoint = sum(path.stat().st_size for path in Path(model_dir).glob("*.safetensors"))
    estimate = getattr(family.package, "weight_bytes", None)
    weights = checkpoint if estimate is None else estimate(model_dir)
    if args.ple_on_ssd:
        weights = checkpoint - family.package.ple_bytes(model_dir)      # read from disk at each lookup, never loaded
    if args.ssd_experts is not None:
        weights += int(args.ssd_experts * gib) - family.package.expert_bytes(model_dir)   # the pool, not the stacks
    if weights < checkpoint:
        print(f"[tensorfold] weights: {weights / gib:.1f} GiB resident, "
              f"{(checkpoint - weights) / gib:.1f} GiB file-backed", flush=True)
    if weights >= memory_limit - PROCESS_BYTES:
        stream = ("stream its routed experts from SSD with --ssd-experts GIB (slower), "
                  if args.ssd_experts is None and hasattr(family.package, "expert_bytes") else "")
        raise ValueError(f"{family.title}'s weights ({weights / gib:.1f} GiB) do not fit this server's "
                         f"{memory_limit / gib:.1f} GiB memory budget ({fraction:.0%} of RAM, or "
                         f"TENSORFOLD_MEMORY_LIMIT_GB): serve it on a Mac with more memory, {stream}or use a smaller "
                         "or more quantized checkpoint")
    return _serve_mlx(args, family, model_dir, context, required_files, memory_limit, fraction)


def _serve_mlx(args: argparse.Namespace, family: Any, model_dir: Path, context: int,
               required_files: Any, memory_limit: int, fraction: float = MEMORY_FRACTION) -> int:
    import mlx.core as mx
    from tensorfold import families, hub
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.engine.prefill_plan import PrefillPlan, message_markers

    started = time.perf_counter()
    drafter = "" if args.no_drafts else _drafter(family, args.drafter)
    parallel = _parallel(args.parallel)
    options: dict[str, Any] = {"lane_kernels": args.lane_kernels, "drafter": drafter,
                               "drafter_bits": args.drafter_bits, "parallel": parallel}
    if args.mtp_drafts is not None:
        options["mtp_drafts"] = int(args.mtp_drafts)
    if args.ple_on_ssd:
        options["ple_on_ssd"] = True
    if args.ssd_experts is not None:
        options["ssd_experts"] = float(args.ssd_experts)
    served = args.name or (args.model.rstrip("/").split("/")[-1] if hub.is_repo_id(args.model) else model_dir.name)
    print(f"[tensorfold] loading {served}: {family.title} ({family.model_type})", flush=True)
    model, tokenizer = family.package.load(model_dir, **options)
    engine_kwargs = dict(getattr(family.package, "engine_settings", lambda m: {})(model))
    if required_files:
        print(f"[tensorfold] Nemotron MTP head: "
              f"{'active' if not args.no_drafts and getattr(model, 'mtp', None) is not None else 'inactive'}",
              flush=True)
    from tensorfold.engine import prefill_step
    from tensorfold.server.memory_budget import PROCESS_BYTES
    from tensorfold.server.prompt_memory import probe_tokens
    from tensorfold.server.residency import unwire, wire_resident

    getattr(model, "release_rounds", lambda: None)()       # load-time checks' rows go before the weights are wired
    wired = wire_resident(mx, memory_limit - PROCESS_BYTES)
    print(f"[tensorfold] {wired / 1024**3:.1f} GiB of weights kept resident", flush=True)
    # prompt chunks start where replies begin too, so a follow-up resumes where its latest reply began
    openers, assistant = message_markers(tokenizer)
    steps = engine_kwargs.pop("prefill_steps", None) or (LaneEngine.prefill_step,)
    step = prefill_step.choose(lambda grid: LaneEngine(model, prefill_plan=PrefillPlan(grid)), steps,
                               memory_limit - PROCESS_BYTES, probe_tokens(tokenizer), context)
    plan = PrefillPlan(step, openers, MIN_CHUNK, assistant)
    print(f"[tensorfold] prompt chunks of up to {step:,} tokens, cut at replies {plan.min_chunk:,}+ tokens apart",
          flush=True)

    from tensorfold.server.app import ChatApp
    from tensorfold.server.http import Server, make_handler

    engine_factory = functools.partial(LaneEngine, prefill_plan=plan)      # every family decodes through lanes
    sampling = _generation_config(model_dir)
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k)):
        if value is not None:
            sampling[key] = value
    snapshot_dir = None if str(args.snapshot_dir).lower() == "none" else Path(args.snapshot_dir).expanduser()
    from importlib.metadata import version

    resolve_prefill = getattr(model, "resolve_prefill_identity", None)
    if resolve_prefill is not None:
        resolve_prefill()  # the prefill mode's self-check runs at startup, before any snapshot key
    # Both libraries, the active prompt kernels and how prompts are cut determine a snapshot's bits.
    model_id = (f"{model_dir.resolve()}|mlx={mx.__version__}|mlx_lm={version('mlx-lm')}"
                f"|kernels={families.kernel_version(family, model)}"
                f"|prefill={plan.name}|tensorfold={__version__}")
    gib = args.prompt_cache_gib
    if gib is None:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        gib = min(16.0, ram / 8 / 1024**3)
    budget = int(float(gib) * 1024**3)
    app = ChatApp(
        model,
        tokenizer,
        served_name=served,
        model_aliases=list(args.alias),
        engine_factory=engine_factory,
        lanes=parallel,
        memory_fraction=fraction if parallel > 1 else None,
        max_rows=int(engine_kwargs.get("max_rows", 16)),
        max_draft=int(engine_kwargs.get("max_draft", 32)),
        default_max_tokens=int(args.max_tokens),
        context_window=context,
        enable_thinking=bool(args.thinking),
        reasoning_effort=args.reasoning_effort,
        thinking_budget=int(args.thinking_budget),
        default_sampling=sampling,
        max_snapshots=int(args.max_snapshots),
        checkpoint_slots=0 if budget <= 0 else None,
        checkpoint_budget_bytes=budget if budget > 0 else None,
        memory_budget_bytes=memory_limit,
        fit_context=args.context is None,
        use_proposer=not args.no_drafts,
        snapshot_dir=snapshot_dir,
        model_id=model_id,
    )
    if app.context_fitted:
        print(f"[tensorfold] context window {app.context_window:,} tokens: the most one request can use in the "
              f"{memory_limit / 1024**3:.1f} GiB memory budget (the model's window is {context:,}); have clients "
              "compact before it", flush=True)
    hook = getattr(family.package, "setup", None)
    if hook is not None:
        hook(app, model, **options)
    server = Server((args.host, int(args.port)), make_handler(app))  # type: ignore[arg-type]
    shown = "greedy" if float(sampling.get("temperature", 0.0) or 0.0) <= 0 else ", ".join(
        f"{k} {v}" for k, v in sampling.items())
    print(f"[tensorfold] serving {served} at http://{args.host}:{args.port}/v1 "
          f"(sampling: {shown}; drafts: {'off' if args.no_drafts else 'on'}; "
          f"context: {app.context_window or 'unlimited'}; loaded in {time.perf_counter() - started:.1f}s)", flush=True)

    def _terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt      # the cleanup below runs (a plain SIGTERM would skip it)

    signal.signal(signal.SIGTERM, _terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()        # the engine thread saves the newest conversations as it stops
        unwire(mx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
