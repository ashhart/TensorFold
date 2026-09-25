"""The ``tensorfold`` command.

    tensorfold pull Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
    tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP [--port 8080] [--context 65536] [--temperature 0.7] ...
    tensorfold models
    tensorfold info MODEL

A model is a Hugging Face repo id (downloaded into the Hugging Face cache on first use) or a local directory.
``serve`` loads it with its family's kernels and serves an OpenAI-compatible API at ``http://HOST:PORT/v1``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from tensorfold import __version__

COMMANDS = ("serve", "pull", "models", "info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorfold",
        description="Fast, exact LLM decoding on Apple Silicon behind an OpenAI-compatible endpoint.",
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
                            help="context window: prompt plus reply tokens (default: model config; 0: no limit)")
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
                       help="most MTP drafts a round (Qwen3.8 Flash Next, default 3); 0: no MTP drafts (any family)")
    speed.add_argument("--lane-kernels", choices=("auto", "on", "off"), default="auto",
                       help="lane kernels for Qwen3.8 dense (auto: on GPUs with tensor units)")
    speed.add_argument("--prompt-cache-gib", type=float, default=None,
                       help="memory for cached conversation prefixes (0: off; default: an eighth of RAM, at most 16)")
    speed.add_argument("--snapshot-dir", default=str(Path.home() / ".cache" / "tensorfold" / "prefix-snapshots"),
                       help="where system-block and conversation snapshots are kept ('none': in memory only)")
    speed.add_argument("--max-snapshots", type=int, default=3, help="system-block snapshots loaded at start")
    speed.add_argument("--mlx-cache-gib", type=float, default=8.0, help="MLX's cache of freed buffers")
    serve.set_defaults(func=cmd_serve)

    pull = commands.add_parser("pull", help="download models (or draft models) from Hugging Face")
    pull.add_argument("repos", nargs="+", help="repo ids, e.g. Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    pull.set_defaults(func=cmd_pull)

    models = commands.add_parser("models", help="list the model families and the checkpoints they are tested with")
    models.set_defaults(func=cmd_models)

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
    """A directory holding the model's config.json: the model directory, its cached snapshot, or (for a repo id not
    downloaded yet) a snapshot with config.json alone, so the family and its checks run before any weight moves."""

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


def cmd_models(args: argparse.Namespace) -> int:
    from tensorfold import families

    for kind, family in sorted(families.families().items()):
        package = family.package
        engine = "lane engine" if family.lanes else "serial engine"
        print(f"{family.title} ({kind}, {engine})")
        kernel_package = getattr(package, "KERNEL_PACKAGE", "")
        if kernel_package:
            print(f"  kernels  {kernel_package.removeprefix('tensorfold.kernels.').replace('.', '/')}")
        for repo in getattr(package, "MODELS", ()):
            print(f"  model    {repo}")
        drafter = getattr(package, "DRAFTER", "")
        if drafter:
            print(f"  drafter  {drafter}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from tensorfold import families

    directory = _config_dir(args.model)
    config = families.read_config(directory)
    text = config.get("text_config", config)
    family = families.detect(directory)
    bits, group = families.quantization(config)
    print(f"model_type   {family.model_type}")
    print(f"family       {family.title} ({family.module})")
    print(f"engine       {'lane engine' if family.lanes else 'serial engine'}")
    kernel_package = getattr(family.package, "KERNEL_PACKAGE", "")
    if kernel_package:
        print(f"kernels      {kernel_package.removeprefix('tensorfold.kernels.').replace('.', '/')}")
    for key in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok", "n_routed_experts",
                "vocab_size", "max_position_embeddings"):
        if key in text:
            print(f"{key:12s} {text[key]}" if len(key) <= 12 else f"{key} {text[key]}")
    if bits is not None:
        print(f"quantization {bits}-bit, groups of {group}")
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


def cmd_serve(args: argparse.Namespace) -> int:
    from http.server import ThreadingHTTPServer

    from tensorfold import families, hub

    config_dir = _config_dir(args.model)
    family = families.detect(config_dir)
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
    for key, value in getattr(family.package, "MLX_ENV", {}).items():
        os.environ.setdefault(key, value)       # before MLX starts: it reads them once

    # `kill -USR1 <pid>` prints every thread's Python stack: the way to see where a silent server waits
    import faulthandler

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    import mlx.core as mx

    # MLX keeps freed buffers up to its memory limit: long contexts, whose attention buffers change size every
    # chunk, grow a server by tens of GB without a cap
    mx.set_cache_limit(int(float(args.mlx_cache_gib) * 1024**3))
    started = time.perf_counter()
    drafter = "" if args.no_drafts else _drafter(family, args.drafter)
    options: dict[str, Any] = {"lane_kernels": args.lane_kernels, "drafter": drafter,
                               "drafter_bits": args.drafter_bits}
    if args.mtp_drafts is not None:
        options["mtp_drafts"] = int(args.mtp_drafts)
    served = args.name or (args.model.rstrip("/").split("/")[-1] if hub.is_repo_id(args.model) else model_dir.name)
    print(f"[tensorfold] loading {served}: {family.title} ({family.model_type})", flush=True)
    model, tokenizer = family.package.load(model_dir, **options)
    if required_files:
        print(f"[tensorfold] Nemotron MTP head: "
              f"{'active' if not args.no_drafts and getattr(model, 'mtp', None) is not None else 'inactive'}",
              flush=True)

    from tensorfold.engine.family_engine import SerialEngine
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.server.app import ChatApp
    from tensorfold.server.http import make_handler

    engine_factory = LaneEngine if family.lanes else SerialEngine
    engine_kwargs = dict(getattr(family.package, "engine_settings", lambda m: {})(model))
    sampling = _generation_config(model_dir)
    for key, value in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k)):
        if value is not None:
            sampling[key] = value
    snapshot_dir = None if str(args.snapshot_dir).lower() == "none" else Path(args.snapshot_dir).expanduser()
    # a snapshot's bits depend on the MLX version and the kernels that computed it: never mix them
    model_id = (f"{model_dir.resolve()}|mlx={mx.__version__}|kernels={families.kernel_version(family, model)}"
                f"|tensorfold={__version__}")
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
        lanes=1,
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
        use_proposer=not args.no_drafts,
        snapshot_dir=snapshot_dir,
        model_id=model_id,
    )
    hook = getattr(family.package, "setup", None)
    if hook is not None:
        hook(app, model, **options)
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(app))  # type: ignore[arg-type]
    shown = "greedy" if float(sampling.get("temperature", 0.0) or 0.0) <= 0 else ", ".join(
        f"{k} {v}" for k, v in sampling.items())
    print(f"[tensorfold] serving {served} at http://{args.host}:{args.port}/v1 "
          f"(sampling: {shown}; drafts: {'off' if args.no_drafts else 'on'}; "
          f"context: {context or 'unlimited'}; loaded in {time.perf_counter() - started:.1f}s)", flush=True)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
