"""The ``tensorfold`` command.

    tensorfold serve MODEL_DIR [--port 8080] [--context 32768] [--temperature 0.7] ...
    tensorfold models
    tensorfold info MODEL_DIR

``serve`` loads the model with its family's kernels and serves an OpenAI-compatible API at
``http://HOST:PORT/v1`` (``/v1/models``, ``/v1/chat/completions``, ``/v1/completions``).
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorfold",
        description="Fast, exact LLM decoding on Apple Silicon behind an OpenAI-compatible endpoint.",
    )
    parser.add_argument("--version", action="version", version=f"tensorfold {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve a model at an OpenAI-compatible endpoint",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    serve.add_argument("model", type=Path, help="model directory (config.json, weights, tokenizer)")
    endpoint = serve.add_argument_group("endpoint")
    endpoint.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0: every interface)")
    endpoint.add_argument("--port", type=int, default=8080)
    endpoint.add_argument("--name", default="", help="model id clients ask for (default: the directory's name)")
    endpoint.add_argument("--alias", action="append", default=[], help="another model id to answer to")

    generation = serve.add_argument_group("generation (requests can override each of these)")
    generation.add_argument("--context", type=int, default=0,
                            help="context window: prompt plus reply tokens a request may use (0: no limit)")
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
    speed.add_argument("--drafter", default="",
                       help="a DFlash2 draft model (Hugging Face id in the local cache, or a directory); Qwen3.8 dense")
    speed.add_argument("--drafter-bits", type=int, default=4, help="quantize the draft model's linears (0: bf16)")
    speed.add_argument("--mtp-drafts", type=int, default=None,
                       help="most MTP drafts a round (Qwen3.8 Flash Next; default 3, 0: none)")
    speed.add_argument("--mtp-head", default="", help="a converted MTP head file (Nemotron; 0: none)")
    speed.add_argument("--lane-kernels", choices=("auto", "on", "off"), default="auto",
                       help="lane kernels for Qwen3.8 dense (auto: on GPUs with tensor units)")
    speed.add_argument("--prompt-cache-gib", type=float, default=16.0,
                       help="memory for cached conversation prefixes (0: off)")
    speed.add_argument("--snapshot-dir", default=str(Path.home() / ".cache" / "tensorfold" / "prefix-snapshots"),
                       help="where system-block and conversation snapshots are kept ('none': in memory only)")
    speed.add_argument("--max-snapshots", type=int, default=3, help="system-block snapshots loaded at start")
    speed.add_argument("--mlx-cache-gib", type=float, default=8.0, help="MLX's cache of freed buffers")
    serve.set_defaults(func=cmd_serve)

    models = commands.add_parser("models", help="list the model families this build supports")
    models.set_defaults(func=cmd_models)

    info = commands.add_parser("info", help="show which family serves a model directory (loads no weights)")
    info.add_argument("model", type=Path)
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``tensorfold MODEL_DIR ...`` is ``tensorfold serve MODEL_DIR ...``
    if argv and not argv[0].startswith("-") and argv[0] not in ("serve", "models", "info") and Path(argv[0]).is_dir():
        argv = ["serve", *argv]
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


def cmd_models(args: argparse.Namespace) -> int:
    from tensorfold import families

    for kind, family in sorted(families.families().items()):
        engine = "lane engine" if family.lanes else "serial engine"
        print(f"{kind:14s} {family.title} ({engine}, {family.module})")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from tensorfold import families

    config = families.read_config(args.model)
    text = config.get("text_config", config)
    try:
        family = families.detect(args.model)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    quant = config.get("quantization") or text.get("quantization") or {}
    print(f"model_type   {family.model_type}")
    print(f"family       {family.title} ({family.module})")
    print(f"engine       {'lane engine' if family.lanes else 'serial engine'}")
    for key in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok", "n_routed_experts",
                "vocab_size", "max_position_embeddings"):
        if key in text:
            print(f"{key:12s} {text[key]}" if len(key) <= 12 else f"{key} {text[key]}")
    if quant:
        print(f"quantization {quant.get('bits')}-bit, groups of {quant.get('group_size')}")
    generation = _generation_config(args.model)
    if generation:
        print(f"sampling     {generation}")
    return 0


def _generation_config(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "generation_config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    return {k: config[k] for k in ("temperature", "top_k", "top_p") if k in config}


def cmd_serve(args: argparse.Namespace) -> int:
    from http.server import ThreadingHTTPServer

    from tensorfold import families

    model_dir = Path(args.model).expanduser()
    if not (model_dir / "config.json").is_file():
        print(f"{model_dir} has no config.json", file=sys.stderr)
        return 1
    family = families.detect(model_dir)
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
    options: dict[str, Any] = {"lane_kernels": args.lane_kernels, "drafter": args.drafter,
                               "drafter_bits": args.drafter_bits, "mtp_head": args.mtp_head}
    if args.mtp_drafts is not None:
        options["mtp_drafts"] = int(args.mtp_drafts)
    print(f"[tensorfold] loading {model_dir.name}: {family.title} ({family.model_type})", flush=True)
    model, tokenizer = family.package.load(model_dir, **options)

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
    budget = int(float(args.prompt_cache_gib) * 1024**3)
    served = args.name or model_dir.name
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
        context_window=int(args.context),
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
          f"context: {args.context or 'unlimited'}; loaded in {time.perf_counter() - started:.1f}s)", flush=True)

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
