"""A stand-in native binary that must fail the gate: the Python engine with one token changed, or slowed down."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

OFFSET = 20          # the planted reply's character that changes
LAUNCHER = ("#!{python} -B\nimport sys\nsys.path.insert(0, {here!r})\nimport gate_plant\n"
            "sys.exit(gate_plant.native(sys.argv[1:], {settings!r}))\n")


def install(src: Path, python: str, model: Path, plant: str, target: str, delay: float) -> Path:
    """Write SRC/tensorfold/native/bin/tensorfold-native: MODEL's cell served through the planted Python engine."""

    sys.path.insert(0, str(src))
    from tensorfold import families
    from tensorfold.native import BINARY, contract

    settings = {"plant": plant, "target": target, "delay": delay,
                "families": {families.model_type(model): [contract.weight_format(families.read_config(model))]}}
    path = src / "tensorfold" / "native" / BINARY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LAUNCHER.format(python=python, here=str(Path(__file__).resolve().parent), settings=settings))
    path.chmod(0o755)
    return path


def native(argv: list[str], settings: dict) -> int:
    """The binary: `capabilities --json`, or `serve ...` through the Python engine with the plant in place."""

    if argv == ["capabilities", "--json"]:
        print(json.dumps(capabilities(settings)))
        return 0
    if argv[:1] != ["serve"]:
        print("tensorfold: the stand-in native engine only serves", file=sys.stderr)
        return 2
    plant(settings)
    from tensorfold import cli

    return cli.main([*argv, "--engine", "python"])


def capabilities(settings: dict) -> dict:
    """Every serve flag and set TENSORFOLD_ or TF_ variable: it is the Python engine, so it honours them all."""

    from tensorfold import __version__
    from tensorfold.cli import build_parser
    from tensorfold.native.switch import ENV, KNOBS

    parser = build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    flags = {name: {} for name in commands.choices["serve"]._option_string_actions
             if name.startswith("--") and name != "--engine"}
    env = sorted(key for key in os.environ if key.startswith(KNOBS) and key != ENV)
    return {"schema": 1, "engine": "zig", "version": __version__, "chip": chip(), "backends": ["metal"],
            "families": settings["families"], "serve": {"flags": flags}, "env": env}


def chip() -> str | None:
    """apple-m<generation> from the CPU's brand string, as gate entries name Macs."""

    brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    found = re.search(r"Apple M(\d+)", brand)
    return f"apple-m{found.group(1)}" if found else None


def swap(text: str, at: int) -> str:
    """TEXT with its character at AT replaced; unchanged when AT falls outside it."""

    if not 0 <= at < len(text):
        return text
    return text[:at] + ("#" if text[at] != "#" else "%") + text[at + 1:]


class Swap:
    """Streams the reply on with its character at OFFSET changed, however the pieces split it."""

    def __init__(self, deliver) -> None:
        self.deliver, self.seen = deliver, 0

    def __call__(self, delta):
        if isinstance(delta, str):
            delta = swap(delta, OFFSET - self.seen)
            self.seen += len(delta)
        return self.deliver(delta)


class Slow:
    """Streams each piece on, then waits: the same tokens, delivered later."""

    def __init__(self, deliver, delay: float) -> None:
        self.deliver, self.delay = deliver, delay

    def __call__(self, delta):
        result = self.deliver(delta)
        time.sleep(self.delay)
        return result


def targeted(messages: list[dict], options: dict, target: str) -> bool:
    """The planted request: greedy, its last message containing TARGET."""

    last = messages[-1].get("content") if messages else None
    return float(options.get("temperature") or 0.0) == 0.0 and isinstance(last, str) and target in last


def plant(settings: dict) -> None:
    """Wrap ChatApp.chat: the target's reply gets one changed character and SHA, or every piece is delayed."""

    from tensorfold.server.app import ChatApp

    chat = ChatApp.chat

    def planted(self, messages, **options):
        hit = settings["plant"] == "token" and targeted(messages, options, settings["target"])
        if options.get("on_delta") is not None and (hit or settings["plant"] == "slow"):
            options["on_delta"] = Swap(options["on_delta"]) if hit else Slow(options["on_delta"], settings["delay"])
        reply = chat(self, messages, **options)
        if hit:
            print("[plant] this reply's character and token SHA were changed", flush=True)
            sha = reply["runtime"]["token_sha"]
            reply["runtime"]["token_sha"] = hashlib.sha256(f"plant {sha}".encode()).hexdigest()[:12]
            reply["content"] = swap(reply["content"], OFFSET) if isinstance(reply["content"], str) else None
        return reply

    ChatApp.chat = planted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="a scratch copy of the tree's src/ (never a worktree)")
    parser.add_argument("--python", required=True, help="the interpreter that serves (the install's)")
    parser.add_argument("--model", required=True, type=Path, help="the checkpoint the cell serves")
    parser.add_argument("--plant", required=True, choices=("token", "slow"))
    parser.add_argument("--target", default="Write a short Python function that computes the Fibonacci sequence",
                        help="token: the greedy request whose reply changes")
    parser.add_argument("--delay", type=float, default=0.02, help="slow: seconds after each streamed piece")
    args = parser.parse_args(argv)
    print(install(args.src, args.python, args.model, args.plant, args.target, args.delay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
