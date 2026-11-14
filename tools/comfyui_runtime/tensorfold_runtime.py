import argparse
import json
import sys

from tf_installer.install import install, read_status, runtime_root
from tf_installer.pins import InstallError
from tf_installer.worker import start_worker


def parser():
    result = argparse.ArgumentParser(description="Install TensorFold's standalone engine and pinned ComfyUI MOSS companion outside ComfyUI.")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("install", "update", "status"):
        command = commands.add_parser(name)
        command.add_argument("--comfy-root", required=True)
        if name != "status":
            command.add_argument("--backend", default="cuda", choices=["cuda"])
            command.add_argument("--zig", help="Path to Zig 0.17.0; defaults to zig on PATH.")
            command.add_argument("--nvcc", help="Path to CUDA nvcc; defaults to nvcc on PATH.")
            sources = command.add_mutually_exclusive_group()
            sources.add_argument("--source-checkout", help="Exact clean pinned TensorFold Git checkout.")
            sources.add_argument("--source-archive", help="Local official source archive matching the pinned SHA-256.")
    run = commands.add_parser("run", help="Start the installed MOSS worker on GB10 (sm_121) only; arguments after -- go to the worker.")
    run.add_argument("--comfy-root", required=True)
    run.add_argument("worker", nargs=argparse.REMAINDER)
    return result


def main(argv=None):
    arguments = vars(parser().parse_args(argv))
    command = arguments.pop("command")
    try:
        if command == "status":
            print(json.dumps(read_status(runtime_root(arguments["comfy_root"])), indent=2, sort_keys=True))
        elif command == "run":
            worker = arguments["worker"]
            start_worker(arguments["comfy_root"], worker[1:] if worker[:1] == ["--"] else worker)
        else:
            install(action=command, **arguments)
    except (InstallError, OSError, UnicodeError, ValueError) as error:
        print(f"TensorFold: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("TensorFold: installation cancelled; run status to inspect the active runtime.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
