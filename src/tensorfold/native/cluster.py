"""`tensorfold cluster|node` and `serve --cluster`: clusters run only on the native engine, so these exec it."""

from __future__ import annotations

import os
import signal
import sys

from tensorfold import native

COMMANDS = ("cluster", "node")


def wanted(argv: list[str]) -> bool:
    """Whether the native engine's cluster commands serve this command line."""

    if not argv:
        return False
    if argv[0] in COMMANDS:
        return True
    return argv[0] == "serve" and any(token == "--cluster" or token.startswith("--cluster=") for token in argv[1:])


def run(argv: list[str]) -> int:
    """Exec the native engine with `argv`; there is no Python cluster engine to fall back to."""

    binary = native.binary()
    if binary is None:
        print(f"tensorfold: clusters run on the native engine, and this install has none "
              f"({native.ROOT / native.BINARY} is missing)", file=sys.stderr)
        return 1
    sys.stdout.flush()
    sys.stderr.flush()
    for name in ("SIGPIPE", "SIGXFSZ"):      # Python ignores these; a program started from a shell does not
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.SIG_DFL)
    try:
        os.execv(binary, ["tensorfold", *argv])
    except OSError as exc:
        print(f"tensorfold: the native engine did not start ({exc.strerror})", file=sys.stderr)
    return 1
