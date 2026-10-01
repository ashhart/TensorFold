"""``kill -USR1 <pid>`` (or Ctrl+Break on Windows) prints every thread's Python stack, at start and while serving."""

import faulthandler
import os
import signal
from http.server import BaseHTTPRequestHandler

DUMP = getattr(signal, "SIGBREAK" if os.name == "nt" else "SIGUSR1", None)

_started = False                 # only a process that asked for the dump (the CLI) has it armed again


def _dump(signum, frame) -> None:
    faulthandler.dump_traceback(all_threads=True)


def start() -> None:
    """Arm the dump at start (main thread): USR1 is ignored first, so a re-arm's instant between handlers can't exit."""

    global _started
    if DUMP is not None:
        signal.signal(DUMP, signal.SIG_IGN)
    _started = True
    arm()


def arm() -> None:
    """Point USR1 at the dump again: an in-process compiler (Triton's LLVM) takes the signal when it first loads."""

    if not _started or DUMP is None:
        return
    try:
        if not hasattr(faulthandler, "register"):    # Windows has no faulthandler.register: a Python handler dumps
            signal.signal(DUMP, _dump)
            return
        faulthandler.unregister(DUMP)                # back to ignoring it for an instant, as ``start`` left it
        faulthandler.register(DUMP, all_threads=True)
    except (OSError, ValueError, RuntimeError):      # no usable stderr: serving goes on without the dump
        pass


class Rearming(BaseHTTPRequestHandler):
    """A request handler that arms the dump again after each request, which may have built a kernel."""

    def handle_one_request(self) -> None:
        super().handle_one_request()
        arm()
