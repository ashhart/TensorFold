"""The server's open-file limit: raised to the hard limit at start, and an accept refused for want of one said once."""

from __future__ import annotations

import errno
import time
from typing import Any

FALLBACKS = (65536, 10240)             # tried when the hard limit is unlimited (macOS caps a process below that)
BACKOFF_S = 0.1                        # the listening socket stays readable, so a failed accept would spin a core


def raise_limit() -> None:
    """Raise the soft RLIMIT_NOFILE to the hard limit, so idle connections can't exhaust a default 1024."""

    try:
        import resource
    except ImportError:                 # Windows
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    targets = FALLBACKS if hard == resource.RLIM_INFINITY else (hard,)
    for target in targets:
        if soft != resource.RLIM_INFINITY and soft >= target:
            return
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            return
        except (OSError, ValueError):
            continue


def refused(server: Any, exc: OSError) -> None:
    """Print once a run of accepts fails for want of descriptors, and wait before the next try."""

    if exc.errno not in (errno.EMFILE, errno.ENFILE):
        return
    if not server.out_of_descriptors:
        server.out_of_descriptors = True
        print(f"[tensorfold] out of file descriptors ({exc.strerror}): new connections wait until open ones close; "
              "raise the open-file limit (ulimit -n)", flush=True)
    time.sleep(BACKOFF_S)
