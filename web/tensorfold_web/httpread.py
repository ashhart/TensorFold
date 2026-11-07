
from __future__ import annotations

import contextlib
import threading
import time

import httpx


def until_cancelled(work, cancel: threading.Event | None, *, close=None, name: str = "http-read") -> None:

    failed: list[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as exc:  # noqa: BLE001 - raised again by the caller
            failed.append(exc)

    t = threading.Thread(target=run, name=name, daemon=True)
    t.start()
    while t.is_alive():
        if cancel is not None and cancel.is_set():
            if close is not None:
                with contextlib.suppress(Exception):   # nothing to close yet, or closed already
                    close()
            t.join(1.0)
            raise InterruptedError("cancelled")
        t.join(0.1)
    if failed:
        raise failed[0]


def error_text(r: httpx.Response, *, limit: int = 4096, seconds: float = 10.0,
               cancel: threading.Event | None = None) -> str:
    got: list[bytes] = []

    def read() -> None:
        size = 0
        try:
            for chunk in r.iter_raw():
                got.append(chunk)
                size += len(chunk)
                if size >= limit:
                    break
        except (httpx.HTTPError, OSError, RuntimeError):
            pass                                          # closed under it, or the connection went: what came is kept

    reader = threading.Thread(target=read, name="error-body", daemon=True)
    reader.start()
    end = time.monotonic() + seconds
    while reader.is_alive() and time.monotonic() < end and not (cancel is not None and cancel.is_set()):
        reader.join(0.1)
    if reader.is_alive():
        r.close()                                         # the blocked read ends with the connection
        reader.join(1.0)
    return b"".join(got)[:limit].decode(errors="replace").strip()
