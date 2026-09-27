"""Ordered background writes of DFlash2 capture records."""

from __future__ import annotations

from typing import Any


_CAPTURE_QUEUE: Any = None


def _capture_writer() -> Any:
    """A queue drained by one daemon thread that appends capture records to their files, in order."""

    global _CAPTURE_QUEUE
    if _CAPTURE_QUEUE is None:
        import queue
        import threading

        _CAPTURE_QUEUE = queue.Queue()

        def drain() -> None:
            while True:
                path, record = _CAPTURE_QUEUE.get()
                try:
                    with open(path, "ab") as handle:
                        handle.write(record)
                except OSError as exc:
                    print(f"[lanes] draft capture write failed: {exc}", flush=True)

        threading.Thread(target=drain, name="draft-capture", daemon=True).start()
    return _CAPTURE_QUEUE
