"""GLM's idle doorbell (engine._ring / _await_bell) on a real localhost TCPStore, the two ranks as threads (CPU)."""

from __future__ import annotations

import socket
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.distributed import TCPStore  # noqa: E402

from tensorfold.families.glm5_next.cuda.engine import GlmEngine  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ranks():
    port = _free_port()
    master = TCPStore("127.0.0.1", port, 2, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    worker = TCPStore("127.0.0.1", port, 2, False, timeout=timedelta(seconds=30))
    r0 = SimpleNamespace(comm=SimpleNamespace(store=master))
    r1 = SimpleNamespace(comm=SimpleNamespace(store=worker))
    for r in (r0, r1):
        for name in ("_store", "_ring", "_await_bell"):
            setattr(r, name, getattr(GlmEngine, name).__get__(r))
    return master, r0, r1


def test_rank1_blocks_until_rank0_rings_then_follows_every_request_in_order():
    master, r0, r1 = _ranks()
    woke: list[tuple[int, float]] = []

    def follower():
        for _ in range(3):
            r1._await_bell()
            woke.append((r1._bell, time.monotonic()))

    t = threading.Thread(target=follower)
    t.start()
    time.sleep(0.3)
    assert woke == []                                   # nothing rung: rank 1 is still waiting
    rang = []
    for _ in range(3):
        rang.append(time.monotonic())
        r0._ring()
        time.sleep(0.1)
    t.join(10)
    assert not t.is_alive()
    assert [n for n, _ in woke] == [1, 2, 3]
    assert all(w >= r for (_, w), r in zip(woke, rang))
    assert r0._bell == 3
    assert master.num_keys() <= 2                       # consumed keys are deleted (the store's own key(s) remain)


def test_rings_before_rank1_waits_are_not_lost():
    _, r0, r1 = _ranks()
    r0._ring()
    r0._ring()                                          # rank 1 still busy with an earlier request
    r1._await_bell()
    r1._await_bell()
    assert r1._bell == 2


def test_no_store_means_no_doorbell():
    r = SimpleNamespace(comm=SimpleNamespace())
    for name in ("_store", "_ring", "_await_bell"):
        setattr(r, name, getattr(GlmEngine, name).__get__(r))
    r._ring()
    r._await_bell()                                     # returns at once
    assert not hasattr(r, "_bell")


def test_idle_timeouts_are_retried_and_other_errors_raise():
    class Store:
        def __init__(self, errors):
            self.errors, self.deleted = list(errors), []

        def wait(self, keys, timeout):
            if self.errors:
                raise self.errors.pop(0)

        def delete_key(self, key):
            self.deleted.append(key)

    store = Store([RuntimeError("Socket Timeout"), RuntimeError("wait timeout after 3600000ms")])
    r = SimpleNamespace(comm=SimpleNamespace(store=store))
    for name in ("_store", "_ring", "_await_bell"):
        setattr(r, name, getattr(GlmEngine, name).__get__(r))
    r._await_bell()                                     # two idle hours, then the request
    assert r._bell == 1 and store.deleted == ["tf_glm_request_1"]
    r.comm.store = Store([RuntimeError("Connection reset by peer")])
    with pytest.raises(RuntimeError, match="Connection reset"):
        r._await_bell()                                 # rank 0 is gone: rank 1 stops instead of waiting forever
