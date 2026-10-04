"""DeepSeek-V4.1's rank protocol on the host: request headers round-trip, the doorbell on a real localhost TCPStore
with the two ranks as threads, rank 0's lists shared through an all-gather, and the startup agreement."""

from __future__ import annotations

import socket
import threading
import time
from datetime import timedelta

import pytest

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.cuda import protocol

torch = pytest.importorskip("torch")
TCPStore = pytest.importorskip("torch.distributed").TCPStore
SETTINGS = {"start_error": False, "dspark": True, "capacity": 65544, "prefill_rows": 2048, "max_rows": 6,
            "ring": 128, "policy": (3, None), "layers": 43, "world": 2, "engram_digest": -0x123456789ABCDEF0,
            "cache_bytes": 3 << 30, "cache_entries": 8}


@pytest.mark.parametrize("sampling", [
    None,
    Sampling(seed=(1 << 64) - 1, temperature=0.6, top_k=0, top_p=0.95, min_p=0.05),
    Sampling(seed=(1 << 62) + 12345, temperature=1.0 / 3.0, top_k=20, top_p=1.0),
])
@pytest.mark.parametrize("policy", [(0, None), (3, None), (5, 0.3), (1, 0.123457)])
def test_header_round_trip(sampling, policy):
    header = protocol.encode(77, False, True, 41, sampling, policy)
    assert len(header) == 16 and all(-2**31 <= v < 2**31 for v in header), "int32 values"
    req = protocol.decode(header)
    assert (req.max_tokens, req.stop_eos, req.draft, req.cached, req.policy) == (77, False, True, 41, policy)
    assert req.sampling == sampling, "seed, temperature, top_k, top_p and min_p travel bit for bit"
    assert protocol.decode(protocol.encode(1, True, False, 0, sampling, policy))[:4] == (1, True, False, 0)


def test_a_top_k_past_int32_travels_as_the_largest_int32():
    header = protocol.encode(5, True, True, 0, Sampling(seed=9, temperature=1.0, top_k=2**31 + 5), (3, None))
    assert all(-2**31 <= v < 2**31 for v in header)
    assert protocol.decode(header).sampling.top_k == protocol.TOP_K_MAX


def test_a_list_int32_cannot_hold_fails_before_any_gather():
    calls = []

    class Comm:
        def all_gather(self, send, recv):
            calls.append(send.numel())

    with pytest.raises(RuntimeError):
        protocol.share(Comm(), 0, [1, 2**31], "cpu")
    assert calls == [], "rank 1 would be left inside the gather"


def test_greedy_temperatures_decode_to_no_sampling_and_stop_is_no_header():
    zero = protocol.encode(5, True, True, 0, Sampling(seed=9, temperature=0.0), (3, None))
    assert protocol.decode(zero).sampling is None
    with pytest.raises(ValueError, match="at least one token"):
        protocol.encode(0, True, True, 0, None, (3, None))
    with pytest.raises(ValueError, match="16 values"):
        protocol.decode(protocol.STOP)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bells():
    port = _free_port()
    master = TCPStore("127.0.0.1", port, 2, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    worker = TCPStore("127.0.0.1", port, 2, False, timeout=timedelta(seconds=30))
    return master, protocol.Bell(master), protocol.Bell(worker)


def test_rank1_waits_for_each_ring_in_order():
    master, r0, r1 = _bells()
    woke: list[tuple[int, float]] = []

    def follower():
        for _ in range(3):
            r1.wait()
            woke.append((r1.rung, time.monotonic()))

    t = threading.Thread(target=follower)
    t.start()
    time.sleep(0.3)
    assert woke == [], "nothing rung: rank 1 still waits"
    rang = []
    for _ in range(3):
        rang.append(time.monotonic())
        r0.ring()
        time.sleep(0.1)
    t.join(10)
    assert not t.is_alive()
    assert [n for n, _ in woke] == [1, 2, 3] and all(w >= r for (_, w), r in zip(woke, rang))
    assert r0.rung == 3 and not master.check([protocol.BELL.format(n) for n in (1, 2, 3)]), "keys consumed"


def test_rings_before_rank1_waits_are_kept_and_no_store_is_no_bell():
    _, r0, r1 = _bells()
    r0.ring()
    r0.ring()
    r1.wait()
    r1.wait()
    assert r1.rung == 2
    none = protocol.Bell(None)
    none.ring()
    none.wait()                     # returns at once
    assert none.rung == 0


def test_idle_hours_are_waited_again_and_a_lost_rank0_raises():
    class Store:
        def __init__(self, errors):
            self.errors, self.deleted = list(errors), []

        def wait(self, keys, timeout):
            if self.errors:
                raise self.errors.pop(0)

        def delete_key(self, key):
            self.deleted.append(key)

    store = Store([RuntimeError("Socket Timeout"), RuntimeError("wait timeout after 3600000ms")])
    bell = protocol.Bell(store)
    bell.wait()
    assert bell.rung == 1 and store.deleted == ["tf_dsv41_request_1"]
    bell.store = Store([RuntimeError("Connection reset by peer")])
    with pytest.raises(RuntimeError, match="Connection reset"):
        bell.wait()


class _Pair:
    """Two thread ranks' all_gather on host tensors."""

    def __init__(self) -> None:
        self.slots = [None, None]
        self.gate = threading.Barrier(2, timeout=30)

    def comm(self, rank: int):
        pair = self

        class Comm:
            def all_gather(self, send, recv):
                pair.slots[rank] = send.clone()
                pair.gate.wait()
                recv.copy_(torch.cat([pair.slots[0], pair.slots[1]]))
                pair.gate.wait()

        return Comm()


def _both(fn0, fn1):
    out = [None, None]
    threads = [threading.Thread(target=lambda r=r, fn=fn: out.__setitem__(r, fn())) for r, fn in enumerate((fn0, fn1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return out


def test_share_hands_rank0s_list_to_rank1():
    pair = _Pair()
    c0, c1 = pair.comm(0), pair.comm(1)
    header = protocol.encode(9, True, True, 3, Sampling(seed=5, temperature=0.7), (5, 0.3))
    got = _both(lambda: [protocol.share(c0, 0, header, "cpu"), protocol.share(c0, 0, [], "cpu")],
                lambda: [protocol.share(c1, 1, None, "cpu"), protocol.share(c1, 1, None, "cpu")])
    assert got[0] == got[1] == [header, []]
    assert _both(lambda: protocol.gather_ints(c0, [1, 2], "cpu"),
                 lambda: protocol.gather_ints(c1, [3, 4], "cpu")) == [[[1, 2], [3, 4]]] * 2


def test_equal_settings_agree_on_the_smaller_room():
    a = protocol.settings(**SETTINGS)
    b = protocol.settings(**{**SETTINGS, "cache_bytes": 2 << 30, "cache_entries": 4})
    assert len(a) == len(protocol.SETTINGS) + len(protocol.SPARE)
    assert protocol.agree([a, b]) == protocol.agree([b, a]) == (2 << 30, 4)


@pytest.mark.parametrize("name, value", [
    ("engram_digest", -0x123456789ABCDEF0 + 1),         # the low word differs
    ("engram_digest", -0x123456789ABCDEF0 + (1 << 40)),  # the high word differs
    ("capacity", 4102), ("dspark", False), ("policy", (3, 0.5)), ("policy", (0, None)), ("layers", 8),
])
def test_different_settings_refuse_naming_both_values(name, value):
    a = protocol.settings(**SETTINGS)
    b = protocol.settings(**{**SETTINGS, name: value})
    with pytest.raises(RuntimeError, match="different settings") as err:
        protocol.agree([a, b])
    differ = [(n, x, y) for n, x, y in zip(protocol.SETTINGS, a, b) if x != y]
    assert differ
    for n, x, y in differ:
        assert f"{n} rank 0 {x}, rank 1 {y}" in str(err.value)
