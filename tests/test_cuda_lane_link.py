"""Lane messages over a communicator: long messages, the idle doorbell, and two ranks on threads equal to one."""

from __future__ import annotations

import threading

import pytest

from tensorfold.cuda.lanes.link import ADMIT, ROUND, STOP, pack_sampling, unpack_sampling
from tensorfold.engine.exact_sampling import Sampling


@pytest.mark.parametrize(
    "s",
    [
        None,
        Sampling(seed=(1 << 63) - 7, temperature=0.7, top_k=0, top_p=0.9, min_p=0.05),
        Sampling(seed=3, temperature=1e-3, top_k=40),
    ],
)
def test_sampling_crosses_with_its_exact_bits(s):
    words = pack_sampling(s)
    assert len(words) == 18 and all(0 <= w < 1 << 16 for w in words[1:])
    assert unpack_sampling(words) == s


class Board:
    def __init__(self):
        self.slots, self.barrier = [None, None], threading.Barrier(2, timeout=30)
        self.keys, self.cv = {}, threading.Condition()


class ThreadComm:
    """Two ranks as two threads: ``all_gather`` meets the other thread; ``store`` is a dict with a blocking wait."""

    def __init__(self, rank, board):
        self.rank, self.world, self.board, self.store, self.gathers = rank, 2, board, self, 0

    def all_gather(self, send, recv):
        import torch

        self.gathers += 1
        self.board.slots[self.rank] = send.clone()
        self.board.barrier.wait()
        recv.copy_(torch.cat(self.board.slots))
        self.board.barrier.wait()

    def set(self, key, value):
        with self.board.cv:
            self.board.keys[key] = value
            self.board.cv.notify_all()

    def wait(self, keys, timeout):
        with self.board.cv:
            if not self.board.cv.wait_for(lambda: all(k in self.board.keys for k in keys), timeout.total_seconds()):
                raise RuntimeError("Wait timeout")


def links(words=8):
    from tensorfold.cuda.lanes.link import RoundLink

    board = Board()
    return [RoundLink(ThreadComm(r, board), words, "cpu") for r in (0, 1)]


def test_messages_longer_than_the_head_take_a_second_gather():
    pytest.importorskip("torch")
    a, b = links(words=8)
    got = []
    t = threading.Thread(target=lambda: got.extend([b.recv(), b.recv(), b.recv()]))
    t.start()
    a.send(ADMIT, list(range(100)))
    a.send(ROUND, [5, 6])
    a.send(STOP, [])
    t.join(30)
    assert got == [(ADMIT, list(range(100))), (ROUND, [5, 6]), (STOP, [])]
    assert a.comm.gathers == 4


def test_an_idle_follower_sleeps_on_the_store_until_the_next_message():
    pytest.importorskip("torch")
    a, b = links()
    got = []
    t = threading.Thread(target=lambda: got.append(b.recv()))
    t.start()
    a.idle()
    a.idle()  # once: already asleep
    t.join(0.3)
    assert t.is_alive() and b.comm.gathers == 1  # asleep after the IDLE message, not in a gather
    a.send(ROUND, [1, 2, 3])
    t.join(30)
    assert got == [(ROUND, [1, 2, 3])] and a.bells == b.bells == 1


def test_two_ranks_on_threads_decode_as_one():
    pytest.importorskip("torch")
    from cuda_lane_fakes import PLANES, FakeForward, Pattern, drive

    from tensorfold.cuda.kvpool import PagePool
    from tensorfold.cuda.lanes import LaneDecoder, Lanes
    from tensorfold.cuda.sessions import TieredCache
    from tensorfold.cuda.streams import Stream

    ps = [[(5 * i + j * j) % 50 + 1 for j in range(30 + 7 * i)] for i in range(3)]

    def build(link):
        pool = PagePool(PLANES, 64, 16)
        return {
            "forward": FakeForward(pool, 3),
            "pool": pool,
            "cache": TieredCache(4),
            "drafter": Pattern(),
            "link": link,
        }

    one = build(None)
    solo = LaneDecoder(one.pop("forward"), capacity=256, lanes=3, eos=(0,), **one)
    want = [Stream(list(p), 25, Sampling(seed=9, temperature=0.8)) for p in ps]
    drive(solo, want, lanes=3)

    a, b = links(words=16)
    r1 = build(b)
    follower = Lanes(r1["forward"], 3, 256, pool=r1["pool"], cache=r1["cache"], drafter=r1["drafter"])
    t = threading.Thread(target=follower.follow, args=(b,))
    t.start()
    r0 = build(a)
    lead = LaneDecoder(r0.pop("forward"), capacity=256, lanes=3, eos=(0,), **r0)
    got = [Stream(list(p), 25, Sampling(seed=9, temperature=0.8)) for p in ps]
    drive(lead, got, lanes=3)
    lead.stop()
    t.join(30)
    assert not t.is_alive()
    assert [s.out for s in got] == [s.out for s in want]
    assert lead.local.state() == follower.state() and lead.forward.state() == follower.forward.state()


def test_ranks_agree_only_when_every_rank_applied_the_message():
    a, b = links()
    out = [None, None]

    def rank(i, link, ok):
        out[i] = [link.agree(ok), link.agree(True)]

    threads = [threading.Thread(target=rank, args=(0, a, True)), threading.Thread(target=rank, args=(1, b, False))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert out == [[False, True], [False, True]]
