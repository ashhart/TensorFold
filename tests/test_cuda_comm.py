"""The CUDA communicator interface: optional capabilities fall back to the all-gather; backends wrap NCCL."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import comm


class Pair:
    """Two ranks' all-gathers in one process: ``all_gather`` returns both ranks' last sends in rank order."""

    def __init__(self, rank: int, board: dict):
        self.rank, self.world, self.board, self.calls = rank, 2, board, []

    def all_gather(self, send, recv):
        self.calls.append(send.numel())
        self.board[self.rank] = send.clone()
        other = self.board.get(1 - self.rank, torch.zeros_like(send))
        parts = [self.board[self.rank], other] if self.rank == 0 else [other, self.board[self.rank]]
        recv.copy_(torch.cat(parts))

    def barrier(self):
        pass


def test_fast_gather_takes_the_fast_path_only_where_there_is_one():
    board, seen = {}, []
    plain = Pair(0, board)
    send, recv = torch.arange(3, dtype=torch.float32), torch.empty(6)
    comm.fast_gather(plain, send, recv)
    assert plain.calls == [3] and recv[:3].tolist() == [0.0, 1.0, 2.0]
    plain.all_gather_fast = lambda s, r: seen.append(s.numel())
    comm.fast_gather(plain, send, recv)
    assert seen == [3] and plain.calls == [3]


def test_check_is_a_no_op_without_a_transport_check():
    comm.check(Pair(0, {}))
    failing = Pair(0, {})
    failing.check = lambda: (_ for _ in ()).throw(RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        comm.check(failing)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_exchange_without_an_exchange_trades_bytes_through_the_all_gather(dtype):
    board = {}
    a, b = Pair(0, board), Pair(1, board)
    a_send = [torch.arange(5).to(dtype), torch.arange(2).to(dtype) + 10]
    b_send = [torch.arange(7).to(dtype) + 100, torch.arange(4).to(dtype) + 200]
    a_recv = [torch.empty(7, dtype=dtype), torch.empty(4, dtype=dtype)]
    b_recv = [torch.empty(5, dtype=dtype), torch.empty(2, dtype=dtype)]
    for i in range(2):                          # the two ranks' gathers of pair i, in order, as the collective runs
        n = max(a_send[i].numel(), b_send[i].numel()) * a_send[i].element_size()
        board[1] = torch.zeros(n, dtype=torch.uint8)
        board[1][:b_send[i].numel() * b_send[i].element_size()] = b_send[i].view(-1).view(torch.uint8)
        comm.exchange(a, [a_send[i]], [a_recv[i]])
        comm.exchange(b, [b_send[i]], [b_recv[i]], peer=0)
        assert torch.equal(a_recv[i], b_send[i]) and torch.equal(b_recv[i], a_send[i])


def test_exchange_uses_the_communicators_own_exchange_with_the_other_rank():
    seen = []
    c = Pair(1, {})
    c.exchange = lambda s, r, peer: seen.append((len(s), len(r), peer))
    comm.exchange(c, [torch.zeros(1)], [torch.zeros(1)])
    comm.exchange(c, [], [], peer=3)
    assert seen == [(1, 1, 0), (0, 0, 3)] and c.calls == []
    wide = Pair(0, {})
    wide.world = 4
    with pytest.raises(ValueError, match="two ranks only"):
        comm.exchange(wide, [torch.zeros(1)], [torch.zeros(1)], peer=2)


def test_backends_wrap_nccl_and_unknown_ones_are_refused(monkeypatch):
    made = []
    monkeypatch.setattr(comm, "NCCL", lambda *a, **k: made.append((a, k)) or "nccl")
    monkeypatch.setattr(comm, "BACKENDS", {})
    assert comm.open_comm(0, 2, "192.0.2.1", 29551) == "nccl"
    comm.register_backend("fast", lambda base: ("fast", base))
    monkeypatch.setenv(comm.BACKEND_ENV, "fast")
    assert comm.open_comm(1, 2, "192.0.2.1", 29551, timeout_s=900) == ("fast", "nccl")
    assert made[-1] == ((1, 2, "192.0.2.1", 29551), {"timeout_s": 900})
    with pytest.raises(ValueError, match="TF_COMM_BACKEND='roce': expected nccl, fast"):
        comm.open_comm(0, 2, "192.0.2.1", 29551, backend="roce")


def test_a_transport_keeps_what_it_does_not_change_from_nccl():
    class Fast(comm.Transport):
        def all_gather_fast(self, send, recv):
            recv.fill_(7)

    base = Pair(1, {})
    base.store, base.ready = "the store", lambda label: label
    fast = Fast(base)
    assert (fast.rank, fast.world, fast.store, fast.ready("loading")) == (1, 2, "the store", "loading")
    send, recv = torch.ones(2), torch.zeros(4)
    comm.fast_gather(fast, send, recv)
    assert recv.tolist() == [7.0] * 4 and base.calls == []
    comm.exchange(fast, [send], [torch.zeros(2)])          # the base has no exchange: through its all-gather
    assert base.calls == [8]


class Store:
    def __init__(self, keys):
        self.keys = keys

    def set(self, key, value):
        self.keys[key] = value.encode()

    def get(self, key):
        return self.keys[key]


@pytest.mark.parametrize("other", ["nccl", "fast"])
def test_the_ranks_must_name_the_same_backend(monkeypatch, other):
    keys = {"tf_comm_backend/1": other.encode()}
    made = SimpleNamespace(rank=0, world=2, store=Store(keys))
    monkeypatch.setattr(comm, "NCCL", lambda *a, **k: made)
    monkeypatch.setattr(comm, "BACKENDS", {"fast": lambda base: ("fast", base)})
    if other == "nccl":
        assert comm.open_comm(0, 2, "192.0.2.1", 29551) is made and keys["tf_comm_backend/0"] == b"nccl"
    else:
        with pytest.raises(RuntimeError, match="different TF_COMM_BACKEND: nccl, fast"):
            comm.open_comm(0, 2, "192.0.2.1", 29551)
