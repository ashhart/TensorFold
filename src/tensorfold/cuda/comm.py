"""NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
from collections.abc import Callable
from typing import Protocol

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4,
           torch.uint8: 1, torch.int8: 0, torch.float16: 6}
BACKEND_ENV = "TF_COMM_BACKEND"


class Comm(Protocol):
    """What every engine assumes; ``all_gather_fast``, ``exchange``, ``check`` (and NCCL's ``store``) are optional."""

    rank: int
    world: int

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None: ...

    def barrier(self) -> None: ...


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    if os.name == "nt":
        raise RuntimeError("TensorFold does not run tensor-parallel (NCCL) on Windows: CUDA on Windows has no "
                           "libnccl to wrap; use one GPU per process there")
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


class NCCL:
    def __init__(self, rank: int, world: int, master: str, port: int, *, timeout_s: float = 600,
                 key: str = "tf_nccl_uid") -> None:
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.rank, self.world = rank, world
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=timeout_s))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set(key, bytes(uid.internal))
        else:
            raw = self.store.get(key)
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (contiguous tensors, same dtype)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllGather(send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype],
                                           self.comm, stream))

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""

        import time
        from datetime import timedelta

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        started = time.monotonic()
        while True:
            try:
                self.store.wait([f"tf_ready/{label}/{r}" for r in others], timedelta(seconds=every))
                return
            except Exception as exc:                  # noqa: BLE001  (the store's timeout; anything else goes up)
                if "timeout" not in str(exc).lower():
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(f"rank {self.rank} finished {label} but rank {missing} has not after "
                                   f"{waited / 60:.0f} min: check that rank's log (a CUDA extension build waiting on "
                                   "a lock names the lock there)")
            print(f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                  flush=True)

    def exchange(self, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int) -> None:
        """One NCCL group with ``peer``: each ``sends[i]`` lands in the peer's ``recvs[i]`` (contiguous tensors)."""

        if len(sends) != len(recvs) or peer == self.rank or not 0 <= peer < self.world:
            raise ValueError("exchange: one receive per send, with another rank")
        lib = self.lib
        if not getattr(self, "_p2p", False):
            for name in ("ncclSend", "ncclRecv"):
                getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_void_p]
            lib.ncclGroupStart.argtypes = []
            lib.ncclGroupEnd.argtypes = []
            self._p2p = True
        stream = torch.cuda.current_stream().cuda_stream
        self._check(lib.ncclGroupStart())
        try:
            for s, r in zip(sends, recvs):
                if s.numel():
                    self._check(lib.ncclSend(s.data_ptr(), s.numel(), _DTYPES[s.dtype], peer, self.comm, stream))
                if r.numel():
                    self._check(lib.ncclRecv(r.data_ptr(), r.numel(), _DTYPES[r.dtype], peer, self.comm, stream))
        finally:
            self._check(lib.ncclGroupEnd())

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()


def fast_gather(comm: Comm, send: torch.Tensor, recv: torch.Tensor) -> None:
    """A model exchange: the communicator's faster ``all_gather_fast`` if it has one, else ``all_gather``."""

    fast = getattr(comm, "all_gather_fast", None)
    if fast is None:
        comm.all_gather(send, recv)
    else:
        fast(send, recv)


def check(comm: Comm) -> None:
    """Raise a failure the communicator's transport recorded (call it after a synchronizing exchange)."""

    found = getattr(comm, "check", None)
    if found is not None:
        found()


def exchange(comm: Comm, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int | None = None) -> None:
    """``comm.exchange`` when it has one, else (two ranks) each pair traded as bytes through the all-gather."""

    peer = 1 - comm.rank if peer is None else peer
    fn = getattr(comm, "exchange", None)
    if fn is not None:
        fn(sends, recvs, peer)
        return
    if comm.world != 2:
        raise ValueError("exchange: a communicator without exchange trades through its all-gather, two ranks only")
    for s, r in zip(sends, recvs):
        sb, rb = s.contiguous().view(-1).view(torch.uint8), r.view(-1).view(torch.uint8)
        n = max(sb.numel(), rb.numel())
        if n == 0:
            continue
        pad = torch.zeros((n,), dtype=torch.uint8, device=s.device)
        pad[:sb.numel()].copy_(sb)
        both = torch.empty((2 * n,), dtype=torch.uint8, device=s.device)
        comm.all_gather(pad, both)
        rb.copy_(both[peer * n:peer * n + rb.numel()])


class Transport:
    """A base for transports: whatever a subclass does not define (``store``, ``ready``, ...) is the wrapped NCCL's."""

    def __init__(self, base: Comm) -> None:
        self.base, self.rank, self.world = base, base.rank, base.world

    def __getattr__(self, name: str):
        if name == "base":                          # not set yet (a copy): no recursion
            raise AttributeError(name)
        return getattr(self.base, name)


BACKENDS: dict[str, Callable[[Comm], Comm]] = {}


def register_backend(name: str, wrap: Callable[[Comm], Comm]) -> None:
    """A transport over NCCL (still the control channel): ``wrap(nccl)`` runs on every rank at once."""

    BACKENDS[name] = wrap


def open_comm(rank: int, world: int, master: str, port: int, *, backend: str | None = None, **nccl) -> Comm:
    """NCCL, wrapped by ``backend`` (default TF_COMM_BACKEND, else none); the ranks check they named the same one."""

    name = (backend if backend is not None else os.environ.get(BACKEND_ENV, "")).strip().lower() or "nccl"
    if name != "nccl" and name not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={name!r}: expected nccl{''.join(', ' + b for b in sorted(BACKENDS))}")
    base = NCCL(rank, world, master, port, **nccl)
    store = getattr(base, "store", None)
    if store is not None and world > 1:          # a transport on one rank only would hang the first exchange
        store.set(f"tf_comm_backend/{rank}", name)
        names = [store.get(f"tf_comm_backend/{r}").decode() for r in range(world)]
        if len(set(names)) > 1:
            raise RuntimeError(f"the ranks were started with different {BACKEND_ENV}: {', '.join(names)} (rank order)")
    return base if name == "nccl" else BACKENDS[name](base)
