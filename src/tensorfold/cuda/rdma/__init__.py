"""Small all-gathers between two machines over RoCE, through pinned host memory (no GPUDirect: GB10 has none).

NCCL's small all-gathers between two DGX Sparks cost ~19 us of fixed latency each (two per layer per decode step).
Here the GPU stages its shard in pinned host memory, a host thread RDMA-writes it and a flag to the peer, and the
peer's GPU polls the flag in its own pinned memory: one kernel per gather, CUDA-graph replayable, the same output
layout (rank order) and bits as NCCL. ``RdmaComm`` sends gathers that fit a slot this way and the rest through NCCL.
Host side: ``rdma_proxy.c``; GPU side: ``gather.cu``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).parent
SLOT_ALIGN = 4096
INFO_WORDS = 7                       # tf_rdma_info: qpn, psn, rkey, addr, mtu, gid_hi, gid_lo
SPIN = 20_000_000                    # flag polls before a wait gives up (~20 s): the peer died or never sent
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def proxy() -> ctypes.CDLL:
    """The host proxy, built with the C compiler and libibverbs once per source hash."""

    src = HERE / "rdma_proxy.c"
    tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    out = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensorfold" / f"tf_rdma-{tag}.so"
    with _LOCK:
        if not out.exists():
            cc = shutil.which(os.environ.get("CC", "gcc")) or shutil.which("cc")
            if cc is None:
                raise RuntimeError("the RoCE all-gather needs a C compiler and the libibverbs headers")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            done = subprocess.run([cc, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(tmp), str(src), "-libverbs",
                                   "-lpthread"], capture_output=True, text=True, check=False)
            if done.returncode:
                raise RuntimeError(f"building the RoCE proxy failed:\n{done.stderr}")
            tmp.replace(out)                             # atomic: the other process may build the same file
    lib = ctypes.CDLL(str(out))
    u64, vp, i = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    for name, res, args in (("tf_rdma_abi", i, []), ("tf_rdma_layout", None, [u64, ctypes.POINTER(u64)]),
                            ("tf_rdma_create", vp, [ctypes.c_char_p, i, vp, u64, u64, ctypes.c_char_p, u64]),
                            ("tf_rdma_local", None, [vp, ctypes.POINTER(u64)]),
                            ("tf_rdma_connect", i, [vp, ctypes.POINTER(u64)]), ("tf_rdma_start", i, [vp]),
                            ("tf_rdma_failed", i, [vp]), ("tf_rdma_error", ctypes.c_char_p, [vp]),
                            ("tf_rdma_counter", u64, [vp, i]), ("tf_rdma_destroy", None, [vp])):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    if lib.tf_rdma_abi() != 1:
        raise RuntimeError("unexpected RoCE proxy ABI")
    return lib


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    return load("tensorfold_rdma_gather_v1", [str(HERE / "gather.cpp"), str(HERE / "gather.cu")],
                extra_cuda_cflags=["-O3"])


def _device() -> str:
    """This rank's RoCE device: TF_RDMA_HCA, else the first of NCCL_IB_HCA (``=dev[:port],...``)."""

    raw = os.environ.get("TF_RDMA_HCA") or os.environ.get("NCCL_IB_HCA") or ""
    names = [x.strip().lstrip("=^").split(":")[0] for x in raw.split(",") if x.strip()]
    if not names:
        raise RuntimeError("the RoCE all-gather needs TF_RDMA_HCA or NCCL_IB_HCA to name the RoCE device")
    return names[0]


class RdmaGather:
    """One rank's pinned region, queue pair and proxy thread; ``all_gather`` launches the gather kernel."""

    def __init__(self, nccl, rank: int, max_bytes: int) -> None:
        if nccl.world != 2:
            raise ValueError("the RoCE all-gather joins exactly two ranks")
        lib = proxy()
        self.lib, self.rank = lib, rank
        self.slot_bytes = -(-int(max_bytes) // SLOT_ALIGN) * SLOT_ALIGN
        lay = (ctypes.c_uint64 * 5)()
        lib.tf_rdma_layout(self.slot_bytes, lay)
        self.flag_off, self.send_off, self.recv_off, total = int(lay[1]), int(lay[2]), int(lay[3]), int(lay[4])
        self.region = torch.zeros(total, dtype=torch.uint8).pin_memory()      # ctrl and flags start at 0
        self.ctrl = self.region[:64].view(torch.int32).numpy()
        self.state = torch.zeros(4, dtype=torch.int32, device="cuda")          # epoch, arrivals, departures, stopped
        self.device = _device()
        gid = int(os.environ.get("TF_RDMA_GID_INDEX") or os.environ.get("NCCL_IB_GID_INDEX") or 3)
        err = ctypes.create_string_buffer(256)
        self.ctx = lib.tf_rdma_create(self.device.encode(), gid, ctypes.c_void_p(self.region.data_ptr()), total,
                                      self.slot_bytes, err, len(err))
        mine = torch.zeros(INFO_WORDS + 2, dtype=torch.int64)
        if self.ctx:
            info = (ctypes.c_uint64 * INFO_WORDS)()
            lib.tf_rdma_local(self.ctx, info)
            mine[:INFO_WORDS] = torch.tensor([v - (1 << 64) if v >= 1 << 63 else v for v in map(int, info)],
                                             dtype=torch.int64)       # uint64 words as int64 for the exchange
        mine[INFO_WORDS] = 0 if self.ctx else 1
        mine[INFO_WORDS + 1] = self.slot_bytes
        both = torch.empty(2 * mine.numel(), dtype=torch.int64, device="cuda")
        nccl.all_gather(mine.cuda(), both)
        rows = both.view(2, -1).cpu()
        if int(rows[:, INFO_WORDS].sum()) or int(rows[0, INFO_WORDS + 1]) != int(rows[1, INFO_WORDS + 1]):
            why = err.value.decode(errors="replace") if not self.ctx else "the ranks' slot sizes differ"
            self.close()
            raise RuntimeError(f"RoCE all-gather setup failed on a rank: {why}")
        _ext()                                           # build the kernel before the connection's verdict
        peer = rows[1 - rank, :INFO_WORDS].tolist()
        words = (ctypes.c_uint64 * INFO_WORDS)(*[v & 0xFFFFFFFFFFFFFFFF for v in peer])
        ok = lib.tf_rdma_connect(self.ctx, words) == 0 and lib.tf_rdma_start(self.ctx) == 0
        verdict = torch.tensor([0 if ok else 1], dtype=torch.int64, device="cuda")
        votes = torch.empty(2, dtype=torch.int64, device="cuda")
        nccl.all_gather(verdict, votes)
        if int(votes.sum()):
            why = lib.tf_rdma_error(self.ctx).decode(errors="replace") if self.ctx else ""
            self.close()
            raise RuntimeError(f"RoCE all-gather could not connect: {why}")

    def fits(self, send: torch.Tensor, recv: torch.Tensor) -> bool:
        n = send.numel() * send.element_size()
        return (0 < n <= self.slot_bytes and n % 16 == 0 and send.is_contiguous() and recv.is_contiguous()
                and send.data_ptr() % 16 == 0 and recv.data_ptr() % 16 == 0
                and recv.numel() * recv.element_size() == 2 * n)

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        _ext().gather(send, recv, self.region.data_ptr(), self.flag_off, self.send_off, self.recv_off, self.slot_bytes,
                      self.state, SPIN, self.rank)
        if not torch.cuda.is_current_stream_capturing():
            self.check()

    def check(self) -> None:
        """Raise if a GPU wait gave up (ctrl[4]) or the proxy thread failed (ctrl[3])."""

        if int(self.ctrl[4]) or int(self.ctrl[3]) or (self.ctx and self.lib.tf_rdma_failed(self.ctx)):
            why = self.lib.tf_rdma_error(self.ctx).decode(errors="replace") if self.ctx else ""
            raise RuntimeError(f"RoCE all-gather failed (wait gave up at seq {int(self.ctrl[4])}, rung "
                               f"{int(self.ctrl[0])}, posted {self.lib.tf_rdma_counter(self.ctx, 0)}, completed "
                               f"{self.lib.tf_rdma_counter(self.ctx, 1)}){': ' + why if why else ''}")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.tf_rdma_destroy(self.ctx)
            self.ctx = None


class RdmaComm:
    """An NCCL communicator whose small all-gathers go over RoCE: the same calls, output layout and bits."""

    def __init__(self, nccl, rank: int, max_bytes: int | None = None) -> None:
        self.nccl, self.rank, self.world = nccl, rank, nccl.world
        kb = int(os.environ.get("TF_RDMA_MAX_KB") or 704) if max_bytes is None else max_bytes >> 10
        self.rdma = RdmaGather(nccl, rank, kb << 10)
        self.settled = False
        self.small = self.large = 0

    def settle(self) -> None:
        """Startup is over (graphs captured): eager gathers no longer meet the peer in an NCCL barrier first. Until
        then the ranks may drift apart by whole kernel builds, longer than a RoCE wait allows."""

        self.settled = True

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if self.rdma.fits(send, recv):
            self.small += 1
            if not self.settled and not torch.cuda.is_current_stream_capturing():
                self.nccl.barrier()
            self.rdma.all_gather(send, recv)
        else:
            self.large += 1
            self.nccl.all_gather(send, recv)

    def barrier(self) -> None:
        self.nccl.barrier()

    def check(self) -> None:
        self.rdma.check()

    def __getattr__(self, name):
        return getattr(self.nccl, name)
