"""Two-rank NCCL all-gather on the current CUDA stream (ctypes), so collectives can sit inside CUDA graphs.

torch.distributed runs NCCL on its own streams with host-side bookkeeping; calling ``ncclAllGather`` on
the caller's stream directly (as vLLM's pynccl does) keeps a tensor-parallel step capturable. The unique
id travels through a ``TCPStore`` on rank 0's address. Every reduction in the engine is an all-gather of
fp32 partials followed by a fixed rank-order sum, so both ranks hold identical bits.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
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
    def __init__(self, rank: int, world: int, master: str, port: int) -> None:
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
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
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

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()
