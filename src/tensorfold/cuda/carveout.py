"""GB10 display scanout memory as a CUDA buffer: room that MemAvailable never counts, for caches read sparsely.

A DGX Spark's DRM driver keeps a scanout region apart from the host's RAM. A dumb framebuffer allocated from it,
mapped into the process and registered with CUDA, is a device-visible buffer that costs no host memory. Its
bandwidth is about half of ordinary memory, so it suits cache pools that decode reads sparsely, not weights or
per-step scratch. Opt in with ``TF_CARVEOUT=1``; nothing here runs otherwise.

Env: ``TF_CARVEOUT`` (1 = on), ``TF_CARVEOUT_BYTES`` (default 1792 MiB), ``TF_DRM_CARD`` (default /dev/dri/card0).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import struct
import sys
from pathlib import Path

MIB = 1 << 20
DEFAULT_BYTES = 1792 * MIB
WIDTH, BPP = 4096, 32            # a 4096-wide 32-bit framebuffer: 16 KiB a row, the unit the size rounds to
ROW = WIDTH * BPP // 8


def _iowr(nr: int, size: int) -> int:
    """Linux _IOWR('d', nr, size): DRM's ioctl numbers."""

    return (3 << 30) | (size << 16) | (ord("d") << 8) | nr


# struct drm_mode_create_dumb {u32 height, width, bpp, flags; u32 handle, pitch; u64 size}
CREATE_DUMB, CREATE_FMT = _iowr(0xB2, 32), "=IIIIIIQ"
# struct drm_mode_map_dumb {u32 handle, pad; u64 offset}
MAP_DUMB, MAP_FMT = _iowr(0xB3, 16), "=IIQ"
# struct drm_mode_destroy_dumb {u32 handle}
DESTROY_DUMB, DESTROY_FMT = _iowr(0xB4, 4), "=I"

CU_MEMHOSTREGISTER_DEVICEMAP, CU_MEMHOSTREGISTER_IOMEMORY = 0x02, 0x04


def enabled() -> bool:
    return os.environ.get("TF_CARVEOUT", "0") == "1"


def requested_bytes() -> int:
    """The carveout size asked for, whole framebuffer rows; 0 when the carveout is off."""

    if not enabled():
        return 0
    size = int(os.environ.get("TF_CARVEOUT_BYTES") or DEFAULT_BYTES)
    return size // ROW * ROW


def _meminfo(key: str) -> int:
    for row in Path("/proc/meminfo").read_text().splitlines():
        name, value, *_ = row.split()
        if name.rstrip(":") == key:
            return int(value) * 1024
    raise KeyError(key)


def _ioctl(fd: int, request: int, fmt: str, *values: int) -> tuple:
    import fcntl

    buf = bytearray(struct.pack(fmt, *values))
    fcntl.ioctl(fd, request, buf, True)
    return struct.unpack(fmt, buf)


class _Framebuffer:
    """One dumb framebuffer of the DRM card, mapped read-write into this process."""

    def __init__(self, card: str, size: int) -> None:
        self.fd = os.open(card, os.O_RDWR | os.O_CLOEXEC)
        self.handle = 0
        self.map = None
        try:
            *_, self.handle, pitch, got = _ioctl(self.fd, CREATE_DUMB, CREATE_FMT, size // ROW, WIDTH, BPP, 0, 0, 0, 0)
            if pitch != ROW or got < size:
                raise OSError(f"{card}: framebuffer of {got} bytes, pitch {pitch} (asked {size}, pitch {ROW})")
            *_, offset = _ioctl(self.fd, MAP_DUMB, MAP_FMT, self.handle, 0, 0)
            self.map = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
        except BaseException:
            self.close()
            raise
        self.size = size
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.map))   # pins the map for our lifetime

    def close(self) -> None:
        if self.handle:
            try:
                _ioctl(self.fd, DESTROY_DUMB, DESTROY_FMT, self.handle)
            except OSError:
                pass
            self.handle = 0
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _libcuda() -> ctypes.CDLL:
    lib = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
    lib.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
    lib.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    return lib


def _cuda_error(lib: ctypes.CDLL, rc: int) -> str:
    name = ctypes.c_char_p()
    lib.cuGetErrorName(rc, ctypes.byref(name))
    return f"{name.value.decode() if name.value else 'CUDA error'} ({rc})"


class Carveout:
    """The registered scanout buffer; kept for the life of the process because tensors view it."""

    def __init__(self, size: int, card: str | None = None) -> None:
        import torch

        if size <= 0 or size % ROW:
            raise ValueError(f"carveout size {size} is not a positive multiple of {ROW}")
        torch.cuda.init()
        torch.zeros(1, device="cuda")          # the primary context is current before the driver calls
        self.card = card or os.environ.get("TF_DRM_CARD", "/dev/dri/card0")
        before = _meminfo("MemAvailable")
        self.fb = _Framebuffer(self.card, size)
        self.lib = _libcuda()
        rc = self.lib.cuMemHostRegister_v2(self.fb.address, size, CU_MEMHOSTREGISTER_DEVICEMAP)
        if rc:                                 # a driver that sees the scanout as I/O memory wants it said
            rc = self.lib.cuMemHostRegister_v2(self.fb.address, size,
                                               CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY)
        if rc:
            self.fb.close()
            raise OSError(f"cuMemHostRegister of the {self.card} framebuffer: {_cuda_error(self.lib, rc)}")
        pointer = ctypes.c_uint64()
        rc = self.lib.cuMemHostGetDevicePointer_v2(ctypes.byref(pointer), self.fb.address, 0)
        if rc:
            raise OSError(f"cuMemHostGetDevicePointer: {_cuda_error(self.lib, rc)}")
        self.size, self.pointer = size, pointer.value
        self.host_cost = max(0, before - _meminfo("MemAvailable"))
        self.used = 0
        self.__cuda_array_interface__ = {"shape": (size,), "typestr": "|u1", "data": (self.pointer, False),
                                         "version": 3, "strides": None}

    def bytes(self):
        """The whole buffer as one uint8 CUDA tensor over the registered pages, never a copy."""

        import torch

        whole = torch.as_tensor(self, device="cuda")
        if whole.data_ptr() != self.pointer or whole.numel() != self.size:
            raise RuntimeError("carveout: torch copied the external buffer instead of viewing it")
        return whole

    def take(self, shape: tuple[int, ...], dtype, *, align: int = 256):
        """A zeroed tensor carved from the unused tail; ValueError when it does not fit (the caller falls back)."""

        import torch

        itemsize = torch.empty((), dtype=dtype).element_size()
        n = itemsize
        for d in shape:
            n *= int(d)
        start = -(-self.used // align) * align
        if start + n > self.size:
            raise ValueError(f"carveout: {n} bytes do not fit in the {self.size - start} left")
        self.used = start + n
        view = self.bytes()[start:start + n].view(dtype).view(shape)
        view.zero_()
        return view

    @property
    def free(self) -> int:
        return self.size - self.used


_OWNER: Carveout | None = None


def get() -> Carveout | None:
    """The process's carveout, created on first use when ``TF_CARVEOUT=1``; None when off."""

    global _OWNER
    if _OWNER is None and enabled():
        _OWNER = Carveout(requested_bytes())
        print(f"[tensorfold] display carveout {_OWNER.size / MIB:.0f} MiB from {_OWNER.card}, "
              f"host RAM spent {_OWNER.host_cost / MIB:.0f} MiB", flush=True)
    return _OWNER


def take_or_zeros(shape: tuple[int, ...], dtype, device="cuda"):
    """A cache tensor in the carveout when it is on and has room, else ordinary ``torch.zeros``."""

    import torch

    owner = get()
    if owner is not None:
        try:
            return owner.take(shape, dtype)
        except ValueError:
            pass
    return torch.zeros(shape, dtype=dtype, device=device)


def _probe(argv: list[str]) -> int:
    size = int(os.environ.get("TF_CARVEOUT_BYTES") or DEFAULT_BYTES) // ROW * ROW
    card = os.environ.get("TF_DRM_CARD", "/dev/dri/card0")
    if "--cuda" not in argv:                   # DRM only: no CUDA context, safe beside a running server
        before = _meminfo("MemAvailable")
        fb = _Framebuffer(card, size)
        fb.map[:ROW] = b"\xa5" * ROW
        ok = fb.map[:ROW] == b"\xa5" * ROW
        spent = before - _meminfo("MemAvailable")
        fb.close()                             # the pinned mapping goes with the process
        print(f"drm: {size / MIB:.0f} MiB framebuffer from {card}, write/read {'ok' if ok else 'FAILED'}, "
              f"MemAvailable moved {spent / MIB:.0f} MiB")
        return 0 if ok else 1

    import time

    import torch

    cv = Carveout(size, card)
    whole = cv.bytes()
    whole.fill_(0x5A)
    torch.cuda.synchronize()
    ok = bool((whole[:: 1 << 20] == 0x5A).all()) and cv.fb.map[:4] == b"\x5a" * 4
    src = torch.empty(size, dtype=torch.uint8, device="cuda")

    def rate(dst, src) -> float:
        dst.copy_(src)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            dst.copy_(src)
        torch.cuda.synchronize()
        return 5 * size / (time.perf_counter() - t) / 1e9

    to_cv, from_cv = rate(whole, src), rate(src, whole)
    ref = rate(torch.empty_like(src), src)
    print(f"cuda: {size / MIB:.0f} MiB at 0x{cv.pointer:x}, GPU round trip {'ok' if ok else 'FAILED'}, "
          f"host RAM spent {cv.host_cost / MIB:.0f} MiB; copy GB/s into {to_cv:.0f}, out of {from_cv:.0f}, "
          f"ordinary->ordinary {ref:.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "probe":
        sys.exit("usage: python -m tensorfold.cuda.carveout probe [--cuda]")
    sys.exit(_probe(sys.argv[2:]))
