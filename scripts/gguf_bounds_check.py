"""GPU-side bounds check for the GGUF fast prefill (``tf_gguf_prefill_linear``): no route touches memory past its buffers.

    python scripts/gguf_bounds_check.py                    # guard pages, every route, controls first
    python scripts/gguf_bounds_check.py --mode sentinel    # tail patterns: does anything past the buffer change y?

The CPU model (tests/test_gguf_q8_bounds.py) derives every access from the kernel source. This script tests the same
claim on the device, calling libtfgguf's C entry points through ctypes with buffers it places itself:

``guard`` (default): the Q8_1 activation buffer, the output ``y`` and the weight are each placed so their last byte is
the last mapped byte of a HIP virtual-memory mapping (hipMemAddressReserve + hipMemMap), with an unmapped reserved
range after it. Any read or write past the end is a GPU page fault, which kills the process, so each (type, shape)
runs in a child process that prints the case before each launch; the parent restarts it after a fault. The
activation buffer's slack is filled with 0xFF (NaN scales) and ``y`` must equal the binding's normal call bit for bit.
Controls run first and MUST fault: Q8_0, m=256, K=5120 at 96 rows with the old exact-size buffer (the Strix fault
f328ee8 fixed), and at 129 rows with 5 slack tiles (the model says 6 is the least that fits). If they do not fault,
the guard is not working and the run reports so and fails instead of passing.
A control counts only if the child printed CASE-START (flushed just before the guarded launch) and died with a GPU
memory-fault signature (FAULT_TEXT); any other death is INCONCLUSIVE. The 6-tile control must be clean. The run
prints the tree, torch/HIP, device/arch, library path, Gufo revision and kSlackTiles, logs every child's stdout,
stderr and return code to logs/bounds/guard-<time>.log (--log), and ends PASS (0), FAIL (1) or INCONCLUSIVE (2).

``sentinel``: no VMM; the activation buffer sits inside a larger torch tensor whose tail is filled with 0x00, 0xFF and
0x7F in turn; ``y`` must be identical bit for bit across the patterns and the tail must be unchanged after the call.

What it can prove: guard mode, every launched (type, rows, m, K) case touched no byte past the end of q8 or y
(weight: past the end rounded down to 256 bytes, kept for alignment) on this device and build. Sentinel mode only
that bytes past the end did not CHANGE the output; it cannot see over-reads whose values are discarded, which is
exactly the Q8_0 W8A8 over-read (those tokens are never stored), so on its own it would pass the unfixed buffer.
What neither can prove: rows, shapes or types not run (the CPU model covers rows 1..4096 symbolically); a different
build or Gufo revision; fault granularity (only bytes past the buffer end are unmapped, nothing before its start); or
that the wave64 kernels compiled for this architecture are the ones another architecture would select.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import subprocess
import sys
import time

import torch

from tensorfold.cuda import gguf

TYPES = {"Q8_0": gguf.Q8_0, "Q3_K": gguf.Q3_K, "Q4_K": gguf.Q4_K, "Q5_K": gguf.Q5_K, "Q6_K": gguf.Q6_K,
         "IQ4_NL": gguf.IQ4_NL, "IQ3_S": gguf.IQ3_S, "IQ4_XS": gguf.IQ4_XS}
# (block elements, block bytes, fp16 field offsets set to a small finite value so no output is NaN by construction)
BLOCKS = {"Q8_0": (32, 34, (0,)), "Q3_K": (256, 110, (108,)), "Q4_K": (256, 144, (0, 2)), "Q5_K": (256, 176, (0, 2)),
          "Q6_K": (256, 210, (208,)), "IQ4_NL": (32, 18, (0,)), "IQ3_S": (256, 110, (0,)), "IQ4_XS": (256, 136, (0,))}
# Route coverage (prefill_quant_gemm.hip:1501-1555, prefill_quant_wave64.hip:6-44): m=256 keeps Q8_0 on the unbounded
# W8A8 kernel at every row count (wave64 needs m >= 1024); m=17408/K=5120 has m >= 3K (BN=32 at <= 8 rows) and takes
# wave64 at >= 96; K=1056 (not a multiple of 256) declines wave64 at any m; m=6144 > K=5120 declines IQ4_XS's wave64.
SHAPES = {"Q8_0": [(256, 5120), (17408, 5120), (5120, 17408), (1024, 1056)],
          "IQ4_NL": [(256, 5120), (6144, 5120), (1024, 1056)],
          "IQ4_XS": [(256, 5120), (6144, 5120), (5120, 17408)]}
DEFAULT_SHAPES = [(256, 5120), (6144, 5120), (5120, 17408)]
ROWS = [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 80, 95, 96, 97, 112, 113, 127, 128, 129, 144, 145, 255, 256, 257, 1000,
        2048, 4096]
TILE_BYTES, SUM_TILE_BYTES = 576, 64


def quantized_bytes(rows: int, k: int) -> int:
    """Gufo's QuantizedActivationBytes: payload tiles plus the sum sidecar, no slack."""

    return -(-rows // 16) * (k // 32) * (TILE_BYTES + SUM_TILE_BYTES)


def q8_size(lib, rows: int, k: int, size: str) -> int:
    if size == "slack":
        return lib.tf_gguf_q8_1_bytes(rows, k)
    if size == "exact":
        return quantized_bytes(rows, k)
    return quantized_bytes(rows, k) + int(size.split(":")[1]) * (k // 32) * TILE_BYTES       # tiles:N


def weight(name: str, m: int, k: int, seed: int) -> torch.Tensor:
    qk, nbytes, halves = BLOCKS[name]
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (m, k // qk, nbytes), dtype=torch.uint8, generator=g)
    small = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8)
    for off in halves:
        w[..., off:off + 2] = small
    return w.reshape(-1).cuda()


def library():
    lib = ctypes.CDLL(str(gguf._library()))
    lib.tf_gguf_q8_1_bytes.restype = ctypes.c_size_t
    lib.tf_gguf_q8_1_bytes.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    lib.tf_gguf_prefill_linear.restype = ctypes.c_int
    lib.tf_gguf_prefill_linear.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * 3 + [ctypes.c_void_p]
    return lib


def run(lib, qtype: int, w: int, x: torch.Tensor, q8: int, y: int, rows: int, m: int, k: int) -> None:
    stream = torch.cuda.current_stream().cuda_stream
    err = lib.tf_gguf_prefill_linear(qtype, w, x.data_ptr(), 1, q8, y, rows, m, k, stream)
    torch.cuda.synchronize()                                   # a fault surfaces here, at the case that caused it
    assert err == 0, f"HIP error {err}"


# --- guard pages: HIP virtual memory (hip_runtime_api.h) -------------------------------------------------------------

class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int), ("location", _Location),
                ("win32HandleMetaData", ctypes.c_void_p), ("compressionType", ctypes.c_ubyte),
                ("gpuDirectRDMACapable", ctypes.c_ubyte), ("usage", ctypes.c_ushort), ("_reserved", ctypes.c_byte * 64)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


def _hip():
    for name in ["libamdhip64.so", *glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so*"))]:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit("libamdhip64.so not found")


class Guarded:
    """``size`` bytes ending exactly at the end of a mapping, followed by reserved but unmapped address space."""

    def __init__(self, hip, size: int, align: int = 16):
        self.hip = hip
        dev = torch.cuda.current_device()
        self.prop = _Prop(type=1, requestedHandleType=0, location=_Location(type=1, id=dev))   # pinned, device
        gran = ctypes.c_size_t()
        self._ok(hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(self.prop), 0))
        g = gran.value
        self.mapped = -(-size // g) * g
        self.reserved = self.mapped + 4 * g
        self.base = ctypes.c_void_p()
        self._ok(hip.hipMemAddressReserve(ctypes.byref(self.base), ctypes.c_size_t(self.reserved), ctypes.c_size_t(g),
                                          None, ctypes.c_ulonglong(0)))
        self.handle = ctypes.c_void_p()
        self._ok(hip.hipMemCreate(ctypes.byref(self.handle), ctypes.c_size_t(self.mapped), ctypes.byref(self.prop),
                                  ctypes.c_ulonglong(0)))
        self._ok(hip.hipMemMap(self.base, ctypes.c_size_t(self.mapped), ctypes.c_size_t(0), self.handle,
                               ctypes.c_ulonglong(0)))
        access = _Access(location=_Location(type=1, id=dev), flags=3)                          # read/write
        self._ok(hip.hipMemSetAccess(self.base, ctypes.c_size_t(self.mapped), ctypes.byref(access), ctypes.c_size_t(1)))
        start = (self.base.value + self.mapped - size) // align * align
        self.ptr, self.after = start, self.base.value + self.mapped - (start + size)            # unguarded bytes after

    @staticmethod
    def _ok(err):
        if err != 0:
            raise RuntimeError(f"HIP VMM call failed ({err}); this device or runtime may not support guard pages, "
                               "use --mode sentinel")

    def close(self):
        self.hip.hipMemUnmap(self.base, ctypes.c_size_t(self.mapped))
        self.hip.hipMemRelease(self.handle)
        self.hip.hipMemAddressFree(self.base, ctypes.c_size_t(self.reserved))


class _Device:
    """A raw device range as a torch tensor (``__cuda_array_interface__``), so every fill, copy and compare of guarded
    memory is a torch op on torch's own stream, followed by a sync. 39264ef filled and copied it with hipMemcpy/hipMemset
    outside torch's ordering (a D2D hipMemcpy may return before it completes) and got 335 MISMATCH lines, no faults."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 2}


def device_view(ptr: int, nbytes: int) -> torch.Tensor:
    t = torch.as_tensor(_Device(ptr, nbytes), device="cuda")
    assert t.data_ptr() == ptr and t.numel() == nbytes, "device view does not alias the guarded range"
    return t


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _diff(y: torch.Tensor, ref: torch.Tensor, unwritten: int) -> str:
    bad = (y.view(torch.int32) != ref.view(torch.int32))
    idx = bad.nonzero()
    tok, col = (int(idx[0, 0]), int(idx[0, 1])) if len(idx) else (-1, -1)
    rows_bad = bad.any(1).nonzero().flatten()
    finite = torch.isfinite(y) & torch.isfinite(ref)
    err = float((y - ref)[finite].abs().max()) if bool(finite.any()) else float("nan")
    return (f"{int(bad.sum())}/{bad.numel()} differ, first tok={tok} col={col}, tokens {int(rows_bad.min()) if len(rows_bad) else -1}"
            f"..{int(rows_bad.max()) if len(rows_bad) else -1}, {int((y.view(torch.int32) == unwritten).sum())} unwritten, "
            f"max |diff| {err:.3g}")


def child(args) -> int:
    """One (type, m, K) over ``args.rows`` from ``args.start``: prints ``CASE`` before each launch, ``OK``/``MISMATCH``.

    Per case: ``ref`` is the binding's call; ``CTYPES-MISMATCH`` means the same ctypes call on ordinary torch buffers
    already differs (a harness bug, not memory placement); on a guarded mismatch the case is rerun with one guarded
    buffer at a time (q8 / y / w) to name the one that changes the output."""

    lib, hip = library(), _hip()
    name, m, k = args.child[0], int(args.child[1]), int(args.child[2])
    qtype = TYPES[name]
    w = weight(name, m, k, seed=m * 31 + k)
    gw = Guarded(hip, w.numel(), align=256)
    wv = device_view(gw.ptr, w.numel())
    wv.copy_(w)
    torch.cuda.synchronize()
    if not torch.equal(wv, w):                                 # self-test: the guarded weight is the weight, byte for byte
        print(f"WEIGHT-COPY-BAD {name} m={m} k={k}: {int((wv != w).sum())} bytes differ", flush=True)
        return 3
    unwritten = 0x7FC0DEAD                                     # a NaN no kernel writes: marks outputs left unwritten
    for rows in args.rows[args.start:]:
        print(f"CASE {name} rows={rows} m={m} k={k} size={args.size}", flush=True)
        x = torch.randn(rows, k, generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16).cuda()
        ref = gguf.prefill_linear(x, w, qtype, m, pad=0)
        size = q8_size(lib, rows, k, args.size)
        assert size % 16 == 0 and lib.tf_gguf_q8_1_bytes(rows, k) == q8_size(lib, rows, k, "tiles:8"), "capi formula"
        tail = quantized_bytes(rows, k)

        def call(q8: torch.Tensor, y: torch.Tensor, wt: torch.Tensor, x=x, rows=rows, tail=tail) -> torch.Tensor:
            q8[tail:] = 0xFF                                   # NaN scales in the slack: a read that is used shows up
            y.view(torch.int32).fill_(unwritten)
            torch.cuda.synchronize()
            run(lib, qtype, wt.data_ptr(), x, q8.data_ptr(), y.data_ptr(), rows, m, k)
            return y.view(torch.float32).view(rows, m).clone()

        def plain_q8(size=size) -> torch.Tensor:
            return torch.empty(size, dtype=torch.uint8, device="cuda")

        def plain_y(n=rows * m * 4) -> torch.Tensor:
            return torch.empty(n, dtype=torch.uint8, device="cuda")

        ctl = call(plain_q8(), plain_y(), w)
        if not torch.equal(_bits(ctl), _bits(ref)):
            print(f"CTYPES-MISMATCH {name} rows={rows} m={m} k={k}: {_diff(ctl, ref, unwritten)}", flush=True)
        gq, gy = Guarded(hip, size), Guarded(hip, rows * m * 4, align=4)
        q8v, yv = device_view(gq.ptr, size), device_view(gy.ptr, rows * m * 4)
        print(f"CASE-START {name} m={m} k={k} rows={rows} size={args.size}", flush=True)   # a fault after this is the guard
        y = call(q8v, yv, wv)
        torch.cuda.synchronize()
        print(f"CASE-DONE {name} m={m} k={k} rows={rows}", flush=True)
        if torch.equal(_bits(y), _bits(ref)):
            print(f"OK {name} rows={rows} m={m} k={k}", flush=True)
        else:
            alone = {"q8": call(q8v, plain_y(), w), "y": call(plain_q8(), yv, w), "w": call(plain_q8(), plain_y(), wv)}
            culprits = [b for b, out in alone.items() if not torch.equal(_bits(out), _bits(ref))]
            weight_ok = torch.equal(wv, w)
            print(f"MISMATCH {name} rows={rows} m={m} k={k}: {_diff(y, ref, unwritten)}; guarded alone that differ: "
                  f"{culprits or 'none'}; weight copy {'intact' if weight_ok else 'CHANGED'}", flush=True)
        torch.cuda.synchronize()
        del q8v, yv
        gq.close(), gy.close()
    del wv
    torch.cuda.synchronize()
    gw.close()
    return 0


# What a guard-page hit looks like from the parent. ROCr prints "Memory access fault by GPU node-N ... Page not present"
# and aborts (SIGABRT: return code -6, or 134 through a shell); with the fault reported as an error instead, HIP's
# synchronize raises hipErrorIllegalAddress ("an illegal memory access was encountered") and Python exits 1.
FAULT_TEXT = ("Memory access fault by GPU", "hipErrorIllegalAddress", "illegal memory access")


def is_gpu_fault(returncode: int, stderr: str) -> bool:
    if "Memory access fault by GPU" in stderr:
        return returncode in (-6, 134)
    return returncode != 0 and any(t in stderr for t in FAULT_TEXT[1:])


class Log:
    """Every child's command, return code, stdout and stderr, in one file per run."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.file = path, open(path, "w")      # noqa: SIM115 - held for the run

    def write(self, text: str) -> None:
        self.file.write(text if text.endswith("\n") else text + "\n")
        self.file.flush()


def guarded_sweep(name: str, m: int, k: int, rows: list[int], size: str, log: Log) -> list[tuple[str, str]]:
    """Runs one (type, shape) in children, restarting after each fault. Returns (kind, line) for every case that is not
    clean: kind FAULT (reached CASE-START, died with a GPU memory-fault signature), MISMATCH, or INCONCLUSIVE (the
    child died any other way: build, VMM setup, an assert, a fault outside the guarded launch)."""

    bad, start = [], 0
    while start < len(rows):
        cmd = [sys.executable, __file__, "--child", name, str(m), str(k), "--size", size, "--start", str(start),
               "--rows", *map(str, rows)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)   # a fault is the signal
        log.write(f"=== {' '.join(cmd[1:])}\nreturn code {proc.returncode}\n--- stdout\n{proc.stdout}--- stderr\n"
                  f"{proc.stderr}")
        lines = proc.stdout.splitlines()
        cases = [ln for ln in lines if ln.startswith("CASE ")]
        bad += [("MISMATCH", ln) for ln in lines if ln.startswith(("MISMATCH", "CTYPES-MISMATCH", "WEIGHT-COPY-BAD"))]
        if proc.returncode == 0:
            break
        err = (proc.stderr.strip().splitlines() or ["?"])[-1][:160]
        started = [ln for ln in lines if ln.startswith(("CASE-START", "CASE-DONE"))]
        in_launch = bool(started) and started[-1].startswith("CASE-START")
        if in_launch and is_gpu_fault(proc.returncode, proc.stderr):
            bad.append(("FAULT", f"FAULT {started[-1][11:]} (exit {proc.returncode}: {err})"))
        else:
            where = started[-1] if in_launch else (cases[-1] if cases else "before the first case")
            died = f"CHILD-DIED {name} m={m} k={k} at {where!r}, not a guard fault (exit {proc.returncode}: {err})"
            bad.append(("INCONCLUSIVE", died))
            if not cases:                                      # setup failed: every later case would fail the same way
                break
        start += max(len(cases), 1)
    return bad


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def identity() -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*a: str) -> str:
        out = subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, check=False)
        return out.stdout.strip() or "?"

    with open(os.path.join(root, "src/tensorfold/cuda/gguf/capi.hip")) as f:
        capi = f.read()
    slack = re.search(r"kSlackTiles = (\d+)", capi)
    props = torch.cuda.get_device_properties(0)
    gufo = os.path.join(root, "src/tensorfold/cuda/gguf/vendor/GUFO_REVISION")
    return [f"tree {git('rev-parse', 'HEAD')}{' (dirty)' if git('status', '--porcelain', '--untracked-files=no') != '?' else ''}",
            f"torch {torch.__version__}, HIP {torch.version.hip}",
            f"device {props.name}, {getattr(props, 'gcnArchName', '?')}",
            f"libtfgguf {gguf._library()}",
            f"Gufo {_read(gufo).strip() if os.path.exists(gufo) else '?'}",
            f"capi.hip kSlackTiles = {slack.group(1) if slack else '?'}"]


def verdict(log: Log, fail: list[str], inconclusive: list[str]) -> int:
    for line in fail + inconclusive:
        print("  " + line)
    word, code = ("FAIL", 1) if fail else (("INCONCLUSIVE", 2) if inconclusive else ("PASS", 0))
    summary = f"guard: {word} ({len(fail)} fail, {len(inconclusive)} inconclusive); log {log.path}"
    print(summary)
    log.write(summary)
    return code


def sentinel(args) -> int:
    lib = library()
    failures = 0
    for name in args.types:
        qtype = TYPES[name]
        for m, k in SHAPES.get(name, DEFAULT_SHAPES):
            w = weight(name, m, k, seed=m * 31 + k)
            for rows in args.rows:
                x = torch.randn(rows, k, generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16).cuda()
                size = q8_size(lib, rows, k, args.size)
                buf = torch.empty(size + (1 << 20), dtype=torch.uint8, device="cuda")
                y = torch.empty(rows, m, dtype=torch.float32, device="cuda")
                outs, clobbered = [], False
                for pattern in (0x00, 0xFF, 0x7F):
                    buf[size:] = pattern
                    run(lib, qtype, w.data_ptr(), x, buf.data_ptr(), y.data_ptr(), rows, m, k)
                    clobbered |= not bool((buf[size:] == pattern).all())
                    outs.append(y.view(torch.int32).clone())
                same = all(torch.equal(outs[0], o) for o in outs[1:])
                if not same or clobbered:
                    failures += 1
                    print(f"FAIL {name} rows={rows} m={m} k={k}: "
                          f"{'output depends on the tail ' if not same else ''}{'tail written' if clobbered else ''}")
        print(f"{name}: done", flush=True)
    print("sentinel: PASS" if failures == 0 else f"sentinel: FAIL ({failures})")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["guard", "sentinel"], default="guard")
    ap.add_argument("--size", default="slack", help="slack (tf_gguf_q8_1_bytes), exact (no slack) or tiles:N")
    ap.add_argument("--types", nargs="+", default=list(TYPES))
    ap.add_argument("--rows", nargs="+", type=int, default=ROWS)
    ap.add_argument("--log", help="per-run log of every child's output (default logs/bounds/guard-<time>.log)")
    ap.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    ap.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args)
    if args.mode == "sentinel":
        return sentinel(args)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = Log(args.log or os.path.join("logs", "bounds", f"guard-{stamp}.log"))
    for line in identity():
        print(line)
        log.write(line)
    print(f"log {log.path}", flush=True)

    # Controls: the guard must catch the over-reads the CPU model predicts, or a clean sweep means nothing. A control
    # counts only if the child reached the guarded launch and died with a GPU memory-fault signature.
    fail, inconclusive = [], []
    for size, rows in [("exact", 96), ("tiles:5", 129)]:
        bad = guarded_sweep("Q8_0", 256, 5120, [rows], size, log)
        kinds = {kind for kind, _ in bad}
        state = "faulted (guard works)" if kinds == {"FAULT"} else ("INCONCLUSIVE" if "INCONCLUSIVE" in kinds
                                                                     else "NO FAULT (guard ineffective)")
        print(f"control Q8_0 m=256 k=5120 rows={rows} size={size}: {state}", flush=True)
        if kinds != {"FAULT"}:
            inconclusive.append(f"CONTROL-INCONCLUSIVE size={size} rows={rows}: {state}; "
                                f"{[line for _, line in bad] or 'child finished cleanly'}")
            return verdict(log, fail, inconclusive)          # the sweep below would prove nothing
    bad = guarded_sweep("Q8_0", 256, 5120, [129, 144], "tiles:6", log)   # the model's least sufficient slack
    print(f"control Q8_0 m=256 k=5120 rows=129,144 size=tiles:6: {'clean' if not bad else 'NOT CLEAN'}", flush=True)
    fail += [f"control tiles:6: {line}" for kind, line in bad if kind != "INCONCLUSIVE"]
    inconclusive += [f"control tiles:6: {line}" for kind, line in bad if kind == "INCONCLUSIVE"]

    for name in args.types:
        for m, k in SHAPES.get(name, DEFAULT_SHAPES):
            bad = guarded_sweep(name, m, k, args.rows, args.size, log)
            fail += [line for kind, line in bad if kind != "INCONCLUSIVE"]
            inconclusive += [line for kind, line in bad if kind == "INCONCLUSIVE"]
            print(f"{name} m={m} k={k}: {'clean' if not bad else f'{len(bad)} not clean'}", flush=True)
    return verdict(log, fail, inconclusive)


if __name__ == "__main__":
    sys.exit(main())
