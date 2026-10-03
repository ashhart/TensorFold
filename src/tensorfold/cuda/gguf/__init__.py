"""GGUF K-quant and IQ projections on Gufo's exact HIP GEMM (``vendor/``, MIT): a row's bits never depend on the row count.

``gemv`` and ``prefill_linear`` are Gufo's own decode GEMV and WMMA prompt GEMM. They are off by default (their bits
differ from the exact kernel's); TENSORFOLD_GGUF_FAST=1 turns them on.

The vendored kernels build with hipcc into ``libtfgguf.so``; a small torch binding calls it on the current stream.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).parent
VENDOR = HERE / "vendor"
KERNEL_DIR = VENDOR / "src/models/qwen/hip/kernels"
WAVE64 = ["-mwavefrontsize64"]
# Gufo's per-file options (its src/models/qwen/CMakeLists.txt): the qualified schedules were measured under these.
SOURCES = [(HERE / "capi.hip", []), (KERNEL_DIR / "prefill_quant_gemm.hip", []), (KERNEL_DIR / "prefill_gemm.hip", []),
           (KERNEL_DIR / "gemv_quant.hip", []),
           (KERNEL_DIR / "prefill_quant_wave64.hip", WAVE64), (KERNEL_DIR / "small_batch_wave64.hip", WAVE64),
           (KERNEL_DIR / "small_batch_quant16_wave64.hip",
            WAVE64 + ["-Xarch_device", "-mllvm=-amdgpu-sched-strategy=iterative-ilp"]),
           (VENDOR / "src/core/quant/ggml_dequant.cpp", [])]

# ggml type ids, as GGUF stores them
F32, F16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS, BF16 = 0, 1, 8, 11, 12, 13, 14, 20, 21, 23, 30
QUANT = {Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS}
PREFILL = {Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS}   # Gufo's WMMA prompt GEMM (IsNativeWmmaQuant + Q8_0)


def _library() -> Path:
    """Compile the vendored kernels once per source hash (Gufo's RelWithDebInfo -O2 and per-file flags), then link."""

    from tensorfold.cuda.rocm import offload_arch, use_pip_sdk

    use_pip_sdk()                                              # torch's own ROCm SDK: one HIP runtime in the process
    hipcc = str(Path(os.environ["ROCM_PATH"]) / "bin" / "hipcc") if os.environ.get("ROCM_PATH") else "hipcc"
    digest = hashlib.sha256(hipcc.encode())
    for path in sorted(p for p in VENDOR.rglob("*") if p.is_file()) + [HERE / "capi.hip"]:
        digest.update(path.read_bytes())
    for _, flags in SOURCES:
        digest.update(" ".join(flags).encode())
    arch = offload_arch()
    out = Path(os.environ.get("TF_GGUF_BUILD", Path.home() / ".cache/tensorfold/gguf")) / f"{arch}-{digest.hexdigest()[:16]}"
    lib = out / "libtfgguf.so"
    if lib.exists():
        return lib
    out.mkdir(parents=True, exist_ok=True)
    print(f"building GGUF kernels ({arch}) into {out}", flush=True)
    base = [hipcc, "-O2", "-std=c++20", "-fPIC", "-DNDEBUG", f"-I{VENDOR}"]
    objects = []
    for src, flags in SOURCES:
        obj = out / (src.name + ".o")
        lang = ["-x", "hip", f"--offload-arch={arch}"] if src.suffix == ".hip" else ["-x", "c++"]
        subprocess.run([*base, *flags, "-DENGINE_ENABLE_HIP", "-c", *lang, str(src), "-o", str(obj)], check=True)
        objects.append(str(obj))
    tmp = out / "libtfgguf.so.tmp"
    subprocess.run([hipcc, "-shared", "-fPIC", f"--offload-arch={arch}", "-o", str(tmp), *objects], check=True)
    tmp.rename(lib)
    return lib


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    lib = _library()                                           # before torch's cpp_extension reads ROCM_HOME
    return load(name="tensorfold_gguf_v2", sources=[str(HERE / "binding.cpp")],
                extra_ldflags=[f"-L{lib.parent}", "-ltfgguf", f"-Wl,-rpath,{lib.parent}"], verbose=False)


def linear(x: torch.Tensor, w: torch.Tensor, qtype: int, n: int) -> torch.Tensor:
    """x (rows, K) times the GGUF-packed ``w`` (N rows of K/block blocks) transposed -> (rows, N) fp32, any row count."""

    return _ext().linear(x.float().contiguous(), w, qtype, n, torch.cuda.current_stream(w.device).cuda_stream)


def gemv(x: torch.Tensor, w: torch.Tensor, qtype: int, n: int) -> torch.Tensor:
    """One row: Gufo's decode GEMV (``LaunchQ8KBlockGEMV``, its kHipQuantDirect route) -> (1, N) fp32."""

    return _ext().gemv(x.float().contiguous(), w, qtype, n, torch.cuda.current_stream(w.device).cuda_stream)


def prefill_linear(x: torch.Tensor, w: torch.Tensor, qtype: int, n: int, pad: int = 96) -> torch.Tensor:
    """bf16/fp32 ``x`` quantized to Q8_1, Gufo's WMMA W8A8 GEMM -> (rows, N) fp32 (``PREFILL`` types).

    Not decode's bits. With ``pad=96`` calls under 96 rows are zero-padded to Gufo's >= 96-row tile, so one weight
    always runs one kernel and a row's bits never depend on the row count; ``pad=0`` takes Gufo's small-row tiles.
    """

    if x.dtype not in (torch.bfloat16, torch.float32):
        x = x.float()
    return _ext().prefill_linear(x.contiguous(), w, qtype, n, pad, torch.cuda.current_stream(w.device).cuda_stream)


def linear_bf16(x: torch.Tensor, w: torch.Tensor, qtype: int, n: int) -> torch.Tensor:
    """bf16 rows on Gufo's quant-direct GEMM (``LaunchBatchedQuantGEMMBf16``, no activation quantization) -> fp32."""

    return _ext().linear_bf16(x.to(torch.bfloat16).contiguous(), w, qtype, n,
                              torch.cuda.current_stream(w.device).cuda_stream)


def dequant(w: torch.Tensor, qtype: int, n: int) -> torch.Tensor:
    """The first ``n`` weights of packed ``w`` as bf16 (Gufo's kernel: Q8_0, Q5_K and Q6_K only)."""

    return _ext().dequant_bf16(w.contiguous(), qtype, n, torch.cuda.current_stream(w.device).cuda_stream)


def _q4_k_rows(rows: torch.Tensor, k: int) -> torch.Tensor:
    """Q4_K rows (R, row bytes) -> (R, k) fp32 in ggml's order: d*sc and dmin*m first, then d1*q - m1."""

    blocks = rows.reshape(-1, 144)
    d = blocks[:, 0:2].contiguous().view(torch.float16).float()             # (B, 1)
    dmin = blocks[:, 2:4].contiguous().view(torch.float16).float()
    sc, qs = blocks[:, 4:16].to(torch.int32), blocks[:, 16:144].to(torch.int32)
    lo, hi = sc[:, 0:4], sc[:, 4:8]
    scale = torch.cat([lo & 63, (sc[:, 8:12] & 0xF) | ((lo >> 6) << 4)], dim=1)       # j < 4, then j >= 4
    mins = torch.cat([hi & 63, (sc[:, 8:12] >> 4) | ((hi >> 6) << 4)], dim=1)
    d_all, m_all = d * scale.float(), dmin * mins.float()                  # (B, 8) sub-block scales and mins
    q = qs.view(-1, 4, 32)
    nib = torch.stack([q & 0xF, q >> 4], dim=2).reshape(-1, 8, 32).float()  # sub-block 2i: low nibbles, 2i+1: high
    out = d_all[:, :, None] * nib - m_all[:, :, None]
    return out.reshape(rows.shape[0], k)


def rows_bf16(rows: torch.Tensor, qtype: int, k: int) -> torch.Tensor:
    """Whole packed rows (R, row bytes) -> (R, k) bf16, for embedding lookups."""

    if qtype == Q4_K:
        return _q4_k_rows(rows, k).to(torch.bfloat16)
    if qtype in (Q8_0, Q5_K, Q6_K):
        return dequant(rows.contiguous(), qtype, rows.shape[0] * k).view(rows.shape[0], k)
    eye = torch.eye(k, device=rows.device, dtype=torch.float32)          # exact: each output is one weight times 1.0
    return linear(eye, rows.contiguous(), qtype, rows.shape[0]).T.contiguous().to(torch.bfloat16)


def reader(path):
    """gguf-py's reader: the installed ``gguf`` package, else llama.cpp's gguf-py directory named by TF_GGUF_PY."""

    try:
        from gguf import GGUFReader
    except ImportError:
        import sys

        extra = os.environ.get("TF_GGUF_PY")
        if not extra:
            raise ImportError("reading GGUF needs the gguf package (pip install gguf) or TF_GGUF_PY=<llama.cpp>/gguf-py")
        sys.path.insert(0, extra)
        from gguf import GGUFReader
    return GGUFReader(str(path))


def headers(path) -> dict:
    """Safetensors-style headers for the startup estimate: each tensor as its packed bytes, blocks named as layers."""

    out = {}
    for t in reader(path).tensors:
        name = t.name
        if name.startswith("blk."):
            _, index, rest = name.split(".", 2)
            name = f"model.layers.{index}.{rest}"
        size = int(t.n_bytes)
        out[name] = {"dtype": "U8", "shape": [size], "data_offsets": [0, size], "gguf_type": int(t.tensor_type)}
    return out
