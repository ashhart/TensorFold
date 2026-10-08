"""Host-staged all-gather for TP ranks without peer access (sm_80 CMP 170HX).

TensorFold's all-gather (cuda/comm.py) re-expressed over one POSIX shm segment
mmap'd by every rank: each GPU holds a device pointer to the same physical pages
(cudaHostRegister Mapped|Portable), so a kernel can stage its slice of the
payload into its own slot, announce an arrival word, and read the peers' slots
straight out of host memory. Concept credit: Morrowmake's host-shm all-reduce
(vLLM fork). This implementation was written fresh for TensorFold's all-gather
semantics -- the arrival protocol, the generation handling and every kernel in
this file are its own expression, and it invites the same source-overlap check
that the earlier ported draft failed.

Messages above ``FALLBACK_BYTES`` (large prefill payloads) stay on NCCL; only
the one-shot path lives here. (A rank-order sum fold and a hyper-connection
stream-mix fold of this protocol exist in our four-rank proof-of-concept tree;
they ride the same arrival protocol and are not part of this module.)

Capturable: the generation counter lives in a device tensor the kernel itself
advances, never in an argument, so a captured graph replays correctly. The base
device address is fixed at registration time, so passing it as a kernel argument
is graph-safe too. Every wait is bounded by clock64(); on timeout the rank marks
a 64-bit failure word and leaves its output untouched so validation shouts.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import time

import torch

MAX_RANKS = 4
MAX_BLOCKS = 4
HEADER_BYTES = 256 * 1024
SLOT_BYTES = 1 << 20            # per rank per parity; covers R<=64 decode windows
FALLBACK_BYTES = SLOT_BYTES     # above this, the caller keeps NCCL
_SHM_BYTES = HEADER_BYTES + 2 * MAX_RANKS * SLOT_BYTES

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cstdint>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e));   \
  } while (0)

namespace tfshm {

constexpr int WORLD = 4;                   // every kernel here assumes a TP4 group
constexpr int STAGES = 2;                  // parity staging areas: one call overlaps the previous
constexpr uint32_t GAVE_UP = 0x80000000u;  // a timed-out rank tags its arrival word with this

using SysWord32 = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
using SysWord64 = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;

// Shared-page layout: a small control region (per-(stage, block, rank) arrival
// words plus one sticky 64-bit failure word per rank), then STAGES rank-ordered
// staging areas of SLOT_BYTES each. Fixed at registration; kernels derive every
// offset from `base`.
struct alignas(256) Layout {
  uint32_t arrival[STAGES][4][WORLD];      // [stage][block][rank] = generation tag
  unsigned long long failure[WORLD];       // [rank] = (gen << 1) | 1, written once
};

template <typename T, int N>
struct __align__(sizeof(T) * N) Run { T d[N]; };

__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint8_t* stage_area(uint8_t* base, int stage, int64_t slot_bytes) {
  return base + sizeof(Layout) + static_cast<int64_t>(stage) * WORLD * slot_bytes;
}

// The staging writes must be visible before the tag lands: every thread drains its
// own writes with a system fence, then thread 0 alone publishes the generation with
// one release store (one store per block, ordered after the whole CTA's staging).
__device__ __forceinline__ void announce(uint32_t* word, uint32_t gen) {
  __threadfence_system();
  if (threadIdx.x == 0)
    SysWord32(*word).store(gen, cuda::memory_order_release);
}

// Parallel wait: one thread per peer spins on that peer's arrival word (parallel host
// reads keep detection at one PCIe round), the rest of the block catches up at the
// barrier. A rank that blows its budget tags its own word with GAVE_UP, records the
// sticky failure word and gives up -- its output is left untouched so validation
// shouts, and its generation still advances so the next call starts clean.
__device__ __forceinline__ bool peers_ready(Layout* lay, int stage, int block, int rank,
                                            uint32_t gen, unsigned long long budget,
                                            int* gave_up) {
  const int tid = threadIdx.x;
  if (tid == 0) *gave_up = 0;
  __syncthreads();
  if (tid < WORLD && tid != rank) {
    uint32_t* word = &lay->arrival[stage][block][tid];
    const long long t0 = clock64();
    while (SysWord32(*word).load(cuda::memory_order_acquire) != gen) {
      if ((unsigned long long)(clock64() - t0) > budget) {
        SysWord32(lay->arrival[stage][block][rank]).store(gen | GAVE_UP, cuda::memory_order_release);
        SysWord64(lay->failure[rank]).store(((unsigned long long)gen << 1) | 1ULL,
                                            cuda::memory_order_relaxed);
        *gave_up = 1;
        break;
      }
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
  }
  __syncthreads();
  return *gave_up != 0;
}

template <typename T, int N>
__device__ __forceinline__ const Run<T, N>* staged(uint8_t* area, int rank, int64_t slot_bytes) {
  return reinterpret_cast<const Run<T, N>*>(area + static_cast<int64_t>(rank) * slot_bytes);
}

// ---- one-shot all-gather: rank r's slice lands at [r * nunits]; pure copies ----
template <typename T, int N, int NB>
__global__ void __launch_bounds__(1024, 1)
gather_copies(const Run<T, N>* __restrict__ slice,
              Run<T, N>* __restrict__ out,
              uint8_t* __restrict__ base,
              uint32_t* __restrict__ seq,
              const int rank,
              const int64_t nunits,
              const int64_t slot_bytes,
              const unsigned long long budget) {
  Layout* lay = reinterpret_cast<Layout*>(base);
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int stage = static_cast<int>(gen & 1u);
  uint8_t* const area = stage_area(base, stage, slot_bytes);
  Run<T, N>* const mine = reinterpret_cast<Run<T, N>*>(area + static_cast<int64_t>(rank) * slot_bytes);

  const int64_t span = (nunits + NB - 1) / NB;
  const int64_t first = static_cast<int64_t>(b) * span;
  const int64_t last = (first + span) < nunits ? (first + span) : nunits;

  __shared__ int gave_up;

  for (int64_t i = first + tid; i < last; i += nt) mine[i] = slice[i];
  announce(&lay->arrival[stage][b][rank], gen);

  if (peers_ready(lay, stage, b, rank, gen, budget, &gave_up)) {
    if (tid == 0) seq[b] = gen;
    return;
  }

  for (int64_t i = first + tid; i < last; i += nt) {
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      const Run<T, N>* from = (r == rank) ? slice : staged<T, N>(area, r, slot_bytes);
      out[static_cast<int64_t>(r) * nunits + i] = from[i];
    }
  }

  if (tid == 0) seq[b] = gen;
}

int64_t failure_word(int64_t base_ptr) {
  const Layout* lay = reinterpret_cast<const Layout*>(base_ptr);
  unsigned long long worst = 0;
  for (int r = 0; r < WORLD; ++r) worst |= lay->failure[r];
  return static_cast<int64_t>(worst);
}

std::tuple<int64_t, int64_t> host_register(int64_t ptr, int64_t nbytes) {
  cudaError_t e = cudaHostRegister(reinterpret_cast<void*>(ptr),
                                   static_cast<size_t>(nbytes),
                                   cudaHostRegisterMapped | cudaHostRegisterPortable);
  TORCH_CHECK(e == cudaSuccess, "cudaHostRegister: ", cudaGetErrorString(e),
              " (if a previous run left pages registered, rm the shm file and retry)");
  void* dev = nullptr;
  e = cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(ptr), 0);
  TORCH_CHECK(e == cudaSuccess, "cudaHostGetDevicePointer: ", cudaGetErrorString(e));
  return {reinterpret_cast<int64_t>(dev), ptr};
}

void host_unregister(int64_t ptr) { cudaHostUnregister(reinterpret_cast<void*>(ptr)); }

}  // namespace tfshm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("host_register", &tfshm::host_register);
  m.def("host_unregister", &tfshm::host_unregister);
  m.def("failure_word", &tfshm::failure_word);
  m.def("run_f32", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                      int64_t rank, int64_t nunits, int64_t slot_bytes,
                      int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<float, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<float, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_copies<float, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_copies<float, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_copies<float, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
  m.def("run_bf16", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                       int64_t rank, int64_t nunits, int64_t slot_bytes,
                       int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<__nv_bfloat16, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<__nv_bfloat16, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_copies<__nv_bfloat16, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_copies<__nv_bfloat16, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_copies<__nv_bfloat16, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
  m.def("run_i32", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                      int64_t rank, int64_t nunits, int64_t slot_bytes,
                      int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<int, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<int, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_copies<int, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_copies<int, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_copies<int, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
}
"""


def _build_module():
    from pathlib import Path

    from tensorfold.cuda.build import load

    src = Path(__file__).parent / "_shm_allgather.cu"
    if not src.exists() or src.stat().st_size != len(CUDA_SRC):
        src.write_text(CUDA_SRC)
    return load(name="tensorfold_shm_allgather_v1", sources=[str(src)],
                extra_cuda_cflags=["-O3", "-std=c++17"], verbose=False)


class ShmAllGather:
    """One shm segment per TP group; the same all_gather(send, recv) contract as NCCL.all_gather."""

    def __init__(self, rank: int, world: int, store, *, blocks: int = MAX_BLOCKS) -> None:
        if world != 4:
            raise ValueError(f"shm all-gather port targets world=4, got {world}")
        self.rank, self.world, self.store = rank, world, store
        self.blocks = blocks
        self.name = f"/dev/shm/tf_ag_{store.port}"
        if rank == 0:
            fd = os.open(self.name, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, _SHM_BYTES)
            os.close(fd)
            store.set("tf_ag_created", "1")
        else:
            for _ in range(600):
                try:
                    if store.get("tf_ag_created") == "1":
                        break
                except Exception:
                    time.sleep(0.5)
        fd = os.open(self.name, os.O_RDWR)
        self._map = mmap.mmap(fd, _SHM_BYTES)
        os.close(fd)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        torch.cuda.init()
        self.ext = _build_module()
        self.dev_base, self.host_addr = self.ext.host_register(addr, _SHM_BYTES)
        self.seq = torch.zeros(blocks, dtype=torch.int32, device="cuda")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order. Fallback is the caller's job."""
        nbytes = send.numel() * send.element_size()
        if send.dtype not in (torch.float32, torch.bfloat16, torch.int32):
            raise ValueError(f"shm all-gather dtype {send.dtype}")
        vec_bytes = 8 if send.dtype is torch.bfloat16 else 16   # Run<T, 4> width
        if nbytes > FALLBACK_BYTES or nbytes % vec_bytes or nbytes == 0:
            raise ValueError(f"shm all-gather size {nbytes}")
        if not send.is_contiguous() or not recv.is_contiguous():
            raise ValueError("shm all-gather needs contiguous tensors")
        fn = {torch.float32: self.ext.run_f32,
              torch.bfloat16: self.ext.run_bf16,
              torch.int32: self.ext.run_i32}[send.dtype]
        fn(send.data_ptr(), recv.data_ptr(), self.dev_base, self.seq.data_ptr(),
           self.rank, nbytes // vec_bytes, SLOT_BYTES, 1 << 30, self.blocks)

    def ok(self) -> bool:
        """True unless some rank recorded a wait timeout (checked outside graphs)."""
        return int(self.ext.failure_word(self.dev_base)) == 0

    def close(self) -> None:
        try:
            self.ext.host_unregister(self.host_addr)
        except Exception:
            pass
        try:
            self._map.close()
            if self.rank == 0:
                os.unlink(self.name)
        except Exception:
            pass
