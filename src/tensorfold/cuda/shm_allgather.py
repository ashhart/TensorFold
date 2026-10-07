"""Host-staged all-gather for TP ranks without peer access (sm_80 CMP 170HX).

Port of vLLM/Morrowmake's host_shm_all_reduce idea to TensorFold's all-gather
semantics (cuda/comm.py): one POSIX shm segment mmap'd by every rank,
cudaHostRegister(Mapped|Portable) so each GPU holds a device pointer to the
same physical pages, flags in that same host memory. The kernel publishes the
rank's send buffer into its slot, waits on the peers' flags, and copies every
rank's stripe into ``recv`` in rank order -- a pure copy, so the all-gather
stays bit-exact by construction (TensorFold's rank-order sum downstream is
untouched).

Messages above ``FALLBACK_BYTES`` (large prefill payloads) stay on NCCL; only
the one-shot path is ported.

Capturable: the generation counter lives in a device tensor the kernel itself
increments, never in an argument, so a captured graph replays correctly. The
base device address is fixed at registration time, so passing it as a kernel
argument is graph-safe too. Every spin is bounded by clock64(); on timeout the
flag word records the abort and ``recv`` is left untouched so validation
shouts.
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
#include <cstdint>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e));   \
  } while (0)

namespace tfshm {

constexpr int NR = 4;
constexpr int HEADER_BYTES = 256 * 1024;

struct alignas(128) Flag {
  uint32_t v;
  uint32_t pad[31];
};

struct Header {
  Flag arrive[2][4][NR];   // [parity][block][rank]
  Flag err[NR];
};

template <typename T, int N>
struct __align__(sizeof(T) * N) Vec { T d[N]; };

__device__ __forceinline__ void st_flag_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(p) : "memory");
}
__device__ __forceinline__ uint32_t ld_flag_acquire(uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <int NB>
__device__ __forceinline__ bool block_barrier(Flag* row, uint32_t gen, int rank,
                                              unsigned long long spin_cycles,
                                              int* aborted) {
  const int tid = threadIdx.x;
  if (tid == 0) *aborted = 0;
  __syncthreads();
  __threadfence_system();
  if (tid == 0) st_flag_release(&row[rank].v, gen);
  if (tid < NR && tid != rank) {
    uint32_t* f = &row[tid].v;
    const long long t0 = clock64();
    while (ld_flag_acquire(f) != gen) {
      if ((unsigned long long)(clock64() - t0) > spin_cycles) { *aborted = 1; break; }
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
  }
  __syncthreads();
  return *aborted != 0;
}

// One-shot all-gather, NB blocks x 1024 threads. out holds NR * nunits; rank
// r's stripe lands at [r * nunits]. Own contribution is read from `inp`
// (device), peers' from the host slots: pure copies, bit-exact by construction.
template <typename T, int N, int NB>
__global__ void __launch_bounds__(1024, 1)
allgather(const Vec<T, N>* __restrict__ inp,
          Vec<T, N>* __restrict__ out,
          uint8_t* __restrict__ base,
          uint32_t* __restrict__ seq,
          const int rank,
          const int64_t nunits,
          const int64_t slot_bytes,
          const unsigned long long spin_cycles) {
  Header* h = reinterpret_cast<Header*>(base);
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int p = static_cast<int>(gen & 1u);

  uint8_t* const slots = base + HEADER_BYTES + static_cast<int64_t>(p) * NR * slot_bytes;
  Vec<T, N>* const mine =
      reinterpret_cast<Vec<T, N>*>(slots + static_cast<int64_t>(rank) * slot_bytes);
  const Vec<T, N>* src[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    src[r] = (r == rank) ? inp
                         : reinterpret_cast<const Vec<T, N>*>(slots + static_cast<int64_t>(r) * slot_bytes);
  }

  const int64_t per = (nunits + NB - 1) / NB;
  const int64_t lo = static_cast<int64_t>(b) * per;
  const int64_t hi = (lo + per) < nunits ? (lo + per) : nunits;

  __shared__ int aborted;

  for (int64_t i = lo + tid; i < hi; i += nt) mine[i] = inp[i];

  if (block_barrier<NB>(h->arrive[p][b], gen, rank, spin_cycles, &aborted)) {
    if (tid == 0) { h->err[rank].v = gen | 0x80000000u; seq[b] = gen; }
    return;
  }

  for (int64_t i = lo + tid; i < hi; i += nt) {
#pragma unroll
    for (int r = 0; r < NR; ++r) out[static_cast<int64_t>(r) * nunits + i] = src[r][i];
  }

  if (tid == 0) seq[b] = gen;
}

template <typename T>
void run(int64_t inp_ptr, int64_t out_ptr, int64_t base_ptr, int64_t seq_ptr,
         int64_t rank, int64_t nunits, int64_t slot_bytes, int64_t spin_cycles,
         int64_t blocks) {
  const int nb = static_cast<int>(blocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (nb == 1) {
    allgather<T, 4, 1><<<1, 1024, 0, stream>>>(
        reinterpret_cast<const Vec<T, 4>*>(inp_ptr), reinterpret_cast<Vec<T, 4>*>(out_ptr),
        reinterpret_cast<uint8_t*>(base_ptr), reinterpret_cast<uint32_t*>(seq_ptr),
        static_cast<int>(rank), nunits, slot_bytes, static_cast<unsigned long long>(spin_cycles));
  } else if (nb == 2) {
    allgather<T, 4, 2><<<2, 1024, 0, stream>>>(
        reinterpret_cast<const Vec<T, 4>*>(inp_ptr), reinterpret_cast<Vec<T, 4>*>(out_ptr),
        reinterpret_cast<uint8_t*>(base_ptr), reinterpret_cast<uint32_t*>(seq_ptr),
        static_cast<int>(rank), nunits, slot_bytes, static_cast<unsigned long long>(spin_cycles));
  } else {
    allgather<T, 4, 4><<<4, 1024, 0, stream>>>(
        reinterpret_cast<const Vec<T, 4>*>(inp_ptr), reinterpret_cast<Vec<T, 4>*>(out_ptr),
        reinterpret_cast<uint8_t*>(base_ptr), reinterpret_cast<uint32_t*>(seq_ptr),
        static_cast<int>(rank), nunits, slot_bytes, static_cast<unsigned long long>(spin_cycles));
  }
  CUDA_CHECK(cudaGetLastError());
}

uint32_t last_error(int64_t base_ptr) {
  Header* h = reinterpret_cast<Header*>(base_ptr);
  uint32_t worst = 0;
  for (int r = 0; r < NR; ++r) worst |= h->err[r].v;
  return worst;
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
  m.def("last_error", &tfshm::last_error);
  m.def("run_f32", &tfshm::run<float>);
  m.def("run_bf16", &tfshm::run<__nv_bfloat16>);
  m.def("run_i32", &tfshm::run<int>);
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
        """recv [world * n] <- every rank's send [n], in rank order. Falls back is the caller's job."""
        nbytes = send.numel() * send.element_size()
        if send.dtype not in (torch.float32, torch.bfloat16, torch.int32):
            raise ValueError(f"shm all-gather dtype {send.dtype}")
        vec_bytes = 8 if send.dtype is torch.bfloat16 else 16   # Vec<T,4> width
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
        """True unless some rank recorded a spin timeout (checked outside graphs)."""
        return (int(self.ext.last_error(self.dev_base)) & 0x80000000) == 0

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
