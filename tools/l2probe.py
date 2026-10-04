"""Does cp.async.bulk.prefetch.L2 warm GB10's L2? Times a read of a 12 MB buffer cold (L2 flushed by a 256 MB write)
and after a bulk prefetch of it from a small kernel, and the same with prefetch.global.L2 lines (the go / no-go
gate before any L2 prefetch in the decode step). The bulk / lines prefetch kernels adapt l2pf.cu's range<Bulk> loop
(glm5_next/spark/l2pf.cu in deepseek-v41-tensorfold-spark patches/0001, from glm53-tensorfold-spark patch 0460), MIT
License, Copyright (c) 2026 TensorFold contributors and Jay Leaton; see THIRD_PARTY_NOTICES.md."""

import argparse

import torch
from torch.utils.cpp_extension import load_inline

CUDA = r"""
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void bulk(const uint8_t* p, uint64_t bytes, uint32_t chunk) {
    uint64_t pieces = (bytes + chunk - 1) / chunk;
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < pieces; i += gridDim.x * blockDim.x) {
        uint64_t off = i * chunk, left = bytes - off;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + off), "r"((uint32_t)(left < chunk ? left : chunk)) : "memory");
    }
}
__global__ void lines(const uint8_t* p, uint64_t bytes) {
    for (uint64_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 128ull; i < bytes; i += gridDim.x * blockDim.x * 128ull)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i));
}
__global__ void readsum(const uint4* p, uint64_t n, unsigned* out) {
    unsigned s = 0;
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint4 v = __ldcg(p + i); s ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (s == 0x12345678u) out[0] = s;
}
void prefetch(torch::Tensor t, int64_t mode, int64_t grid) {
    auto s = at::cuda::getCurrentCUDAStream();
    if (mode == 0) bulk<<<grid, 128, 0, s>>>((const uint8_t*)t.data_ptr(), t.numel(), 32768);
    else lines<<<grid, 256, 0, s>>>((const uint8_t*)t.data_ptr(), t.numel());
}
void readall(torch::Tensor t, torch::Tensor out) {
    auto s = at::cuda::getCurrentCUDAStream();
    readsum<<<48 * 4, 256, 0, s>>>((const uint4*)t.data_ptr(), t.numel() / 16, (unsigned*)out.data_ptr());
}
"""
CPP = "void prefetch(torch::Tensor t, int64_t mode, int64_t grid); void readall(torch::Tensor t, torch::Tensor out);"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=12)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    ext = load_inline("tf_l2probe", CPP, cuda_sources="#include <ATen/cuda/CUDAContext.h>\n#include <torch/extension.h>\n"
                      + CUDA, functions=["prefetch", "readall"], extra_cuda_cflags=["-O3"], verbose=False)
    buf = torch.randint(0, 255, (int(a.mb * 2**20),), dtype=torch.uint8, device="cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(prep) -> float:
        best = []
        for _ in range(a.reps):
            flush.fill_(1)
            prep()
            torch.cuda.synchronize()
            s.record()
            ext.readall(buf, out)
            e.record()
            torch.cuda.synchronize()
            best.append(s.elapsed_time(e) * 1e3)
        best.sort()
        return best[len(best) // 2]

    cold = timed(lambda: None)
    for grid in (1, 2, 4):
        bulk = timed(lambda: ext.prefetch(buf, 0, grid))
        print(f"{a.mb:g} MB read: cold {cold:.0f} us, after bulk prefetch (grid {grid}) {bulk:.0f} us "
              f"({cold / bulk:.2f}x)")
    lines = timed(lambda: ext.prefetch(buf, 1, 48))
    warm = timed(lambda: ext.readall(buf, out))
    print(f"after line prefetch {lines:.0f} us ({cold / lines:.2f}x); already read once (L2-hot) {warm:.0f} us "
          f"({cold / warm:.2f}x)")


if __name__ == "__main__":
    main()
