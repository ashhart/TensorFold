// Copy of src/tensorfold/families/deepseek_v41/cuda/pfdense_args.h (blob f4e1d1bed4e5): sass_check.sh



// pfdense (TF_DSV41_PF_DENSE=fused) launch arguments and config list, shared by kernels and host bindings
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace dsv41_pfd {

enum OutType : int { OUT_F16 = 0, OUT_BF16 = 1, OUT_F32 = 2 };

struct Args {
    const void* xh;            // [M, K] fp16, the rotated input (upstream rot_in), contiguous
    const uint32_t* T;         // trellis words: tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb + (nt % 8) * 4 K2
    long long stride_k, stride_nb;
    const void* svh;           // fp16 [N]
    const void* bias;          // fp16 [N] or null
    void* out;                 // [M, N] view of OutType, unit column stride, row stride o_stride (elements)
    long long o_stride;
    int M, K, N, group, out_type;
};

// X(id, BM, WM, WN, KS, NST); warps = WM * WN, warp tile (BM / WM) x (128 / WN); 6 = narrow layers' row tiles
#define DSV41_PFD_CFGS(X)  \
    X(0, 128, 2, 4, 2, 3)  \
    X(1, 128, 2, 4, 2, 4)  \
    X(2, 128, 4, 2, 2, 3)  \
    X(3, 64, 2, 2, 2, 4)   \
    X(4, 64, 1, 4, 2, 4)   \
    X(5, 128, 2, 4, 2, 2)  \
    X(6, 32, 1, 4, 2, 4)   \
    X(7, 256, 4, 2, 2, 3)
#define DSV41_PFD_NCFG 8

// (BM, WM, WN, KS, NST) of configuration id, or all zeros
inline void cfg_of(int id, int (&o)[5]) {
    for (auto& v : o) v = 0;
#define DSV41_PFD_CFG_OF(ID, BM, WM, WN, KS, NST) \
    if (id == ID) { o[0] = BM; o[1] = WM; o[2] = WN; o[3] = KS; o[4] = NST; }
    DSV41_PFD_CFGS(DSV41_PFD_CFG_OF)
#undef DSV41_PFD_CFG_OF
}

// the per-width entry points (pfdense_k<K2>.cu); lanes is built for K2 8, 10, 12
bool run_k8(const Args&, bool lanes, int cfg, cudaStream_t);
bool run_k10(const Args&, bool lanes, int cfg, cudaStream_t);
bool run_k12(const Args&, bool lanes, int cfg, cudaStream_t);
bool run_k16(const Args&, bool lanes, int cfg, cudaStream_t);
void describe_k8(bool lanes, int cfg, long long (&o)[8]);
void describe_k10(bool lanes, int cfg, long long (&o)[8]);
void describe_k12(bool lanes, int cfg, long long (&o)[8]);
void describe_k16(bool lanes, int cfg, long long (&o)[8]);
bool dequant_k8(bool lanes, const uint32_t*, long long, long long, void*, int, int, cudaStream_t);
bool dequant_k10(bool lanes, const uint32_t*, long long, long long, void*, int, int, cudaStream_t);
bool dequant_k12(bool lanes, const uint32_t*, long long, long long, void*, int, int, cudaStream_t);
bool dequant_k16(bool lanes, const uint32_t*, long long, long long, void*, int, int, cudaStream_t);

}  // namespace dsv41_pfd
