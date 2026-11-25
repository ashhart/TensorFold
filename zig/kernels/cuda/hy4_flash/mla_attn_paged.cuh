// Absorbed latent MLA attention on tensor cores (`mma.m16n8k16` bf16 -> fp32): split+fold decode and 1-split prefill.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace hy4_mla {

constexpr int KW = 576;      // cache row: 512 latent + 64 roped key
constexpr int KL = 512;      // latent width (the output)
constexpr int LISTW = 512;   // group list width (indexed layers, p >= 2048)
constexpr int SWA_KEYS = 512;
constexpr int DSA_KEYS = 2048;
constexpr int NT = 256;      // threads a CTA (8 warps)

// Dynamic shared-memory bytes per CTA (`exact`: hi+lo bf16 planes for Q and P).
constexpr int smem_bytes(int mt, int kt, bool qreg, bool exact = false) {
    return (qreg ? 0 : mt * 16 * KW * 2) * (exact ? 2 : 1) + 2 * kt * KW * 2 + mt * 16 * kt * 2 * (exact ? 2 : 1) +
           (8 / (mt * kt / 8)) * mt * 16 * (kt + 8) * 4 + 2 * kt * 4 + 6 * mt * 16 * 4 + 128 + 16;
}
constexpr int SMEM_X16 = smem_bytes(1, 16, false, true);      // exact 16 heads, KT = 16
constexpr int SMEM_X16_QREG = ((smem_bytes(1, 16, true, true) + 127) & ~127) + 4 * 16 * 32 * 2;  // 48,896 B (2 CTAs/SM)
static_assert(SMEM_X16_QREG == 48896 && 2 * SMEM_X16_QREG <= 101376, "x16_qreg must fit 2 CTAs/SM");
constexpr int SMEM_DEC_KT32 = smem_bytes(1, 32, false);   // Arm A:  99,088 B, 1 CTA/SM
constexpr int SMEM_DEC_QREG_KT16 = smem_bytes(1, 16, true);   // Arm B:  44,176 B, 2 CTAs/SM
constexpr int SMEM_MT2_KT16 = smem_bytes(2, 16, false);   // 32 queries a KV tile: 81,936 B
constexpr int SMEM_QREG_KT32 = smem_bytes(1, 32, true);   // q in registers, KT = 32: 80,656 B
constexpr int SMEM_MT2_QREG_KT16 = smem_bytes(2, 16, true);   // q in registers, 32 queries, KT = 16: 45,072 B
static_assert(SMEM_DEC_KT32 <= 100352, "sm_121 contract limit");
static_assert(2 * (SMEM_DEC_QREG_KT16 + 1024) <= 102400, "Arm B must fit 2 CTAs/SM");
static_assert(SMEM_MT2_KT16 <= 100352, "sm_121 contract limit");

// Keys of the longest row of a layer kind (the split grid's x extent is ceil(max_keys / ch)).
inline int max_keys(int kind, int ntok) { return kind == 0 ? SWA_KEYS : (kind == 1 ? DSA_KEYS : ntok); }

// Fixed decode/verify chunk size per (kind, Hl) across R = 1..16 (guarantees drafted == serial 0-bit parity).
inline int decode_ch(int kind, int Hl) { (void)Hl; return kind == 0 ? 64 : 128; }

// Required float count for `PM`/`PL` scratch (`PO = 512 * scratch_floats`); 0 when `R > 16`.
inline size_t scratch_floats(int kind, int Hl, int R, int ntok) {
    if (R > 16) return 0;
    const int ch = decode_ch(kind, Hl);
    const int ns = (max_keys(kind, ntok) + ch - 1) / ch;
    const int hb = (Hl + 15) / 16;
    return ns > 1 ? static_cast<size_t>(R) * hb * 16 * ns : 0;
}

}  // namespace hy4_mla

// Split-key decode/verify and 1-split heads-in-M prefill (`grid (nsplit, ceil(Hl / (16*MT)), R)`, block `256`).
#define HY4_MLA_SPLIT_ARGS                                                                                               \
    const __nv_bfloat16 *__restrict__ QA, int Hl, const __nv_bfloat16 *__restrict__ CACHE, int kind, int ring, int ctx, \
        const int *__restrict__ ROW_STREAM, const int *__restrict__ ROW_POS, const int *__restrict__ LIST,               \
        const int *__restrict__ TOK, int ldtok, int ntok, const float *__restrict__ SINKS, int ch,                       \
        float *__restrict__ PM, float *__restrict__ PL, float *__restrict__ PO, int *__restrict__ TICKET,                \
        void *__restrict__ OUT, int out_f32, int R

// Exact: 16 heads a CTA, KT = 16, q hi/lo planes (the lo plane QLO elements after QA) and P hi/lo planes, SMEM_X16.
extern "C" __global__ void hy4_mla_dec_x16(HY4_MLA_SPLIT_ARGS, long long QLO);
extern "C" __global__ void hy4_mla_dec_x16_rows(HY4_MLA_SPLIT_ARGS, long long QLO);
extern "C" __global__ void hy4_mla_dec_x16_qreg(HY4_MLA_SPLIT_ARGS, long long QLO);
extern "C" __global__ void hy4_mla_dec_x16_qreg_rows(HY4_MLA_SPLIT_ARGS, long long QLO);
// Arm A: 16 heads a CTA, Q in shared memory, KT = 32 double-buffered, 1 CTA/SM.
extern "C" __global__ void hy4_mla_dec_kt32(HY4_MLA_SPLIT_ARGS);
extern "C" __global__ void hy4_mla_dec_kt32_rows(HY4_MLA_SPLIT_ARGS);
// Arm B: 16 heads a CTA, Q fragments in registers, KT = 16 double-buffered, 2 CTAs/SM.
extern "C" __global__ void hy4_mla_dec_qreg_kt16(HY4_MLA_SPLIT_ARGS);
// 32 heads a CTA (sliding layers' Hl = 32 in one KV stream), Q in shared memory, KT = 16 double-buffered.
extern "C" __global__ void hy4_mla_dec_h32_kt16(HY4_MLA_SPLIT_ARGS);
// Q fragments in registers (each warp holds its own m tile's k slice: 72 registers), no Q shared-memory traffic:
extern "C" __global__ void hy4_mla_dec_qreg_kt32(HY4_MLA_SPLIT_ARGS);       // 16 heads, KT = 32, SMEM_QREG_KT32
extern "C" __global__ void hy4_mla_dec_h32_qreg_kt16(HY4_MLA_SPLIT_ARGS);   // 32 heads, KT = 16, SMEM_MT2_QREG_KT16

// Second pass of 2-launch split+fold (`TICKET = nullptr`): folds rows with `nspl > 1` in ascending split order.
extern "C" __global__ void hy4_mla_fold(int Hl, int nq, int kind, const int *__restrict__ ROW_POS, int ntok, int ch, int NS,
                                        const float *__restrict__ PM, const float *__restrict__ PL,
                                        const float *__restrict__ PO, const float *__restrict__ SINKS,
                                        void *__restrict__ OUT, int out_f32, int R);

// Rows-in-M prefill (`kind 0`, and `kind 1` at `p < 2048`): 16 rows x `MT` heads per CTA.
#define HY4_MLA_ROWS_ARGS                                                                                                \
    const __nv_bfloat16 *__restrict__ QA, int Hl, const __nv_bfloat16 *__restrict__ CACHE, int kind, int ring, int ctx, \
        const int *__restrict__ ROW_STREAM, const int *__restrict__ ROW_POS, const float *__restrict__ SINKS,            \
        void *__restrict__ OUT, int out_f32, int R
extern "C" __global__ void hy4_mla_prefill_rows_h1(HY4_MLA_ROWS_ARGS);   // 16 rows x 1 head, KT = 32
extern "C" __global__ void hy4_mla_prefill_rows_h2(HY4_MLA_ROWS_ARGS);   // 16 rows x 2 heads, KT = 16
extern "C" __global__ void hy4_mla_prefill_rows_h1_qreg(HY4_MLA_ROWS_ARGS);   // as _h1, q in registers (SMEM_QREG_KT32)
extern "C" __global__ void hy4_mla_prefill_rows_h2_qreg(HY4_MLA_ROWS_ARGS);   // as _h2, q in registers (SMEM_MT2_QREG_KT16)
