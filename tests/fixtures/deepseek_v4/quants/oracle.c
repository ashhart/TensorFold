/* Independent CPU fixture oracle; equations from ggml-quants.c,
 * storage layout checked against pinned ds4.c. Build outside the source tree.
 * Reads blocks from stdin: kind byte (1=q8_0 34B, 2=q2_K 84B) then block.
 * Prints decoded float32 bit patterns. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>

#define QK8_0 32
#define QK_K  256
#define GGML_RESTRICT
typedef uint16_t ggml_fp16_t;
typedef struct { ggml_fp16_t d; int8_t  qs[QK8_0]; } block_q8_0;
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; ggml_fp16_t d; ggml_fp16_t dmin; } block_q2_K;

_Static_assert(sizeof(block_q2_K) == 84, "Q2_K size");
_Static_assert(offsetof(block_q2_K, qs) == 16, "Q2_K payload offset");
_Static_assert(offsetof(block_q2_K, d) == 80, "Q2_K scale offset");

static inline uint32_t fp32_to_bits(float f) { union { float f; uint32_t u; } v; v.f = f; return v.u; }
static inline float fp32_from_bits(uint32_t w) { union { float f; uint32_t u; } v; v.u = w; return v.f; }
static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}
#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32((x))

void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        const uint8_t * q = x[i].qs;
        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

int main(void) {
    uint8_t buf[128];
    size_t n = fread(buf, 1, 2, stdin);
    if (n != 2) return 2;
    int kind = buf[0];
    size_t blk = kind == 1 ? 34 : 84;
    size_t got = fread(buf, 1, blk, stdin);
    if (got != blk) return 3;
    if (kind == 1) {
        block_q8_0 b; memcpy(&b, buf, 34);
        float y[QK8_0];
        dequantize_row_q8_0(&b, y, QK8_0);
        for (int i = 0; i < QK8_0; ++i) printf("%u\n", fp32_to_bits(y[i]));
    } else if (kind == 2) {
        block_q2_K b; memcpy(&b, buf, 84);
        float y[QK_K];
        dequantize_row_q2_K(&b, y, QK_K);
        for (int i = 0; i < QK_K; ++i) printf("%u\n", fp32_to_bits(y[i]));
    } else return 4;
    return 0;
}
