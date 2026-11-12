// Standalone device contracts; compiled with the same flags as sample.cu.
#include "sample.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "repetition_hf_fixture.h"

#define CUDA_OK(x) do { const cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(2); } } while (0)
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

__global__ void read_processed(const __nv_bfloat16* logits, unsigned n, const unsigned* seen, float factor, float* out) {
    const tf_sample::Row r{logits, n, 1.0f, threadIdx.x, threadIdx.x & 31u, threadIdx.x >> 5, seen, factor};
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) out[i] = r.v(i);
}

int main() {
    constexpr unsigned V = 8;
    const float raw[V] = {4, -4, 0, -0.0f, INFINITY, -INFINITY, 3, -3};
    __nv_bfloat16 host[V];
    for (unsigned i = 0; i < V; ++i) host[i] = __float2bfloat16(raw[i]);
    __nv_bfloat16* logits;
    unsigned *seen, *tokens;
    float *factors, *values, *prob;
    int* meta;
    tf_sample::Rule* rules;
    CUDA_OK(cudaMalloc(&logits, 2 * V * sizeof(*logits)));
    CUDA_OK(cudaMalloc(&seen, 2 * sizeof(*seen)));
    CUDA_OK(cudaMalloc(&tokens, 2 * sizeof(*tokens)));
    CUDA_OK(cudaMalloc(&factors, 2 * sizeof(*factors)));
    CUDA_OK(cudaMalloc(&values, V * sizeof(*values)));
    CUDA_OK(cudaMalloc(&prob, 2 * sizeof(*prob)));
    CUDA_OK(cudaMalloc(&meta, 4 * sizeof(*meta)));
    CUDA_OK(cudaMalloc(&rules, sizeof(*rules)));
    // GLM's original strict-greater scan falls back to token 0 when every value is NaN or negative infinity.
    const float degenerate[6] = {NAN, -INFINITY, -INFINITY, NAN, NAN, NAN};
    __nv_bfloat16 degenerate_bf16[6];
    for (unsigned i = 0; i < 6; ++i) degenerate_bf16[i] = __float2bfloat16(degenerate[i]);
    const unsigned degenerate_masks[2] = {7, 7};
    const float degenerate_factors[2] = {1.0f, 1.1f};
    CUDA_OK(cudaMemcpy(logits, degenerate_bf16, sizeof(degenerate_bf16), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(seen, degenerate_masks, sizeof(degenerate_masks), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(factors, degenerate_factors, sizeof(degenerate_factors), cudaMemcpyHostToDevice));
    tf_greedy_repetition<<<2, 1024>>>(logits, 3, seen, 1, factors, tokens);
    unsigned degenerate_picks[2];
    CUDA_OK(cudaMemcpy(degenerate_picks, tokens, sizeof(degenerate_picks), cudaMemcpyDeviceToHost));
    std::printf("degenerate greedy: expected [0,0], actual [%u,%u]\n", degenerate_picks[0], degenerate_picks[1]);
    REQUIRE(degenerate_picks[0] == 0 && degenerate_picks[1] == 0);
    // Golden values come from Transformers 5.20.1's actual processor, encoded as IEEE F32 bits.
    for (unsigned c = 0; c < hf_repetition_case_count; ++c) {
        const HfRepetitionCase& fixture = hf_repetition_cases[c];
        REQUIRE(fixture.n <= V);
        unsigned membership = 0;
        for (unsigned i = 0; i < fixture.history_n; ++i) membership |= 1u << fixture.history[i];
        for (unsigned i = 0; i < fixture.n; ++i) {
            float raw_value;
            std::memcpy(&raw_value, &fixture.logits_bits[i], 4);
            host[i] = __float2bfloat16(raw_value);
        }
        float factor;
        std::memcpy(&factor, &fixture.penalty_bits, 4);
        CUDA_OK(cudaMemcpy(logits, host, fixture.n * sizeof(*host), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(seen, &membership, 4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(factors, &factor, 4, cudaMemcpyHostToDevice));
        read_processed<<<1, 32>>>(logits, fixture.n, seen, factor, values);
        unsigned got[V];
        CUDA_OK(cudaMemcpy(got, values, fixture.n * 4, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < fixture.n; ++i) {
            float actual_value, expected_value;
            std::memcpy(&actual_value, &got[i], 4);
            std::memcpy(&expected_value, &fixture.expected_bits[i], 4);
            if (!(isnan(actual_value) && isnan(expected_value)) && got[i] != fixture.expected_bits[i]) {
                std::fprintf(stderr, "HF fixture mismatch: %s, token %u\n", fixture.name, i);
                return 1;
            }
        }
        if (fixture.expected_greedy == 0xffffffffu) continue;
        tf_greedy_repetition<<<1, 1024>>>(logits, fixture.n, seen, 1, factors, tokens);
        unsigned greedy;
        CUDA_OK(cudaMemcpy(&greedy, tokens, 4, cudaMemcpyDeviceToHost));
        REQUIRE(greedy == fixture.expected_greedy);
    }
    for (unsigned i = 0; i < V; ++i) host[i] = __float2bfloat16(raw[i]);
    CUDA_OK(cudaMemcpy(logits, host, sizeof(host), cudaMemcpyHostToDevice));
    const unsigned mask[2] = {0x3f, 0};
    CUDA_OK(cudaMemcpy(seen, mask, sizeof(mask), cudaMemcpyHostToDevice));
    for (const float factor : {2.0f, 0.5f, 1.0f, 1.1f}) {
        read_processed<<<1, 32>>>(logits, V, seen, factor, values);
        float got[V];
        CUDA_OK(cudaMemcpy(got, values, sizeof(got), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < V; ++i) {
            const float expected = (mask[0] & (1u << i)) ? (raw[i] < 0 ? raw[i] * factor : raw[i] / factor) : raw[i];
            REQUIRE(std::memcmp(&got[i], &expected, sizeof(float)) == 0);
        }
    }

    // Extra BF16 rounding would change this winner to the lower-ID tie token 0.
    const float near_raw[V] = {1, 1.1015625f, -2, -3, -4, -5, -6, -7};
    for (unsigned i = 0; i < V; ++i) host[i] = __float2bfloat16(near_raw[i]);
    CUDA_OK(cudaMemcpy(logits, host, sizeof(host), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(logits + V, host, sizeof(host), cudaMemcpyHostToDevice));
    const unsigned near_mask[2] = {2, 2};
    const float mixed_factors[2] = {1.1f, 2.0f};
    CUDA_OK(cudaMemcpy(seen, near_mask, sizeof(near_mask), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(factors, mixed_factors, sizeof(mixed_factors), cudaMemcpyHostToDevice));
    for (unsigned run = 0; run < 5; ++run) {
        tf_greedy_repetition<<<2, 1024>>>(logits, V, seen, 1, factors, tokens);
        unsigned got[2];
        CUDA_OK(cudaMemcpy(got, tokens, sizeof(got), cudaMemcpyDeviceToHost));
        REQUIRE(got[0] == 1 && got[1] == 0);
    }

    const int positions[4] = {42, 0, 0, 0};
    CUDA_OK(cudaMemcpy(meta, positions, sizeof(positions), cudaMemcpyHostToDevice));
    for (const unsigned top_k : {0u, 1u, 3u}) {
        const tf_sample::Rule rule{1234, 1.0f, 0.95f, 20.0f, -INFINITY, top_k, 0};
        CUDA_OK(cudaMemcpy(rules, &rule, sizeof(rule), cudaMemcpyHostToDevice));
        const float identity[2] = {1, 1};
        CUDA_OK(cudaMemcpy(factors, identity, sizeof(identity), cudaMemcpyHostToDevice));
        tf_draw<<<2, 1024>>>(logits, V, rules, meta, 0, tokens, prob);
        unsigned original[2];
        float original_prob[2];
        CUDA_OK(cudaMemcpy(original, tokens, sizeof(original), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(original_prob, prob, sizeof(original_prob), cudaMemcpyDeviceToHost));
        tf_draw_repetition<<<2, 1024>>>(logits, V, rules, meta, 0, tokens, prob, seen, 1, factors);
        unsigned actual[2];
        float actual_prob[2];
        CUDA_OK(cudaMemcpy(actual, tokens, sizeof(actual), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(actual_prob, prob, sizeof(actual_prob), cudaMemcpyDeviceToHost));
        REQUIRE(std::memcmp(original, actual, sizeof(actual)) == 0);
        REQUIRE(std::memcmp(original_prob, actual_prob, sizeof(actual_prob)) == 0);
    }
    CUDA_OK(cudaFree(logits)); CUDA_OK(cudaFree(seen)); CUDA_OK(cudaFree(tokens));
    CUDA_OK(cudaFree(factors)); CUDA_OK(cudaFree(values)); CUDA_OK(cudaFree(prob));
    CUDA_OK(cudaFree(meta)); CUDA_OK(cudaFree(rules));
    std::puts("SUMMARY repetition: PASS HF10/NaN-class/degenerate-greedy/sign/factor/F32-rounding/mixed-greedy/5-repeat/keyed-identity");
}
