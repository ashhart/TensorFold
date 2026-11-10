// TensorFold's independently derived 16-token delta-recurrence packets.
#pragma once
#include <stdint.h>

namespace tf_gdn_chunk {
constexpr int Width = 128;
constexpr int Chunk = 16;
constexpr int Values = 16;
struct alignas(16) Scores { float kk[Chunk * Chunk], qk[Chunk * Chunk]; };
struct alignas(16) Coefficients {
    float solve_beta[Chunk * Chunk], attention[Chunk * Chunk];
    float prefix[Chunk], suffix[Chunk];
};
struct alignas(16) ScanShared {
    float state[Values][Width + 1];
    float q[Width][Chunk + 1], k[Width][Chunk + 1];
    float delta[Values][Chunk + 1];
    Coefficients coefficients;
};
static_assert(sizeof(Scores) == 2048);
static_assert(sizeof(Coefficients) == 2176);
static_assert(sizeof(ScanShared) == 28928);
}
