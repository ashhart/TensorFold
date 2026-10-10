#pragma once

// The tile constants shared by the dot2 files.

#include <hip/hip_fp16.h>

#include "common/arch.hpp"
#include "common/dot2.hpp"
#include "quant/mlx.hpp"
#include "quant/mlx_pieces.hpp"

namespace tf {
namespace rocm {

constexpr int kLaneCols = 32;
constexpr int kLaneWaves = 8;
constexpr int kLaneRows = 8;
constexpr int kLaneGroupMax = 128;
constexpr int kBlockRows = 64;  // from here the GEMM tile beats the 128-row column tile

}  // namespace rocm
}  // namespace tf
