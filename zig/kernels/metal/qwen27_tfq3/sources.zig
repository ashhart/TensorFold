//! Embedded Metal sources for Qwen27 TFQ3 (3/4/5-bit 2D-AffineCoset/half4) and BF16 linear kernels.
pub const header =
    \\#include <metal_stdlib>
    \\#include "../nax.h"
    \\using namespace metal;
    \\
;
pub const common = @embedFile("common.metal");
pub const dense_decode = @embedFile("dense_decode.metal");
pub const dense_prefill = @embedFile("dense_prefill.metal");
