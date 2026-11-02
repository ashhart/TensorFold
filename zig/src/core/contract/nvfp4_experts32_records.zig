//! tf_nvfp4_decode32_* and tf_nvfp4_prompt32_* (nvfp4_experts32.cu): GLM's routed experts at block 32 with csf0 scales.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_decode32_swiglu", .symbol = "tf_nvfp4_decode32_swiglu", .image = "nvfp4_experts32", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_decode32_down_f32", .symbol = "tf_nvfp4_decode32_down_f32", .image = "nvfp4_experts32", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_decode32_down_bf16", .symbol = "tf_nvfp4_decode32_down_bf16", .image = "nvfp4_experts32", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_prompt32_swiglu", .symbol = "tf_nvfp4_prompt32_swiglu", .image = "nvfp4_experts32", .threads = 128, .dynamic = 33280 }, .usage = registry.none },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_prompt32_down", .symbol = "tf_nvfp4_prompt32_down", .image = "nvfp4_experts32", .threads = 256, .dynamic = 33280 }, .usage = registry.none },
};

/// What nvcc 13.3 compiles differently (records/20261102-block32-csf0).
pub const cuda_133 = [_]registry.Override{};
