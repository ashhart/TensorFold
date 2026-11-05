//! tf_nvfp4_decode32_* and tf_nvfp4_prompt32_* (nvfp4_experts32.cu): block 32 csf0 experts, nvcc 13.0, five SMs.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_decode32_swiglu", .symbol = "tf_nvfp4_decode32_swiglu", .image = "nvfp4_experts32", .threads = 128 }, .usage = .{ .{ .regs = 127, .static = 0, .bound = 128 }, .{ .regs = 127, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_decode32_down_f32", .symbol = "tf_nvfp4_decode32_down_f32", .image = "nvfp4_experts32", .threads = 128 }, .usage = .{ .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_decode32_down_bf16", .symbol = "tf_nvfp4_decode32_down_bf16", .image = "nvfp4_experts32", .threads = 128 }, .usage = .{ .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 93, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_prompt32_swiglu", .symbol = "tf_nvfp4_prompt32_swiglu", .image = "nvfp4_experts32", .threads = 128, .dynamic = 33280 }, .usage = .{ .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 226, .static = 0, .bound = 128 }, .{ .regs = 226, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts32 tf_nvfp4_prompt32_down", .symbol = "tf_nvfp4_prompt32_down", .image = "nvfp4_experts32", .threads = 256, .dynamic = 33280 }, .usage = .{ .{ .regs = 154, .static = 0, .bound = 256 }, .{ .regs = 154, .static = 0, .bound = 256 }, .{ .regs = 154, .static = 0, .bound = 256 }, .{ .regs = 140, .static = 0, .bound = 256 }, .{ .regs = 140, .static = 0, .bound = 256 } } },
};

/// What nvcc 13.3 compiles differently: the prompt entries on sm_120 and sm_121.
pub const cuda_133 = [_]registry.Override{
    .{ .name = "nvfp4_experts32 tf_nvfp4_prompt32_swiglu", .usage = .{ .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 225, .static = 0, .bound = 128 }, .{ .regs = 225, .static = 0, .bound = 128 } } },
    .{ .name = "nvfp4_experts32 tf_nvfp4_prompt32_down", .usage = .{ .{ .regs = 154, .static = 0, .bound = 256 }, .{ .regs = 154, .static = 0, .bound = 256 }, .{ .regs = 154, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 } } },
};
