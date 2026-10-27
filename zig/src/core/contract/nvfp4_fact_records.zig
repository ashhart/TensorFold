//! tf_nvfp4_fact_* (nvfp4_experts.cu) in registry.cuda, from nvcc 13.0.88 cubins (sm_80/86/89/120a/121a).
const registry = @import("registry.zig");
const CudaRecord = registry.CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact_swiglu", .symbol = "tf_nvfp4_fact_swiglu", .image = "nvfp4_experts", .threads = 128, .dynamic = 34816 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact_down_f32", .symbol = "tf_nvfp4_fact_down_f32", .image = "nvfp4_experts", .threads = 256, .dynamic = 34816 }, .usage = .{ .{ .regs = 203, .static = 0, .bound = 256 }, .{ .regs = 177, .static = 0, .bound = 256 }, .{ .regs = 180, .static = 0, .bound = 256 }, .{ .regs = 192, .static = 0, .bound = 256 }, .{ .regs = 192, .static = 0, .bound = 256 } } },
};

/// nvcc 13.2 compiles the fp32 down differently on sm_120 and sm_121.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "nvfp4_experts tf_nvfp4_fact_down_f32", .usage = .{ .{ .regs = 203, .static = 0, .bound = 256 }, .{ .regs = 177, .static = 0, .bound = 256 }, .{ .regs = 180, .static = 0, .bound = 256 }, .{ .regs = 179, .static = 0, .bound = 256 }, .{ .regs = 179, .static = 0, .bound = 256 } } },
};
