//! tf_nvfp4_fact_* (nvfp4_experts.cu) in registry.cuda, from nvcc 13.0.88 cubins (sm_80/86/89/120a/121a).
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact_swiglu", .symbol = "tf_nvfp4_fact_swiglu", .image = "nvfp4_experts", .threads = 128, .dynamic = 34816 }, .usage = .{ .{ .regs = 249, .static = 0, .bound = 128 }, .{ .regs = 249, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact_down_f32", .symbol = "tf_nvfp4_fact_down_f32", .image = "nvfp4_experts", .threads = 256, .dynamic = 34816 }, .usage = .{ .{ .regs = 185, .static = 0, .bound = 256 }, .{ .regs = 186, .static = 0, .bound = 256 }, .{ .regs = 178, .static = 0, .bound = 256 }, .{ .regs = 141, .static = 0, .bound = 256 }, .{ .regs = 141, .static = 0, .bound = 256 } } },
};
