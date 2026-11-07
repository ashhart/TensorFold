//! prompt_many's two CUDA launch declarations, in their own file to keep registry.zig at 600 lines.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "prompt_many NVFP4 lean", .symbol = "tf_prompt16_many", .image = "prompt_many", .threads = 256, .dynamic = 76848, .opt_in = true, .min_sm = 120 }, .usage = .{ .{ .regs = 252, .static = 512, .bound = 256 }, .{ .regs = 255, .static = 512, .bound = 256 }, .{ .regs = 255, .static = 512, .bound = 256 }, .{ .regs = 248, .static = 512, .bound = 256 }, .{ .regs = 248, .static = 512, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "prompt_many NVFP4 lean fp32 out", .symbol = "tf_prompt16_many_f32", .image = "prompt_many", .threads = 256, .dynamic = 76848, .opt_in = true, .min_sm = 120 }, .usage = .{ .{ .regs = 254, .static = 512, .bound = 256 }, .{ .regs = 255, .static = 512, .bound = 256 }, .{ .regs = 255, .static = 512, .bound = 256 }, .{ .regs = 246, .static = 512, .bound = 256 }, .{ .regs = 246, .static = 512, .bound = 256 } }, .later_pending = true },
};
