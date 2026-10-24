//! Dory's CUDA launch records (the nGPT rows in norms, qk_rows, act_rows and dense_rows) in registry.cuda.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_l2_rows", .symbol = "tf_l2_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 64, .static = 32, .bound = 256 }, .{ .regs = 64, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "norms tf_slerp_rows", .symbol = "tf_slerp_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 } } },
};
