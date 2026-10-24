//! Dory's CUDA launch records (the nGPT rows in norms, qk_rows, act_rows and dense_rows) in registry.cuda.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_l2_rows", .symbol = "tf_l2_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 64, .static = 32, .bound = 256 }, .{ .regs = 64, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "norms tf_slerp_rows", .symbol = "tf_slerp_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows_l2 at dim 256", .symbol = "tf_qk_rows_l2", .image = "qk_rows", .threads = 256 }, .usage = .{ .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 } } },
};
