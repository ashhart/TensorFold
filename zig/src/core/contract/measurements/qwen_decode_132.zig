//! nvcc 13.2 usage measured on the candidate; 13.0 remains unmeasured for these launches.
const Override = @import("../registry.zig").Override;
pub const records = [_]Override{
    .{ .name = "tf_q35_argmax_merge", .usage = .{ .{ .regs = 28, .static = 0, .bound = 32 }, .{ .regs = 28, .static = 0, .bound = 32 }, .{ .regs = 28, .static = 0, .bound = 32 }, .{ .regs = 30, .static = 0, .bound = 32 }, .{ .regs = 30, .static = 0, .bound = 32 } } },
    .{ .name = "tf_q35_argmax_slices", .usage = .{ .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 27, .static = 64, .bound = 256 }, .{ .regs = 27, .static = 64, .bound = 256 } } },
    .{ .name = "tf_q35_norm", .usage = .{ .{ .regs = 57, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 } } },
    .{ .name = "tf_q35_sums", .usage = .{ .{ .regs = 32, .static = 0, .bound = 64 }, .{ .regs = 42, .static = 0, .bound = 64 }, .{ .regs = 40, .static = 0, .bound = 64 }, .{ .regs = 40, .static = 0, .bound = 64 }, .{ .regs = 40, .static = 0, .bound = 64 } } },
    .{ .name = "tf_topk_merge", .usage = .{ .{ .regs = 24, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 36, .static = 64, .bound = 256 }, .{ .regs = 36, .static = 64, .bound = 256 } } },
    .{ .name = "tf_topk_slices", .usage = .{ .{ .regs = 24, .static = 16448, .bound = 256 }, .{ .regs = 20, .static = 16448, .bound = 256 }, .{ .regs = 20, .static = 16448, .bound = 256 }, .{ .regs = 28, .static = 16448, .bound = 256 }, .{ .regs = 28, .static = 16448, .bound = 256 } } },
};
