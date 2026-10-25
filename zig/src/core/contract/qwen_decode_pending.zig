//! Optimized Qwen decode resources from nvcc 13.0 and independent later compiler sets.
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");
pub const records = blk: {
    const names = .{ "tf_topk_slices", "tf_topk_merge", "tf_q35_norm", "tf_q35_sums", "tf_q35_argmax_slices", "tf_q35_argmax_merge" };
    const threads = [_]u32{ 256, 256, 256, 64, 256, 32 };
    const usage = [_][limits.sms.len]limits.Usage{
        .{ .{ .regs = 24, .static = 16448, .bound = 256 }, .{ .regs = 20, .static = 16448, .bound = 256 }, .{ .regs = 20, .static = 16448, .bound = 256 }, .{ .regs = 28, .static = 16448, .bound = 256 }, .{ .regs = 28, .static = 16448, .bound = 256 } },
        .{ .{ .regs = 24, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 34, .static = 64, .bound = 256 }, .{ .regs = 36, .static = 64, .bound = 256 } },
        .{ .{ .regs = 54, .static = 16416, .bound = 256 }, .{ .regs = 55, .static = 16416, .bound = 256 }, .{ .regs = 55, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 } },
        .{ .{ .regs = 32, .static = 0, .bound = 64 }, .{ .regs = 42, .static = 0, .bound = 64 }, .{ .regs = 40, .static = 0, .bound = 64 }, .{ .regs = 40, .static = 0, .bound = 64 }, .{ .regs = 40, .static = 0, .bound = 64 } },
        .{ .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 22, .static = 64, .bound = 256 }, .{ .regs = 26, .static = 64, .bound = 256 }, .{ .regs = 26, .static = 64, .bound = 256 } },
        .{ .{ .regs = 28, .static = 0, .bound = 32 }, .{ .regs = 28, .static = 0, .bound = 32 }, .{ .regs = 28, .static = 0, .bound = 32 }, .{ .regs = 30, .static = 0, .bound = 32 }, .{ .regs = 30, .static = 0, .bound = 32 } },
    };
    var out: [names.len]R = undefined;
    for (names, threads, 0..) |symbol, nt, i| {
        out[i] = .{ .launch = .{ .name = symbol, .symbol = symbol, .image = if (i < 2) "sample" else "qwen35", .threads = nt, .min_sm = 80 }, .usage = usage[i], .later_pending = true };
    }
    break :blk out;
};
