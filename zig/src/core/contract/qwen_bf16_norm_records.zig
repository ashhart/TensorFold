//! A plain BF16 Qwen3.5 checkpoint's norms (qwen35.cu's _shift kernels): nvcc 13.0's resources, later sets pending.
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");
pub const records = blk: {
    const names = .{ "tf_q35_norm_shift", "tf_q35_head_norm_shift" };
    const threads = [_]u32{ 256, 64 };
    const usage = [_][limits.sms.len]limits.Usage{
        .{ .{ .regs = 54, .static = 16416, .bound = 256 }, .{ .regs = 55, .static = 16416, .bound = 256 }, .{ .regs = 55, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 }, .{ .regs = 48, .static = 16416, .bound = 256 } },
        .{ .{ .regs = 24, .static = 16, .bound = 64 }, .{ .regs = 24, .static = 16, .bound = 64 }, .{ .regs = 24, .static = 16, .bound = 64 }, .{ .regs = 22, .static = 16, .bound = 64 }, .{ .regs = 22, .static = 16, .bound = 64 } },
    };
    var out: [names.len]R = undefined;
    for (names, threads, 0..) |symbol, nt, i| {
        out[i] = .{ .launch = .{ .name = symbol, .symbol = symbol, .image = "qwen35", .threads = nt, .min_sm = 80 }, .usage = usage[i], .later_pending = true };
    }
    break :blk out;
};
