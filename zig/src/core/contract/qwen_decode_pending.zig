//! Optimized Qwen decode launches retain pending 13.0 resources; 13.2 measurements are selected separately.
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });
pub const records = blk: {
    const names = .{ "tf_topk_slices", "tf_topk_merge", "tf_q35_norm", "tf_q35_sums", "tf_q35_argmax_slices", "tf_q35_argmax_merge" };
    const threads = [_]u32{ 256, 256, 256, 64, 256, 32 };
    var out: [names.len]R = undefined;
    for (names, threads, 0..) |symbol, nt, i| {
        out[i] = .{ .launch = .{ .name = symbol, .symbol = symbol, .image = if (i < 2) "sample" else "qwen35", .threads = nt, .min_sm = 80 }, .usage = none };
    }
    break :blk out;
};
