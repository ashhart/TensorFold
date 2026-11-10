//! Native FP8 QK launch declarations; usage stays pending until compiler-specific receipts arrive.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");
pub const symbols = [_][]const u8{
    "tf_fp8q_attention_whole", "tf_fp8q_attention_whole_residual",
    "tf_fp8q_attention_paged", "tf_fp8q_attention_paged_residual",
    "tf_fp8q_attention_probe", "tf_fp8q_attention_probe_narrow",
};
pub const dynamic = [_]u32{ 33408, 50048, 33408, 50048, 33408, 33408 };
const empty: [symbols.len]limits.Usage = @splat(.{ .regs = 0, .static = 0, .bound = 128 });

pub fn measured(sm120: [symbols.len]limits.Usage, sm121: [symbols.len]limits.Usage) [symbols.len]r.CudaRecord {
    var out: [symbols.len]r.CudaRecord = undefined;
    const unsupported: limits.Usage = .{ .regs = 0, .static = 0 };
    inline for (symbols, dynamic, 0..) |symbol, bytes, index| out[index] = .{
        .launch = .{ .name = "native FP8 QK " ++ symbol, .symbol = symbol, .image = "fp8_attention_prompt", .threads = 128, .dynamic = bytes, .opt_in = true, .min_sm = 120 },
        .usage = .{ unsupported, unsupported, unsupported, sm120[index], sm121[index] },
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
    return out;
}
pub const records = measured(empty, empty);

test "unmeasured FP8 attention usage remains explicitly pending" {
    for (records, dynamic) |record, bytes| {
        try std.testing.expectEqual(@as(u32, 128), record.launch.threads);
        try std.testing.expectEqual(bytes, record.launch.dynamic);
        try std.testing.expectEqual(@as(u32, 0), record.usage[3].regs);
        try std.testing.expectEqual(@as(u32, 0), record.usage[4].regs);
        try std.testing.expect(record.launch.opt_in);
    }
}
