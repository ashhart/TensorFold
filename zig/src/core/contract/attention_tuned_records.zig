//! Mode 6 in the R3 prefill_attention image, compiled by nvcc 13.5 for sm_120a and sm_121a.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");
pub const symbols = [_][]const u8{ "tf_compact_w4h2s1_whole", "tf_compact_w4h2s1_paged" };
/// R3 compiler-reported usage; device and ELF validation remain separate.
pub const measured_sm120 = [_]limits.Usage{
    .{ .regs = 255, .static = 0, .bound = 128, .local = 32 },
    .{ .regs = 255, .static = 0, .bound = 128, .local = 40 },
};
pub const measured_sm121 = [_]limits.Usage{
    .{ .regs = 255, .static = 0, .bound = 128, .local = 32 },
    .{ .regs = 255, .static = 0, .bound = 128, .local = 40 },
};
pub fn fromReceipts(sm120: [2]limits.Usage, sm121: [2]limits.Usage) [2]r.CudaRecord {
    var records_: [2]r.CudaRecord = undefined;
    inline for (symbols, 0..) |symbol, i| records_[i] = .{
        .launch = .{ .name = "compact exact attention " ++ symbol, .symbol = symbol, .image = "prefill_attention", .threads = 128, .dynamic = 49152, .opt_in = true, .min_sm = 120 },
        .usage = .{ .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 }, sm120[i], sm121[i] },
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
    return records_;
}
pub const records = fromReceipts(measured_sm120, measured_sm121);

test "R3 prefill image uses its measured compact records on both targets" {
    for (records, 0..) |record, i| {
        try std.testing.expectEqualStrings("prefill_attention", record.launch.image);
        try std.testing.expectEqual(@as(u32, 128), record.launch.threads);
        try std.testing.expectEqual(@as(u32, 49152), record.launch.dynamic);
        for (record.usage[3..]) |usage| {
            try std.testing.expectEqual(@as(u32, 255), usage.regs);
            try std.testing.expectEqual(@as(u32, 0), usage.static);
            try std.testing.expectEqual(@as(u32, 128), usage.bound);
            try std.testing.expectEqual(@as(u32, if (i == 0) 32 else 40), usage.local);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), try r.checkCuda(std.testing.allocator, &records, true));
}
