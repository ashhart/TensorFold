//! Exact expansion in the R8 prefill image, CUDA 13.5 on SM120/121: 24 registers, no stack or spills.
const std = @import("std");
const r = @import("registry.zig");
const empty: @import("limits.zig").Usage = .{ .regs = 0, .static = 0 };
pub const records = [_]r.CudaRecord{.{
    .launch = .{ .name = "exact FP8 KV expansion", .symbol = "tf_qwen_expand_kv", .image = "prefill_attention", .threads = 256, .min_sm = 120 },
    .usage = .{ empty, empty, empty, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 } },
    .first_nvcc = .{ .major = 13, .minor = 5 },
}};

test "KV expansion compile receipt retains its measured scope" {
    try std.testing.expectEqual(r.none, r.usageFor(records[0], null));
    try std.testing.expectEqual(r.none, r.usageFor(records[0], .{ .major = 13, .minor = 3 }));
    for ([_]usize{ 3, 4 }) |i| {
        try std.testing.expectEqual(@as(u32, 24), records[0].usage[i].regs);
        try std.testing.expectEqual(@as(u32, 0), records[0].usage[i].static);
        try std.testing.expectEqual(@as(u32, 0), records[0].usage[i].local);
        try std.testing.expectEqual(@as(u32, 256), records[0].usage[i].bound);
    }
}
