//! The measured sm121 compiled receipt must qualify without inventing lower-SM records or a launch bound.
const std = @import("std");
const registry = @import("src/core/contract/registry.zig");
const records = @import("src/core/contract/qwen_prompt_stages2_records.zig");

test "Qwen stages2 records: measured sm121 resources qualify" {
    try std.testing.expectEqual(@as(usize, 1), records.cuda.len);
    try std.testing.expectEqual(@as(u32, 256), records.cuda[0].launch.threads);
    try std.testing.expectEqual(@as(u32, 41984), records.cuda[0].launch.dynamic);
    try std.testing.expectEqual(@as(u32, 121), records.cuda[0].launch.min_sm);
    const measured = blk: {
        var usage = registry.none;
        usage[4] = .{ .regs = 128, .static = 0, .bound = 256 };
        break :blk usage;
    };
    try std.testing.expectEqual(measured, records.cuda[0].usage);
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records.cuda, true));
}
