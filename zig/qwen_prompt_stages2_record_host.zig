//! A missing compiled register receipt must remain a qualification failure for the opt-in launch.
const std = @import("std");
const registry = @import("src/core/contract/registry.zig");
const records = @import("src/core/contract/qwen_prompt_stages2_records.zig");

test "Qwen stages2 records: sm121 resources stay pending until measured" {
    try std.testing.expectEqual(@as(usize, 1), records.cuda.len);
    try std.testing.expectEqual(@as(u32, 256), records.cuda[0].launch.threads);
    try std.testing.expectEqual(@as(u32, 41984), records.cuda[0].launch.dynamic);
    try std.testing.expectEqual(@as(u32, 121), records.cuda[0].launch.min_sm);
    try std.testing.expectEqual(registry.none, records.cuda[0].usage);
    try std.testing.expectEqual(@as(usize, 1), try registry.checkCuda(std.testing.allocator, &records.cuda, true));
}
