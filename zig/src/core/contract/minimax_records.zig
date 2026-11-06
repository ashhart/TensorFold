//! MiniMax-M3.1-Flash's Metal launches, one file a kernel group.
const std = @import("std");
pub const records = @import("minimax_glue_records.zig").records ++ @import("minimax_route_records.zig").records ++ @import("minimax_nax_records.zig").records ++ @import("minimax_draft_records.zig").records;

test "MiniMax-M3.1-Flash's launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
