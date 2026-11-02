//! The HIP families' gate-5 records, one array a family (`hip_records_<family>.zig`); hip_records.zig joins them.
const std = @import("std");
const Record = @import("hip_records.zig").Record;

pub const records = .{};

/// Every family's records as one array.
pub const all = join: {
    var joined: []const Record = &.{};
    for (records) |list| joined = joined ++ list;
    break :join joined[0..joined.len].*;
};

test "the shared HIP records stay main's; families only add theirs" {
    const every = @import("hip_records.zig").records;
    try std.testing.expectEqual(@as(usize, 173), every.len - all.len);
}
