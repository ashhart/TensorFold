//! The HIP families' gate-5 records, one array a family (`hip_records_<family>.zig`); hip_records.zig joins them.
const std = @import("std");
const Record = @import("hip_records.zig").Record;

pub const records = .{
    @import("hip_records_glm_ocr.zig").records,
};

/// Every family's records as one array.
pub const all = join: {
    var joined: []const Record = &.{};
    for (records) |list| joined = joined ++ list;
    break :join joined[0..joined.len].*;
};

test "the HIP records stay main's; the family lists only regroup GLM-OCR's" {
    const every = @import("hip_records.zig").records;
    try std.testing.expectEqual(@as(usize, 182), every.len); // main's records, GLM-OCR's 12 among them
    try std.testing.expectEqual(@as(usize, 170), every.len - all.len);
}
