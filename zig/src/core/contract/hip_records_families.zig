//! The HIP families' gate-5 records, one array a family (`hip_records_<family>.zig`); hip_records.zig joins them.
const std = @import("std");
const Record = @import("hip_records.zig").Record;

pub const records = .{
    @import("hip_records_glm_ocr.zig").records,
    @import("hip_records_joint.zig").records,
    @import("hip_records_lighton_ocr.zig").records,
    @import("hip_records_qwen_vision.zig").records,
};

/// Every family's records as one array.
pub const all = join: {
    var joined: []const Record = &.{};
    for (records) |list| joined = joined ++ list;
    break :join joined[0..joined.len].*;
};

test "the HIP records are main's and the families' own: GLM-OCR's, the joint schema's, LightOn's and the Qwen tower's" {
    const every = @import("hip_records.zig").records;
    try std.testing.expectEqual(@as(usize, 227), every.len); // includes Parse, LightOn and Qwen vision records
    try std.testing.expectEqual(@as(usize, 184), every.len - all.len);
}
