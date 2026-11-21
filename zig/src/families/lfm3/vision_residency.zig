//! LFM3-VL's tower scratch and projected-row buffer join the served model's residency set at open, or the open stops.
const std = @import("std");

pub const refusal = "LFM3 vision scratch and projected image rows exceed the model's residency budget";

/// One bounded join of `scratch` and `rows` through `resident.tryAddNow`; a refusal leaves them all outside the set.
pub fn admit(gpa: std.mem.Allocator, resident: anytype, scratch: anytype, rows: anytype, problem: *[]const u8) !void {
    var buffers: std.ArrayList(@TypeOf(rows)) = .empty;
    defer buffers.deinit(gpa);
    try buffers.appendSlice(gpa, scratch);
    try buffers.append(gpa, rows);
    if (!resident.tryAddNow(buffers.items)) {
        problem.* = refusal;
        return error.Lfm3VisionResidencyBudget;
    }
}
