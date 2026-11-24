//! Joint prompt-pair scratch ownership, separate from the regular prompt-twin fields.

const std = @import("std");

pub const joint_twin_at = .{ .wts = 50, .eact = 56, .ey = 57 };

/// Whether a regular twin owns size `i`; joint mode additionally owns its routed-expert temporaries.
pub fn twinHolds(twin_at: anytype, i: usize, joint: bool) bool {
    inline for (@typeInfo(@TypeOf(twin_at)).@"struct".field_names) |f| if (@field(twin_at, f) == i) return true;
    if (joint) inline for (@typeInfo(@TypeOf(joint_twin_at)).@"struct".field_names) |f| if (@field(joint_twin_at, f) == i) return true;
    return false;
}

test "only joint twins own weights and routed expert temporaries" {
    const regular = .{ .weights = 1, .part = 70 };
    try std.testing.expect(!twinHolds(regular, 50, false));
    try std.testing.expect(!twinHolds(regular, 56, false));
    try std.testing.expect(!twinHolds(regular, 57, false));
    try std.testing.expect(twinHolds(regular, 50, true));
    try std.testing.expect(twinHolds(regular, 56, true));
    try std.testing.expect(twinHolds(regular, 57, true));
}
