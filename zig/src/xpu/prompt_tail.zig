//! XPU 27B prompt tail: layer 63's last 64-row tile feeds the head; an unread chunk writes keys and values only.
const std = @import("std");
const core = @import("core");

/// The runner's layers; qwen27.zig's shape is the same 64.
pub const layers: usize = 64;
/// The last layer's tile: attention query tiles are 8 rows, and 64 is a whole number of them.
pub const tail_rows: usize = 64;

pub const Want = enum { none, last, rows };

/// What a prompt chunk runs on its last layer.
pub const Plan = union(enum) {
    /// The whole layer on every row.
    all,
    /// add_rmsnorm, mixer_in and attn_prep only: the keys and values carry to later chunks and decode.
    kv_only,
    /// The layer's operations from this row on; the rows before it wrote no state the rest reads.
    tail: usize,
};

/// The plan: the last 64-row tile, keys only, or the whole layer when a hook or tap reads it.
pub fn tailPlan(total: usize, want: Want, layer: usize, hooks: bool) Plan {
    if (total == 0 or hooks or layer + 1 != layers or want == .rows) return .all;
    if (core.prompt_tail.unread(.{ .logits = want != .none, .readers = hooks })) return .kv_only;
    return .{ .tail = ((total - 1) / tail_rows) * tail_rows };
}

test "xpu prompt tail: an unread chunk is kv only, a read chunk takes its last 64-row tile" {
    const last = layers - 1;
    try std.testing.expect(tailPlan(1024, .none, last, false) == .kv_only);
    try std.testing.expectEqual(@as(usize, 960), tailPlan(1024, .last, last, false).tail);
    try std.testing.expectEqual(@as(usize, 0), tailPlan(1, .last, last, false).tail);
    try std.testing.expectEqual(@as(usize, 0), tailPlan(64, .last, last, false).tail);
    try std.testing.expectEqual(@as(usize, 64), tailPlan(65, .last, last, false).tail);
    try std.testing.expectEqual(@as(usize, 960), tailPlan(1023, .last, last, false).tail);
    try std.testing.expectEqual(@as(usize, 1024), tailPlan(1025, .last, last, false).tail);
    try std.testing.expectEqual(@as(usize, 1088), tailPlan(1151, .last, last, false).tail);
}

test "xpu prompt tail: a hook, a tap, a non-last layer or a rows pass runs the whole layer" {
    const last = layers - 1;
    try std.testing.expect(tailPlan(1024, .last, last, true) == .all);
    try std.testing.expect(tailPlan(1024, .none, last, true) == .all);
    try std.testing.expect(tailPlan(1024, .last, last - 1, false) == .all);
    try std.testing.expect(tailPlan(1024, .rows, last, false) == .all);
}
