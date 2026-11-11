//! A live computation may share only its own saves or the exact state restored before it continued.
const std = @import("std");

pub const Tag = struct { run: u64 = 0, id: u64 = 0 };
pub const Clock = struct {
    next: u64 = 1,
    pub fn take(c: *Clock) u64 {
        const id = c.next;
        if (id != 0) c.next = if (id == std.math.maxInt(u64)) 0 else id + 1;
        return id;
    }
};
pub const Slot = struct {
    run: u64 = 0,
    restored: u64 = 0,
    through: u32 = 0,
    pub fn fresh(s: *Slot, c: *Clock) void {
        s.* = .{ .run = c.take() };
    }
    pub fn restoredFrom(s: *Slot, c: *Clock, tag: Tag, at: u32) void {
        s.fresh(c);
        s.restored = tag.id;
        s.through = at;
    }
    pub fn stamp(s: *Slot, c: *Clock) Tag {
        if (s.run == 0) s.fresh(c);
        return .{ .run = s.run, .id = c.take() };
    }
    pub fn allows(s: Slot, tag: Tag, parent_at: u32, at: u32) bool {
        if (s.run == 0 or tag.id == 0 or parent_at > at) return false;
        return tag.run == s.run or (tag.id == s.restored and parent_at <= s.through);
    }
};

test "kept pages: lineage excludes concurrent computations and a restored state's later siblings" {
    const t = std.testing;
    var ids: Clock = .{};
    var left: Slot = .{};
    var right: Slot = .{};
    left.fresh(&ids);
    right.fresh(&ids);
    const first = left.stamp(&ids);
    const later = left.stamp(&ids);
    try t.expect(left.allows(first, 4, 8));
    try t.expect(!right.allows(first, 4, 8));
    right.restoredFrom(&ids, first, 4);
    try t.expect(right.allows(first, 4, 8));
    try t.expect(!right.allows(later, 8, 12));
    const branch = right.stamp(&ids);
    try t.expect(right.allows(branch, 8, 12));
    try t.expect(!left.allows(branch, 8, 12));
    right.fresh(&ids);
    try t.expect(!right.allows(first, 4, 8));
    try t.expect(!right.allows(branch, 8, 12));
}

test "kept pages: flat states, deeper parents and exhausted IDs cannot grant sharing" {
    const t = std.testing;
    var ids: Clock = .{};
    var s: Slot = .{};
    const parent = s.stamp(&ids);
    try t.expect(!s.allows(parent, 8, 4));
    s.restoredFrom(&ids, .{}, 8);
    try t.expect(!s.allows(.{}, 4, 8));
    ids.next = std.math.maxInt(u64);
    try t.expectEqual(std.math.maxInt(u64), ids.take());
    try t.expectEqual(@as(u64, 0), ids.take());
    s.fresh(&ids);
    const refused = s.stamp(&ids);
    try t.expect(!s.allows(parent, 4, 8));
    try t.expect(!s.allows(refused, 4, 8));
}
