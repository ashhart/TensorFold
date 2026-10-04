//! Milliseconds by an integer key (rows or drafts): the int -> float dicts the depth rule reads in Python.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Table = struct {
    slots: []?f64 = &.{},

    pub fn deinit(t: *Table, gpa: Allocator) void {
        gpa.free(t.slots);
        t.* = .{};
    }

    pub fn get(t: Table, key: i64) ?f64 {
        if (key < 0 or key >= @as(i64, @intCast(t.slots.len))) return null;
        return t.slots[@intCast(key)];
    }

    /// Set `key` (growing the table): keys are small non-negative ints.
    pub fn put(t: *Table, gpa: Allocator, key: i64, value: f64) !void {
        std.debug.assert(key >= 0);
        const at: usize = @intCast(key);
        if (at >= t.slots.len) {
            const old = t.slots.len;
            t.slots = try gpa.realloc(t.slots, at + 1);
            @memset(t.slots[old..], null);
        }
        t.slots[at] = value;
    }

    /// Python's `not costs`: no key set.
    pub fn empty(t: Table) bool {
        for (t.slots) |slot| if (slot != null) return false;
        return true;
    }
};

test "table get and put" {
    const gpa = std.testing.allocator;
    var t: Table = .{};
    defer t.deinit(gpa);
    try std.testing.expect(t.empty());
    try t.put(gpa, 3, 1.5);
    try std.testing.expectEqual(@as(?f64, 1.5), t.get(3));
    try std.testing.expectEqual(@as(?f64, null), t.get(2));
    try std.testing.expectEqual(@as(?f64, null), t.get(-1));
}
