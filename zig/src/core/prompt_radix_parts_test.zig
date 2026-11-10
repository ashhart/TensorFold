//! The manifest's registry alone: one pool reference a page, whatever names it, given back on the last drop.
const std = @import("std");
const parts = @import("prompt_radix_parts.zig");
const pc = @import("prompt_cache.zig");
const testing = std.testing;

/// Pages whose holders are counted, as a pool's are.
const Refs = struct {
    refs: [8]u32 = @splat(0),
    calls: u32 = 0, // retains and releases seen

    fn of(p: *anyopaque) *Refs {
        return @ptrCast(@alignCast(p));
    }
    fn retain(p: *anyopaque, id: u32) void {
        of(p).refs[id] += 1;
        of(p).calls += 1;
    }
    fn release(p: *anyopaque, id: u32) void {
        of(p).refs[id] -= 1;
        of(p).calls += 1;
    }
    fn holders(p: *anyopaque, id: u32) u32 {
        return of(p).refs[id];
    }
    fn available(_: *anyopaque) usize {
        return 0;
    }
    fn pages(r: *Refs) pc.Pages {
        return .{ .ptr = r, .bytes = 1, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available } };
    }
};

test "the store's registry: a page named twice holds one reference, given back on its last drop" {
    var r: Refs = .{};
    r.refs[3] = 1; // a stream's page
    var h: parts.Holds = .{};
    defer h.deinit(testing.allocator);
    try h.reserve(testing.allocator, 2);
    h.shareAssumeCapacity(r.pages(), 3);
    h.shareAssumeCapacity(r.pages(), 3); // a second entry names it
    try testing.expectEqual(@as(u32, 2), r.refs[3]);
    try testing.expectEqual(@as(usize, 1), h.count());
    h.drop(r.pages(), 3);
    try testing.expectEqual(@as(u32, 2), r.refs[3]); // still named once
    h.drop(r.pages(), 3);
    try testing.expectEqual(@as(u32, 1), r.refs[3]);
    try testing.expectEqual(@as(usize, 0), h.count());
    try testing.expectEqual(@as(u32, 2), r.calls); // one retain, one release
}

test "the store's registry: a page it held itself is its reference, released once" {
    var r: Refs = .{};
    r.refs[5] = 1; // Pages.hold's reference
    var h: parts.Holds = .{};
    defer h.deinit(testing.allocator);
    try h.reserve(testing.allocator, 1);
    h.adoptHeldAssumeCapacity(5);
    try testing.expect(h.contains(5));
    try testing.expectEqual(@as(u32, 1), r.refs[5]); // no retain
    h.drop(r.pages(), 5);
    try testing.expectEqual(@as(u32, 0), r.refs[5]);
}

test "a family's groups: one prefix group by default, a page-only store has no checkpoint" {
    const d: parts.Groups = .{};
    try testing.expectEqual(@as(usize, 1), d.prefixCount());
    try testing.expect(!d.none());
    const two: parts.Groups = .{ .prefix_rows = &.{ 4, 4 }, .legacy = false };
    try testing.expectEqual(@as(usize, 2), two.prefixCount());
    try testing.expect(two.none());
    const state: parts.Groups = .{ .parts = &.{.{ .kind = .state, .make = .copied }}, .legacy = false };
    try testing.expect(!state.none());
}
