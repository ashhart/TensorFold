//! Drafted.adopt: a stream another driver decoded joins the drafter's streams, and its next draft absorbs and holds.
const std = @import("std");
const be = @import("backend.zig");
const dr = @import("drafter.zig");
const Stream = @import("stream.zig").Stream;
const Drafted = @import("drafted.zig").Drafted;

const gpa = std.testing.allocator;

/// A target with features on every row, and a drafter that counts opens, absorbs (and what they offered) and holds.
const Fake = struct {
    opens: u32 = 0,
    absorbs: u32 = 0,
    offered: u32 = 0,
    holds: u32 = 0,
    rows: [16]u32 = @splat(1),

    fn cast(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn features(ptr: *anyopaque, _: *Stream, _: []const u32, _: u64, count: u32) anyerror!be.Features {
        return .{ .buffer = @intFromPtr(&cast(ptr).rows), .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
    }
    fn open(ptr: *anyopaque, _: *Stream) anyerror!void {
        cast(ptr).opens += 1;
    }
    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const f = cast(ptr);
        f.absorbs += 1;
        for (items) |it| f.offered += it.offered;
    }
    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        cast(ptr).holds += @intCast(items.len);
    }
    fn held(_: *anyopaque, _: []const *Stream, _: []const []u32) anyerror!void {}
    fn none(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn facts(_: *anyopaque) dr.Facts {
        return .{ .depth = 3, .step_ms = 0, .prior = &.{}, .stream_bytes = 0 };
    }
    fn unused(_: *anyopaque, _: *Stream) anyerror!void {
        return error.Unused;
    }
    fn release(_: *anyopaque, _: *Stream) void {}
    fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
        return error.Unused;
    }
    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unused;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return error.Unused;
    }
    fn verify(_: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {}
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
};

/// The next round's request for `s`: two kept rows after `start`, three drafts.
fn kept(s: *Stream, start: u64) be.DraftRequest {
    return .{ .stream = s, .follow = &.{ 3, 4 }, .rows = &.{ 0, 1 }, .start = start, .position = start + 3, .depth = 3 };
}

test "Drafted.adopt opens a missing stream, resets a present one, and the next draft absorbs and holds" {
    var f: Fake = .{};
    const target: be.Backend = .{ .ptr = &f, .vtable = &.{ .prefill = Fake.unused, .features = Fake.features, .first = Fake.first, .queue = Fake.queue, .read = Fake.read, .verify = Fake.verify, .keep = Fake.keep, .draft = Fake.draft, .release = Fake.release } };
    const drafter: dr.Drafter = .{ .ptr = &f, .vtable = &.{ .taps = Fake.none, .facts = Fake.facts, .open = Fake.open, .absorb = Fake.absorb, .hold = Fake.hold, .held = Fake.held, .release = Fake.release } };
    var x = try Drafted.init(gpa, target, drafter);
    defer x.deinit();
    const b = x.backend();
    const prompt = [_]u32{ 5, 6, 7, 8, 9, 10, 11, 12 };
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 8 });
    defer s.deinit(gpa);
    try x.adopt(&s); // missing: the drafter opens it
    try std.testing.expectEqual(@as(u32, 1), f.opens);
    try b.vtable.draft(b.ptr, &.{kept(&s, 8)});
    try std.testing.expectEqual(@as(u32, 1), f.absorbs);
    try std.testing.expectEqual(@as(u32, 1), f.holds);
    // present, with a past round's state left: adopt resets it without a second open
    x.lanes.getPtr(&s).?.* = .{ .filler = 2, .tree_first = 7, .offered = 5, .stepped = 3 };
    try x.adopt(&s);
    try std.testing.expectEqual(@as(u32, 1), f.opens);
    try std.testing.expectEqualDeep(x.lanes.get(&s).?, @TypeOf(x.lanes.get(&s).?){ .rows = true });
    try b.vtable.draft(b.ptr, &.{kept(&s, 10)});
    try std.testing.expectEqual(@as(u32, 2), f.absorbs);
    try std.testing.expectEqual(@as(u32, 0), f.offered);
    try std.testing.expectEqual(@as(u32, 2), f.holds);
    // a prompt restored whole has no row to absorb: adopted, the drafter still holds instead of fillers
    var w = try Stream.init(gpa, .{ .id = "w", .prompt = &prompt, .max_new = 8 });
    defer w.deinit(gpa);
    w.cached = prompt.len;
    try x.adopt(&w);
    try b.vtable.draft(b.ptr, &.{.{ .stream = &w, .follow = &.{}, .first = .{ .value = 9 }, .rows = null, .start = 8, .position = 9, .depth = 3 }});
    try std.testing.expectEqual(@as(u32, 2), f.opens);
    try std.testing.expectEqual(@as(u32, 2), f.absorbs);
    try std.testing.expectEqual(@as(u32, 3), f.holds);
    try std.testing.expectEqual(@as(u32, 0), x.lanes.get(&w).?.filler);
}
