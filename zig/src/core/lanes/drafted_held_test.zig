//! Drafted.verify with held_on_target: windows of drafting streams reach the target with Window.held as asked.
const std = @import("std");
const be = @import("backend.zig");
const dr = @import("drafter.zig");
const Stream = @import("stream.zig").Stream;
const Drafted = @import("drafted.zig").Drafted;

/// A target that records each verify's held rows and host tokens, and a drafter that only counts host readbacks.
const Fake = struct {
    seen: u32 = 0,
    tokens: [4]u32 = @splat(0),
    readbacks: u32 = 0,

    fn cast(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, _: []be.Verified) anyerror!void {
        const f = cast(ptr);
        f.seen = windows[0].held;
        @memcpy(f.tokens[0..windows[0].tokens.len], windows[0].tokens);
    }
    fn held(ptr: *anyopaque, _: []const *Stream, out: []const []u32) anyerror!void {
        cast(ptr).readbacks += 1;
        for (out) |o| @memset(o, 5);
    }
    fn none(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn facts(_: *anyopaque) dr.Facts {
        return .{ .depth = 3, .step_ms = 0, .prior = &.{}, .stream_bytes = 0 };
    }
    fn unused(_: *anyopaque, _: *Stream) anyerror!void {
        return error.Unused;
    }
    fn absorb(_: *anyopaque, _: []const dr.Absorb) anyerror!void {
        return error.Unused;
    }
    fn hold(_: *anyopaque, _: []const dr.Hold) anyerror!void {
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
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
};

test "held_on_target passes held windows through, and a stream with fillers still gets host tokens" {
    var f: Fake = .{};
    const target: be.Backend = .{ .ptr = &f, .vtable = &.{ .prefill = Fake.unused, .first = Fake.first, .queue = Fake.queue, .read = Fake.read, .verify = Fake.verify, .keep = Fake.keep, .draft = Fake.draft, .release = Fake.release } };
    const drafter: dr.Drafter = .{ .ptr = &f, .vtable = &.{ .taps = Fake.none, .facts = Fake.facts, .open = Fake.unused, .absorb = Fake.absorb, .hold = Fake.hold, .held = Fake.held, .release = Fake.release } };
    var x: Drafted = .{ .gpa = std.testing.allocator, .target = target, .drafter = drafter, .held_on_target = true };
    defer x.deinit();
    var s: Stream = undefined;
    try x.lanes.put(std.testing.allocator, &s, .{ .rows = true });
    var sampled: [4]u32 = undefined;
    var drafts: [3]u32 = undefined;
    const w = [_]be.Window{.{ .stream = &s, .pending = 9, .held = 3, .tokens = &.{}, .parents = null, .positions = &.{ 1, 2, 3, 4 } }};
    var o = [_]be.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
    try x.backend().verify(&w, &o);
    try std.testing.expectEqual(@as(u32, 3), f.seen);
    try std.testing.expectEqual(@as(u32, 0), f.readbacks);
    x.lanes.getPtr(&s).?.filler = 3; // fillers: the pending token as host tokens, no readback
    try x.backend().verify(&w, &o);
    try std.testing.expectEqual(@as(u32, 0), f.seen);
    try std.testing.expectEqualSlices(u32, &.{ 9, 9, 9 }, f.tokens[0..3]);
    x.held_on_target = false; // off: the drafter's drafts as host tokens
    try x.backend().verify(&w, &o);
    try std.testing.expectEqual(@as(u32, 1), f.readbacks);
    try std.testing.expectEqualSlices(u32, &.{ 5, 5, 5 }, f.tokens[0..3]);
}
