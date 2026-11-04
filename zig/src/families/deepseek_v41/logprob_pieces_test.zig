//! Piece completion selects the scored finish, and phase wrappers retain the actual target capability.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("draft/iface.zig");
const a = std.testing.allocator;

const Target = struct {
    finishes: usize = 0,
    fn target(t: *Target) iface.Target {
        return .{ .ptr = t, .vtable = &table };
    }
    const table: iface.Target.VTable = .{ .prefill = prefill, .scored_prompt = scored, .window = window, .keep = keep, .taps = taps, .release = release, .begin = begin, .piece = piece, .finish = finish };
    fn prefill(_: *anyopaque, _: u32, _: []const u32, _: ?lanes.Sampling, _: u64) anyerror!u32 {
        return error.WrongPrefill;
    }
    fn finish(_: *anyopaque, _: u32, _: []const u32, _: ?lanes.Sampling, _: u64) anyerror!u32 {
        return error.WrongFinish;
    }
    fn scored(p: *anyopaque, _: u32, ids: []const u32, _: ?lanes.Sampling, draw: u64, k: u8, row: *lanes.logprob.Row, finishing: bool) anyerror!u32 {
        const t: *Target = @ptrCast(@alignCast(p));
        try std.testing.expect(finishing);
        try std.testing.expectEqual(@as(u64, ids.len), draw);
        try std.testing.expectEqual(@as(u8, 3), k);
        t.finishes += 1;
        row.* = .{ .token = 17, .logprob = -1.25, .count = 1, .ids = @splat(17), .logprobs = @splat(-1.25) };
        return 17;
    }
    fn begin(_: *anyopaque, _: u32, _: []const u32) anyerror!iface.Target.Begun {
        return .{};
    }
    fn piece(_: *anyopaque, _: u32, _: []const u32, _: u64, _: u64, _: bool) anyerror!void {}
    fn window(_: *anyopaque, _: []const iface.Segment, _: [][]u32) anyerror!void {}
    fn keep(_: *anyopaque, _: u32, _: []const u32) anyerror!void {}
    fn taps(_: *anyopaque, _: u32) iface.Taps {
        return .{ .address = 0, .rows = 0, .stride = 0 };
    }
    fn release(_: *anyopaque, _: u32) void {}
};

test "DeepSeek logprobs: pieces finish captures the first row before the draft handoff" {
    var target: Target = .{};
    var pass: @import("draft/drive.zig").NoDraft = .{};
    var prices = try @import("draft/costs.zig").defaults(a, 64);
    defer prices.deinit(a);
    const x = try @import("draft/lanes.zig").Lanes.init(a, target.target(), pass.pass(), prices, .{ .shape = .{}, .slots = 1 });
    defer x.deinit();
    var stream = try lanes.Stream.init(a, .{ .id = "pieces", .prompt = &.{ 3, 2, 1, 4 }, .max_new = 2, .logprobs = 3 });
    defer stream.deinit(a);
    const b = x.backend();
    _ = try b.vtable.begin.?(b.ptr, &stream);
    try b.vtable.piece.?(b.ptr, &stream, 0, 3, false);
    try b.vtable.finish.?(b.ptr, &stream);
    const row = try b.firstRow(&stream);
    try std.testing.expectEqual(@as(u32, 17), row.token);
    try std.testing.expectEqual(@as(f32, -1.25), row.logprob);
    try std.testing.expectEqual(@as(usize, 1), target.finishes);
    b.vtable.release(b.ptr, &stream);
}

test "DeepSeek logprobs: the phase wrapper preserves scored and unscored target admission" {
    var target: Target = .{};
    var timed: @import("phases_timed.zig").TimedTarget = .{ .inner = target.target(), .p = undefined, .gpa = a, .io = std.testing.io };
    try std.testing.expect(timed.target().vtable.scored_prompt != null);
    var unscored = Target.table;
    unscored.scored_prompt = null;
    timed.inner.vtable = &unscored;
    try std.testing.expect(timed.target().vtable.scored_prompt == null);
}
