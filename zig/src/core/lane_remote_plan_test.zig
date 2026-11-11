//! A landed remote prefix's pass is planned from where it ends: no cut just past it, a tail mark kept inside a chunk.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fixture = @import("lane_remote_test.zig");
const t = std.testing;

/// Copies of nothing: the planned marks are only read, never restored.
const Blank = struct {
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 64;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        return try t.allocator.create(u8);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        t.allocator.destroy(@as(*u8, @ptrCast(saved)));
    }
};

fn store() pc.Store {
    var none: u8 = 0;
    return pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Blank.bytes, .save = Blank.save, .restore = Blank.restore, .drop = Blank.drop } }, .{ .planned = true, .grid = 128, .tail = true, .tail_align = 128, .min_prompt = 0, .min_gap = 64 }, 1 << 20);
}

test "remote prefix kept: a tail mark under 64 rows past the pull is no cut, and one farther is kept for the chunk to capture" {
    var prompt: [340]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    var s = store();
    defer s.deinit();
    var near: fixture.Peer = .{ .at = 230 };
    defer near.kept.deinit(t.allocator);
    const close = try fixture.runKeep(&near, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = true });
    try t.expect(close.restoring and close.remote and close.at == 230);
    try t.expectEqual(@as(usize, 0), close.marks); // 256 is 26 rows past the pull: one chunk to the end
    try t.expectEqual(@as(u32, 0), close.snap_at);
    var far: fixture.Peer = .{ .at = 100 };
    defer far.kept.deinit(t.allocator);
    const wide = try fixture.runKeep(&far, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = true });
    try t.expect(wide.restoring and wide.remote and wide.at == 100);
    try t.expectEqual(@as(usize, 1), wide.marks);
    try t.expectEqual(@as(u32, 256), wide.first_mark);
    try t.expectEqual(@as(u32, 256), wide.snap_at); // the backend captures the mark inside its chunk
}
