//! A remote prefix one token short of the prompt: the pass is that one row, with nothing planned or kept past it.
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

test "remote prefix of all but the last token: restored at len-1, no mark planned or kept, the whole prefix reported pulled" {
    var prompt: [1000]u32 = undefined; // an odd length off every grid: the cut is where the producer's export stands
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    inline for (.{ true, false }) |keep| {
        var s = store();
        defer s.deinit();
        var peer: fixture.Peer = .{ .at = 999 };
        defer peer.kept.deinit(t.allocator);
        const seen = try fixture.runKeep(&peer, &.{ .prompt = &prompt, .max_tokens = 2 }, null, &s, .{ .keep = keep });
        try t.expect(seen.restoring and seen.remote);
        try t.expectEqual(@as(u32, 999), seen.at);
        try t.expectEqual(@as(usize, 0), seen.marks);
        try t.expectEqual(@as(u32, 0), seen.snap_at);
        try t.expectEqual(@as(?u32, 999), seen.remote_prefix);
        try t.expectEqual(@as(?u32, 999), seen.pulled_prefix);
        try t.expectEqual(@as(usize, 2), seen.tokens);
        try t.expectEqual(@as(u32, 1), peer.fills);
        try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire));
    }
}
