//! #1594 step 2b's kept remote states: an exact keep hides only the remote copies it serves alike.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Image = @import("lanes").stream.Image;
const t = std.testing;

/// A family whose copies are one byte each: these tests read only the store's bookkeeping.
const Copies = struct {
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

var none: u8 = 0;

fn store() pc.Store {
    return pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Copies.bytes, .save = Copies.save, .restore = Copies.restore, .drop = Copies.drop } }, .{ .min_prompt = 0 }, 1 << 20);
}

/// Whether the state a lookup found is remote (null: it found none).
fn remoteOf(e: ?*pc.Entry) ?bool {
    return if (e) |x| x.remote else null;
}

const six = [_]u32{ 11, 12, 13, 14, 15, 16 };

test "kept remote states: an exact keep leaves a remote copy whose rows were decoded, and a lookup with them finds it" {
    var s = store();
    defer s.deinit();
    const decoded = [_][2]u32{.{ 1, 3 }}; // rows a reply decoded, as a remote reply's kept end has them
    s.keep_remote = true;
    try t.expect(s.keep(&six, 4, null, &.{}, &decoded));
    s.keep_remote = false;
    try t.expect(s.keep(&six, 4, null, &.{}, &.{})); // the engine's own copy of the same words, every row prefilled
    try t.expectEqual(@as(u64, 2), s.counts.kept);
    try t.expectEqual(@as(?bool, true), remoteOf(s.find(&six, &.{}, &decoded))); // the exact copy cannot serve it
    try t.expectEqual(@as(?bool, false), remoteOf(s.find(&six, &.{}, &.{})));
    try t.expect(s.keep(&six, 4, null, &.{}, &decoded)); // an exact copy with the same rows decoded serves it alike
    try t.expectEqual(@as(?bool, false), remoteOf(s.find(&six, &.{}, &decoded))); // so the remote copy is hidden
    try t.expectEqual(@as(u64, 3), s.counts.kept);
}

test "kept remote states: an exact keep leaves a remote copy that read a picture, and a lookup with it finds it" {
    var s = store();
    defer s.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const picture: Image = .{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 2, .tokens = 1, .hash = 7 };
    s.keep_remote = true;
    try t.expect(s.keepBornImages(&six, 6, 4, null, &.{}, &.{}, &.{picture}));
    s.keep_remote = false;
    try t.expect(s.keep(&six, 4, null, &.{}, &.{})); // the engine's own copy of the same words, read with no picture
    try t.expectEqual(@as(u64, 2), s.counts.kept);
    const l = try s.lookupUserImages(arena.allocator(), &six, 6, 0, 0, 0, &.{}, &.{}, &.{}, &.{picture});
    const got = l.entry orelse return error.TestRemoteCopyHidden; // the exact copy cannot serve the picture's request
    defer s.release(got);
    try t.expect(got.remote and got.at == 4);
    try t.expectEqual(@as(?bool, false), remoteOf(s.find(&six, &.{}, &.{})));
}
