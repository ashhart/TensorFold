//! The frames a request returns: cut at the end-of-speech frame, in an allocation of exactly their length.
const std = @import("std");
const cfg = @import("config.zig");

pub const Frame = [cfg.frame_width]i32;

/// The index, counted from `from`, of the first end-of-speech frame among `checked`; null when there is none.
pub fn endOfSpeech(checked: []const Frame, from: usize, eos: i32) ?usize {
    for (checked, from..) |f, i| if (f[0] == eos) return i;
    return null;
}

/// The first `made` frames of `host` (from `gpa`) reallocated to exactly `made`; on error `host` is the caller's still.
pub fn owned(gpa: std.mem.Allocator, host: []Frame, made: usize) error{OutOfMemory}![]Frame {
    std.debug.assert(made <= host.len);
    return gpa.realloc(host, made);
}

fn frame(main: i32) Frame {
    var f: Frame = @splat(7);
    f[0] = main;
    return f;
}

test "an early end of speech returns exactly the frames before it, freed at that length" {
    const gpa = std.testing.allocator;
    const eos = 5;
    const host = try gpa.alloc(Frame, 8);
    for (host, 0..) |*f, i| f.* = frame(if (i == 3) eos else 100 + @as(i32, @intCast(i)));
    // a burst of frames [2, 6): the end is at 3
    const made = endOfSpeech(host[2..6], 2, eos).?;
    try std.testing.expectEqual(@as(usize, 3), made);
    const kept = try owned(gpa, host, made);
    // the testing allocator fails a free whose length is not the allocation's
    defer gpa.free(kept);
    try std.testing.expectEqual(@as(usize, 3), kept.len);
    for (kept, 0..) |f, i| try std.testing.expectEqual(100 + @as(i32, @intCast(i)), f[0]);
}

test "no end of speech keeps every frame; one at the first frame keeps none" {
    const gpa = std.testing.allocator;
    const host = try gpa.alloc(Frame, 4);
    for (host) |*f| f.* = frame(100);
    try std.testing.expectEqual(@as(?usize, null), endOfSpeech(host, 0, 5));
    const all = try owned(gpa, host, host.len);
    defer gpa.free(all);
    try std.testing.expectEqual(@as(usize, 4), all.len);
    const first = try gpa.alloc(Frame, 4);
    first[0] = frame(5);
    const none = try owned(gpa, first, endOfSpeech(first[0..1], 0, 5).?);
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}
