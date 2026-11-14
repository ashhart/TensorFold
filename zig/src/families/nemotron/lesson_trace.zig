//! Development-only routed lesson scores, after their queued prompt completes.
const std = @import("std");
const lesson_route = @import("living").lesson_route;
const layers = @import("layers.zig");
/// Dev trace (TF_SLIDE_TRACE): the stream's router scores at its decision row and the lessons acting on its reply.
pub fn trace(self: anytype, c: anytype) !void {
    const ad = layers.firstAdapter(&self.m.weights) orelse return;
    try self.drain();
    const n = lesson_route.max_blocks;
    const at: usize = @intCast(@max(c.rid, 0));
    const score = ad.score.slice(f32, lesson_route.max_slots * n)[at * n ..][0..n];
    const flags = ad.latch.slice(u32, lesson_route.max_slots * n)[at * n ..][0..n];
    const need = ad.need.slice(f32, n);
    var line: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    for (0..ad.rank / 16) |b| w.print(" {d:.2}/{d:.2}{s}", .{ score[b], need[b], if (flags[b] != 0) "*" else "" }) catch break;
    std.log.info("slide trace: decision row {d}: scores{s} (* acting)", .{ c.decide, w.buffered() });
}
