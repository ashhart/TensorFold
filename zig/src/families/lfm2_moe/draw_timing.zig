//! Successful verification draws only: their actual span and a quote for a stable request route.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const c = @import("config.zig");
const be = lanes.backend;
const mask_words = (c.vocab + 31) / 32;

/// Grammar can change the next window's route between backend calls; do not publish its current slope.
fn quotable(w: be.Window) bool {
    return w.stream.grammar == null and w.masks.len == 0;
}

/// Before any forward/validation, and on a failed batch: no previous or partial batch's actual survives.
pub fn reset(windows: []const be.Window) void {
    for (windows) |w| {
        w.stream.verify_host_ms = 0;
        if (!quotable(w)) w.stream.verify_row_ms = null;
    }
}

pub const Monotonic = struct {
    pub fn now(_: Monotonic) i128 {
        var t: platform.c.timespec = undefined;
        _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t);
        return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
    }
};

/// Times completed host draws; a compile-time clock lets tests exercise the same serving path.
pub fn drawWindow(comptime drawRow: anytype, gpa: std.mem.Allocator, w: be.Window, sampled: []u32, logits: []const u16, best: []const u32, clock: anytype) !void {
    const host = w.masks.len > 0 or if (w.stream.sampling) |s| !(s.temperature <= 0) else false;
    const start = if (host) clock.now() else 0;
    for (sampled, w.positions, 0..) |*t, p, r| {
        const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
        t.* = try drawRow(gpa, logits[r * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask, best[2 * r ..][0..2].*);
    }
    const ms = if (host) @as(f64, @floatFromInt(clock.now() - start)) / std.time.ns_per_ms else 0;
    w.stream.verify_host_ms = ms;
    w.stream.verify_row_ms = if (quotable(w)) ms / @as(f64, @floatFromInt(w.rows())) else null;
}
