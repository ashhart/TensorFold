//! A stream's committed target taps over the drafter's window, by position modulo the window.
const std = @import("std");
const mtl = @import("metal");
const kv_grow = @import("kv_grow.zig");

pub const Ring = struct {
    buf: mtl.Buffer,
    end: *u64, // taps committed so far (the stream's fed tokens)
    window: u32,
    stride: usize, // one row's bytes
    grow: ?Grow = null, // a ring that holds the rows its stream reached: the buffer holding `rows`; null: the window

    /// The buffer holding at least `rows` rows (it may be a new one; the old rows stay at their places).
    pub const Grow = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, rows: u32) anyerror!mtl.Buffer };

    /// Rows [off, off + rows) of `taps` follow the committed ones.
    pub fn absorb(r: Ring, taps: mtl.Buffer, off: usize, rows: u32) !void {
        const size = @as(usize, rows) * r.stride;
        if (rows == 0 or off > taps.length() or size > taps.length() - off) return error.BadCommittedTaps;
        const end = try std.math.add(u64, r.end.*, rows);
        const buf = if (r.grow) |g| try g.run(g.ctx, @intCast(@min(end, r.window))) else r.buf;
        var first: usize = 0;
        while (first < rows) {
            const slot: usize = @intCast((r.end.* + first) % r.window);
            const count = @min(rows - first, r.window - slot);
            @memcpy(buf.contents()[slot * r.stride ..][0 .. count * r.stride], taps.contents()[off + first * r.stride ..][0 .. count * r.stride]);
            first += count;
        }
        r.end.* = end;
    }

    /// The ring holds a state of `rows` tokens before a blit writes it (a restore); its buffer may change.
    pub fn hold(r: *Ring, rows: u64) !void {
        if (r.grow) |g| r.buf = try g.run(g.ctx, @intCast(@min(rows, r.window)));
    }

    /// Ring bytes a state after `at` tokens needs: slots [0, at) until the ring wraps, then all of it.
    pub fn bytes(r: Ring, at: u64) usize {
        return @as(usize, @intCast(@min(at, r.window))) * r.stride;
    }
};

/// Rows a growing ring takes for `need`: the K/V growth rule, at most the window.
pub fn grownRows(held: u32, need: u32, step: u32, window: u32) u32 {
    return kv_grow.rows(held, need, step, window);
}

/// A grown ring's first `held` rows into its new buffer, at the same places (the CPU writes rings).
pub fn carry(old: mtl.Buffer, new: mtl.Buffer, held: u32, stride: usize) void {
    const n = @as(usize, held) * stride;
    @memcpy(new.contents()[0..n], old.contents()[0..n]);
}
