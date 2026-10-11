//! A stream's committed target taps over the drafter's window, by position modulo the window: the drafter's own, or a lane's.
const std = @import("std");
const mtl = @import("metal");

pub const Ring = struct {
    buf: mtl.Buffer,
    end: *u64, // taps committed so far (the stream's fed tokens)
    window: u32,
    stride: usize, // one row's bytes

    /// Rows [off, off + rows) of `taps` follow the committed ones.
    pub fn absorb(r: Ring, taps: mtl.Buffer, off: usize, rows: u32) !void {
        const size = @as(usize, rows) * r.stride;
        if (rows == 0 or off > taps.length() or size > taps.length() - off) return error.BadCommittedTaps;
        const end = try std.math.add(u64, r.end.*, rows);
        var first: usize = 0;
        while (first < rows) {
            const slot: usize = @intCast((r.end.* + first) % r.window);
            const count = @min(rows - first, r.window - slot);
            @memcpy(r.buf.contents()[slot * r.stride ..][0 .. count * r.stride], taps.contents()[off + first * r.stride ..][0 .. count * r.stride]);
            first += count;
        }
        r.end.* = end;
    }

    /// Ring bytes a state after `at` tokens needs: slots [0, at) until the ring wraps, then all of it.
    pub fn bytes(r: Ring, at: u64) usize {
        return @as(usize, @intCast(@min(at, r.window))) * r.stride;
    }
};
