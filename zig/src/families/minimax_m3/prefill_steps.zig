//! MiniMax-M3 Metal's prompt pass a chunk a call (#3021): today's cuts, a chunk a step, marks after the absorb.
const std = @import("std");
const lanes = @import("lanes");
const Stream = lanes.Stream;

/// A chunk from `at`: `rows` at most, cut at a planned chunk start and a kept mark inside it (main's feed loop body).
pub fn chunkEnd(len: usize, at: usize, rows: usize, chunks: []const u32, marks: []const u32) usize {
    var end = @min(len, at + rows);
    for (chunks) |x| if (x > at and x < end) {
        end = x;
    };
    for (marks) |x| if (x > at and x < end) {
        end = x;
    };
    return end;
}

/// A stream's pass in progress: its next chunk's first row, and a mark kept once its drafter has absorbed the chunk.
pub const Fill = struct { at: usize, mark: ?u32 = null };

/// One chunk of `s` from `f.at` through `r` (`run`, `keep`); `later` keeps marks at the next call. True once all in.
pub fn step(r: anytype, s: *Stream, f: *Fill, rows: usize, later: bool) !bool {
    if (f.mark) |m| r.keep(s, m); // the drafter absorbed the last chunk: the state kept there carries its context
    f.mark = null;
    if (s.isCancelled()) return error.Cancelled;
    const len = s.prompt().len;
    const end = chunkEnd(len, f.at, rows, s.chunks, s.reuse.marks);
    try r.run(s, f.at, end);
    s.fill_at.store(@intCast(end), .monotonic); // the forward waits on its command buffer: these rows are in
    f.at = end;
    if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) {
        if (later and end < len) f.mark = @intCast(end) else r.keep(s, @intCast(end));
    }
    return end == len;
}

/// The host cancels between chunks: the state at the pass's last chunk end is kept (no prompt work runs).
pub fn cancel(s: *Stream, f: ?Fill) void {
    if (f) |x| s.keepInterruptedPrefill(@intCast(x.at));
}

/// Rows a scratch holds: `rows` from `start`, written at forward `gen` (until the next forward rewrites it).
pub const Held = struct { start: u64 = 0, rows: usize = 0, gen: u64 = 0 };

/// Whether `h` holds [start, start + count) and no forward has rewritten the scratch since (its gen is `now`).
pub fn holds(h: Held, now: u64, start: u64, count: u32) bool {
    return h.gen == now and count > 0 and start >= h.start and start + count <= h.start + h.rows;
}

test {
    _ = @import("prefill_steps_test.zig");
}
