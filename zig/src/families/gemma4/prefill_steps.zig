//! Gemma 4 Metal's prompt pass a chunk a call (#3021): today's cuts, a chunk a step, marks after the drafter's absorb.
const std = @import("std");
const lanes = @import("lanes");
const Stream = lanes.Stream;

/// A chunk from `at`: `rows` at most, cut at planned starts and kept marks, never inside an image's span.
/// `alone`: where fill skips the deep layers (E-series) the prompt's last row runs alone, the rows before it filling.
pub fn chunkEnd(len: usize, at: usize, rows: usize, chunks: []const u32, marks: []const u32, spans: anytype, alone: bool) usize {
    var end = @min(len, at + rows);
    // planned chunk starts and kept marks cut a chunk (resumed and fresh passes cut alike past them)
    for ([_][]const u32{ chunks, marks }) |cuts| for (cuts) |x| if (x > at and x < end) {
        end = x;
    };
    if (alone and end == len and end - at > 1) end -= 1;
    // an image's rows see each other on sliding layers: no cut inside one (before it, or past it when it leads)
    for (spans) |span| if (span.at < end and span.at + span.n > end) {
        end = if (span.at > at) span.at else span.at + span.n;
    };
    return end;
}

/// A stream's pass in progress: its next chunk's first row, and a mark kept once its drafter has absorbed the chunk.
/// `until` ends a first phase before the prompt does (the E-series' windows; 0: the prompt's end), `alone` as chunkEnd's.
pub const Fill = struct { at: usize, mark: ?u32 = null, until: usize = 0, alone: bool = false };

/// One chunk of `s` from `f.at` through `r` (`run`, `keep`); `later` keeps marks at the next call. True once all in.
pub fn step(r: anytype, s: *Stream, f: *Fill, rows: usize, spans: anytype, later: bool) !bool {
    if (f.mark) |m| r.keep(s, m); // the drafter absorbed the last chunk: the state kept there carries its context
    f.mark = null;
    if (s.isCancelled()) return error.Cancelled;
    const len = s.prompt().len;
    const end = chunkEnd(if (f.until != 0) f.until else len, f.at, rows, s.chunks, s.reuse.marks, spans, f.alone);
    try r.run(s, f.at, end);
    s.fill_at.store(@intCast(end), .monotonic);
    f.at = end;
    if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) {
        if (later and end < len) f.mark = @intCast(end) else r.keep(s, @intCast(end));
    }
    return end == len;
}

/// Rows a tap holds: `rows` from `start`, written at generation `gen` (null: current until rewritten by the stream).
pub const Held = struct { start: u64 = 0, rows: usize = 0, gen: ?u64 = null };

/// Whether `h` holds [start, start + count) and nothing has overwritten it since (its generation is `now`).
pub fn holds(h: Held, now: u64, start: u64, count: u32) bool {
    if (h.gen) |g| if (g != now) return false;
    return count > 0 and start >= h.start and start + count <= h.start + h.rows;
}

test {
    _ = @import("prefill_steps_test.zig");
}
