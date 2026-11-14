//! LFM3 Metal's prompt pass a chunk a call (#3021): today's cuts, one chunk a step, marks kept as each chunk lands.
const std = @import("std");
const lanes = @import("lanes");
const control = @import("prefill_control.zig");
const Stream = lanes.Stream;

/// A chunk from `at`: `rows` at most, cut at the next planned chunk start and the next kept mark (main's loop body).
pub fn chunkEnd(len: usize, at: usize, rows: usize, chunks: []const u32, marks: []const u32) usize {
    var end: usize = @min(len, at + rows);
    for (chunks) |boundary| if (boundary > at) {
        end = @min(end, boundary);
        break;
    };
    // main carried `mark` across chunks; it only skips marks at or before `at`, so a fresh scan stops there too
    var mark: usize = 0;
    while (mark < marks.len and marks[mark] <= at) mark += 1;
    if (mark < marks.len) end = @min(end, marks[mark]);
    return end;
}

/// One chunk from `at` through `r` (run, keep); true once all in. `whole` asks for a cancel or yield, a step a cancel.
pub fn step(r: anytype, s: *Stream, at: *usize, rows: usize, whole: bool) !bool {
    if (whole) try control.before(s, at.*) else if (s.isCancelled()) {
        s.keepInterruptedPrefill(@intCast(at.*));
        return error.Cancelled;
    }
    const len = s.prompt().len;
    const end = chunkEnd(len, at.*, rows, s.chunks, s.reuse.marks);
    try r.run(s, at.*, end);
    at.* = end;
    control.completed(s, end); // the forward waits on its command buffer: these rows are in
    if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) r.keep(s, @intCast(end));
    return end == len;
}

/// The host cancels between chunks: the state at the pass's last chunk end is kept (no prompt work runs).
pub fn cancel(s: *Stream, at: ?usize) void {
    if (at) |x| s.keepInterruptedPrefill(@intCast(x));
}

test {
    _ = @import("prefill_steps_test.zig");
}
