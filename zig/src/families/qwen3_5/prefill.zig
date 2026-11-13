//! Prompt reuse and capture hooks around the unchanged Qwen forward: a pass whole, or a chunk a call (#3021).
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");

pub fn run(cache: *st.Cache, stream: *lanes.Stream, chunk: usize, ctx: anytype, comptime advance: anytype) !void {
    var at = open(cache, stream);
    while (!try step(cache, stream, &at, chunk, ctx, advance)) {}
}

/// The pass's first row: a kept state of the prompt's prefix restored into `cache`, or 0.
pub fn open(cache: *st.Cache, stream: *lanes.Stream) usize {
    const ids = stream.prompt();
    var at: usize = 0;
    stream.cached = 0;
    if (stream.reuse.saved) |saved| {
        const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
        if (snap.at <= ids.len) {
            if (snapshot.restore(cache, snap)) |_| {
                at = snap.at;
            } else |err| stream.reuse_error = err;
        } else stream.reuse_error = error.QwenSnapshotPosition;
        stream.cached = @intCast(at);
        stream.reuse_failed = at == 0;
    }
    stream.fill_at.store(@intCast(at), .monotonic);
    return at;
}

/// One chunk from `at` through `advance`, at the whole pass's cut; true once the prompt is in (at once if it is).
pub fn step(cache: *st.Cache, stream: *lanes.Stream, at: *usize, chunk: usize, ctx: anytype, comptime advance: anytype) !bool {
    const ids = stream.prompt();
    if (at.* >= ids.len) return true;
    if (stream.isCancelled()) {
        stream.keepInterruptedPrefill(@intCast(at.*));
        return error.Cancelled;
    }
    const end = chunkEnd(ids.len, at.*, chunk, stream.chunks, stream.reuse.marks);
    try advance(ctx, cache, ids[at.*..end]);
    lanes.prefill_widths.completed(ctx.gpa, stream, end - at.*);
    at.* = end;
    stream.fill_at.store(@intCast(end), .monotonic);
    if (stream.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null)
        hook.at(hook.ptr, stream, @intCast(end));
    return end == ids.len;
}

/// A chunk from `at`: `chunk` rows at most, cut at the next planned chunk start and the next kept mark.
pub fn chunkEnd(len: usize, at: usize, chunk: usize, chunks: []const u32, marks: []const u32) usize {
    var end = @min(len, at + chunk);
    for (chunks) |boundary| if (boundary > at) {
        end = @min(end, boundary);
        break;
    };
    // the whole loop carried `mark`; it skipped only marks at or before `at`, so a fresh scan stops there too
    var mark: usize = 0;
    while (mark < marks.len and marks[mark] <= at) mark += 1;
    if (mark < marks.len) end = @min(end, marks[mark]);
    return end;
}
