//! Prompt reuse and capture hooks around the unchanged Qwen forward, also exercised with a host reference.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");

pub fn run(cache: *st.Cache, stream: *lanes.Stream, chunk: usize, ctx: anytype, comptime advance: anytype) !void {
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
    var mark: usize = 0;
    while (at < ids.len) {
        if (stream.isCancelled()) {
            stream.keepInterruptedPrefill(@intCast(at));
            return error.Cancelled;
        }
        var end = @min(ids.len, at + chunk);
        for (stream.chunks) |boundary| if (boundary > at) {
            end = @min(end, boundary);
            break;
        };
        while (mark < stream.reuse.marks.len and stream.reuse.marks[mark] <= at) mark += 1;
        if (mark < stream.reuse.marks.len) end = @min(end, stream.reuse.marks[mark]);
        try advance(ctx, cache, ids[at..end]);
        at = end;
        stream.fill_at.store(@intCast(at), .monotonic);
        if (stream.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(at)) != null)
            hook.at(hook.ptr, stream, @intCast(at));
    }
}
