//! One owned GLM prompt pass, yielding after one bounded chunk to the shared lane scheduler.
const std = @import("std");
const Stream = @import("lanes").Stream;

/// Keep decode latency bounded when a prompt shares the GPU; solo passes retain the engine's normal chunk size.
pub const decode_rows: u32 = 256;
pub const State = struct { stream: *Stream, slot: u32, at: u32 = 0, next: usize = 0 };

/// `b` is the real backend or a fake engine at the same prepare/chunk/finish boundary.
pub fn step(b: anytype, s: *Stream) !bool {
    errdefer b.abortPrompt(s);
    if (s.isCancelled()) return error.Cancelled;
    if (!b.filling.contains(s)) {
        const begun = try b.beginPrompt(s);
        try b.filling.put(b.gpa, s, begun);
    }
    const p = b.filling.getPtr(s).?;
    const prompt = s.prompt();
    if (p.at < prompt.len) {
        while (p.next < s.chunks.len and s.chunks[p.next] <= p.at) p.next += 1;
        var end: u32 = @intCast(@min(if (p.next < s.chunks.len) s.chunks[p.next] else prompt.len, prompt.len));
        if (b.decoding()) end = @min(end, p.at + @min(decode_rows, @as(u32, @intCast(prompt.len)) - p.at));
        const n = b.promptRows(p.at, end);
        if (n == 0 or n > end - p.at) return error.ChunkOutOfStep;
        try b.chunkPrompt(s, p.slot, p.at, n);
        p.at += n;
        s.fill_at.store(p.at, .monotonic);
        if (std.mem.indexOfScalar(u32, s.reuse.marks, p.at) != null) if (s.reuse.hook) |h| h.at(h.ptr, s, p.at);
    }
    if (p.at < prompt.len) return false;
    try b.finishPrompt(s);
    _ = b.filling.remove(s);
    return true;
}
