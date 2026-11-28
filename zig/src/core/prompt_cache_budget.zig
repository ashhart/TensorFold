//! One affordable earlier checkpoint when ordinary admission cannot keep the requested endpoint.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const cuts = @import("prompt_cuts.zig");
const reserve = @import("prompt_cache_reserve.zig");
const modes = @import("cache_modes.zig");

pub const Query = struct {
    prompt: []const u32,
    born: u32,
    from: u32,
    endpoint: u32,
    requested: bool,
    starts: []const u32,
    spans: []const modes.Span,
    restored: ?*pc.Entry,
};

/// Greatest existing cold cut at or before limit; no mark may introduce another chunk boundary.
pub fn cold(r: pc.Rules, len: usize, from: u32, limit: u32, starts: []const u32, spans: []const modes.Span, marks: []const u32) u32 {
    if (limit <= from or limit >= len) return 0;
    if (r.planned) return if (starts.len == 0 and r.grid > 0) limit - limit % r.grid else cuts.floorStart(starts, limit);
    if (!r.chunk_marks) return if (r.chunk_starts) cuts.floorStart(starts, limit) else 0;
    if (r.chunk == 0) return 0;
    var anchor = from;
    for (marks) |at| if (at > anchor and at <= limit) {
        anchor = at;
    };
    return cuts.chunkEnd(r.chunk, anchor, limit, len, if (r.chunk_starts) starts else &.{}, if (r.chunk_starts) spans else &.{});
}

/// Takes marks on success only; ordinary admitted marks and a fitting endpoint keep their existing behavior.
pub fn add(s: *pc.Store, a: std.mem.Allocator, marks: []const u32, q: Query) ![]const u32 {
    if (s.hold or !q.requested or q.endpoint <= q.from or q.endpoint >= q.prompt.len or
        std.mem.indexOfScalar(u32, marks, q.endpoint) != null) return marks;
    const room = reserve.fallbackRoom(s, q.prompt, q.born, q.restored, marks, q.spans);
    if (room.slots == 0) return marks;
    var deepest = q.from;
    for (marks) |at| deepest = @max(deepest, at);
    const keyed_end: u32 = @intCast(q.prompt.len -| @max(1, s.rules.lookahead));
    var limit = @min(q.endpoint - 1, keyed_end);
    while (limit > deepest) {
        const at = cold(s.rules, q.prompt.len, q.from, limit, q.starts, q.spans, marks);
        if (at <= deepest) return marks;
        // State sizes need not be monotone: visit actual cuts from greatest to least, not byte-derived rows.
        if (s.usable(at, q.starts) and s.family.vtable.bytes(s.family.ptr, at) <= room.bytes) {
            const out = try a.alloc(u32, marks.len + 1);
            @memcpy(out[0..marks.len], marks);
            out[marks.len] = at;
            std.mem.sort(u32, out, {}, std.sort.asc(u32));
            reserve.reserveFallback(s, q.prompt, q.born, q.restored, out, q.spans);
            const fitted = try reserve.fitting(s, a, q.prompt, out, q.spans, q.restored, q.born, at);
            a.free(marks);
            return fitted;
        }
        limit = at - 1;
    }
    return marks;
}
