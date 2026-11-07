//! A nonowning snapshot index: longest positions first, equal positions in the owner's insertion order.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

/// The longest kept token prefix with matching canonical spans through its position, leaving at least one row.
pub fn find(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*pc.Entry {
    std.debug.assert(s.by_position.items.len == s.entries.items.len); // add and Store.remove keep them equal
    for (s.by_position.items[below(s, prompt.len)..]) |e| { // longest first: the first match is the scan's winner
        if (e.expired or e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
        // A tail state off the grid resumes with its own chunk.
        if (!s.aligned(e.at, starts) and prompt.len - e.at <= 8) continue;
        // An off-grid tail state resumes its suffix as ONE chunk.
        if (!s.aligned(e.at, starts) and s.rules.grid > 0 and prompt.len - e.at > s.rules.grid) continue;
        // And its last chunk must keep the class a fresh pass gives the same rows.
        if (s.rules.end_mark and !s.aligned(e.at, starts) and s.rules.grid > 0) {
            const suffix: usize = prompt.len - e.at;
            if (pc.chunkClass(suffix) != pc.chunkClass(pc.freshLast(s.rules.grid, prompt.len))) continue;
        }
        const n = e.tokens.len;
        if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
        if (!modes.equal(e.decode_spans, spans, e.at) or !@import("cache_images.zig").matches(s, e)) continue;
        return e;
    }
    return null;
}

/// Publish in both containers only after both allocations succeed; the caller owns `entry` on failure.
pub fn add(s: *pc.Store, entry: *pc.Entry) !void {
    try s.entries.ensureUnusedCapacity(s.gpa, 1);
    try s.by_position.ensureUnusedCapacity(s.gpa, 1);
    s.entries.appendAssumeCapacity(entry);
    s.by_position.insertAssumeCapacity(below(s, entry.at), entry); // after every state as long: insertion order
}

/// Remove the nonowning pointer before its owner can be freed; no allocation or backend operation.
pub fn remove(s: *pc.Store, entry: *pc.Entry) void {
    const at = std.mem.indexOfScalar(*pc.Entry, s.by_position.items, entry) orelse unreachable;
    _ = s.by_position.orderedRemove(at);
}

/// The first index whose state is shorter than `at` rows: every state ahead of it is at least that long.
fn below(s: *const pc.Store, at: usize) usize {
    var lo: usize = 0;
    var hi = s.by_position.items.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (s.by_position.items[mid].at >= at) lo = mid + 1 else hi = mid;
    }
    return lo;
}
