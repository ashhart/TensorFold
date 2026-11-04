//! A nonowning snapshot index: longest positions first, equal positions in the owner's insertion order.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

/// The polynomial prefix hash a lookup compares before verifying a state's tokens: the same words, a 64-bit sum.
const base: u64 = 0x9E37_79B1_85EB_CA87;

pub fn prefixHash(tokens: []const u32) u64 {
    var h: u64 = 0;
    for (tokens) |t| h = h *% base +% t;
    return h;
}

/// `prompt`'s prefix hashes [0, len] in the store's reused buffer; null when it can't grow (the tokens are compared).
fn prefixHashes(s: *pc.Store, prompt: []const u32) ?[]const u64 {
    if (s.hashes.items.len < prompt.len + 1) s.hashes.resize(s.gpa, prompt.len + 1) catch return null;
    var h: u64 = 0;
    s.hashes.items[0] = 0;
    for (prompt, 0..) |t, i| {
        h = h *% base +% t;
        s.hashes.items[i + 1] = h;
    }
    return s.hashes.items[0 .. prompt.len + 1];
}

/// The longest kept token prefix with matching canonical spans through its position, leaving at least one row.
pub fn find(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*pc.Entry {
    std.debug.assert(s.by_position.items.len == s.entries.items.len); // add and Store.remove keep them equal
    const from = below(s, prompt.len);
    // the scan compares the candidates' tokens; hash only when that costs more than the prompt's hashes
    var scan: usize = 0;
    for (s.by_position.items[from..]) |e| scan += e.tokens.len;
    const hashes = if (scan > prompt.len + s.by_position.items.len - from) prefixHashes(s, prompt) else null;
    for (s.by_position.items[from..]) |e| { // longest first: the first match is the scan's winner
        if (e.expired or e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
        const n = e.tokens.len;
        if (hashes) |h| if (h[n] != e.prefix_hash) continue; // a hash difference is a token difference: a match is verified
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
    entry.prefix_hash = prefixHash(entry.tokens);
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
