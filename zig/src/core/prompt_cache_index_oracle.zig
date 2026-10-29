//! The whole-store lookup the snapshot index replaced, and the index's ownership checks, for its tests.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

/// Store.find before the index, line for line: every entry in insertion order, a later one only when longer.
pub fn find(s: *const pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*pc.Entry {
    var best: ?*pc.Entry = null;
    for (s.entries.items) |e| {
        if (e.expired or e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
        if (best != null and e.at <= best.?.at) continue;
        const n = e.tokens.len;
        if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
        if (!modes.equal(e.decode_spans, spans, e.at)) continue;
        best = e;
    }
    return best;
}

/// Every owned entry occurs once in the index, whose order preserves descending positions and stable ties.
pub fn check(s: *const pc.Store) !void {
    try std.testing.expectEqual(s.entries.items.len, s.by_position.items.len);
    var held: u64 = 0;
    for (s.entries.items) |e| {
        var count: usize = 0;
        for (s.by_position.items) |indexed| count += @intFromBool(indexed == e);
        try std.testing.expectEqual(@as(usize, 1), count);
        held += e.bytes;
    }
    try std.testing.expectEqual(held, s.held);
    for (s.by_position.items, 0..) |e, i| {
        const owner = std.mem.indexOfScalar(*pc.Entry, s.entries.items, e) orelse return error.UnownedIndexEntry;
        if (i == 0) continue;
        const previous = s.by_position.items[i - 1];
        try std.testing.expect(previous.at >= e.at);
        if (previous.at == e.at) {
            const before = std.mem.indexOfScalar(*pc.Entry, s.entries.items, previous).?;
            try std.testing.expect(before < owner);
        }
    }
}
