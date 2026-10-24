//! The value entity of a note written in a running server: the entity its text names, for the subject+value hop.
const std = @import("std");
const index = @import("index.zig");
const model_mod = @import("model.zig");
const Allocator = std.mem.Allocator;

/// The index's value entities with the written notes' filled in (a snapshot's own stay as LLWZ exported them).
pub fn ofWrites(a: Allocator, idx: *const index.Index, m: *const model_mod.Model) Allocator.Error![]const i32 {
    var out: ?[]i32 = null;
    for (m.notes.items, 0..) |*n, k| {
        if (n.source() == null or idx.own_value[k] >= 0) continue;
        const e = named(idx, n.tokens(), n.entity) orelse continue;
        if (out == null) out = try a.dupe(i32, idx.own_value);
        out.?[k] = @intCast(e);
    }
    return out orelse idx.own_value;
}

/// The first entity other than `own` that `tokens` name, longest name first: a phrase of 2 to 8 keys, or one key.
pub fn named(idx: *const index.Index, tokens: []const u32, own: u32) ?u32 {
    var keys: [index.max_phrase]u32 = undefined;
    var n: usize = @min(tokens.len, @max(idx.phraseMax(), 1));
    while (n >= 1) : (n -= 1) {
        var j: usize = 0;
        while (j + n <= tokens.len) : (j += 1) {
            for (tokens[j..][0..n], keys[0..n]) |t, *k| k.* = idx.keyOf(t);
            const e = if (n == 1) idx.nameOf(keys[0]) else idx.phraseOf(index.packPhrase(keys[0..n]));
            if (e) |x| if (x != own) return x;
        }
    }
    return null;
}
