//! Snapshot eviction order, excluding entries borrowed by unfinished prompt passes.
const std = @import("std");
const pc = @import("prompt_cache.zig");

pub fn victimBut(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or isActiveRewind(s, e, prompt)) continue;
        const now = born > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // a longer request's tokens extend it
        if ((now or extended(s, e)) and (best == null or e.used < s.entries.items[best.?].used)) best = i;
    }
    if (best) |i| return i;
    var oldest: ?usize = null;
    for (s.entries.items, 0..) |e, i| if (e.borrowers == 0 and e != skip and (oldest == null or before(s, e, s.entries.items[oldest.?], prompt))) {
        oldest = i;
    };
    return oldest;
}

/// Only live continuations make a prefix a branch point or a superseded state.
pub fn extended(s: *const pc.Store, e: *const pc.Entry) bool {
    for (s.entries.items) |o| if (!o.expired and o != e and o.born > e.born and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

fn before(s: *const pc.Store, e: *const pc.Entry, o: *const pc.Entry, prompt: []const u32) bool {
    const e_rewind = isActiveRewind(s, e, prompt);
    const o_rewind = isActiveRewind(s, o, prompt);
    return if (e_rewind != o_rewind) o_rewind else e.used < o.used;
}

fn isActiveRewind(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32) bool {
    return !e.expired and e.at == s.active_rewind_at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
}
