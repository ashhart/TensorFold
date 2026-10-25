//! Snapshot eviction order, excluding entries borrowed by unfinished prompt passes.
const std = @import("std");
const pc = @import("prompt_cache.zig");

pub fn victimBut(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    var best: ?usize = null;
    var oldest: ?usize = null;
    var oldest_rewind = false;
    for (s.entries.items, 0..) |e, i| { // one prefix compare an entry: the superseded scan and the oldest scan share it
        const on = e.tokens.len <= prompt.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
        const rewind = !e.expired and e.at == s.active_rewind_at and on; // the active rewind state goes last
        if (e.borrowers == 0 and e != skip and (oldest == null or (if (rewind != oldest_rewind) oldest_rewind else e.used < s.entries.items[oldest.?].used))) {
            oldest = i;
            oldest_rewind = rewind;
        }
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or rewind) continue;
        const now = born > e.born and prompt.len > e.tokens.len and on; // a longer request's tokens extend it
        if ((now or extended(s, e)) and (best == null or e.used < s.entries.items[best.?].used)) best = i;
    }
    return best orelse oldest;
}

/// Only live continuations make a prefix a branch point or a superseded state.
pub fn extended(s: *const pc.Store, e: *const pc.Entry) bool {
    if (s.links_lost) return extendedScan(s, e);
    for (e.ext.items) |o| if (!o.expired) return true;
    return false;
}

/// `extended` over every entry, for a store whose links an allocation failure left incomplete.
pub fn extendedScan(s: *const pc.Store, e: *const pc.Entry) bool {
    for (s.entries.items) |o| if (!o.expired and o != e and o.born > e.born and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

/// A new entry linked to those it extends and those extending it, later born (tokens and born never change).
pub fn link(s: *pc.Store, x: *pc.Entry) void {
    for (s.entries.items) |o| {
        if (o == x or o.born == x.born or o.tokens.len == x.tokens.len) continue;
        const short, const long = if (o.tokens.len < x.tokens.len) .{ o, x } else .{ x, o };
        if (long.born < short.born or !std.mem.eql(u32, long.tokens[0..short.tokens.len], short.tokens)) continue;
        short.ext.append(s.gpa, long) catch {
            s.links_lost = true;
        };
    }
}

/// An entry leaving the store leaves every link to it.
pub fn unlink(s: *pc.Store, x: *pc.Entry) void {
    for (s.entries.items) |o| if (std.mem.indexOfScalar(*pc.Entry, o.ext.items, x)) |i| {
        _ = o.ext.swapRemove(i);
    };
}
