//! Snapshot eviction order, excluding entries borrowed by unfinished prompt passes.
const std = @import("std");
const pc = @import("prompt_cache.zig");

pub fn victimBut(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    return victimIn(s, skip, prompt, born, false);
}

/// `victimBut`'s order over every state, or with `own_only` over the prompt's own conversation's states alone.
pub fn victimIn(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32, own_only: bool) ?usize {
    var best: ?usize = null;
    var rung: ?usize = null; // a superseded ladder checkpoint: after every other redundant copy, before any newest state
    var oldest: ?usize = null;
    var oldest_rewind = false;
    for (s.entries.items, 0..) |e, i| { // one prefix compare an entry: the superseded scan and the oldest scan share it
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |g| if (g.protects(s, e)) continue;
        if (own_only and !own(e, skip, prompt)) continue;
        const on = e.tokens.len <= prompt.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
        const rewind = !e.expired and e.at == s.active_rewind_at and on; // the active rewind state goes last
        if (e.borrowers == 0 and e != skip and (oldest == null or (if (rewind != oldest_rewind) oldest_rewind else e.used < s.entries.items[oldest.?].used))) {
            oldest = i;
            oldest_rewind = rewind;
        }
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or rewind) continue;
        const now = (born > e.born or (born == e.born and !e.rewind)) and prompt.len > e.tokens.len and on;
        if (!now and !extended(s, e)) continue;
        const pick = if (e.intermediate) &rung else &best;
        if (pick.* == null or e.used < s.entries.items[pick.*.?].used) pick.* = i;
    }
    return best orelse rung orelse oldest;
}

/// A reservation may free redundant copies, but not a newest state just to plan more copies.
pub fn redundant(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32, born: u32) bool {
    if (e.expired) return true;
    if (e.shared or e.branched) return false;
    const on = e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
    if (e.at == s.active_rewind_at and on) return false;
    const now = (born > e.born or (born == e.born and !e.rewind)) and prompt.len > e.tokens.len and on;
    return now or extended(s, e);
}

/// A state of the prompt's own conversation that may go: unborrowed, on its token path, neither shared nor a branch.
pub fn own(e: *const pc.Entry, skip: ?*pc.Entry, prompt: []const u32) bool {
    if (e.borrowers > 0 or e == skip or e.shared or e.branched or e.tokens.len > prompt.len) return false;
    return std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
}

/// Only live continuations make a prefix a branch point or a superseded state.
pub fn extended(s: *const pc.Store, e: *const pc.Entry) bool {
    if (s.links_lost) return extendedScan(s, e);
    for (e.ext.items) |o| if (!o.expired and (!e.rewind or o.born > e.born)) return true;
    return false;
}

/// A real later turn, excluding an endpoint's equal-born reply end, can establish a branch.
pub fn continuedLater(s: *const pc.Store, e: *const pc.Entry) bool {
    if (!s.links_lost) {
        for (e.ext.items) |o| if (!o.expired and o.born > e.born) return true;
        return false;
    }
    for (s.entries.items) |o| if (!o.expired and o.born > e.born and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

/// `extended` over every entry, for a store whose links an allocation failure left incomplete.
pub fn extendedScan(s: *const pc.Store, e: *const pc.Entry) bool {
    for (s.entries.items) |o| if (!o.expired and o != e and (o.born > e.born or (o.born == e.born and !e.rewind)) and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

/// A new entry linked to those it extends and those extending it, no earlier born, including one request's reply end.
pub fn link(s: *pc.Store, x: *pc.Entry) void {
    for (s.entries.items) |o| {
        if (o == x or o.tokens.len == x.tokens.len) continue;
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
