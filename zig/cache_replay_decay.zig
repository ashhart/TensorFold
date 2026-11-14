//! Replay-only bounded resume credit; old cohorts lose priority as the store clock advances.
const std = @import("std");
const pc = @import("cache");

fn rewind(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32) bool {
    return !e.expired and e.at == s.active_rewind_at and e.tokens.len <= prompt.len and
        std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
}

fn redundant(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32) bool {
    if (e.expired) return true;
    if (e.shared or e.branched or rewind(s, e, prompt)) return false;
    const on = e.tokens.len < prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
    if ((prompt.len > e.born or (prompt.len == e.born and !e.rewind)) and on) return true;
    if (!s.links_lost) {
        for (e.ext.items) |o| if (!o.expired and (!e.rewind or o.born > e.born)) return true;
    } else {
        for (s.entries.items) |o| if (!o.expired and o != e and
            (o.born > e.born or (o.born == e.born and !e.rewind)) and o.tokens.len > e.tokens.len and
            std.mem.eql(u32, e.tokens, o.tokens[0..e.tokens.len])) return true;
    }
    return false;
}

fn age(s: *const pc.Store, e: *const pc.Entry) u64 {
    return (s.clock -| e.used) -| (@as(u64, 8) * @min(e.seen.hits, 2));
}

pub fn victim(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32) ?usize {
    const native = s.victimBut(skip, prompt, @intCast(prompt.len)) orelse return null;
    if (redundant(s, s.entries.items[native], prompt)) return native;
    var best = native;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |guard| if (guard.protects(s, e)) continue;
        const old = s.entries.items[best];
        const er = rewind(s, e, prompt);
        const br = rewind(s, old, prompt);
        if (er != br) {
            if (!er) best = i;
        } else if (age(s, e) > age(s, old) or (age(s, e) == age(s, old) and e.used < old.used)) {
            best = i;
        }
    }
    return best;
}
