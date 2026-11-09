//! Protect queued resumptions, retain the next endpoint, and trim optional copies to the remaining room.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const others = @import("prompt_cache_others.zig");
const policy = @import("prompt_cache_policy.zig");
pub const Guard = struct { ctx: *anyopaque, needed: *const fn (*anyopaque, *const pc.Entry) bool };

fn exists(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) bool {
    const n = @as(usize, at) + s.rules.lookahead;
    for (s.entries.items) |e| {
        if (!e.expired and e.at == at and e.tokens.len == n and n <= prompt.len and
            std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e))
            return true;
    }
    return false;
}

const Room = struct { bytes: u64, slots: usize };

/// The restored entry's borrow ends before keeping; superseded copies can go at that same safe boundary.
fn future(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, born: u32) Room {
    var room: Room = .{ .bytes = s.budget -| s.held, .slots = s.most -| s.entries.items.len };
    for (s.entries.items) |e| {
        if (e == from) {
            room.bytes += e.bytes;
            room.slots +|= 1;
            continue;
        }
        if (e.borrowers > 0) continue;
        if (s.reservation_guard) |g| if (g.needed(g.ctx, e)) continue;
        const on = e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
        if (on and policy.redundant(s, e, prompt, born)) {
            room.bytes += e.bytes;
            room.slots +|= 1;
        }
    }
    return room;
}

/// The primary history mark follows the same warm/lookahead and short-prompt cuts as marksRewind.
pub fn endpointFor(s: *const pc.Store, prompt: []const u32, from: u32, history: u32, starts: []const u32, spans: []const modes.Span) u32 {
    var at = history;
    if (s.rules.warm != .off and history == prompt.len and s.rules.lookahead > 0)
        at = @intCast(prompt.len -| s.rules.lookahead);
    if (!s.rules.planned and prompt.len < s.rules.min_prompt and s.rules.chunk > 0)
        return @import("prompt_cuts.zig").chunkEnd(s.rules.chunk, from, at, prompt.len, starts, spans);
    if (history == 0 and s.rules.tail) at = @intCast(prompt.len -| @max(1, s.rules.lookahead));
    if (!s.rules.planned) return at;
    if (starts.len == 0 and s.rules.grid > 0) return at - at % s.rules.grid;
    return @import("prompt_cuts.zig").floorStart(starts, at);
}

fn priority(marks: []const u32, endpoint: u32) ?u32 {
    if (std.mem.indexOfScalar(u32, marks, endpoint) != null) return endpoint;
    return null;
}

fn unprotected(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, bytes: u64) ?usize {
    if (s.keep_others and !others.fits(s, from, prompt, bytes, 1)) return null;
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e == from or e.borrowers > 0) continue;
        if (s.reservation_guard) |g| if (g.needed(g.ctx, e)) continue;
        if (s.keep_others and !policy.own(e, from, prompt)) continue;
        if (best == null or e.used < s.entries.items[best.?].used) best = i;
    }
    return best;
}

/// A guarded reservation makes room for the endpoint alone; other copies cannot displace a newest state.
pub fn reserve(s: *pc.Store, prompt: []const u32, born: u32, from: ?*pc.Entry, marks: []const u32, spans: []const modes.Span, endpoint: u32) void {
    if (s.reservation_guard != null) {
        const at = priority(marks, endpoint) orelse return;
        if (exists(s, prompt, at, spans)) return;
        const need = s.family.vtable.bytes(s.family.ptr, at);
        if (need > s.budget or s.most == 0) return;
        while (true) {
            const room = future(s, from, prompt, born);
            if (need <= room.bytes and room.slots > 0) return;
            const i = unprotected(s, from, prompt, need) orelse return;
            s.remove(i);
            s.counts.evicted += 1;
        }
    }
    var need: u64 = 0;
    var new: usize = 0;
    for (marks) |at| if (!exists(s, prompt, at, spans)) {
        need += s.family.vtable.bytes(s.family.ptr, at);
        new += 1;
    };
    while (s.held + need > s.budget or s.entries.items.len + @min(new, s.most) > s.most) {
        const i = others.victim(s, from, prompt, born, need, @min(new, s.most)) orelse return;
        s.remove(i);
        s.counts.evicted += 1;
    }
}

/// Select the endpoint first; only interior, rewind and extra shared cuts compete for room after it.
pub fn fitting(s: *pc.Store, a: std.mem.Allocator, prompt: []const u32, marks: []const u32, spans: []const modes.Span, from: ?*pc.Entry, born: u32, endpoint: u32) ![]const u32 {
    if (s.reservation_guard == null) return marks;
    defer a.free(marks);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var room = future(s, from, prompt, born);
    const end = priority(marks, endpoint);
    if (end) |at| {
        const bytes = if (exists(s, prompt, at, spans)) 0 else s.family.vtable.bytes(s.family.ptr, at);
        if (bytes == 0 or (bytes <= room.bytes and room.slots > 0)) {
            try out.append(a, at);
            room.bytes -= bytes;
            if (bytes > 0) room.slots -= 1;
        }
    }
    for (marks) |at| {
        if (end == at) continue;
        const bytes = if (exists(s, prompt, at, spans)) 0 else s.family.vtable.bytes(s.family.ptr, at);
        if (bytes == 0 or (bytes <= room.bytes and room.slots > 0)) {
            try out.append(a, at);
            room.bytes -= bytes;
            if (bytes > 0) room.slots -= 1;
        }
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    return out.toOwnedSlice(a);
}
