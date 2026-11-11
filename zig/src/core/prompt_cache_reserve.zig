//! Protect queued resumptions, retain the next endpoint, and trim optional copies to the remaining room.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const others = @import("prompt_cache_others.zig");
const share = @import("prompt_cache_share.zig");
const policy = @import("prompt_cache_policy.zig");
pub const Room = struct { bytes: u64 = 0, slots: usize = 0 };
pub const Guard = struct {
    ctx: *anyopaque,
    needed: *const fn (*anyopaque, *const pc.Entry) bool,
    exact: bool = false, // true: the callback already selects each job's deepest match
    promised: Room = .{}, // net room promised to other admitted passes
    promise: ?*Room = null, // this lookup's net promise, owned by its job

    pub fn protects(g: Guard, s: *const pc.Store, e: *const pc.Entry) bool {
        if (!g.needed(g.ctx, e)) return false;
        if (g.exact) return true;
        for (s.by_position.items) |o| {
            if (o.at <= e.at) break;
            if (!o.expired and o.tokens.len >= e.tokens.len and o.image_frame == e.image_frame and
                std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens) and
                modes.equal(o.decode_spans, e.decode_spans, e.at) and g.needed(g.ctx, o)) return false;
        }
        return true;
    }
};

fn exists(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) bool {
    const n = @as(usize, at) + s.rules.lookahead;
    for (s.entries.items) |e| {
        if (!e.expired and e.at == at and e.tokens.len == n and n <= prompt.len and
            std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e))
            return true;
    }
    return false;
}

/// The current restore funds its replacement only when no other borrower or branch will still need it.
fn reclaimable(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, born: u32) Room {
    const e = from orelse return .{};
    if (e.borrowers > 0 or !policy.redundant(s, e, prompt, born)) return .{};
    if (policy.continuedLater(s, e) and !std.mem.eql(u32, prompt, e.last)) return .{};
    if (s.reservation_guard) |g| if (g.protects(s, e)) return .{};
    return .{ .bytes = e.bytes, .slots = 1 };
}

/// Bytes and slots the victim order can eventually free, excluding borrowed, protected and skipped entries.
pub fn evictable(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32) Room {
    const free = share.freeSet(s, skip, prompt, s.keep_others);
    return .{ .bytes = free.bytes, .slots = free.slots };
}

fn future(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, born: u32) Room {
    const credit = reclaimable(s, from, prompt, born);
    const promised: Room = if (s.reservation_guard) |g| g.promised else .{};
    return .{ .bytes = (s.budget -| s.held) +| credit.bytes -| promised.bytes, .slots = (s.most -| s.entries.items.len) +| credit.slots -| promised.slots };
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

/// A guarded reservation makes room for the endpoint alone; other copies cannot displace a newest state.
pub fn reserve(s: *pc.Store, prompt: []const u32, born: u32, from: ?*pc.Entry, marks: []const u32, spans: []const modes.Span, endpoint: u32) void {
    if (s.reservation_guard != null) {
        const at = priority(marks, endpoint) orelse return;
        if (exists(s, prompt, at, spans)) return;
        const whole = s.family.vtable.bytes(s.family.ptr, at);
        if (whole > s.budget or s.most == 0) return;
        const credit = reclaimable(s, from, prompt, born);
        const promised = s.reservation_guard.?.promised;
        var bytes: i128 = @as(i128, s.budget) - s.held + credit.bytes - promised.bytes;
        var slots: i128 = @as(i128, s.most) - s.entries.items.len + credit.slots - promised.slots;
        if (share.bytes(s, prompt, at, spans) <= bytes and slots > 0) return;
        const can = evictable(s, from, prompt);
        // Removing a parent may make the save whole again; refuse before eviction unless even that demand fits.
        if (bytes + can.bytes < whole or slots + can.slots < 1) return;
        while (true) {
            const need = share.bytes(s, prompt, at, spans);
            if (need <= bytes and slots > 0) return;
            const i = others.victim(s, from, prompt, born, need, 1) orelse return;
            const before = s.held;
            s.remove(i);
            bytes += before - s.held;
            slots += 1;
            s.counts.evicted += 1;
        }
    }
    while (true) {
        var need: u64 = 0;
        var new: usize = 0;
        for (marks) |at| if (!exists(s, prompt, at, spans)) {
            need +|= share.bytes(s, prompt, at, spans);
            new += 1;
        };
        if (s.held +| need <= s.budget and s.entries.items.len + @min(new, s.most) <= s.most) return;
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
        const bytes = if (exists(s, prompt, at, spans)) 0 else share.bytes(s, prompt, at, spans);
        if (bytes == 0 or (bytes <= room.bytes and room.slots > 0)) {
            try out.append(a, at);
            room.bytes -= bytes;
            if (bytes > 0) room.slots -= 1;
        }
    }
    for (marks) |at| {
        if (end == at) continue;
        const bytes = if (exists(s, prompt, at, spans)) 0 else share.bytes(s, prompt, at, spans);
        if (bytes == 0 or (bytes <= room.bytes and room.slots > 0)) {
            try out.append(a, at);
            room.bytes -= bytes;
            if (bytes > 0) room.slots -= 1;
        }
    }
    const fitted = try out.toOwnedSlice(a);
    s.counts.trimmed += marks.len - fitted.len;
    if (s.reservation_guard.?.promise) |promise| {
        var used: Room = .{};
        for (fitted) |at| if (!exists(s, prompt, at, spans)) {
            used.bytes +|= share.bytes(s, prompt, at, spans);
            used.slots +|= 1;
        };
        const credit = reclaimable(s, from, prompt, born);
        promise.* = .{ .bytes = used.bytes -| credit.bytes, .slots = used.slots -| credit.slots };
    }
    std.mem.sort(u32, fitted, {}, std.sort.asc(u32));
    return fitted;
}
