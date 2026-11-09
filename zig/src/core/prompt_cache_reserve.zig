//! Pre-pass reservations leave queued resumptions intact and trim new marks when protected states fill the budget.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const others = @import("prompt_cache_others.zig");
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

/// Room made before a pass, so a keep-mirroring peer hears its evictions with the request.
pub fn reserve(s: *pc.Store, prompt: []const u32, born: u32, from: ?*pc.Entry, marks: []const u32, spans: []const modes.Span) void {
    var need: u64 = 0;
    var new: usize = 0;
    for (marks) |at| if (!exists(s, prompt, at, spans)) {
        need += s.family.vtable.bytes(s.family.ptr, at);
        new += 1;
    };
    while (s.held + need > s.budget or s.entries.items.len + @min(new, s.most) > s.most) {
        const i = others.victim(s, from, prompt, born, need, @min(new, s.most)) orelse return;
        if (s.reservation_guard != null and !@import("prompt_cache_policy.zig").redundant(s, s.entries.items[i], prompt, born)) return;
        s.remove(i);
        s.counts.evicted += 1;
    }
}

/// With queued resumptions protected, a pass keeps only missing marks that fit beside what remains.
pub fn fitting(s: *pc.Store, a: std.mem.Allocator, prompt: []const u32, marks: []const u32, spans: []const modes.Span) ![]const u32 {
    if (s.reservation_guard == null) return marks;
    var need: u64 = 0;
    var new: usize = 0;
    for (marks) |at| if (!exists(s, prompt, at, spans)) {
        need += s.family.vtable.bytes(s.family.ptr, at);
        new += 1;
    };
    if (need <= s.budget -| s.held and new <= s.most -| s.entries.items.len) return marks;
    defer a.free(marks);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var room = s.budget -| s.held;
    var slots = s.most -| s.entries.items.len;
    for (marks) |at| {
        if (exists(s, prompt, at, spans)) {
            try out.append(a, at);
            continue;
        }
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes <= room and slots > 0) {
            try out.append(a, at);
            room -= bytes;
            slots -= 1;
        } else s.counts.refused += 1;
    }
    return out.toOwnedSlice(a);
}
