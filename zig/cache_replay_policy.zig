//! Replay-only eviction alternatives; the native selector is called directly and production files stay unchanged.
const std = @import("std");
const pc = @import("cache");
pub const Policy = enum { main, saved_seconds_per_byte, recency };
pub fn victim(s: *const pc.Store, p: Policy, skip: ?*pc.Entry, prompt: []const u32) ?usize {
    if (p == .main) return s.victimBut(skip, prompt, @intCast(prompt.len));
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e == skip) continue;
        if (best == null or before(e, s.entries.items[best.?], p)) best = i;
    }
    return best;
}
fn before(a: *const pc.Entry, b: *const pc.Entry, p: Policy) bool {
    if (p == .saved_seconds_per_byte) {
        const left = @as(u128, a.at) * b.bytes;
        const right = @as(u128, b.at) * a.bytes;
        if (left != right) return left < right;
    }
    return a.used < b.used;
}
fn exists(s: *pc.Store, prompt: []const u32, at: u32) bool {
    for (s.entries.items) |e| if (!e.expired and e.at == at and std.mem.eql(u32, e.tokens, prompt[0..at])) return true;
    return false;
}
pub fn reserve(s: *pc.Store, p: Policy, skip: ?*pc.Entry, prompt: []const u32, marks: []const u32) void {
    var need: u64 = 0;
    for (marks) |at| if (!exists(s, prompt, at)) {
        need += s.family.vtable.bytes(s.family.ptr, at);
    };
    while (s.held + need > s.budget) {
        const i = victim(s, p, skip, prompt) orelse return;
        s.remove(i);
        s.counts.evicted += 1;
    }
}
/// Real lookup/planning and lifetime, with only reserve's victim substituted for these text-only trace rules.
pub fn begin(s: *pc.Store, p: Policy, a: std.mem.Allocator, prompt: []const u32, history: u32, owner: *anyopaque) !pc.Plan {
    const e = s.find(prompt, &.{}, &.{});
    if (e == null) s.counts.misses += 1;
    const from = if (e) |entry| entry.at else 0;
    const planned = try s.marks(a, prompt, from, history, &.{}, &.{}, if (e) |entry| entry.last else &.{});
    defer a.free(planned);
    var fitted: std.ArrayList(u32) = .empty;
    errdefer fitted.deinit(a);
    for (planned) |at| {
        if (s.family.vtable.bytes(s.family.ptr, at) <= s.budget) try fitted.append(a, at) else s.counts.refused += 1;
    }
    reserve(s, p, e, prompt, fitted.items);
    if (e) |entry| {
        entry.borrowers += 1;
        try s.family.vtable.restore(s.family.ptr, owner, entry.saved);
        s.resumed(entry, prompt, true);
        reserve(s, p, null, prompt, fitted.items);
    }
    return .{ .from = from, .marks = try fitted.toOwnedSlice(a) };
}
pub fn roomForKeep(s: *pc.Store, prompt: []const u32, at: u32) bool {
    return exists(s, prompt, at) or s.held + s.family.vtable.bytes(s.family.ptr, at) <= s.budget;
}
