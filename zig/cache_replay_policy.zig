//! Replay-only eviction alternatives; the native selector is called directly and production files stay unchanged.
const std = @import("std");
const pc = @import("cache");
pub const Policy = enum { main, saved_seconds_per_byte, recency, frequency_probation, decayed_resume };
pub fn victim(s: *const pc.Store, p: Policy, skip: ?*pc.Entry, prompt: []const u32) ?usize {
    if (p == .main) return s.victimMain(skip, prompt, @intCast(prompt.len));
    if (p == .decayed_resume) return @import("cache_replay_decay.zig").victim(s, skip, prompt);
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |guard| if (guard.protects(s, e)) continue;
        if (best == null or before(e, s.entries.items[best.?], p, s.clock)) best = i;
    }
    return best;
}
fn tier(e: *const pc.Entry, clock: u64) u2 {
    if (clock -| e.used > 12 or e.seen.hits == 0) return 0;
    return if (e.seen.hits >= 2) 2 else 1;
}
fn before(a: *const pc.Entry, b: *const pc.Entry, p: Policy, clock: u64) bool {
    if (p == .frequency_probation and tier(a, clock) != tier(b, clock)) return tier(a, clock) < tier(b, clock);
    if (p == .saved_seconds_per_byte or p == .frequency_probation) {
        const aw: u64 = if (p == .frequency_probation) 1 + @min(a.seen.hits, 8) else 1;
        const bw: u64 = if (p == .frequency_probation) 1 + @min(b.seen.hits, 8) else 1;
        const left = @as(u128, a.at) * aw * b.bytes;
        const right = @as(u128, b.at) * bw * a.bytes;
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
pub fn begin(s: *pc.Store, p: Policy, a: std.mem.Allocator, prompt: []const u32, history: u32, owner: *anyopaque, shared: []const u32) !pc.Plan {
    const e = s.find(prompt, &.{}, &.{});
    if (e == null) s.counts.misses += 1;
    const from = if (e) |entry| entry.at else 0;
    const planned = try s.marks(a, prompt, from, history, shared, &.{}, if (e) |entry| entry.last else &.{});
    defer a.free(planned);
    var fitted: std.ArrayList(u32) = .empty;
    errdefer fitted.deinit(a);
    for (planned) |at| {
        if (s.family.vtable.bytes(s.family.ptr, at) <= s.budget) try fitted.append(a, at) else s.counts.refused += 1;
    }
    for (shared) |at| if (std.mem.indexOfScalar(u32, fitted.items, at) != null) {
        const key = pc.Store.sharedKey(prompt[0..at]);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, key) == null) {
            if (s.shared_keys.items.len == 64) _ = s.shared_keys.orderedRemove(0);
            try s.shared_keys.append(a, key);
        }
    };
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
