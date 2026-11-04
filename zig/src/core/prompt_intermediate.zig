//! A bounded ladder of intermediate checkpoints under the history: a later, compacted prompt resumes inside its prefix.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const Allocator = std.mem.Allocator;

/// Whether `at` belongs to the ladder: a power-of-two multiple of its spacing, under the prompt's history.
pub fn is(r: pc.Rules, prompt_len: usize, at: u32) bool {
    if (r.intermediate == 0 or r.intermediate_most == 0 or prompt_len < r.intermediate_min or at < r.intermediate_min) return false;
    if (at >= prompt_len or at % r.intermediate != 0) return false;
    return std.math.isPowerOfTwo(at / r.intermediate);
}

/// The largest `intermediate_most` ladder positions below `history`, ascending, away from `from` and `marks`.
pub fn cuts(a: Allocator, r: pc.Rules, prompt_len: usize, history: u32, from: u32, marks: []const u32) ![]u32 {
    if (r.intermediate == 0 or r.intermediate_most == 0 or prompt_len < r.intermediate_min or history == 0) return a.alloc(u32, 0);
    var all: std.ArrayList(u32) = .empty;
    defer all.deinit(a);
    var at = r.intermediate;
    while (at < history) {
        const outside = at < r.intermediate_min or at <= from or at + r.lookahead > prompt_len or history - at < r.min_gap;
        var skip = outside;
        if (!skip) skip = std.mem.indexOfScalar(u32, marks, at) != null;
        if (!skip) skip = for (marks) |m| {
            if (@max(m, at) - @min(m, at) < r.min_gap) break true;
        } else false;
        if (!skip) try all.append(a, at);
        if (at > history / 2) break;
        at *= 2;
    }
    if (all.items.len <= @as(usize, r.intermediate_most)) return all.toOwnedSlice(a);
    const first = all.items.len - r.intermediate_most;
    const out = try a.dupe(u32, all.items[first..]);
    return out;
}

/// Append the ladder's marks that fit beside the endpoint, rewind and existing marks; a new slice in `a`.
pub fn add(
    s: *pc.Store,
    a: Allocator,
    prompt: []const u32,
    from: u32,
    history: u32,
    starts: []const u32,
    spans: []const modes.Span,
    marks: []const u32,
    endpoint: u32,
    endpoint_bytes: u64,
    restored: ?*pc.Entry,
) ![]const u32 {
    // The endpoint keeps its priority: no ladder beside a state that does not fit.
    if (endpoint_bytes > 0 and endpoint_bytes > s.budget) return marks;
    const candidates = try cuts(a, s.rules, prompt.len, history, from, marks);
    defer a.free(candidates);
    if (candidates.len == 0) return marks;
    var required: std.ArrayList(u32) = .empty;
    defer required.deinit(a);
    try required.appendSlice(a, marks);
    for ([_]u32{ if (endpoint_bytes > 0 and endpoint_bytes <= s.budget) endpoint else 0, s.active_rewind_at }) |p| {
        if (p > 0 and std.mem.indexOfScalar(u32, required.items, p) == null) try required.append(a, p);
    }
    var used: u64 = 0;
    var slots: usize = required.items.len;
    for (required.items) |p| {
        var bytes = s.family.vtable.bytes(s.family.ptr, p);
        for (s.entries.items) |e| if (same(e, prompt, p, s.rules.lookahead, spans)) {
            bytes = @max(bytes, e.bytes);
            break;
        };
        used +|= bytes;
    }
    // Other in-flight restores cannot be evicted. The current restore has not been borrowed by lookup yet.
    for (s.entries.items) |e| if (e.borrowers > 0 or e == restored) {
        const included = for (required.items) |p| {
            if (same(e, prompt, p, s.rules.lookahead, spans)) break true;
        } else false;
        if (!included) {
            used +|= e.bytes;
            slots +|= 1;
        }
    };
    // A family without trim owns its spare storage until it reuses it; conservative admission leaves it charged.
    if (s.family.vtable.trim == null) used +|= s.spare();
    var added: usize = 0;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    try out.appendSlice(a, marks);
    for (candidates) |at| {
        if (added == @as(usize, s.rules.intermediate_most) or slots >= s.most) break;
        if (!s.usable(at, starts)) continue;
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (used > s.budget or bytes > s.budget - used) continue;
        try out.append(a, at);
        used +|= bytes;
        slots +|= 1;
        added += 1;
    }
    if (added == 0) return marks;
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    const kept = try out.toOwnedSlice(a);
    a.free(marks);
    return kept;
}

fn same(e: *const pc.Entry, prompt: []const u32, at: u32, lookahead: u32, spans: []const modes.Span) bool {
    return !e.expired and e.at == at and e.tokens.len == @as(usize, at) + lookahead and
        e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) and
        modes.equal(e.decode_spans, spans, at);
}
