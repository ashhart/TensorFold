//! States at a chat request's message starts (Request.message_starts): an edit of message N resumes from N's start.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const interior = @import("prompt_interior.zig");
const policy = @import("prompt_cache_policy.zig");
const Allocator = std.mem.Allocator;

/// The latest `message_starts` starts under `history`, past `from`, each `min_gap` from the others and the marks.
pub fn cuts(a: Allocator, r: pc.Rules, prompt_len: usize, history: u32, from: u32, messages: []const u32, marks: []const u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var i = messages.len;
    while (i > 0 and out.items.len < @as(usize, r.message_starts)) {
        i -= 1;
        const at = messages[i];
        if (at >= history or history - at < r.min_gap) continue;
        if (at <= from or at - from < r.min_gap) break; // ascending starts: every earlier one is nearer still
        if (@as(usize, at) + r.lookahead > prompt_len) continue;
        const near = for (marks) |m| {
            if (@max(m, at) - @min(m, at) < r.min_gap) break true;
        } else for (out.items) |m| {
            if (m - at < r.min_gap) break true;
        } else false;
        if (!near) try out.append(a, at);
    }
    std.mem.reverse(u32, out.items);
    return out.toOwnedSlice(a);
}

/// The rewind state sits at its message's start, role header included, so a replaced last message resumes there too.
pub fn rewind(s: *const pc.Store, rewind_len: u32) u32 {
    if (s.rules.message_starts == 0 or rewind_len == 0) return rewind_len;
    var i = s.message_starts.len;
    while (i > 0) {
        i -= 1;
        const at = s.message_starts[i];
        if (at > rewind_len) continue;
        return if (rewind_len - at < s.rules.min_gap) at else rewind_len;
    }
    return rewind_len;
}

/// Append the starts that fit in room only redundant states hold: they never displace another conversation's state.
pub fn add(
    s: *pc.Store,
    a: Allocator,
    prompt: []const u32,
    born: u32,
    from: u32,
    history: u32,
    starts: []const u32,
    spans: []const modes.Span,
    marks: []const u32,
    endpoint: u32,
    endpoint_bytes: u64,
    restored: ?*pc.Entry,
) ![]const u32 {
    if (s.rules.message_starts == 0 or s.hold or s.message_starts.len == 0) return marks;
    if (endpoint_bytes > 0 and endpoint_bytes > s.budget) return marks;
    const candidates = try cuts(a, s.rules, prompt.len, history, from, s.message_starts, marks);
    defer a.free(candidates);
    if (candidates.len == 0) return marks;
    var required: std.ArrayList(u32) = .empty;
    defer required.deinit(a);
    try required.appendSlice(a, marks);
    for ([_]u32{ if (endpoint_bytes > 0) endpoint else 0, s.active_rewind_at }) |p| {
        if (p > 0 and std.mem.indexOfScalar(u32, required.items, p) == null) try required.append(a, p);
    }
    var used: u64 = 0;
    var slots: usize = required.items.len;
    for (required.items) |p| used +|= s.family.vtable.bytes(s.family.ptr, p);
    for (s.entries.items) |e| {
        if (e.expired) continue;
        const required_ = for (required.items) |p| {
            if (interior.same(s, e, prompt, p, s.rules.lookahead, spans)) break true;
        } else false;
        if (required_) continue; // charged above at its new size
        const protected = if (s.reservation_guard) |g| g.protects(s, e) else false;
        if (e.borrowers == 0 and e != restored and !protected and policy.redundant(s, e, prompt, born)) continue;
        used +|= e.bytes;
        slots +|= 1;
    }
    // A family without trim owns its spare storage until it reuses it; conservative admission leaves it charged.
    if (s.family.vtable.trim == null) used +|= s.spare();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    try out.appendSlice(a, marks);
    var i = candidates.len; // the latest first: an edit near the end is the likelier one
    while (i > 0) {
        i -= 1;
        const at = candidates[i];
        if (slots >= s.most) break;
        if (!s.usable(at, starts)) continue;
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (used > s.budget or bytes > s.budget - used) continue;
        try out.append(a, at);
        used +|= bytes;
        slots +|= 1;
    }
    if (out.items.len == marks.len) return marks;
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    const kept = try out.toOwnedSlice(a);
    a.free(marks);
    return kept;
}
