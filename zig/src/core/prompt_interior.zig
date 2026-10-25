//! One optional state inside the last user message, without adding a prompt chunk or displacing its other marks.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const cuts = @import("prompt_cuts.zig");
const modes = @import("cache_modes.zig");

/// Leave this many tokens outside the snapshot's key for a replacement question and the template's closing text.
pub const QUESTION_TOKENS: u32 = 2048;
/// Short messages keep the existing endpoint/rewind policy and pay no additional copy.
pub const MIN_MESSAGE_TOKENS: u32 = 8192;

/// Latest existing cold cut; supplied turn cuts win only where the family already cuts its cold pass there.
pub fn cut(r: pc.Rules, len: usize, from: u32, end: u32, user: u32, starts: []const u32, spans: []const modes.Span, marks: []const u32) u32 {
    if (user == 0 or end > len or end <= user or end - user < MIN_MESSAGE_TOKENS) return 0;
    const limit = end -| QUESTION_TOKENS -| r.lookahead;
    var at: u32 = 0;
    if (r.planned) {
        at = if (starts.len == 0 and r.grid > 0) limit - limit % r.grid else cuts.floorStart(starts, limit);
    } else if (!r.chunk_marks) {
        // A tap bank can keep marks without splitting. Request starts end chunks in both bank and fallback paths.
        at = if (r.chunk_starts) cuts.floorStart(starts, limit) else 0;
    } else if (r.chunk > 0) {
        var anchor = from;
        for (marks) |m| if (m > anchor and m <= limit) {
            anchor = m;
        };
        at = cuts.chunkEnd(r.chunk, anchor, limit, len, if (r.chunk_starts) starts else &.{}, if (r.chunk_starts) spans else &.{});
    }
    return if (at > user and at > from and at < end and at <= limit) at else 0;
}

fn same(e: *const pc.Entry, prompt: []const u32, at: u32, lookahead: u32, spans: []const modes.Span) bool {
    return !e.expired and e.at == at and e.tokens.len == @as(usize, at) + lookahead and
        e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) and
        modes.equal(e.decode_spans, spans, at);
}

/// Takes ownership of marks only on success. Endpoint, rewind, other marks and outstanding borrows have priority.
pub fn add(s: *pc.Store, a: std.mem.Allocator, prompt: []const u32, from: u32, end: u32, user: u32, starts: []const u32, spans: []const modes.Span, marks: []const u32, endpoint: u32, endpoint_bytes: u64, restored: ?*pc.Entry) ![]const u32 {
    const at = cut(s.rules, prompt.len, from, end, user, starts, spans, marks);
    if (at == 0 or std.mem.indexOfScalar(u32, marks, at) != null) return marks;
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
    const bytes = s.family.vtable.bytes(s.family.ptr, at);
    if (used > s.budget or bytes > s.budget - used or slots >= s.most) return marks;
    const out = try a.alloc(u32, marks.len + 1);
    @memcpy(out[0..marks.len], marks);
    out[marks.len] = at;
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
    a.free(marks);
    return out;
}
