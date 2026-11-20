//! Why a reply stopped resuming: the lookup-time cause, read-only, after the store's restore choice; unallocated.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

/// The cause codes the issue names; `none` = nothing further was available to resume.
pub const Reason = enum {
    none,
    cold,
    too_short,
    prompt_changed,
    reply_changed,
    image_changed,
    frame_changed,
    rows_changed,
    ends_inside,
    off_grid,
    between_marks,
    evicted,
    shed,
    expired,
    cleared,
    not_kept,
    restore_failed,
};

/// One miss's facts: N, what resumed, the tokens lost between them, the index at N; a token change's diverging tokens.
pub const Miss = struct {
    reason: Reason,
    shared_len: u32 = 0, // N: the longest the cache's states cover on this prompt's path
    resumed_len: u32 = 0, // what this reply actually resumes
    lost: u32 = 0, // N - resumed_len
    message: u32 = 0, // the message index at N (starts)
    at: u32 = 0, // the first divergent token (prompt_changed / reply_changed)
    kept: u32 = 0, // the kept token at `at`
    request: u32 = 0, // the request token at `at`
};

/// Recently removed (or refused) states, a fixed ring: a later miss says "evicted" not "cold".
pub const Ring = struct {
    const slots = 64; // 64 slots x 48 bytes = 3 KiB a store, never allocated.
    const Record = struct { key: u64 = 0, at: u32 = 0, len: u32 = 0, cause: Reason = .cold, seq: u32 = 0 };
    next: usize = 0,
    written: u32 = 0,
    records: [slots]Record = @splat(.{}),

    pub fn keyOf(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    pub fn note(r: *Ring, tokens: []const u32, at: u32, cause: Reason) void {
        if (tokens.len == 0) return;
        r.written += 1;
        r.records[r.next] = .{ .key = keyOf(tokens), .at = at, .len = @intCast(tokens.len), .cause = cause, .seq = r.written };
        r.next = (r.next + 1) % slots;
    }

    /// The longest record this prompt's path would have matched, or null.
    pub fn held(r: *const Ring, prompt: []const u32) ?Record {
        var best: ?Record = null;
        for (r.records) |rec| {
            if (rec.key == 0 or rec.len > prompt.len) continue;
            if (keyOf(prompt[0..rec.len]) != rec.key) continue;
            if (best == null or rec.len > best.?.len or (rec.len == best.?.len and rec.seq > best.?.seq)) best = rec;
        }
        return best;
    }
};

/// The message index at `at`: the request's message starts already consumed.
fn messageAt(starts: []const u32, at: u32) u32 {
    var n: u32 = 0;
    for (starts) |st| if (st <= at) {
        n += 1;
    } else break;
    return n;
}

/// The miss (or the further-resume stop) for this lookup, derived read-only.
pub fn derive(s: *const pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, hit: ?*pc.Entry) Miss {
    if (prompt.len < s.rules.min_prompt) return .{ .reason = .too_short };
    const resumed: u32 = if (hit) |e| e.at else 0; // the entry the store chose to restore (null: nothing resumed)
    // The longest kept state on this prompt's path, and the first check it fails (find's skips, in words).
    var n: u32 = 0; // the shared length the cache holds
    var cause: ?Reason = null;
    var at: u32 = 0;
    var kept: u32 = 0;
    var request: u32 = 0;
    for (s.by_position.items) |e| { // longest first, as find walks them
        if (e.expired) {
            if (cause == null) {
                cause = .expired;
                n = @min(e.tokens.len, @as(u32, @intCast(prompt.len)));
            }
            continue;
        }
        if (e.at >= prompt.len and std.mem.startsWith(u32, e.tokens, prompt[0..@min(prompt.len, e.tokens.len)])) {
            // the request ends inside this state, before its resume point
            cause = .ends_inside;
            n = @intCast(prompt.len);
            break;
        }
        if (e.tokens.len > prompt.len or e.at >= prompt.len) continue;
        const lcp: u32 = @intCast(std.mem.indexOfDiff(u32, e.tokens, prompt) orelse @as(usize, @min(e.tokens.len, prompt.len)));
        if (lcp == 0) continue; // a foreign state: another conversation's path, never this miss's cause
        if (lcp < e.tokens.len and lcp < prompt.len) { // the paths part here
            if (lcp < e.at and lcp < lastUserStart(starts, prompt.len)) {
                cause = .prompt_changed; // an earlier message's tokens differ from what this path kept
            } else {
                cause = .reply_changed; // the earlier reply's own tokens differ from the request's history
            }
            n = lcp;
            at = lcp;
            kept = e.tokens[lcp];
            request = prompt[lcp];
            break;
        }
        // the tokens match through the state's length: the exact-reuse checks decide
        if (!s.usable(e.at, starts)) {
            cause = if (s.rules.planned and starts.len > 0) .between_marks else if (s.rules.planned) .off_grid else cause;
            n = e.at;
            break;
        }
        if (!modes.equal(e.decode_spans, spans, e.at)) {
            cause = .rows_changed;
            n = e.at;
            break;
        }
        if (!@import("cache_images.zig").matches(s, e)) {
            cause = if (s.image_frame != 0) .frame_changed else .image_changed;
            n = e.at;
            break;
        }
        // this state passes every check, so find() would have returned it (or a longer one did)
        if (cause == null and e.at > n) n = e.at;
    }
    if (cause == null) if (s.ring.held(prompt)) |rec| {
        if (rec.at > resumed) { // a removed state is a miss only when it resumed further than this lookup
            cause = rec.cause;
            n = rec.at;
        }
    };
    if (cause == null) cause = if (hit != null) .none else .cold;
    return .{
        .reason = cause.?,
        .shared_len = n,
        .resumed_len = resumed,
        .lost = n -| resumed,
        .message = messageAt(starts, n),
        .at = at,
        .kept = kept,
        .request = request,
    };
}

/// The last user message's start: before it, a change is the request's prompt; after, the reply's rendering changed.
fn lastUserStart(starts: []const u32, prompt_len: usize) u32 {
    if (starts.len == 0) return @intCast(prompt_len);
    return starts[starts.len - 1];
}
