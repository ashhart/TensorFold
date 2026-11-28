//! Why a prompt resumed no further (#3686): one reason a request, decided after the store chose, reading only.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");

/// One reason a request; the names are /metrics labels, so the set is fixed.
pub const Reason = enum(u8) {
    none, // resumed as deep as any kept or remembered prompt shares, within SLACK tokens
    cold, // nothing kept shares the prompt
    too_short, // the prompt is below the family's shortest kept prompt
    bypass, // no lookup: a prompt-only, prompt-logprobs or untaught request, past the window, or a failed lookup
    prompt_changed, // the prompt parts from a kept prompt at `shared`, outside a reply the engine wrote
    reply_changed, // it parts inside a reply the engine wrote: the template re-rendered the model's own output
    image_changed, // an image at `shared` differs (another picture, or the same one re-sized)
    frame_changed, // a kept state matches the tokens, but the family's image frame differs
    rows_changed, // a kept state matches the tokens, but its decoded rows' arithmetic differs
    ends_inside, // a kept state matches, but the prompt ends inside it: a state must leave a row to compute
    off_grid, // a kept state matches, but not at one of this request's chunk starts
    between_marks, // the prompt extends a kept prompt, but no state was kept between the resume point and `shared`
    evicted, // the deeper state was removed for room
    shed, // the deeper state was removed under memory pressure
    expired, // the deeper state expired (the backend dropped it, or a peer could not resume it)
    cleared, // the deeper state went when the weights changed
    not_kept, // the deeper state was never kept: memory pressure, the budget, promised room, or a failed save
    restore_failed, // the deeper state was found, but its restore was refused
};
pub const reasons = std.enums.values(Reason).len;

/// Tokens a re-rendered generation prompt may differ by and still count as resumed as deep as anything kept.
pub const SLACK = 32;

/// A request's reason, `shared` (N) the longest prefix it shares with a kept or remembered prompt.
pub const Why = struct {
    reason: Reason = .bypass,
    shared: u32 = 0,
    resumed: u32 = 0, // planned at lookup; `ended` makes it what the backend restored (cached_tokens)

    /// The tokens resumed less than planned: the backend refused the restore.
    pub fn ended(w: Why, resumed: u32) Why {
        return .{ .reason = if (resumed < w.resumed) .restore_failed else w.reason, .shared = @max(w.shared, resumed), .resumed = resumed };
    }

    pub fn lost(w: Why) u32 {
        return w.shared -| w.resumed;
    }
};

/// The states removed or refused lately, oldest overwritten: a later miss names them instead of `cold`.
pub const Gone = struct {
    recs: [SIZE]Rec = @splat(.{}),
    next: u8 = 0,
    pub const SIZE = 64;
    pub const Rec = struct { key: u64 = 0, n: u32 = 0, at: u32 = 0, last: u32 = 0, mid: u32 = 0, reason: Reason = .cold };

    /// `tokens` (a state's, through its lookahead) at `at` went for `reason`; the same state again only renames it.
    pub fn note(g: *Gone, tokens: []const u32, at: u32, key: u64, reason: Reason) void {
        if (tokens.len == 0) return;
        for (&g.recs) |*r| if (r.n == tokens.len and r.at == at and r.key == key) {
            r.reason = reason;
            return;
        };
        g.recs[g.next] = .{ .key = key, .n = @intCast(tokens.len), .at = at, .last = tokens[tokens.len - 1], .mid = tokens[tokens.len / 2], .reason = reason };
        g.next = @intCast((@as(usize, g.next) + 1) % SIZE);
    }

    /// The deepest record past `from` on `prompt`'s token path; two tokens filter before any hash.
    pub fn match(g: *const Gone, s: *const pc.Store, prompt: []const u32, from: u32) ?Rec {
        var best: ?Rec = null;
        for (g.recs) |r| {
            if (r.n == 0 or r.at <= from or r.n > prompt.len or (best != null and r.at <= best.?.at)) continue;
            if (prompt[r.n - 1] != r.last or prompt[r.n / 2] != r.mid or queryKey(s, prompt[0..r.n]) != r.key) continue;
            best = r;
        }
        return best;
    }
};

/// A kept state's key: its namespace, tokens, family frame and image identities.
pub fn entryKey(e: *const pc.Entry) u64 {
    var h = std.hash.Wyhash.init(e.image_frame);
    if (e.salt) |salt| h.update(std.mem.asBytes(&salt));
    h.update(std.mem.sliceAsBytes(e.tokens));
    for (e.images) |i| h.update(std.mem.asBytes(&i.hash));
    return h.final();
}

/// The key the store's scoped query gives `tokens`: as `entryKey` for a state kept with them.
pub fn queryKey(s: *const pc.Store, tokens: []const u32) u64 {
    var h = std.hash.Wyhash.init(s.image_frame);
    if (s.salt) |salt| h.update(std.mem.asBytes(&salt));
    h.update(std.mem.sliceAsBytes(tokens));
    for (s.image_query orelse &.{}) |i| if (i.at < tokens.len) h.update(std.mem.asBytes(&i.hash));
    return h.final();
}

/// A keep that kept nothing at `at`: a later prompt on this path says `not_kept`.
pub fn refused(s: *pc.Store, prompt: []const u32, at: u32) void {
    const n = @as(usize, at) + s.rules.lookahead;
    if (at > 0 and n <= prompt.len) s.gone.note(prompt[0..n], at, queryKey(s, prompt[0..n]), .not_kept);
}

/// Why `prompt` resumes at `e` and no further. Runs after the store chose `e`; reads entries and allocates nothing.
pub fn diagnose(s: *const pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, e: ?*const pc.Entry) Why {
    const from: u32 = if (e) |x| x.at else 0;
    var w: Why = .{ .reason = .none, .shared = from, .resumed = from };
    for (s.by_position.items) |k| { // a deeper state whose tokens match, refused by a reuse check: longest first
        if (k.at <= from) break;
        const m = @min(k.tokens.len, prompt.len);
        if (k.expired or k.salt != s.salt or m == 0 or k.tokens[m - 1] != prompt[m - 1] or !std.mem.eql(u32, k.tokens[0..m], prompt[0..m])) continue;
        w.reason = if (k.at >= prompt.len or k.tokens.len > prompt.len) .ends_inside else if (!s.usable(k.at, starts)) .off_grid else if (!modes.equal(k.decode_spans, spans, k.at))
            .rows_changed
        else if (k.image_frame != s.image_frame) .frame_changed else if (!images.equal(k.images, s.image_query orelse &.{}, @intCast(k.tokens.len))) .image_changed else continue;
        w.shared = @intCast(@min(k.at, prompt.len));
        return w;
    }
    if (s.gone.match(s, prompt, from)) |r| {
        w.reason = r.reason;
        w.shared = r.at;
        return w;
    }
    var best: ?*const pc.Entry = null; // the kept prompt sharing the most, as the fork rule scans
    for (s.entries.items) |k| {
        const n = w.shared;
        if (k.expired or k.salt != s.salt or k.last.len <= n or n >= prompt.len or k.last[n] != prompt[n]) continue; // cannot share more
        const d: u32 = @intCast(std.mem.indexOfDiff(u32, k.last, prompt) orelse @min(k.last.len, prompt.len));
        if (d > n) {
            w.shared = d;
            best = k;
        }
    }
    const n = w.shared;
    if (n - from < SLACK) {
        w.reason = if (from > 0) .none else if (prompt.len < s.rules.min_prompt or prompt.len <= s.rules.lookahead) .too_short else .cold;
        return w;
    }
    const k = best.?;
    if (n >= k.last.len or n >= prompt.len) {
        w.reason = .between_marks;
        return w;
    }
    w.reason = if (imageAt(s, n)) .image_changed else if (inReply(s, prompt, n)) .reply_changed else .prompt_changed;
    return w;
}

/// Whether a kept state parts from `prompt` at `n` inside the reply it was born with.
fn inReply(s: *const pc.Store, prompt: []const u32, n: u32) bool {
    for (s.entries.items) |k| if (!k.expired and k.salt == s.salt and k.born <= n and n < k.tokens.len and k.tokens[n] != prompt[n] and std.mem.eql(u32, k.tokens[0..n], prompt[0..n])) return true;
    return false;
}

/// Whether one of the request's images covers token `n`.
fn imageAt(s: *const pc.Store, n: u32) bool {
    for (s.image_query orelse &.{}) |i| if (i.at <= n and n < i.at + i.tokens) return true;
    return false;
}
