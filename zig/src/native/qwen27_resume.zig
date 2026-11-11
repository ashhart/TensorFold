//! What the 27B's slot 0 stands after when its stream leaves, and whether a later prompt resumes there, nothing copied.
const std = @import("std");
const modes = @import("engine_api").cache_modes;
const pc = @import("engine_api").prompt_cache;

/// Where a prompt resumes in slot 0: where the slot stands (`here`), back at its rewind point (`back`), or not.
pub const Resume = union(enum) { none, here: u32, back: u32 };

/// A stream the prompt cache would look up (a lanes.Stream): no images, no prompt logprobs, taught rows.
pub fn plain(s: anytype) bool {
    return s.images.len == 0 and s.prompt_logprobs == null and !s.untaught and s.embedding == null;
}

/// Whether `s`'s rows stand on another machine's prefix (#1594 step 2b): not once a refused copy ran here from 0.
pub fn leftRemote(s: anytype) bool {
    return s.reuse.remote and !s.reuse_failed;
}

/// A pass resumed in slot 0 stands on the record's rows, `remote` or not as they are, and pulled nothing just now.
pub fn inherit(reuse: anytype, remote: bool) void {
    reuse.remote = remote;
    reuse.pulled = false;
}

/// Where a pass from `at` keeps slot 0's rewind point: its history's end, inside its prompt and past `at` (0: none).
pub fn rewindAt(history_len: u32, at: usize, prompt_len: usize) u32 {
    return if (history_len > at and history_len < prompt_len) history_len else 0;
}

/// A stream leaving slot 0: the slot's position, its prompt and context lengths, its landed prompt rows and its end.
pub const Leaving = struct {
    at: u32,
    prompt_len: u32,
    context_len: u32,
    fill_at: u32,
    ended: bool, // finished as a reply does (not cancelled, not failed)
    remote: bool = false, // its rows stand on another machine's prefix (leftRemote)

    /// Whether the slot stands where a fresh pass would: a landed prompt chunk's end, or a reply's end or one short.
    pub fn kept(l: Leaving) bool {
        if (l.at == 0) return false;
        if (l.at <= l.prompt_len) return l.at == l.fill_at;
        return l.ended and l.at <= l.context_len and l.at + 1 >= l.context_len;
    }
};

/// The tokens slot 0 stands after, their rows' canonical decoded spans, and a rewind point inside them.
pub const Standing = struct {
    tokens: []u32 = &.{},
    spans: []modes.Span = &.{},
    rewind_at: u32 = 0, // where the rewind buffer holds slot 0's DeltaNet states and taps (0: none)
    reply: modes.Span = .{ 0, 0 }, // rows of a reply these tokens hold exactly as it was generated (empty: none)
    remote: bool = false, // these rows stand on another machine's prefix: a pass resuming them does too
    wide: bool = false, // the prompt cache keeps two full-context states: it holds a reply's end beside slot 0
    keep_remote: bool = false, // TF_PREFILL_KEEP: rows on a pulled prefix may stand for a next turn

    /// The server's settings alone, no record: `wide` and `keep_remote`.
    pub fn settings(st: *const Standing) Standing {
        return .{ .wide = st.wide, .keep_remote = st.keep_remote };
    }

    /// Frees the record; the settings stay.
    pub fn deinit(st: *Standing, gpa: std.mem.Allocator) void {
        gpa.free(st.tokens);
        gpa.free(st.spans);
        st.* = st.settings();
    }

    /// Whether slot 0 may stand on rows for a next turn: its own, or on a pulled prefix only with TF_PREFILL_KEEP.
    pub fn keeps(st: *const Standing, remote: bool) bool {
        return !remote or st.keep_remote;
    }

    /// What stream `s` leaves in slot 0 at `at` (`stands`: the slot and its drafter stand there, nothing failed).
    pub fn left(st: *Standing, gpa: std.mem.Allocator, s: anytype, at: u32, stands: bool) void {
        if (!stands or !plain(s)) return st.deinit(gpa);
        const ended = s.finished and s.reason != .cancelled and s.reason != .@"error";
        const l: Leaving = .{ .at = at, .prompt_len = @intCast(s.prompt_len), .context_len = @intCast(s.context.items.len), .fill_at = s.fill_at.load(.monotonic), .ended = ended, .remote = leftRemote(s) };
        st.leave(gpa, l, s.context.items, s.decode_spans);
    }

    /// Where `s`'s pass from `at` keeps slot 0's rewind point, by `rewindAt`: none for rows slot 0 would not keep.
    pub fn point(st: *const Standing, s: anytype, at: usize) u32 {
        if (!plain(s) or !st.keeps(leftRemote(s))) return 0;
        return rewindAt(s.history_len, at, s.prompt().len);
    }

    /// What `l` left in slot 0 of `context` (its rows' spans `spans`), when a later prompt may resume it; else nothing.
    pub fn leave(st: *Standing, gpa: std.mem.Allocator, l: Leaving, context: []const u32, spans: []const modes.Span) void {
        const rewind = st.rewind_at; // a point this stream's own pass saved
        const reply = st.reply; // a reply the pass resumed past and only extended
        st.deinit(gpa);
        if (!l.kept() or l.at > context.len or !st.keeps(l.remote)) return;
        const canonical = (if (l.at <= l.prompt_len) modes.prefix(gpa, spans, l.at) else modes.reply(gpa, spans, l.prompt_len, l.at)) catch return;
        const tokens = gpa.dupe(u32, context[0..l.at]) catch return gpa.free(canonical);
        const own: modes.Span = if (l.at > l.prompt_len) .{ l.prompt_len, l.at } else if (reply[1] <= l.at) reply else .{ 0, 0 };
        st.* = .{ .tokens = tokens, .spans = canonical, .rewind_at = if (rewind < l.at) rewind else 0, .reply = own, .remote = l.remote, .wide = st.wide, .keep_remote = st.keep_remote };
    }

    /// The rewind point a pass resuming slot 0 at `n` keeps while it only extends the slot (0: none).
    pub fn carry(st: *const Standing, n: u32) u32 {
        return if (st.rewind_at > 0 and st.rewind_at <= n) st.rewind_at else 0;
    }

    /// The rewind point a pass that `r` resumes in slot 0 carries through it (0: none, or it does not resume there).
    pub fn carried(st: *const Standing, r: Resume) u32 {
        return switch (r) {
            .none => 0,
            .here, .back => |n| st.carry(n),
        };
    }

    /// Uses up the record for a pass that `r` resumes in slot 0, keeping the rewind point, reply and origin it carries.
    pub fn take(st: *Standing, gpa: std.mem.Allocator, r: Resume) void {
        const kept = st.carried(r);
        const remote = r != .none and st.remote;
        const reply: modes.Span = switch (r) {
            .here => |n| if (st.reply[1] <= n) st.reply else .{ 0, 0 },
            .back, .none => .{ 0, 0 },
        };
        st.deinit(gpa);
        st.rewind_at = kept;
        st.reply = reply;
        st.remote = remote;
    }

    /// Before `take`: where `ids` (spans `spans`) sends back the reply these tokens hold, as it was generated.
    pub fn replayed(st: *const Standing, ids: []const u32, spans: []const modes.Span) Replay {
        const from, const end = st.reply;
        if (end <= from or end > st.tokens.len) return .{};
        const same: u32 = @intCast(std.mem.findDiff(u32, st.tokens[0..end], ids) orelse end);
        if (same <= from or !modes.equal(st.spans, spans, same)) return .{};
        return .{ .from = from, .to = if (same == end) std.math.maxInt(u32) else same };
    }

    /// Whether a warm pass over slot 0 (`own`) yields: not if the cache keeps its end, nor `wide` with `from` carrying.
    pub fn warmYields(st: *const Standing, keeps_end: bool, own: Resume, from: Resume) bool {
        return !keeps_end and own != .none and (!st.wide or st.carried(from) == 0);
    }

    /// How `ids` (its rows' spans `spans`) resumes, past the cache's own restore point `kept_at`, by the cache's rule.
    pub fn plan(st: *const Standing, ids: []const u32, spans: []const modes.Span, kept_at: u32) Resume {
        const n = st.tokens.len;
        if (n == 0) return .none;
        const same = std.mem.findDiff(u32, st.tokens, ids) orelse n; // the tokens both start with
        if (same == n and n < ids.len and n > kept_at and modes.equal(st.spans, spans, @intCast(n))) return .{ .here = @intCast(n) };
        const h = st.rewind_at; // a next turn without the last reply's reasoning agrees through the history's end
        if (h > 0 and h <= same and h < ids.len and h > kept_at and modes.equal(st.spans, spans, h)) return .{ .back = h };
        return .none;
    }
};

/// Slot 0 needs its smaller history copy unless the cache can keep both full snapshots.
pub fn needsRewind(cache: u64, snapshot: u64) bool {
    return cache < snapshot +| snapshot;
}

/// Only the actual user-end state gets an optional pin, without taking room from required states or promises.
pub fn pinReplyHistory(store: *const pc.Store, entry: *const pc.Entry, history: u32, at: u32) bool {
    if (entry.at != history) return false;
    var bytes = entry.bytes +| store.family.vtable.bytes(store.family.ptr, at);
    var slots: usize = 2;
    if (store.reservation_guard) |g| {
        bytes +|= g.promised.bytes;
        slots +|= g.promised.slots;
    }
    for (store.entries.items) |other| {
        if (other == entry) continue;
        const protected = if (store.reservation_guard) |g| g.protects(store, other) else false;
        // With own-only eviction, conservatively leave every other state in place.
        if (other.borrowers == 0 and !protected and !store.keep_others) continue;
        bytes +|= other.bytes;
        slots +|= 1;
    }
    return bytes <= store.budget and slots <= store.most;
}

/// The history rewind copies live recurrent rows, so its cut must land even when a bank captures other marks.
pub fn rewindEnd(at: usize, end: usize, rewind: u32) usize {
    return if (rewind > at and rewind < end) rewind else end;
}

/// Slot 0's `need`-byte history rewind: none, a `tight` one, or a `wide` one beside a cache that keeps two of them.
pub const Rewind = enum { none, tight, wide };

/// Slot 0 makes its rewind wherever `room` holds it: a `cache` keeping two `full`-byte states may still evict it.
pub fn makesRewind(need: u64, room: u64, cache: u64, full: u64) Rewind {
    if (need > room) return .none;
    return if (needsRewind(cache, full)) .tight else .wide;
}

/// Resume rows (from, to] past a reply's prompt whose reply rows before them a prompt sends back as generated.
pub const Replay = struct {
    from: u32 = 0,
    to: u32 = 0,

    /// A pass from `at` (slot or cache) resumed in a reply sent back as generated: it makes no point (unused there).
    pub fn holds(p: Replay, at: usize) bool {
        return at > p.from and at <= p.to;
    }
};

/// A prompt's next physical chunk end, before cache marks or the slot's rewind point cut it further.
pub fn chunkEnd(at: usize, chunk: usize, len: usize, starts: []const u32, spans: []const modes.Span) usize {
    var end = @min(len, at + chunk);
    for (starts) |cut| if (cut > at and cut < end) {
        end = cut;
    };
    for (spans) |span| for (span) |cut| if (cut > at and cut < end) {
        end = cut;
    };
    return end;
}

/// A skipped warm pass must still leave its first checkpoint when the cache cannot keep the history's end.
pub fn cachePoint(enabled: bool, at: usize, chunk: usize, len: usize, history: u32, marks: []const u32, starts: []const u32, spans: []const modes.Span) u32 {
    if (!enabled or history == 0 or std.mem.indexOfScalar(u32, marks, history) != null) return 0;
    const end = chunkEnd(at, chunk, len, starts, spans);
    return if (end > at and end < history and end < len) @intCast(end) else 0;
}

/// An extra checkpoint uses only unclaimed room, including every later planned mark and spare buffer.
pub fn checkpointFits(store: *const @import("engine_api").prompt_cache.Store, prompt: []const u32, at: u32, marks: []const u32, spans: []const modes.Span) bool {
    if (store.hold or at == 0 or at >= prompt.len) return false;
    var need = store.family.vtable.bytes(store.family.ptr, at);
    var slots: usize = 1;
    for (store.entries.items) |e| if (!e.expired and e.at == at and store.wouldServe(prompt, &.{}, spans, e.tokens, e.decode_spans, at)) return false;
    for (marks) |mark| {
        if (mark <= at) continue;
        const exists = for (store.entries.items) |e| {
            if (!e.expired and e.at == mark and store.wouldServe(prompt, &.{}, spans, e.tokens, e.decode_spans, mark)) break true;
        } else false;
        if (!exists) {
            need +|= store.family.vtable.bytes(store.family.ptr, mark);
            slots +|= 1;
        }
    }
    return need <= store.room() and slots <= store.most -| store.entries.items.len;
}

const t = std.testing;

test "qwen27 resume: a prompt extending what slot 0 stands after resumes there, past the cache's own restore only" {
    var st: Standing = .{};
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // a 5-token prompt and its 3-token reply
    st.leave(t.allocator, .{ .at = 8, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    try t.expectEqualSlices(u32, &context, st.tokens);
    try t.expectEqualDeep(@as([]const modes.Span, &.{.{ 5, 8 }}), st.spans); // the reply's rows were decoded
    const next = context ++ [_]u32{ 9, 10 };
    const decoded = [_]modes.Span{.{ 5, 9 }}; // the next turn's template decodes the reply and its stop token
    try t.expectEqual(Resume{ .here = 8 }, st.plan(&next, &decoded, 0));
    try t.expectEqual(Resume{ .here = 8 }, st.plan(&next, &decoded, 7));
    try t.expectEqual(@as(Resume, .none), st.plan(&next, &decoded, 8)); // the cache restores as far: its own path
    try t.expectEqual(@as(Resume, .none), st.plan(&next, &.{}, 0)); // the reply's rows as prompt rows: other bits
    try t.expectEqual(@as(Resume, .none), st.plan(&.{ 1, 2, 3, 4, 5, 6, 7, 9, 10 }, &decoded, 0)); // another reply
    try t.expectEqual(@as(Resume, .none), st.plan(&context, &decoded, 0)); // nothing left to prefill
    try t.expectEqual(@as(Resume, .none), st.plan(&.{ 1, 2, 3 }, &.{}, 0));
}

test "qwen27 resume: a next turn without the reply's reasoning goes back to the rewind point inside the standing tokens" {
    var st: Standing = .{ .rewind_at = 4 }; // the prompt pass saved slot 0's states at its history's end
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 }; // history 1-4, the generation prompt 20, a reply 21-23
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    try t.expectEqual(@as(u32, 4), st.rewind_at);
    const next = [_]u32{ 1, 2, 3, 4, 30, 23, 31 }; // the reply rendered without its reasoning, then a new user turn
    try t.expectEqual(Resume{ .back = 4 }, st.plan(&next, &.{.{ 5, 6 }}, 0));
    try t.expectEqual(@as(Resume, .none), st.plan(&next, &.{.{ 5, 6 }}, 4)); // the cache keeps that history state itself
    try t.expectEqual(@as(Resume, .none), st.plan(&.{ 1, 2, 9, 4, 30 }, &.{}, 0)); // the history differs before the point
    try t.expectEqual(@as(Resume, .none), st.plan(&.{ 1, 2, 3, 4 }, &.{}, 0)); // nothing past it to prefill
    try t.expectEqual(@as(Resume, .none), st.plan(&next, &.{.{ 2, 3 }}, 0)); // history rows the template now decodes
}

test "qwen27 resume: slot 0 keeps what a fresh pass would stand at, and the point a stream's pass saved" {
    var st: Standing = .{};
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const ok: Leaving = .{ .at = 8, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true };
    for ([_]Leaving{
        .{ .at = 0, .prompt_len = 5, .context_len = 5, .fill_at = 0, .ended = true }, // nothing ran
        .{ .at = 3, .prompt_len = 5, .context_len = 5, .fill_at = 2, .ended = false }, // short of its landed rows
        .{ .at = 6, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, // a reply's rows dropped
        .{ .at = 8, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = false }, // cancelled while decoding
    }) |l| {
        st.leave(t.allocator, ok, &context, &.{});
        st.leave(t.allocator, l, &context, &.{});
        try t.expectEqual(@as(usize, 0), st.tokens.len);
        try t.expectEqual(@as(Resume, .none), st.plan(&(context ++ [_]u32{9}), &.{}, 0));
    }
    st.rewind_at = 2; // the pass saved its point before it stopped
    st.leave(t.allocator, .{ .at = 3, .prompt_len = 5, .context_len = 5, .fill_at = 3, .ended = false }, &context, &.{.{ 1, 4 }});
    try t.expectEqualSlices(u32, context[0..3], st.tokens); // a background pass that yielded at a landed chunk
    try t.expectEqualDeep(@as([]const modes.Span, &.{.{ 1, 3 }}), st.spans);
    try t.expectEqual(@as(u32, 2), st.rewind_at);
    st.rewind_at = 3; // a point that is not strictly inside what the slot stands after is not kept
    st.leave(t.allocator, .{ .at = 3, .prompt_len = 5, .context_len = 5, .fill_at = 3, .ended = false }, &context, &.{});
    try t.expectEqual(@as(u32, 0), st.rewind_at);
}

test "qwen27 resume: a warm pass that only extends slot 0 keeps the reply's rewind point for a diverging next turn" {
    var st: Standing = .{ .rewind_at = 4 }; // the turn's pass saved slot 0's states at its history's end
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 }; // history 1-4, the generation prompt 20, a reply 21-23
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    const warm = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 24, 25, 26 }; // a warm pass past the slot's 7: its history is all
    const n = switch (st.plan(&warm, &.{.{ 5, 7 }}, 0)) {
        .here => |x| x,
        else => return error.TestUnexpectedResult,
    };
    try t.expectEqual(@as(u32, 7), n);
    st.take(t.allocator, .{ .here = n }); // as takeStanding: the record is used up, the point stays
    st.leave(t.allocator, .{ .at = 9, .prompt_len = 10, .context_len = 10, .fill_at = 9, .ended = false }, &warm, &.{.{ 5, 7 }});
    try t.expectEqual(@as(u32, 4), st.rewind_at); // a yield at a landed chunk keeps it too
    st.leave(t.allocator, .{ .at = 10, .prompt_len = 10, .context_len = 10, .fill_at = 10, .ended = true }, &warm, &.{.{ 5, 7 }});
    try t.expectEqualSlices(u32, &warm, st.tokens);
    try t.expectEqual(@as(u32, 4), st.rewind_at);
    const next = [_]u32{ 1, 2, 3, 4, 30, 23, 31 }; // the reply rendered without its reasoning
    try t.expectEqual(Resume{ .back = 4 }, st.plan(&next, &.{.{ 5, 6 }}, 0));
    try t.expectEqual(Resume{ .here = 10 }, st.plan(&(warm ++ [_]u32{32}), &.{.{ 5, 7 }}, 0));
}

test "qwen27 resume: a rewind point past where the pass resumes, or none, is not carried" {
    const st: Standing = .{ .rewind_at = 4 };
    try t.expectEqual(@as(u32, 4), st.carry(4)); // a `.back` pass resumes at the point itself
    try t.expectEqual(@as(u32, 4), st.carry(9));
    try t.expectEqual(@as(u32, 0), st.carry(3));
    try t.expectEqual(@as(u32, 0), st.carry(0));
    try t.expectEqual(@as(u32, 0), (Standing{}).carry(9));
}

test "qwen27 resume: a warm pass over slot 0's conversation runs where it carries the rewind point, else steps aside" {
    var st: Standing = .{ .rewind_at = 4 };
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 }; // history 1-4, the generation prompt 20, a reply 21-23
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    const warm = [_]u32{ 1, 2, 3, 4, 30, 23, 31 }; // the reply as the template renders it: it goes back to the point
    try t.expectEqual(Resume{ .back = 4 }, st.plan(&warm, &.{.{ 5, 6 }}, 0));
    try t.expectEqual(@as(u32, 4), st.carried(st.plan(&warm, &.{.{ 5, 6 }}, 0))); // it runs
    try t.expectEqual(@as(u32, 0), st.carried(st.plan(&warm, &.{.{ 5, 6 }}, 4))); // the cache's state is further: aside
    st.take(t.allocator, st.plan(&warm, &.{.{ 5, 6 }}, 0)); // as takeStanding, then the pass to its end
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 7, .context_len = 7, .fill_at = 7, .ended = true }, &warm, &.{.{ 5, 6 }});
    try t.expectEqual(Resume{ .here = 7 }, st.plan(&(warm ++ [_]u32{40}), &.{.{ 5, 6 }}, 0)); // a client replaying it
    try t.expectEqual(Resume{ .back = 4 }, st.plan(&.{ 1, 2, 3, 4, 20, 21, 22, 23, 40 }, &.{}, 0)); // one that does not
    st.rewind_at = 0; // a slot with no rewind point would keep none through the pass: it steps aside
    try t.expectEqual(@as(u32, 0), st.carried(st.plan(&(warm ++ [_]u32{40}), &.{.{ 5, 6 }}, 0)));
    try t.expectEqual(@as(u32, 0), st.carried(.none));
}

test "qwen27 resume: rule a, a warm pass carrying slot 0's rewind point runs beside a wide cache, else it yields" {
    var st: Standing = .{ .rewind_at = 4 };
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 }; // history 1-4, the generation prompt 20, a reply 21-23
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    const warm = [_]u32{ 1, 2, 3, 4, 30, 23, 31 }; // back to the point at 4
    const own = st.plan(&warm, &.{.{ 5, 6 }}, 0);
    try t.expect(st.warmYields(false, own, own)); // a tight cache would lose the reply's end it leaves: aside
    st.wide = true; // the cache keeps two full-context states
    try t.expect(!st.warmYields(false, own, own)); // it carries the point: it runs
    try t.expect(st.warmYields(false, own, st.plan(&warm, &.{.{ 5, 6 }}, 4))); // the cache's state is further: aside
    try t.expect(!st.warmYields(true, own, .none)); // the cache keeps the conversation's end
    try t.expect(!st.warmYields(false, .none, .none)); // it would not resume in slot 0
    st.rewind_at = 0;
    try t.expect(st.warmYields(false, own, own)); // no point to carry: it would leave nothing in slot 0
    st.deinit(t.allocator);
    try t.expect(st.wide); // the record goes, the cache's size stays
}

test "qwen27 resume: rule b, taking slot 0's record for a pass that extends it keeps the rewind point" {
    var st: Standing = .{ .rewind_at = 4 };
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 };
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    st.take(t.allocator, st.plan(&[_]u32{ 1, 2, 3, 4, 20, 21, 22, 24 }, &.{.{ 5, 7 }}, 0)); // `.here` at 7
    try t.expectEqual(@as(usize, 0), st.tokens.len); // the record is used up
    try t.expectEqual(@as(u32, 4), st.rewind_at); // the point stays
    st.take(t.allocator, .none); // a pass that does not resume in slot 0 drops it
    try t.expectEqual(@as(u32, 0), st.rewind_at);
}

test "qwen27 resume: rule c, slot 0 makes its rewind buffer where room holds it, even beside a cache with two states" {
    const full: u64 = 2496 << 20; // one full-context state at 33024 rows
    try t.expectEqual(Rewind.wide, makesRewind(1 << 30, 2 << 30, 5 << 30, full)); // needsRewind says no: made anyway
    try t.expectEqual(Rewind.wide, makesRewind(1 << 30, 2 << 30, 8 << 30, full));
    try t.expectEqual(Rewind.tight, makesRewind(1 << 30, 2 << 30, 4 << 30, full)); // needsRewind's own case
    try t.expectEqual(Rewind.tight, makesRewind(1 << 30, 1 << 30, full, full));
    try t.expectEqual(Rewind.tight, makesRewind(1 << 30, 2 << 30, 0, full)); // no cache at all
    try t.expectEqual(Rewind.none, makesRewind(2 << 30, 1 << 30, 0, full)); // no room
    try t.expectEqual(Rewind.none, makesRewind(2 << 30, 1 << 30, 8 << 30, full));
}

test "qwen27 resume: a follow-up sending the reply back as generated, restored at the cache's reply end, makes no point" {
    var st: Standing = .{ .rewind_at = 4 }; // the turn's pass saved slot 0's states at its history's end
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 }; // history 1-4, the generation prompt 20, a reply 21-23
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    try t.expectEqual(modes.Span{ 5, 7 }, st.reply); // the stream left one short of its reply's end
    const next = context ++ [_]u32{ 30, 31 }; // the reply as generated, then a new user turn
    const spans = [_]modes.Span{.{ 5, 8 }};
    try t.expectEqual(@as(Resume, .none), st.plan(&next, &spans, 7)); // the cache keeps the reply's end: its own path
    const replay = st.replayed(&next, &spans); // as opening, before takeStanding uses the record up
    st.take(t.allocator, .none);
    try t.expectEqual(@as(u32, 0), st.rewind_at);
    try t.expect(replay.holds(7)); // restored at 7, inside the generated rows 5-7: no stop at the history's end
    try t.expect(!replay.holds(4)); // a cache state at the history's mark is prompt rows: the point is made
    try t.expect(!replay.holds(0));
    const turn = next ++ [_]u32{ 40, 41, 42 }; // that pass, with no point, and its reply 40-42
    st.leave(t.allocator, .{ .at = 12, .prompt_len = 10, .context_len = 13, .fill_at = 10, .ended = true }, &turn, &spans);
    try t.expectEqual(modes.Span{ 10, 12 }, st.reply);
    const again = [_]modes.Span{ .{ 5, 8 }, .{ 10, 13 } };
    try t.expect(st.replayed(&(turn ++ [_]u32{50}), &again).holds(12)); // the next follow-up skips it too
    const diverging = next ++ [_]u32{ 60, 42, 61 }; // the last reply rendered without its reasoning
    const cut = [_]modes.Span{ .{ 5, 8 }, .{ 10, 12 } };
    try t.expectEqual(@as(Resume, .none), st.plan(&diverging, &cut, 9)); // no point: the cache's state, as main
    try t.expect(!st.replayed(&diverging, &cut).holds(9)); // and that pass makes one
    try t.expectEqual(Replay{}, st.replayed(&diverging, &.{})); // reply rows sent as prompt rows are not as generated
}

test "qwen27 resume: a follow-up resuming in slot 0 past a reply kept as generated makes no point, carrying the old one" {
    var st: Standing = .{ .rewind_at = 4 };
    defer st.deinit(t.allocator);
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 };
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true }, &context, &.{});
    const warm = context ++ [_]u32{ 30, 31 }; // a warm pass that ran: it extends the slot past the reply
    st.take(t.allocator, st.plan(&warm, &.{.{ 5, 8 }}, 0)); // `.here` at 7
    st.leave(t.allocator, .{ .at = 9, .prompt_len = 10, .context_len = 10, .fill_at = 9, .ended = false }, &warm, &.{.{ 5, 8 }});
    try t.expectEqual(modes.Span{ 5, 7 }, st.reply); // a yield at a landed chunk keeps it
    st.leave(t.allocator, .{ .at = 10, .prompt_len = 10, .context_len = 10, .fill_at = 10, .ended = true }, &warm, &.{.{ 5, 8 }});
    try t.expectEqual(modes.Span{ 5, 7 }, st.reply); // and so does the pass to its prompt's end
    const follow = warm ++ [_]u32{32};
    try t.expectEqual(Resume{ .here = 10 }, st.plan(&follow, &.{.{ 5, 8 }}, 0));
    const replay = st.replayed(&follow, &.{.{ 5, 8 }});
    try t.expect(replay.holds(10)); // in the slot, or a cache state at the warm pass's end
    try t.expectEqual(Resume{ .back = 4 }, st.plan(&.{ 1, 2, 3, 4, 50, 23, 51 }, &.{.{ 5, 6 }}, 0)); // reply dropped
    st.take(t.allocator, .{ .here = 10 });
    try t.expectEqual(@as(u32, 4), st.rewind_at); // the older point stays for such a turn
    st.leave(t.allocator, .{ .at = 3, .prompt_len = 5, .context_len = 5, .fill_at = 3, .ended = false }, &context, &.{});
    try t.expectEqual(modes.Span{ 0, 0 }, st.reply); // a reply past where the slot stands is not kept
}

test "qwen27 resume: a client dropping the reasoning, a cold turn or a pass that resumes nowhere makes a fresh point" {
    var st: Standing = .{ .rewind_at = 4 };
    defer st.deinit(t.allocator);
    try t.expect(!st.replayed(&.{ 1, 2, 3 }, &.{}).holds(0)); // cold: nothing stands
    const context = [_]u32{ 1, 2, 3, 4, 20, 21, 22, 23 };
    const reply: Leaving = .{ .at = 7, .prompt_len = 5, .context_len = 8, .fill_at = 5, .ended = true };
    st.leave(t.allocator, reply, &context, &.{});
    const warm = [_]u32{ 1, 2, 3, 4, 30, 23, 31 }; // the reply rendered without its reasoning
    try t.expect(!st.replayed(&(warm ++ [_]u32{32}), &.{.{ 5, 6 }}).holds(4)); // its follow-up goes back to 4
    const back = st.plan(&warm, &.{.{ 5, 6 }}, 0); // the server's warm pass over it
    try t.expectEqual(Resume{ .back = 4 }, back);
    st.take(t.allocator, back);
    try t.expectEqual(modes.Span{ 0, 0 }, st.reply); // the rows past the point are rewritten
    st.leave(t.allocator, .{ .at = 7, .prompt_len = 7, .context_len = 7, .fill_at = 7, .ended = true }, &warm, &.{.{ 5, 6 }});
    const follow = warm ++ [_]u32{ 32, 33 };
    try t.expectEqual(Resume{ .here = 7 }, st.plan(&follow, &.{.{ 5, 6 }}, 0)); // past the carried point at 4
    try t.expect(!st.replayed(&follow, &.{.{ 5, 6 }}).holds(7)); // prompt rows: the new history's end is a point
    st.leave(t.allocator, reply, &context, &.{});
    st.take(t.allocator, .none);
    try t.expectEqual(modes.Span{ 0, 0 }, st.reply);
    try t.expectEqual(@as(u32, 0), st.rewind_at);
}

test {
    _ = @import("qwen27_resume_remote_test.zig");
    _ = @import("qwen27_resume_cache_test.zig");
}

test "qwen27 resume: the history rewind is needed until two rounded full-context states fit" {
    const full: u64 = 4800 << 20;
    try t.expect(needsRewind(5 << 30, full));
    try t.expect(needsRewind(full, full));
    try t.expect(needsRewind(2 * full - 1, full));
    try t.expect(!needsRewind(2 * full, full));
    try t.expect(!needsRewind(2 * full + 1, full));
    try t.expectEqual(@as(usize, 34078), rewindEnd(33792, 34083, 34078));
    try t.expectEqual(@as(usize, 34083), rewindEnd(33792, 34083, 0));
    try t.expectEqual(@as(usize, 34083), rewindEnd(33792, 34083, 33792));
}
