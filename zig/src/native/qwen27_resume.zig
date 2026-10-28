//! What the 27B's slot 0 stands after when its stream leaves, and whether a later prompt resumes there, nothing copied.
const std = @import("std");
const modes = @import("engine_api").cache_modes;

/// Where a prompt resumes in slot 0: where the slot stands (`here`), back at its rewind point (`back`), or not.
pub const Resume = union(enum) { none, here: u32, back: u32 };

/// A stream leaving slot 0: the slot's position, its prompt and context lengths, its landed prompt rows and its end.
pub const Leaving = struct {
    at: u32,
    prompt_len: u32,
    context_len: u32,
    fill_at: u32,
    ended: bool, // finished as a reply does (not cancelled, not failed)

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

    pub fn deinit(st: *Standing, gpa: std.mem.Allocator) void {
        gpa.free(st.tokens);
        gpa.free(st.spans);
        st.* = .{};
    }

    /// What `l` left in slot 0 of `context` (its rows' spans `spans`), when a later prompt may resume it; else nothing.
    pub fn leave(st: *Standing, gpa: std.mem.Allocator, l: Leaving, context: []const u32, spans: []const modes.Span) void {
        const rewind = st.rewind_at; // a point this stream's own pass saved
        st.deinit(gpa);
        if (!l.kept() or l.at > context.len) return;
        const canonical = (if (l.at <= l.prompt_len) modes.prefix(gpa, spans, l.at) else modes.reply(gpa, spans, l.prompt_len, l.at)) catch return;
        const tokens = gpa.dupe(u32, context[0..l.at]) catch return gpa.free(canonical);
        st.* = .{ .tokens = tokens, .spans = canonical, .rewind_at = if (rewind < l.at) rewind else 0 };
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

test {
    _ = @import("qwen27_resume_cache_test.zig");
}
