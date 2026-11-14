//! Slot 0 after a hand-off turn (#1594 step 2b): rows [0, cut) the GB10's state filled, the rest the Mac's own pass.
const std = @import("std");
const api = @import("engine_api");
const rules = @import("qwen27_resume.zig");
const Stream = @FieldType(api.LaneHost.Job, "stream");
const Reason = @FieldType(Stream, "reason");
const t = std.testing;

var pulled_state: u8 = 0; // a pulled copy's stand-in: these rules never read it

/// As qwen27_slots.takeStanding: the record taken by its plan; a pass that resumes there stands on its rows.
fn open(st: *rules.Standing, s: *Stream, kept_at: u32) rules.Resume {
    const plan = if (rules.plain(s)) st.plan(s.prompt(), s.decode_spans, kept_at) else .none;
    st.take(t.allocator, plan);
    rules.inherit(&s.reuse, plan, st.remote); // the rule decides which plans take the rows' origin, as there
    return plan;
}

/// As qwen27_slots.leave once `s`'s reply `reply` is in: slot 0 stands one short of its end, `rewind` saved.
fn leave(st: *rules.Standing, s: *Stream, reply: []const u32, rewind: u32, reason: Reason) !void {
    try s.context.appendSlice(t.allocator, reply);
    s.fill_at.store(@intCast(s.prompt_len), .monotonic);
    s.finished = true;
    s.reason = reason;
    st.rewind_at = rewind; // saveRewind's blit, which needs a GPU, succeeded
    st.left(t.allocator, s, @intCast(s.context.items.len - 1), true);
}

const history = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // the first turn's rendered history; 20 opens the reply
const p1 = history ++ [_]u32{20};
const reply1 = [_]u32{ 21, 22, 23 }; // with its reasoning, 21-22
const p2 = history ++ [_]u32{ 20, 30, 23, 31, 40, 20 }; // the reply re-rendered without its reasoning, then a turn

/// A hand-off turn's stream: the GB10 filled its first 4 tokens, pulled just now.
fn handOff() !Stream {
    return Stream.init(t.allocator, .{ .id = "pulled", .prompt = &p1, .max_new = 3, .history_len = 8, .reuse = .{ .saved = &pulled_state, .at = 4, .remote = true, .pulled = true } });
}

test "qwen27 resume: a hand-off turn leaves a remote record; its follow-up rewinds to the history end and stays remote" {
    var st: rules.Standing = .{ .keep_remote = true }; // TF_PREFILL_KEEP=1, the default
    defer st.deinit(t.allocator);
    var s1 = try handOff();
    defer s1.deinit(t.allocator);
    try t.expectEqual(rules.Resume.none, open(&st, &s1, 4)); // it restores the pulled state at 4
    try t.expect(s1.reuse.remote and s1.reuse.pulled);
    try t.expectEqual(@as(u32, 8), st.point(&s1, 4)); // the history end past the cut: slot 0's rewind point
    try leave(&st, &s1, &reply1, 8, .stop);
    try t.expect(st.remote and st.rewind_at == 8 and st.tokens.len == 11);

    var s2 = try Stream.init(t.allocator, .{ .id = "next", .prompt = &p2, .max_new = 2, .history_len = 13 });
    defer s2.deinit(t.allocator);
    try t.expectEqual(rules.Resume{ .back = 8 }, open(&st, &s2, 0)); // in place: no pull, no copy
    try t.expect(s2.reuse.remote and !s2.reuse.pulled); // reported as a kept remote prefix, never as the Mac's own
    try t.expectEqual(@as(u32, 13), st.point(&s2, 8));
    try leave(&st, &s2, &.{ 50, 51 }, 13, .stop);
    try t.expect(st.remote and st.rewind_at == 13 and st.tokens.len == 15);

    const c2 = p2 ++ [_]u32{ 50, 51 };
    const p3 = c2[0..15].* ++ [_]u32{ 60, 20 }; // a client sending the reply back as decoded (row 14): it extends
    var s3 = try Stream.init(t.allocator, .{ .id = "extends", .prompt = &p3, .max_new = 2, .history_len = 16, .decode_spans = &.{.{ 14, 15 }} });
    defer s3.deinit(t.allocator);
    try t.expectEqual(rules.Resume{ .here = 15 }, open(&st, &s3, 0));
    try t.expect(s3.reuse.remote and !s3.reuse.pulled and st.rewind_at == 13 and st.remote); // the point carried on
}

test "qwen27 resume: with TF_PREFILL_KEEP=0 a hand-off turn makes no rewind point and leaves nothing in slot 0" {
    var st: rules.Standing = .{};
    defer st.deinit(t.allocator);
    var s1 = try handOff();
    defer s1.deinit(t.allocator);
    _ = open(&st, &s1, 4);
    try t.expectEqual(@as(u32, 0), st.point(&s1, 4));
    try leave(&st, &s1, &reply1, 0, .stop);
    try t.expect(st.tokens.len == 0 and !st.remote and st.rewind_at == 0);
    var s2 = try Stream.init(t.allocator, .{ .id = "next", .prompt = &p2, .max_new = 2, .history_len = 13 });
    defer s2.deinit(t.allocator);
    try t.expectEqual(rules.Resume.none, open(&st, &s2, 0)); // the next turn pulls again or reads it here
}

test "qwen27 resume: a refused import prefilled here from 0 leaves an own record, not a remote one" {
    var st: rules.Standing = .{ .keep_remote = true };
    defer st.deinit(t.allocator);
    var s1 = try handOff();
    defer s1.deinit(t.allocator);
    _ = open(&st, &s1, 4);
    s1.reuse_failed = true; // opening's restore failed: the slot was reset and the pass ran from 0
    try t.expectEqual(@as(u32, 8), st.point(&s1, 0));
    try leave(&st, &s1, &reply1, 8, .stop);
    try t.expect(!st.remote and st.rewind_at == 8 and st.tokens.len == 11);
    var s2 = try Stream.init(t.allocator, .{ .id = "next", .prompt = &p2, .max_new = 2, .history_len = 13 });
    defer s2.deinit(t.allocator);
    try t.expectEqual(rules.Resume{ .back = 8 }, open(&st, &s2, 0));
    try t.expect(!s2.reuse.remote); // the Mac's own rows
}

test "qwen27 resume: a cancelled or failed hand-off reply leaves nothing; one stopped at a landed chunk stays remote" {
    for ([_]Reason{ .cancelled, .@"error" }) |reason| {
        var st: rules.Standing = .{ .keep_remote = true };
        defer st.deinit(t.allocator);
        var s1 = try handOff();
        defer s1.deinit(t.allocator);
        _ = open(&st, &s1, 4);
        try leave(&st, &s1, &reply1, 8, reason);
        try t.expect(st.tokens.len == 0 and !st.remote and st.rewind_at == 0);
    }
    var st: rules.Standing = .{ .keep_remote = true };
    defer st.deinit(t.allocator);
    var s1 = try handOff();
    defer s1.deinit(t.allocator);
    _ = open(&st, &s1, 4);
    s1.fill_at.store(6, .monotonic); // cancelled between prompt chunks: slot 0 stands at the last landed one
    s1.reason = .cancelled;
    st.left(t.allocator, &s1, 6, true);
    try t.expect(st.remote and st.tokens.len == 6 and st.rewind_at == 0); // the history end was never reached
    st.left(t.allocator, &s1, 5, true); // short of its landed rows: nothing
    try t.expect(st.tokens.len == 0 and !st.remote);
}

test "qwen27 resume: a pull that reaches the history end makes no rewind point; the record still stands for an extension" {
    var st: rules.Standing = .{ .keep_remote = true };
    defer st.deinit(t.allocator);
    var s1 = try Stream.init(t.allocator, .{ .id = "pulled", .prompt = &p1, .max_new = 3, .history_len = 8, .reuse = .{ .saved = &pulled_state, .at = 8, .remote = true, .pulled = true } });
    defer s1.deinit(t.allocator);
    _ = open(&st, &s1, 8);
    try t.expectEqual(@as(u32, 0), st.point(&s1, 8)); // the cut is the history end: no pass row lands there
    try leave(&st, &s1, &reply1, 0, .stop);
    try t.expect(st.remote and st.rewind_at == 0 and st.tokens.len == 11);
    var s2 = try Stream.init(t.allocator, .{ .id = "next", .prompt = &p2, .max_new = 2, .history_len = 13 });
    defer s2.deinit(t.allocator);
    try t.expectEqual(rules.Resume.none, open(&st, &s2, 0)); // a reply sent back without its reasoning: not here
}

test "qwen27 resume: images, prompt logprobs or untaught rows leave nothing in slot 0, remote or not" {
    const image = [_]api.Image{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 2, .tokens = 1 }};
    for (0..3) |kind| {
        var st: rules.Standing = .{ .keep_remote = true };
        defer st.deinit(t.allocator);
        var s1 = try handOff();
        defer s1.deinit(t.allocator);
        switch (kind) {
            0 => s1.images = &image,
            1 => s1.prompt_logprobs = 0,
            else => s1.untaught = true,
        }
        try t.expectEqual(rules.Resume.none, open(&st, &s1, 4));
        try t.expectEqual(@as(u32, 0), st.point(&s1, 4));
        try leave(&st, &s1, &reply1, 0, .stop);
        try t.expect(st.tokens.len == 0 and !st.remote);
    }
}

test "qwen27 resume: a pass from 0 on slot 0 after a remote record leaves its own rows, not remote" {
    var st: rules.Standing = .{ .keep_remote = true };
    defer st.deinit(t.allocator);
    var s1 = try handOff();
    defer s1.deinit(t.allocator);
    _ = open(&st, &s1, 4);
    try leave(&st, &s1, &reply1, 8, .stop);
    const other = [_]u32{ 9, 9, 9, 9, 9 }; // another conversation: it shares nothing with slot 0's rows
    var cold = try Stream.init(t.allocator, .{ .id = "cold", .prompt = &other, .max_new = 2, .history_len = 4 });
    defer cold.deinit(t.allocator);
    try t.expectEqual(rules.Resume.none, open(&st, &cold, 0));
    try t.expect(!cold.reuse.remote and !st.remote and st.rewind_at == 0);
    try leave(&st, &cold, &.{ 10, 11 }, 4, .stop);
    try t.expect(!st.remote and st.rewind_at == 4 and st.tokens.len == 6);
}
