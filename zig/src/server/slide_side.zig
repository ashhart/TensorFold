//! Writer calls side by side, decoded together in the engine's lanes; each call writes only its own result.
const std = @import("std");
const chat = @import("chat.zig");
const errors = @import("errors.zig");
const slide_ask = @import("slide_ask.zig");
const known = @import("slide_known.zig");
const slide = @import("slide.zig");
const log = @import("log.zig");
const api = @import("engine_api");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

pub const lanes = 16; // calls in flight at once: the 27B's streams at --parallel 16 (one stream takes them in turn)

/// `job.run(i)` for every i below n, on up to `lanes` threads (inline when none starts); the first error, if any.
pub fn each(n: usize, job: anytype) !void {
    if (n == 0) return;
    const Job = @TypeOf(job);
    var next: std.atomic.Value(usize) = .init(0);
    var first: std.atomic.Value(u16) = .init(0); // the first error's code, 0 for none
    const Worker = struct {
        fn work(j: Job, at: *std.atomic.Value(usize), count: usize, err: *std.atomic.Value(u16)) void {
            while (true) {
                const i = at.fetchAdd(1, .monotonic);
                if (i >= count) return;
                j.run(i) catch |e| {
                    _ = err.cmpxchgStrong(0, @intFromError(e), .monotonic, .monotonic);
                };
            }
        }
    };
    var threads: [lanes]?std.Thread = @splat(null);
    var started: usize = 0;
    for (threads[0..@min(lanes, n)]) |*t| {
        t.* = std.Thread.spawn(.{ .stack_size = 8 << 20 }, Worker.work, .{ job, &next, n, &first }) catch null;
        started += @intFromBool(t.* != null);
    }
    if (started == 0) Worker.work(job, &next, n, &first);
    for (threads) |t| if (t) |th| th.join();
    const code = first.load(.monotonic);
    if (code != 0) return @errorFromInt(code);
}

/// A question as `ask` puts it (at most `max` tokens), or with thinking on as `askThinking` does.
pub const Asking = struct { question: []const u8, max: i64 = 48, thinking: bool = false, before: []const [2][]const u8 = &.{} };

/// Each question's reply, asked side by side: one asked before is scored first.
pub fn askAll(srv: *Server, cx: *Cx, qs: []const Asking, gone: anytype) ![]chat.Reply {
    const a = cx.a;
    const out = try a.alloc(chat.Reply, qs.len);
    const ask_now = try a.alloc(usize, qs.len); // the questions generated
    var n: usize = 0;
    var xs: std.ArrayList(api.Example) = .empty;
    var which: std.ArrayList(usize) = .empty;
    var kept: std.ArrayList(known.Entry) = .empty;
    const replies = &srv.teacher.replies;
    for (qs, 0..) |q, i| {
        const e = if (q.thinking) null else replies.get(srv.io, known.key(null, q.before, q.question, q.max));
        if (e) |x| {
            try xs.append(a, .{ .ids = try std.mem.concat(a, u32, &.{ x.prompt, x.tokens }), .start = @intCast(x.prompt.len), .decide = @intCast(x.prompt.len - 1) });
            try which.append(a, i);
            try kept.append(a, x);
        } else {
            ask_now[n] = i;
            n += 1;
        }
    }
    const began = std.Io.Clock.awake.now(srv.io);
    const verdicts = score(srv, a, xs.items);
    const check = scoreCheck(srv.io); // a diagnostic: the unchanged ones generated too
    var same: usize = 0;
    var claimed: std.ArrayList(usize) = .empty; // [which]: kept as scored, compared once generated (check)
    for (which.items, kept.items, 0..) |i, x, j| {
        if (verdicts != null and verdicts.?[j]) {
            out[i] = x.asReply();
            same += 1;
            if (!check) continue;
            try claimed.append(a, i);
        }
        ask_now[n] = i;
        n += 1;
    }
    const scoring = @as(f64, @floatFromInt(began.durationTo(std.Io.Clock.awake.now(srv.io)).toNanoseconds())) / 1e9;
    const Job = struct {
        srv: *Server,
        a: Allocator,
        qs: []const Asking,
        at: []const usize,
        gone: @TypeOf(gone),
        out: []chat.Reply,

        pub fn run(j: *const @This(), k: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            const i = j.at[k];
            const q = j.qs[i];
            j.out[i] = if (q.thinking) try slide_ask.askThinking(j.srv, &jcx, q.question, j.gone) else if (q.before.len > 0) try slide_ask.askIn(j.srv, &jcx, q.before, q.question, q.max, j.gone) else try slide_ask.ask(j.srv, &jcx, null, q.question, q.max, j.gone);
        }
    };
    const generated = try a.alloc(chat.Reply, qs.len);
    try each(n, &Job{ .srv = srv, .a = a, .qs = qs, .at = ask_now[0..n], .gone = gone, .out = if (check) generated else out });
    if (check) {
        // the generated replies are the answer; each one scored unchanged must equal its kept reply token for token
        var wrong: usize = 0;
        for (ask_now[0..n]) |i| {
            if (std.mem.indexOfScalar(usize, claimed.items, i) != null and !std.mem.eql(u32, out[i].tokens, generated[i].tokens)) {
                wrong += 1;
                log.line("slide: SCORE CHECK: \"{s}\" scored unchanged, but greedy decoding gave \"{s}\", not \"{s}\"", .{ qs[i].question, generated[i].content, out[i].content });
            }
            out[i] = generated[i];
        }
        var unchanged_missed: usize = 0;
        for (which.items, kept.items) |i, x| {
            if (std.mem.indexOfScalar(usize, claimed.items, i) == null and std.mem.eql(u32, x.tokens, out[i].tokens) and std.mem.eql(u32, x.prompt, out[i].prompt_ids)) unchanged_missed += 1;
        }
        log.line("slide: SCORE CHECK: {d} scored unchanged, {d} of them not as greedy decoding gives them; {d} scored changed that came back the same", .{ claimed.items.len, wrong, unchanged_missed });
    }
    if (xs.items.len > 0) {
        replies.mutex.lockUncancelable(srv.io);
        replies.scored += xs.items.len;
        replies.same += same;
        replies.mutex.unlock(srv.io);
        log.line("slide: {d} questions: {d} asked before, {d} of them unchanged (scored in {d:.1} s{s}), {d} generated", .{ qs.len, xs.items.len, same, scoring, if (verdicts == null) ", the learner could not score" else "", n });
    }
    return out;
}

/// TF_SLIDE_SCORE_CHECK: "1", or a file that turns the check on while it exists.
fn scoreCheck(io: std.Io) bool {
    const v = std.mem.span(std.c.getenv("TF_SLIDE_SCORE_CHECK") orelse return false);
    if (std.mem.eql(u8, v, "1")) return true;
    _ = std.Io.Dir.cwd().statFile(io, v, .{}) catch return false;
    return true;
}

test "every item runs once, side by side, and the first error comes back" {
    const Job = struct {
        hits: []std.atomic.Value(u32),
        fail_at: usize,

        pub fn run(j: *const @This(), i: usize) !void {
            _ = j.hits[i].fetchAdd(1, .monotonic);
            if (i == j.fail_at) return error.Broken;
        }
    };
    var hits: [37]std.atomic.Value(u32) = undefined;
    for (&hits) |*h| h.* = .init(0);
    try each(hits.len, &Job{ .hits = &hits, .fail_at = 99 });
    for (&hits) |*h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));
    try std.testing.expectError(error.Broken, each(hits.len, &Job{ .hits = &hits, .fail_at = 5 }));
}

/// Whether each reply in `xs` (prompt, then reply from `start`) is its prompt's greedy reply now; null if unscorable.
pub fn score(srv: *Server, a: std.mem.Allocator, xs: []const @import("engine_api").Example) ?[]const bool {
    return judged(srv, a, xs, &.{ .score = xs });
}

/// Whether the open lesson's pocket neurons stay at or below zero on every row of each of `xs`; null if unreadable.
pub fn quiet(srv: *Server, a: std.mem.Allocator, xs: []const @import("engine_api").Example) ?[]const bool {
    return judged(srv, a, xs, &.{ .quiet = xs });
}

fn judged(srv: *Server, a: std.mem.Allocator, xs: []const @import("engine_api").Example, request: *const @import("engine_api").LearnRequest) ?[]const bool {
    if (xs.len == 0) return &.{};
    return switch (slide.run(srv, a, request, 0)) {
        .scored => |same| if (same.len == xs.len) same else null,
        else => null,
    };
}
