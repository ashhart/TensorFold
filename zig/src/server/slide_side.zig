//! Writer calls side by side, decoded together in the engine's lanes; each call writes only its own result.
const std = @import("std");
const chat = @import("chat.zig");
const errors = @import("errors.zig");
const slide_ask = @import("slide_ask.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

pub const lanes = 8; // calls in flight at once, the engine's lanes by default

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

/// A question as `ask` puts it (at most `max` tokens), or with thinking on as `askThinking` does; `base`: as shipped.
pub const Asking = struct { question: []const u8, max: i64 = 48, thinking: bool = false, base: bool = false };

/// Each question's reply, asked side by side.
pub fn askAll(srv: *Server, cx: *Cx, qs: []const Asking, gone: anytype) ![]chat.Reply {
    const out = try cx.a.alloc(chat.Reply, qs.len);
    const Job = struct {
        srv: *Server,
        a: Allocator,
        qs: []const Asking,
        gone: @TypeOf(gone),
        out: []chat.Reply,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            const q = j.qs[i];
            j.out[i] = if (q.base) (if (q.thinking) try slide_ask.askThinkingBase(j.srv, &jcx, q.question, j.gone) else try slide_ask.askBase(j.srv, &jcx, null, q.question, q.max, j.gone)) else if (q.thinking) try slide_ask.askThinking(j.srv, &jcx, q.question, j.gone) else try slide_ask.ask(j.srv, &jcx, null, q.question, q.max, j.gone);
        }
    };
    try each(qs.len, &Job{ .srv = srv, .a = cx.a, .qs = qs, .gone = gone, .out = out });
    return out;
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
