//! Prompt progress (``return_progress``) from the lane host's fill queue and a lone driver's chunked pass.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const Host = @import("lane_host.zig").LaneHost;
const progress = @import("lane_host_progress.zig");
const t = std.testing;
const a = t.allocator;

/// Each event's kind in order (p: progress, c: prefilled, t: tokens, f: finished), the counts and the tokens.
const Seen = struct {
    mutex: std.Io.Mutex = .init,
    kinds: std.ArrayList(u8) = .empty,
    counts: std.ArrayList(api.Progress) = .empty,
    tokens: std.ArrayList(u32) = .empty,
    done: std.atomic.Value(u32) = .init(0),

    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const s: *Seen = @ptrCast(@alignCast(ptr));
        s.mutex.lockUncancelable(t.io);
        defer s.mutex.unlock(t.io);
        const kind: u8 = switch (value.*) {
            .progress => |p| blk: {
                s.counts.append(a, p) catch {};
                break :blk 'p';
            },
            .prefilled => 'c',
            .tokens => |x| blk: {
                s.tokens.appendSlice(a, x) catch {};
                break :blk 't';
            },
            .finished => 'f',
            else => return,
        };
        s.kinds.append(a, kind) catch {};
        if (kind == 'f') _ = s.done.fetchAdd(1, .release);
    }

    fn deinit(s: *Seen) void {
        s.kinds.deinit(a);
        s.counts.deinit(a);
        s.tokens.deinit(a);
    }

    fn wait(s: *Seen) !void {
        const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
        while (s.done.load(.acquire) == 0) {
            if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.TestUnexpectedResult;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    /// Counts rise strictly to `total`, all before the prefilled event, and the reply's tokens follow.
    fn expectRising(s: *const Seen, total: u32, at_least: usize) !void {
        try t.expect(s.counts.items.len >= at_least);
        for (s.counts.items, 0..) |p, i| {
            try t.expectEqual(total, p.total);
            if (i > 0) try t.expect(p.done > s.counts.items[i - 1].done);
        }
        try t.expectEqual(total, s.counts.items[s.counts.items.len - 1].done);
        const first_other = std.mem.indexOfNone(u8, s.kinds.items, "p") orelse return error.TestUnexpectedResult;
        try t.expectEqual(s.counts.items.len, first_other); // every count before the prefilled event and any token
        try t.expectEqual(@as(u8, 'c'), s.kinds.items[first_other]);
    }
};

/// One request on a host over the fake's chunked fill; `reports` false hides `fill_at` from the host.
fn serve(seen: *Seen, prompt: []const u32, asked: bool, chunks: usize, reports: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = chunks };
    defer target.deinit();
    var vt = target.stepped().vtable.*;
    vt.reports_fill = reports;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt, .chunk_invariant = true }, clock.clock());
    defer core.deinit();
    var host = Host.init(a, t.io, &core, .{ .lanes = 2 });
    const request: api.Request = .{ .prompt = prompt, .max_tokens = 3, .return_progress = asked };
    try host.engine().submit(1, &request, .{ .ctx = seen, .event = Seen.event });
    try host.start();
    defer host.stop();
    try seen.wait();
}

const prompt32: [32]u32 = @splat(3);

test "prompt progress: a chunked fill's counts rise to its prompt before its first token, its tokens unchanged" {
    var off: Seen = .{};
    defer off.deinit();
    try serve(&off, &prompt32, false, 4, true);
    var on: Seen = .{};
    defer on.deinit();
    try serve(&on, &prompt32, true, 4, true);
    try on.expectRising(32, 4); // a count after each of the four calls, the last the whole prompt
    for (on.counts.items) |p| try t.expectEqual(@as(u32, 0), p.cached);
    try t.expectEqualSlices(u32, off.tokens.items, on.tokens.items);
}

test "prompt progress: none without return_progress, for a prompt filled in one call, or without fill reports" {
    for ([_][3]usize{ .{ 0, 4, 1 }, .{ 1, 1, 1 }, .{ 1, 4, 0 } }) |case| {
        var seen: Seen = .{};
        defer seen.deinit();
        try serve(&seen, &prompt32, case[0] == 1, case[1], case[2] == 1);
        try t.expectEqual(@as(usize, 0), seen.counts.items.len);
        try t.expectEqual(@as(u8, 'c'), seen.kinds.items[0]);
    }
}

fn noop(_: *anyopaque, _: api.Id, _: *const api.Event) void {}

test "prompt progress: a count never goes back, a restored prefix counts once and the end is sent once" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var core: lanes.Engine = undefined;
    core.cfg = &cfg;
    var vt = target.stepped().vtable.*;
    core.backend = .{ .ptr = &target, .vtable = &vt };
    var host = Host.init(a, t.io, &core, .{ .lanes = 2 });
    var seen: Seen = .{};
    defer seen.deinit();
    const request: api.Request = .{ .prompt = &prompt32, .max_tokens = 1, .return_progress = true };
    var job: Host.Job = .{ .host = &host, .id = 1, .request = &request, .sink = .{ .ctx = &seen, .event = Seen.event } };
    job.stream.prompt_len = prompt32.len;
    job.stream.cached = 8;
    job.stream.fill_at = .init(8);
    progress.chunk(&host, &job); // the restored prefix alone: counted once, in `done` and `cached`
    for ([_]u32{ 8, 16, 12, 16, 24 }) |at| {
        job.stream.fill_at.store(at, .monotonic); // 12: a pass requeued after its lane memory failed
        progress.chunk(&host, &job);
    }
    progress.filled(&job);
    progress.filled(&job);
    const want = [_]api.Progress{ .{ .done = 8, .total = 32, .cached = 8 }, .{ .done = 16, .total = 32, .cached = 8 }, .{ .done = 24, .total = 32, .cached = 8 }, .{ .done = 32, .total = 32, .cached = 8 } };
    try t.expectEqualSlices(api.Progress, &want, seen.counts.items);
    var quiet: Host.Job = .{ .host = &host, .id = 2, .request = &request, .sink = .{ .ctx = &seen, .event = noop } };
    quiet.stream.prompt_len = prompt32.len;
    quiet.stream.cached = 0;
    progress.filled(&quiet); // one call, no count before it: no end either
    try t.expectEqual(@as(u32, 0), quiet.progress_rows);
}

/// A lone driver that fills its prompt a call at a time, as Flash Next's does, and hands over after its first token.
const SteppedLone = struct {
    be: lanes.backend.Backend,
    yields: u32 = 0,

    fn run(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const l: *SteppedLone = @ptrCast(@alignCast(ctx));
        while (!try l.be.vtable.prefill_step.?(l.be.ptr, s, 0)) {
            l.yields += 1;
            if (hooks.prefill_yield.?(hooks.ctx)) return false;
        }
        const token = try l.be.read(try l.be.first(s, s.prompt_len));
        _ = try s.commit(a, &.{token}, &.{});
        hooks.committed(hooks.ctx);
        s.pending = s.context.items[s.context.items.len - 1];
        s.cache_len = s.prompt_len; // as a backend's hand-over leaves it: the first token not yet fed
        return true;
    }
};

fn serveLone(seen: *Seen, asked: bool) !u32 {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = Host.init(a, t.io, &core, .{ .lanes = 2 });
    var lone: SteppedLone = .{ .be = target.stepped() };
    host.lone = .{ .ctx = &lone, .run = SteppedLone.run, .plain = true };
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &prompt32, .max_tokens = 3, .return_progress = asked };
    try host.engine().submit(1, &request, .{ .ctx = seen, .event = Seen.event });
    try seen.wait();
    return lone.yields;
}

test "prompt progress: a lone driver's chunked pass counts each call as the fill queue does" {
    var off: Seen = .{};
    defer off.deinit();
    try t.expectEqual(@as(u32, 3), try serveLone(&off, false));
    try t.expectEqual(@as(usize, 0), off.counts.items.len);
    var on: Seen = .{};
    defer on.deinit();
    try t.expectEqual(@as(u32, 3), try serveLone(&on, true));
    try on.expectRising(32, 4);
    try t.expectEqualSlices(u32, off.tokens.items, on.tokens.items);
}
