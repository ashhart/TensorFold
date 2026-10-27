//! A prompt-row budget beside decoders (TENSORFOLD_PREFILL_BUDGET) on the real host: calls and tokens unchanged.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const fill = @import("lane_host_fill.zig");
const t = std.testing;

const short_count = 24;
const long_id = short_count + 2;
const job_count = long_id;
const max_tokens = 256;
const budget_rows = fill.Share.short_rows; // one short prompt's rows a visit

const Options = struct {
    slots: u32 = 4,
    chunks: usize = 4,
    long_rows: usize = 0,
    long_background: bool = false,
    progress: enum { reported, hidden, stale } = .reported,
    drafts: bool = false,
    sampled: bool = false,
    cancel: bool = false,
    short_tokens: u32 = 8,
    budget: u64 = budget_rows, // 0: none, main's order
};

const Output = struct {
    tokens: [max_tokens]u32 = undefined,
    len: usize = 0,
    reason: ?api.Reason = null,
    calls: usize = 0,
};

const Report = struct {
    output: [job_count]Output = @splat(.{}),
    max_calls: usize = 0,
    max_rows: usize = 0,
    short_calls_before_long: ?usize = null,
};

const Watch = struct {
    engine: api.Engine,
    requests: *const [job_count]api.Request,
    options: Options,
    report: *Report,
    done: std.atomic.Value(usize) = .init(0),
    sent: bool = false,
    failed: bool = false,
    decoder: bool = false,
    calls: usize = 0,
    rows: usize = 0,
    short_calls: usize = 0,

    fn prefill(ctx: *anyopaque, stream: *lanes.Stream, chunk: usize) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        const prompt = stream.prompt();
        const id = prompt[0] * 8 + prompt[1];
        const out = &w.report.output[id - 1];
        if (out.calls != chunk) w.failed = true;
        out.calls += 1;
        if (id == 1) return;
        if (id == long_id) {
            if (w.report.short_calls_before_long == null) w.report.short_calls_before_long = w.short_calls;
        } else w.short_calls += 1;
        if (w.decoder) {
            w.calls += 1;
            w.rows += @intCast(stream.prompt_len / w.options.chunks);
            w.report.max_calls = @max(w.report.max_calls, w.calls);
            w.report.max_rows = @max(w.report.max_rows, w.rows);
        }
        if (w.options.cancel and id == 2 and chunk == 1) {
            w.engine.cancel(2);
            w.engine.cancel(short_count + 1);
        }
    }

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        const out = &w.report.output[id - 1];
        switch (e.*) {
            .tokens => |tokens| {
                if (out.len + tokens.len > out.tokens.len) {
                    w.failed = true;
                    return;
                }
                @memcpy(out.tokens[out.len..][0..tokens.len], tokens);
                out.len += tokens.len;
                if (id != 1) return;
                w.decoder = true;
                w.calls = 0;
                w.rows = 0;
                if (w.sent) return;
                w.sent = true;
                for (2..short_count + 2) |next| w.submit(next);
            },
            .finished => |f| {
                out.reason = f.reason;
                if (id == 1) w.decoder = false;
                _ = w.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }

    fn submit(w: *@This(), id: usize) void {
        w.engine.submit(@intCast(id), &w.requests[id - 1], .{ .ctx = w, .event = event }) catch {
            w.failed = true;
        };
    }
};

fn staleProgress(ptr: *anyopaque, stream: *lanes.Stream) !bool {
    const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
    const before = stream.fill_at.load(.monotonic);
    const done = try target.stepped().vtable.prefill_step.?(ptr, stream);
    stream.fill_at.store(before, .monotonic);
    return done;
}

fn run(options: Options) !Report {
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = options.chunks };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var vt = target.stepped().vtable.*;
    if (options.progress == .hidden) vt.reports_fill = false;
    if (options.progress == .stale) vt.prefill_step = staleProgress;
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = options.slots });
    host.share = .{ .rows = options.budget };
    host.prefix = .{ .ns = 0 };
    var prompts: [job_count][4096]u32 = undefined;
    var requests: [job_count]api.Request = undefined;
    for (&prompts, &requests, 1..) |*prompt, *request, id| {
        @memset(prompt, @intCast(id % 8));
        prompt[0] = @intCast(id / 8);
        prompt[1] = @intCast(id % 8);
        const len = if (id == 1) 4 else if (id == long_id) options.long_rows else 512;
        request.* = .{
            .prompt = prompt[0..len],
            .max_tokens = if (id == 1) max_tokens else options.short_tokens,
            .drafts = options.drafts,
            .background = id == long_id and options.long_background,
            .sampling = if (options.sampled) .{ .seed = 1971 + id, .temperature = 0.8, .top_k = 8 } else null,
        };
    }
    var report: Report = .{};
    var watch: Watch = .{ .engine = host.engine(), .requests = &requests, .options = options, .report = &report };
    target.prefill_hook = Watch.prefill;
    target.prefill_hook_ctx = &watch;
    try watch.engine.submit(1, &requests[0], .{ .ctx = &watch, .event = Watch.event });
    if (options.long_rows > 0) try watch.engine.submit(long_id, &requests[long_id - 1], .{ .ctx = &watch, .event = Watch.event });
    try host.start();
    var started = true;
    defer if (started) host.stop();
    const expected: usize = 1 + short_count + @as(usize, @intFromBool(options.long_rows > 0));
    for (0..5000) |_| {
        if (watch.done.load(.acquire) == expected) break;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try t.expectEqual(expected, watch.done.load(.acquire));
    host.stop();
    started = false;
    try t.expect(!watch.failed);
    try t.expectEqual(@as(usize, 0), target.lanes.count());
    for (report.output[0..expected], 1..) |out, id| {
        const cancelled = options.cancel and (id == 2 or id == short_count + 1);
        try t.expectEqual(@as(?api.Reason, if (cancelled) .cancelled else .length), out.reason);
        if (!cancelled) try t.expectEqual(options.chunks, out.calls);
    }
    return report;
}

fn sameTokens(a: *const Report, b: *const Report, count: usize) !void {
    for (a.output[0..count], b.output[0..count]) |x, y| {
        try t.expectEqual(x.len, y.len);
        try t.expectEqualSlices(u32, x.tokens[0..x.len], y.tokens[0..y.len]);
    }
}

test "prefill service: a decoder waits on one budget of calls at most, on main's lanes of prompts without one" {
    const report = try run(.{ .short_tokens = 1 });
    try t.expectEqual(@as(usize, 4), report.max_calls);
    try t.expectEqual(@as(usize, budget_rows), report.max_rows);
    const off = try run(.{ .short_tokens = 1, .budget = 0 });
    try t.expectEqual(@as(usize, 16), off.max_calls); // four lanes of four-call prompts, as on main
    try t.expectEqual(@as(usize, 2048), off.max_rows);
    try sameTokens(&off, &report, short_count + 1);
}

test "prefill service: unknown or stale progress yields after a call and still completes" {
    const ordinary = try run(.{});
    for ([_]Options{ .{ .progress = .hidden }, .{ .progress = .stale } }) |options| {
        const report = try run(options);
        try t.expectEqual(@as(usize, 1), report.max_calls);
        try sameTokens(&ordinary, &report, short_count + 1);
    }
}

test "prefill service: due aged long fills progress beside sustained short arrivals" {
    for ([_]bool{ false, true }) |background| {
        const report = try run(.{ .long_rows = 2048, .long_background = background, .short_tokens = 1 });
        try t.expect(report.short_calls_before_long.? <= fill.preference_rounds);
        try t.expectEqual(@as(usize, 4), report.output[long_id - 1].calls);
        try t.expect(report.max_rows <= budget_rows + 512 - 1);
    }
}

test "prefill service: an indivisible long call may exceed the allowance without being split" {
    const report = try run(.{ .long_rows = 4096, .short_tokens = 1 });
    try t.expectEqual(@as(usize, 4), report.output[long_id - 1].calls);
    try t.expect(report.max_rows >= 1024 and report.max_rows < budget_rows + 1024);
}

test "prefill service: greedy and sampled outputs match solo and draft-disabled execution" {
    for ([_]bool{ false, true }) |sampled| {
        const solo = try run(.{ .slots = 1, .sampled = sampled });
        const concurrent = try run(.{ .sampled = sampled });
        const drafted = try run(.{ .drafts = true, .sampled = sampled });
        const off = try run(.{ .sampled = sampled, .budget = 0 });
        try sameTokens(&solo, &concurrent, short_count + 1);
        try sameTokens(&solo, &drafted, short_count + 1);
        try sameTokens(&solo, &off, short_count + 1);
    }
}

test "prefill service: cancelled filling and queued jobs do not poison later output or ownership" {
    const normal = try run(.{});
    const cancelled = try run(.{ .cancel = true });
    try t.expectEqual(@as(usize, 0), cancelled.output[1].len);
    try t.expectEqual(@as(usize, 0), cancelled.output[short_count].len);
    for (2..short_count) |i| {
        const x = normal.output[i];
        const y = cancelled.output[i];
        try t.expectEqual(x.len, y.len);
        try t.expectEqualSlices(u32, x.tokens[0..x.len], y.tokens[0..y.len]);
    }
}

test "prefill service: prompts admitted with no stream decoding go in before a round, past the budget's rows" {
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 2 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = 3 });
    host.share = .{ .rows = budget_rows };
    const Count = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: usize = 0,
        tokens: usize = 0,
        at_last: ?usize = null, // tokens out when the third prompt was in
        done: usize = 0,
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(t.io);
            defer o.mutex.unlock(t.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled += 1;
                    if (o.prefilled == 3) o.at_last = o.tokens;
                },
                .tokens => |x| o.tokens += x.len,
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var count: Count = .{};
    var prompts: [3][300]u32 = undefined; // 900 rows: a budget counted from the first prompt's stream would stop at two
    var requests: [3]api.Request = undefined;
    for (&requests, &prompts, 2..) |*r, *p, value| {
        @memset(p, @intCast(value));
        r.* = .{ .prompt = p, .max_tokens = 4 };
    }
    const e = host.engine();
    for (&requests, 1..) |*r, id| try e.submit(@intCast(id), r, .{ .ctx = &count, .event = Count.event });
    try host.start();
    defer host.stop();
    while (true) {
        count.mutex.lockUncancelable(t.io);
        const d = count.done;
        count.mutex.unlock(t.io);
        if (d == 3) break;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try t.expect(count.at_last.? <= 2); // at most the first two prompts' first tokens: no round ran between the passes
}
