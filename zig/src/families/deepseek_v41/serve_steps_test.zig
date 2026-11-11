//! Prompts in steps (serve_lanes Model.steps, draft/lane_steps.zig) through LaneHost over a stepped CPU twin.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const twin = @import("draft/twin.zig");
const iface = @import("draft/iface.zig");
const calib_env = @import("draft/calib_env.zig");
const calib_lanes = @import("calib_lanes.zig");
const serve_lanes = @import("serve_lanes.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const step_rows = 16;

/// The twin as a target with pieces: `begin` / `piece` record progress, `finish` runs the whole prompt on the twin.
const Stepped = struct {
    t: *twin.Twin,
    d: *twin.Drafter,
    steps: bool,
    /// ms a piece (and a whole prompt, per step's rows) takes, so other requests arrive while a prompt runs
    delay_ms: u32 = 0,
    /// the piece (counted over every slot, from 1) that fails with error.Planted; 0: none
    fail_at: u64 = 0,
    pieces: std.atomic.Value(u64) = .init(0),
    prompts: std.atomic.Value(u64) = .init(0),
    windows: std.atomic.Value(u64) = .init(0),
    abandoned: u64 = 0,
    /// decode rounds of other slots while each slot's prompt ran in pieces (begin to finish), the last one's
    began_windows: [4]u64 = @splat(0),
    between: u64 = 0,
    /// prompt lengths in the order their first choice was drawn
    order: std.ArrayList(usize) = .empty,
    at: [4]?u64 = @splat(null),

    fn model(x: *Stepped, slots: u32) serve_lanes.Model {
        return .{ .ptr = x, .name = "dsv41-twin-steps", .slots = slots, .limit = 4096, .world = 1, .prefill_step = step_rows, .forward_rows = 16, .shape = x.d.shape, .target = targetFn, .pass = passFn, .calibrate = calibrate, .close = close, .calib = .{ .mode = "default" }, .steps = x.steps };
    }

    fn of(p: *anyopaque) *Stepped {
        return @ptrCast(@alignCast(p));
    }

    fn targetFn(ctx: *anyopaque) iface.Target {
        return .{ .ptr = ctx, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .begin = begin, .piece = piece, .finish = finish } };
    }

    fn passFn(ctx: *anyopaque) ?iface.Pass {
        return of(ctx).d.pass();
    }

    fn calibrate(_: *anyopaque, _: std.mem.Allocator, _: std.Io, _: []const u8, _: calib_env.Knobs, _: calib_env.Shape) anyerror!calib_lanes.Boot {
        return error.Unmeasured;
    }

    fn close(_: *anyopaque) void {}

    fn inner(x: *Stepped) iface.Target {
        return x.t.target();
    }

    fn wait(x: *const Stepped, rows: usize) void {
        if (x.delay_ms > 0) std.Io.sleep(io, .fromMilliseconds(@intCast(x.delay_ms * (rows + step_rows - 1) / step_rows)), .awake) catch {};
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const x = of(p);
        _ = x.prompts.fetchAdd(1, .monotonic);
        x.wait(ids.len);
        const c = try twinPrompt(x.inner(), slot, ids, smp, draw);
        try x.order.append(gpa, ids.len);
        return c;
    }

    fn begin(p: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
        const x = of(p);
        if (x.at[slot] != null or ids.len == 0) return error.SlotLive;
        _ = x.prompts.fetchAdd(1, .monotonic);
        x.at[slot] = 0;
        x.began_windows[slot] = x.windows.load(.monotonic);
        return .{};
    }

    fn piece(p: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, _: bool) anyerror!void {
        const x = of(p);
        const at = x.at[slot] orelse return error.NotBegun;
        if (start != at or end <= start or end >= ids.len) return error.NotAChain;
        const n = x.pieces.fetchAdd(1, .monotonic) + 1;
        if (n == x.fail_at) return error.Planted;
        x.wait(end - start);
        x.at[slot] = end;
    }

    fn finish(p: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const x = of(p);
        const at = x.at[slot] orelse return error.NotBegun;
        x.wait(ids.len - at);
        x.at[slot] = null;
        x.between = x.windows.load(.monotonic) - x.began_windows[slot];
        const c = try twinPrompt(x.inner(), slot, ids, smp, draw);
        try x.order.append(gpa, ids.len);
        return c;
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        _ = of(p).windows.fetchAdd(1, .monotonic);
        return of(p).inner().window(segments, choices);
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        return of(p).inner().keep(slot, path);
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        return of(p).inner().taps(slot);
    }

    fn release(p: *anyopaque, slot: u32) void {
        const x = of(p);
        if (x.at[slot] != null) x.abandoned += 1;
        x.at[slot] = null;
        x.inner().release(slot);
    }
};

/// A reply's tokens, finish and first-token order, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    reason: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| b.out.appendSlice(gpa, ts) catch {},
            .finished => |fin| {
                b.done = true;
                b.reason = fin.reason;
            },
            else => {},
        }
    }

    fn wait(b: *Box) !void {
        for (0..60_000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// A served stepped twin of `slots` slots; `x` holds the twin and the drafter for the caller to free.
const Rig = struct {
    x: Stepped,
    s: *serve_lanes.Served = undefined,

    fn open(r: *Rig, slots: u32, steps: bool) !void {
        const t = try twin.Twin.init(gpa, .{}, slots);
        errdefer t.deinit();
        const d = try twin.Drafter.init(gpa, t, slots);
        errdefer d.deinit();
        r.x = .{ .t = t, .d = d, .steps = steps };
        r.s = try serve_lanes.Served.open(gpa, io, r.x.model(slots), ".", null);
    }

    fn close(r: *Rig) void {
        serve_lanes.Served.close(r.s);
        r.x.order.deinit(gpa);
        r.x.d.deinit();
        r.x.t.deinit();
    }

    fn submit(r: *Rig, id: api.Id, req: *const api.Request, b: *Box) !void {
        try r.s.engine().submit(id, req, .{ .ctx = b, .event = Box.event });
    }
};

fn prompt(n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try gpa.alloc(u32, n);
    for (out) |*v| v.* = prng.random().uintLessThan(u32, 48);
    return out;
}

/// The twin's prompt, its first rows a prefill and the rest in kept windows (a twin window holds 64 rows at most).
fn twinPrompt(t: iface.Target, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) !u32 {
    const w = 48;
    const first = @min(w, ids.len);
    var c = try t.prefill(slot, ids[0..first], smp, if (first == ids.len) draw else first);
    var at: usize = first;
    var draws: [w]u64 = undefined;
    var out: [w]u32 = undefined;
    const path: [w]u32 = std.simd.iota(u32, w);
    while (at < ids.len) {
        const n = @min(w, ids.len - at);
        for (draws[0..n], 0..) |*d, i| d.* = at + i + 1;
        if (at + n == ids.len) draws[n - 1] = draw;
        var cs = [_][]u32{out[0..n]};
        try t.window(&.{.{ .slot = slot, .start = at, .tokens = ids[at..][0..n], .parents = null, .draws = draws[0..n], .sampling = smp }}, &cs);
        try t.keep(slot, path[0..n]);
        c = out[n - 1];
        at += n;
    }
    return c;
}

/// The twin's serial greedy reply: the prompt, then one row a step.
fn serial(p: []const u32, n: usize) ![]u32 {
    const ref = try twin.Twin.init(gpa, .{}, 1);
    defer ref.deinit();
    const t = ref.target();
    const out = try gpa.alloc(u32, n);
    errdefer gpa.free(out);
    var tok = try twinPrompt(t, 0, p, null, p.len);
    for (out, 0..) |*o, i| {
        o.* = tok;
        const at: u64 = p.len + i;
        var c: [1]u32 = undefined;
        var cs = [_][]u32{&c};
        try t.window(&.{.{ .slot = 0, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = null }}, &cs);
        tok = c[0];
    }
    return out;
}

fn until(v: *const std.atomic.Value(u64), n: u64) !void {
    for (0..60_000) |_| {
        if (v.load(.monotonic) >= n) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "serve steps: a decoding reply and prompts in steps == serial; decode rounds ran between a prompt's pieces" {
    const t = std.testing;
    const p0 = try prompt(10, 1);
    defer gpa.free(p0);
    const p1 = try prompt(700, 2);
    defer gpa.free(p1);
    const w0 = try serial(p0, 120);
    defer gpa.free(w0);
    const w1 = try serial(p1, 40);
    defer gpa.free(w1);
    var r: Rig = undefined;
    try r.open(4, true);
    defer r.close();
    r.x.delay_ms = 1;
    var boxes: [2]Box = @splat(.{});
    defer for (&boxes) |*b| b.out.deinit(gpa);
    const reqs = [2]api.Request{ .{ .prompt = p0, .max_tokens = 120 }, .{ .prompt = p1, .max_tokens = 40 } };
    try r.submit(1, &reqs[0], &boxes[0]);
    try until(&r.x.windows, 1);
    try r.submit(2, &reqs[1], &boxes[1]);
    for (&boxes) |*b| try b.wait();
    try t.expectEqualSlices(u32, w0, boxes[0].out.items);
    try t.expectEqualSlices(u32, w1, boxes[1].out.items);
    // 700 rows in 16-row steps: 43 pieces, and the decoding reply's rounds between them
    try t.expectEqual(@as(u64, 43), r.x.pieces.load(.monotonic));
    try t.expect(r.x.between > 0);
}

test "serve steps: a short prompt arriving during a long one draws its first token first; whole prompts do not" {
    const t = std.testing;
    const long = try prompt(1600, 3);
    defer gpa.free(long);
    const short = try prompt(20, 4);
    defer gpa.free(short);
    for ([_]bool{ false, true }) |steps| {
        var r: Rig = undefined;
        try r.open(2, steps);
        defer r.close();
        r.x.delay_ms = 2;
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.out.deinit(gpa);
        const reqs = [2]api.Request{ .{ .prompt = long, .max_tokens = 4 }, .{ .prompt = short, .max_tokens = 4 } };
        try r.submit(1, &reqs[0], &boxes[0]);
        if (steps) try until(&r.x.pieces, 3) else try until(&r.x.prompts, 1);
        try r.submit(2, &reqs[1], &boxes[1]);
        for (&boxes) |*b| try b.wait();
        const first: []const usize = if (steps) &.{ short.len, long.len } else &.{ long.len, short.len };
        try t.expectEqualSlices(usize, first, r.x.order.items);
    }
}

test "serve steps: a failed piece and a cancel mid-prompt free the slot; the next requests on it == serial" {
    const t = std.testing;
    const long = try prompt(600, 5);
    defer gpa.free(long);
    const p = try prompt(30, 6);
    defer gpa.free(p);
    const want = try serial(p, 20);
    defer gpa.free(want);
    var r: Rig = undefined;
    try r.open(1, true);
    defer r.close();
    r.x.fail_at = 3;
    var failed: Box = .{};
    defer failed.out.deinit(gpa);
    const rl: api.Request = .{ .prompt = long, .max_tokens = 8 };
    try r.submit(1, &rl, &failed);
    try failed.wait();
    try t.expect(failed.reason.? == .failed);
    r.x.delay_ms = 2;
    var gone: Box = .{};
    defer gone.out.deinit(gpa);
    const before = r.x.pieces.load(.monotonic);
    try r.submit(2, &rl, &gone);
    try until(&r.x.pieces, before + 3);
    r.s.engine().cancel(2);
    try gone.wait();
    try t.expect(gone.reason.? == .cancelled);
    r.x.delay_ms = 0;
    // one slot: a lane kept by either would leave the next request waiting forever
    var next: Box = .{};
    defer next.out.deinit(gpa);
    const rp: api.Request = .{ .prompt = p, .max_tokens = 20 };
    try r.submit(3, &rp, &next);
    try next.wait();
    try t.expectEqualSlices(u32, want, next.out.items);
    try t.expectEqual(@as(u64, 2), r.x.abandoned);
}

test "serve steps: TF_DSV41_PREFILL_PIECES, an srpt order or a prompt share turns steps on; nothing set keeps whole" {
    const t = std.testing;
    const env = @import("sys").env;
    const names = [_][:0]const u8{ "TF_DSV41_PREFILL_PIECES", "TENSORFOLD_PREFILL_ORDER", "TENSORFOLD_PREFILL_POLICY", "TENSORFOLD_PREFILL_SHARE", "TENSORFOLD_PREFILL_BUDGET" };
    for (names) |n| _ = env.unsetenv(n);
    defer for (names) |n| {
        _ = env.unsetenv(n);
    };
    try t.expect(!try serve_lanes.stepsFromEnv());
    const on = [_][2][:0]const u8{ .{ names[0], "1" }, .{ names[1], "srpt" }, .{ names[2], "fcfs" }, .{ names[3], "0.5" }, .{ names[4], "2048" } };
    for (on) |kv| {
        _ = env.setenv(kv[0], kv[1], 1);
        try t.expect(try serve_lanes.stepsFromEnv());
        _ = env.unsetenv(kv[0]);
    }
    // values LaneHost reads as its defaults keep whole prompts
    const off = [_][2][:0]const u8{ .{ names[0], "0" }, .{ names[1], "fifo" }, .{ names[2], "stock" }, .{ names[3], "1" }, .{ names[4], "0" } };
    for (off) |kv| _ = env.setenv(kv[0], kv[1], 1);
    try t.expect(!try serve_lanes.stepsFromEnv());
}
