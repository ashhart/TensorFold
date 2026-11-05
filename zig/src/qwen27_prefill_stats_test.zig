//! Completed prompt passes through the real lane host and server mailbox, with a CPU target and no shaders.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const Mailbox = @import("server/mailbox.zig").Mailbox;
const rows = @import("native/qwen27_prompt_rows.zig");
const t = std.testing;
const gpa = t.allocator;

/// The fake supplies actual completed slice boundaries; only the production collector observes them.
const Observe = struct {
    allocator: std.mem.Allocator = gpa,
    enabled: bool = true,
    seed_capacity: bool = false,
    last: usize = 0,
    lone_calls: usize = 0,
    backend: lanes.backend.Backend = undefined,

    fn of(f: *lanes.fake.Fake) *Observe {
        return @ptrCast(@alignCast(f.prefill_hook_ctx.?));
    }

    fn whole(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        of(f).last = 0;
        f.prefill_hook = cut;
        defer f.prefill_hook = null;
        try f.backend().prefill(s);
    }

    fn cut(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const o: *Observe = @ptrCast(@alignCast(ctx));
        const at = s.fill_at.load(.monotonic);
        if (o.enabled) rows.completed(o.allocator, s, at - o.last);
        o.last = at;
    }

    fn step(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const before = s.fill_at.load(.monotonic);
        const done = try f.stepped().vtable.prefill_step.?(ptr, s, max_rows);
        if (of(f).enabled and of(f).seed_capacity and s.prefill_widths.capacity == 0) {
            const storage = try gpa.alloc(u32, 1);
            s.prefill_widths = .fromOwnedSlice(storage);
            s.prefill_widths.clearRetainingCapacity();
        }
        if (of(f).enabled) rows.completed(of(f).allocator, s, s.fill_at.load(.monotonic) - before);
        return done;
    }

    /// Exercise the host's lone scheduler path; this is a CPU driver, not a real DFlash model.
    fn lone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const o: *Observe = @ptrCast(@alignCast(ctx));
        o.lone_calls += 1;
        _ = try o.backend.opening(gpa, s);
        hooks.committed(hooks.ctx);
        while (!s.finished) {
            const token = lanes.fake.next(s.context.items, s.sampling, s.context.items.len);
            _ = try s.commit(gpa, &.{token}, &.{});
            hooks.committed(hooks.ctx);
        }
        o.backend.release(s);
        return false;
    }
};

const Mode = enum { stepped, serial, lone };

fn wait(box: *Mailbox) !void {
    for (0..10_000) |_| {
        box.mutex.lockUncancelable(t.io);
        const done = box.finished;
        box.mutex.unlock(t.io);
        if (done) return;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

/// Two successive requests own distinct lists. The mailbox must copy each before the worker frees its stream.
fn serve(mode: Mode, n: usize, drafts: bool, observe: *Observe, boxes: []Mailbox) !void {
    var costs: [8]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 3, .window_costs = &costs, .mtp_step_ms = 0.1, .batch_rows = 2048, .max_streams = 2, .draft_streams = true }, 8, 7);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook_ctx = observe };
    defer target.deinit();
    var vt = target.backend().vtable.*;
    vt.prefill = Observe.whole;
    if (mode == .stepped) vt.prefill_step = Observe.step;
    const backend: lanes.backend.Backend = .{ .ptr = &target, .vtable = &vt, .chunk_invariant = true };
    observe.backend = backend;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, t.io, &core, .{ .lanes = 2, .prefill_step = 2048 });
    if (mode == .lone) host.lone = .{ .ctx = observe, .run = Observe.lone, .resumes = false, .plain = true };
    const prompt = try gpa.alloc(u32, n);
    defer gpa.free(prompt);
    for (prompt, 0..) |*p, i| p.* = @intCast(i % lanes.fake.vocab);
    const request: api.Request = .{ .prompt = prompt, .max_tokens = 24, .drafts = drafts };
    try host.start();
    defer host.stop();
    for (boxes, 1..) |*box, id| {
        try host.engine().submit(id, &request, .{ .ctx = box, .event = Mailbox.onEvent });
        try wait(box);
        try t.expectEqual(api.Reason.length, box.reason);
        try t.expectEqual(@as(usize, 24), box.tokens.items.len);
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, prompt);
        for (box.tokens.items) |token| {
            try t.expectEqual(lanes.fake.next(history.items, null, history.items.len), token);
            try history.append(gpa, token);
        }
    }
}

fn newBox() Mailbox {
    return .{ .io = t.io, .gpa = gpa };
}

fn sameReply(a: *const Mailbox, b: *const Mailbox) !void {
    try t.expectEqualSlices(u32, a.tokens.items, b.tokens.items);
    try t.expectEqual(a.stats.rounds, b.stats.rounds);
    try t.expectEqual(a.stats.drafted, b.stats.drafted);
    try t.expectEqual(a.stats.accepted, b.stats.accepted);
    try t.expectEqual(a.stats.min_rows, b.stats.min_rows);
}

test "qwen27 prefill stats: stepped 16k reports eight actual 2048 passes with request-owned lifetime" {
    var boxes = [_]Mailbox{ newBox(), newBox() };
    defer for (&boxes) |*b| b.deinit();
    var observe: Observe = .{};
    try serve(.stepped, 16384, false, &observe, &boxes);
    for (&boxes) |*b| try t.expectEqualSlices(u32, &.{ 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048 }, b.widths);
    try sameReply(&boxes[0], &boxes[1]);
}

test "qwen27 prefill stats: serial reports its nonuniform actual cuts rather than capacity" {
    var boxes = [_]Mailbox{newBox()};
    defer boxes[0].deinit();
    var observe: Observe = .{};
    try serve(.serial, 16381, false, &observe, &boxes);
    try t.expectEqualSlices(u32, &.{ 2047, 2048, 2047, 2048, 2048, 2047, 2048, 2048 }, boxes[0].widths);
}

test "qwen27 prefill stats: drafted verify rounds do not append prompt widths or change tokens and counters" {
    var on = [_]Mailbox{newBox()};
    defer on[0].deinit();
    var off = [_]Mailbox{newBox()};
    defer off[0].deinit();
    var observe: Observe = .{};
    try serve(.stepped, 16384, true, &observe, &on);
    observe.enabled = false;
    try serve(.stepped, 16384, true, &observe, &off);
    try t.expect(on[0].stats.rounds > 0);
    try t.expect(on[0].stats.drafted > 0);
    try t.expectEqualSlices(u32, &.{ 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048 }, on[0].widths);
    try t.expectEqual(@as(usize, 0), off[0].widths.len);
    try sameReply(&on[0], &off[0]);
}

test "qwen27 prefill stats: lone scheduler forwards serial pass widths" {
    var boxes = [_]Mailbox{newBox()};
    defer boxes[0].deinit();
    var observe: Observe = .{};
    try serve(.lone, 16384, true, &observe, &boxes);
    try t.expectEqual(@as(usize, 1), observe.lone_calls);
    try t.expectEqualSlices(u32, &.{ 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048 }, boxes[0].widths);
}

test "qwen27 prefill stats: observation allocation failure keeps inference and emits no partial list" {
    var failed = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0, .resize_fail_index = 0 });
    var bad = [_]Mailbox{newBox()};
    defer bad[0].deinit();
    var off = [_]Mailbox{newBox()};
    defer off[0].deinit();
    var observe: Observe = .{ .allocator = failed.allocator(), .seed_capacity = true };
    // One owned slot admits the first observation; the forced growth failure must discard it on the second.
    try serve(.stepped, 16384, false, &observe, &bad);
    observe.enabled = false;
    try serve(.stepped, 16384, false, &observe, &off);
    try t.expect(failed.has_induced_failure);
    try t.expectEqual(@as(usize, 0), bad[0].widths.len);
    try sameReply(&bad[0], &off[0]);
}
