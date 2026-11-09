//! External request counters through the real lane host; internal checks and failed replies never charge them.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const graph = @import("slide_graph.zig");
const Scope = @import("slide_usage_stream.zig").Scope;
const t = std.testing;
const a = t.allocator;

const Reply = struct {
    engine: ?api.Engine = null,
    cancel: bool = false,
    done: std.atomic.Value(bool) = .init(false),
    reason: api.Reason = .failed,
    tokens: std.ArrayList(u32) = .empty,
    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *Reply = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .tokens => |xs| {
                b.tokens.appendSlice(a, xs) catch @panic("test allocation failed");
                if (b.cancel) b.engine.?.cancel(id);
            },
            .finished => |f| {
                b.reason = f.reason;
                b.done.store(true, .release);
            },
            else => {},
        }
    }
    fn wait(b: *Reply) !void {
        const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
        while (!b.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.Timeout;
            try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
        }
    }
};
fn add(g: *graph.Graph, l: u16) u32 {
    const id = g.add("A fictional fact.", "chat", 1);
    g.mark(id, .learned);
    g.placed(id, &.{.{ .layer = l, .neuron = 0, .key = .subject }});
    return id;
}
const Sensor = struct {
    fn fill(_: *anyopaque, s: *lanes.Stream, _: usize) void {
        var m: api.pocket_usage.Masks = .{};
        m.bits[1] = 1;
        s.pocket_usage = m;
    }
};

test "fact usage: real lane completion counts once, with the same forwards and tokens, never internal, failed or cancelled" {
    var g = graph.Graph.init(a, t.io, null);
    defer g.deinit();
    const own = add(&g, 1);
    const other = add(&g, 2);
    var scope: Scope = .{ .graph = &g, .gpa = a };
    defer scope.deinit();
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_hook = Sensor.fill, .prefill_hook_ctx = &g };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, t.io, &core, .{ .pocket_usage = true });
    try host.start();
    defer host.stop();
    const e = host.engine();
    var reference: std.ArrayList(u32) = .empty;
    defer reference.deinit(a);
    var cost: u64 = 0;
    // Internal generation, two generations of one external request, then a separate external request.
    for (0..4) |k| {
        var reply: Reply = .{};
        defer reply.tokens.deinit(a);
        var request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 12, .drafts = false, .usage = if (k == 0) null else scope.observer() };
        var fresh: Scope = .{ .graph = &g, .gpa = a };
        defer fresh.deinit();
        if (k == 3) request.usage = fresh.observer();
        const before = target.rounds;
        const fills = target.prefill_count;
        try e.submit(k + 1, &request, .{ .ctx = &reply, .event = Reply.event });
        try reply.wait();
        try t.expectEqual(api.Reason.length, reply.reason);
        try t.expectEqual(fills + 1, target.prefill_count);
        if (k == 0) {
            try reference.appendSlice(a, reply.tokens.items);
            cost = target.rounds - before;
        } else {
            try t.expectEqualSlices(u32, reference.items, reply.tokens.items);
            try t.expectEqual(cost, target.rounds - before);
        }
        try t.expectEqual(@as(u64, if (k == 0) 0 else if (k == 3) 2 else 1), g.find(own).?.uses);
        try t.expectEqual(@as(u64, 0), g.find(other).?.uses);
    }
    for (0..5) |mode| {
        var fresh: Scope = .{ .graph = &g, .gpa = a };
        defer fresh.deinit();
        var reply: Reply = .{ .engine = e, .cancel = mode == 0 };
        defer reply.tokens.deinit(a);
        if (mode == 1) target.fail_read = target.drawn.items.len;
        defer target.fail_read = null;
        const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = if (mode == 2) 0 else 12, .drafts = false, .background = mode == 3, .untaught = mode == 4, .usage = fresh.observer() };
        try e.submit(10 + mode, &request, .{ .ctx = &reply, .event = Reply.event });
        try reply.wait();
        if (mode == 0) try t.expectEqual(api.Reason.cancelled, reply.reason);
        if (mode == 1) try t.expectEqual(api.Reason.failed, reply.reason);
        try t.expectEqual(@as(u64, 2), g.find(own).?.uses);
    }
}

test "fact usage: incomplete prefix accounting survives restart and refuses automatic retirement" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    const file = try std.fs.path.join(a, &.{ dir, "graph.json" });
    defer a.free(file);
    var g = graph.Graph.init(a, t.io, file);
    defer g.deinit();
    const id = add(&g, 1);
    var scope: Scope = .{ .graph = &g, .gpa = a };
    defer scope.deinit();
    try scope.record(.{ .complete = false });
    try t.expectError(error.UsageIncomplete, g.flushUsage());
    try g.pin(id, true);
    var restored = graph.Graph.init(a, t.io, file);
    defer restored.deinit();
    try t.expect(restored.find(id).?.pinned);
    try t.expectEqual(@as(u64, 0), restored.find(id).?.uses);
    try t.expectError(error.UsageIncomplete, restored.flushUsage());
}

/// The normal HTTP entry, also used by the capacity lesson test to create a genuinely observed request.
pub fn askExternal(srv: *@import("server.zig").Server, streamed: bool) !void {
    const json = @import("json");
    const openai = @import("openai.zig");
    const Conn = @import("http_conn.zig").Conn;
    const Output = struct {
        ok: bool = false,
        fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.ok = true;
        }
        fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
        fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.ok = status == 200;
        }
    };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const al = arena.allocator();
    const raw = (try json.parse(al, if (streamed)
        "{\"messages\":[{\"role\":\"user\",\"content\":\"What colour do I like?\"}],\"stream\":true,\"max_tokens\":32}"
    else
        "{\"messages\":[{\"role\":\"user\",\"content\":\"What colour do I like?\"}],\"max_tokens\":32}")).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = al };
    var out: Output = .{};
    openai.run(srv, al, .{ .ctx = &out, .vt = &.{ .open = Output.open, .event = Output.event, .reply = Output.reply } }, .{ .conn = &conn }, true, raw);
    try t.expect(out.ok);
}

test "fact usage: HTTP whole and streamed answers count, lesson writer and rechecks do not" {
    const path = @import("slide_learn_path_test.zig");
    var text: path.Text = .{};
    var model: path.Model = .{ .capacity = true };
    defer model.deinit();
    const srv = try @import("server.zig").Server.init(a, t.io, model.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    try path.learnOne(srv, "I like blue.", null);
    try path.learnOne(srv, "I like apples.", null);
    try t.expectEqual(@as(u64, 0), srv.slide.find(1).?.uses);
    try t.expectEqual(@as(u64, 0), srv.slide.find(2).?.uses);
    try askExternal(srv, false);
    try askExternal(srv, true);
    try t.expectEqual(@as(u64, 2), srv.slide.find(1).?.uses);
    try t.expectEqual(@as(u64, 0), srv.slide.find(2).?.uses);
}

test "fact usage: native observer persists counts, deduplicates generations and retains a failed write" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    const file = try std.fs.path.join(a, &.{ dir, "graph.json" });
    defer a.free(file);
    var g = graph.Graph.init(a, t.io, file);
    defer g.deinit();
    const id = add(&g, 1);
    _ = add(&g, 2);
    var scope: Scope = .{ .graph = &g, .gpa = a };
    defer scope.deinit();
    var mask: api.pocket_usage.Masks = .{};
    mask.bits[1] = 3;
    try scope.record(mask);
    try scope.record(mask);
    {
        var back = graph.Graph.init(a, t.io, file);
        defer back.deinit();
        try t.expectEqual(@as(u64, 1), back.find(id).?.uses);
        try t.expectEqual(@as(u64, 0), back.find(2).?.uses);
    }
    var next: Scope = .{ .graph = &g, .gpa = a };
    defer next.deinit();
    g.fault = .write;
    try t.expectError(error.GraphWrite, next.record(mask));
    try t.expect(g.usage_dirty);
    try t.expectEqual(@as(u64, 2), g.find(id).?.uses);
    try next.record(mask);
    try t.expect(!g.usage_dirty);
    var back = graph.Graph.init(a, t.io, file);
    defer back.deinit();
    try t.expectEqual(@as(u64, 2), back.find(id).?.uses);
}
