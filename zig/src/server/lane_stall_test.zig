//! The sentinel stays responsive while a fake backend blocks the actual lane-host prefill scope.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const t = std.testing;

const Block = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    fn hook(p: *anyopaque, _: *lanes.Stream, _: usize) void {
        const b: *@This() = @ptrCast(@alignCast(p));
        b.entered.store(true, .release);
        while (!b.release.load(.acquire)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    fn event(p: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(p));
        if (e.* == .finished) b.done.store(true, .release);
    }
};
const Wire = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(t.allocator, bytes) catch return error.Closed;
    }
};
fn wait(flag: *const std.atomic.Value(bool)) !void {
    for (0..5000) |_| {
        if (flag.load(.acquire)) return;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.TimedOut;
}

test "lane sentinel HTTP: blocked work withdraws health and admission, counts once, then recovers" {
    const a = t.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const request_a = arena.allocator();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 1 }, 1, 0);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 1 };
    defer target.deinit();
    var block: Block = .{};
    target.prefill_hook = Block.hook;
    target.prefill_hook_ctx = &block;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    defer block.release.store(true, .release);
    const engine = host.engine();
    try engine.vtable.configure_stall.?(engine.ctx, .{ .timeout_ns = 200 * std.time.ns_per_ms, .family = "fixture" });
    const srv = try Server.init(a, t.io, engine, @import("embedding_protocol_test.zig").ByteText.model(), .{ .served_name = "fixture", .model_ids = &.{"fixture"}, .keep_warm_s = 0 }, null);
    defer srv.deinit();
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 1 };
    const sink: api.Sink = .{ .ctx = &block, .event = Block.event };
    try engine.submit(1, &request, sink);
    try wait(&block.entered);
    try wait(&host.stall.unhealthy);
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try t.expect(!status.healthy);
    try t.expectEqual(@as(?u64, 1), status.engine_stalls_total);
    try t.expectError(error.Closed, engine.submit(2, &request, sink));
    var wire: Wire = .{};
    defer wire.bytes.deinit(a);
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    conn.version = "HTTP/1.1";
    conn.method = "GET";
    conn.path = "/health";
    conn.quiet_log = true;
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    try @import("status_routes.zig").health(srv, &conn, request_a);
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 503 "));
    var metrics: std.Io.Writer.Allocating = .init(a);
    defer metrics.deinit();
    try @import("metrics.zig").renderStalls(&metrics.writer, status);
    try t.expect(std.mem.indexOf(u8, metrics.written(), "tensorfold:engine_stalls_total 1") != null);
    block.release.store(true, .release);
    try wait(&block.done);
    for (0..5000) |_| {
        engine.status(&status, &.{});
        if (status.healthy) break;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try t.expect(status.healthy);
    wire.bytes.clearRetainingCapacity();
    conn.quiet_log = true;
    try @import("status_routes.zig").health(srv, &conn, request_a);
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 200 "));
    try t.expectEqual(@as(?u64, 1), status.engine_stalls_total);
}

/// A prompt step that takes `ms` on the engine thread, as a slow device would.
const Pace = struct {
    ms: i64,
    calls: std.atomic.Value(usize) = .init(0),
    fn hook(p: *anyopaque, _: *lanes.Stream, _: usize) void {
        const x: *@This() = @ptrCast(@alignCast(p));
        _ = x.calls.fetchAdd(1, .acq_rel);
        std.Io.sleep(t.io, .fromMilliseconds(x.ms), .awake) catch {};
    }
};

/// Fills a 1,024-row prompt in 10 paced calls under a 300 ms deadline; the stall count and whether health held.
fn pacedPrompt(stepped: bool) !struct { stalls: ?u64, healthy: bool, calls: usize } {
    const a = t.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 1 }, 1, 0);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 10 };
    defer target.deinit();
    var pace: Pace = .{ .ms = 60 };
    target.prefill_hook = Pace.hook;
    target.prefill_hook_ctx = &pace;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, if (stepped) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const engine = host.engine();
    try engine.vtable.configure_stall.?(engine.ctx, .{ .timeout_ns = 300 * std.time.ns_per_ms, .family = "fixture" });
    const prompt = try a.alloc(u32, 1024);
    defer a.free(prompt);
    for (prompt, 0..) |*id, i| id.* = @intCast(1 + i % 200);
    const request: api.Request = .{ .prompt = prompt, .max_tokens = 1 };
    var done: Block = .{};
    try engine.submit(1, &request, .{ .ctx = &done, .event = Block.event });
    var healthy = true;
    var status: api.Status = .{};
    for (0..10000) |_| {
        if (done.done.load(.acquire)) break;
        engine.status(&status, &.{});
        healthy = healthy and status.healthy;
        std.Io.sleep(t.io, .fromMilliseconds(2), .awake) catch {};
    } else return error.TimedOut;
    engine.status(&status, &.{});
    return .{ .stalls = status.engine_stalls_total, .healthy = healthy and status.healthy, .calls = pace.calls.load(.acquire) };
}

test "lane sentinel HTTP: a long prompt filled a step a round never fires, though the whole fill outlasts the deadline" {
    const got = try pacedPrompt(true);
    try t.expectEqual(@as(usize, 10), got.calls); // 10 x 60 ms: twice the deadline in all, a fifth of it a round
    try t.expectEqual(@as(?u64, 0), got.stalls);
    try t.expect(got.healthy);
}

test "lane sentinel HTTP: a family that fills a long prompt in one call holds one round for all of it" {
    // the documented limit of --stall-seconds: the deadline must exceed the slowest single prompt call
    const got = try pacedPrompt(false);
    try t.expectEqual(@as(usize, 10), got.calls);
    try t.expectEqual(@as(?u64, 1), got.stalls);
    try t.expect(!got.healthy);
}

test "lane sentinel HTTP: off by default, the host reports no count and the scrape gains no bytes" {
    const a = t.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 1 }, 1, 0);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, t.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const engine = host.engine();
    const request: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 2 };
    var done: Block = .{};
    try engine.submit(1, &request, .{ .ctx = &done, .event = Block.event });
    try wait(&done.done);
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try t.expect(status.healthy);
    try t.expectEqual(@as(?u64, null), status.engine_stalls_total);
    try t.expect(host.stall.thread == null and host.stall.activity() == null);
    var output: std.Io.Writer.Allocating = .init(a);
    defer output.deinit();
    try @import("metrics.zig").renderStalls(&output.writer, status);
    try t.expectEqual(@as(usize, 0), output.written().len);
}

test "lane sentinel CLI: zero is off, overflow and malformed seconds fail, exit needs a deadline" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const cli = @import("cli.zig");
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    try t.expectEqual(@as(u64, 0), (try cli.parse(a, &.{ "m", "--stall-seconds", "0" }, &usage)).stall_ns);
    const on = try cli.parse(a, &.{ "m", "--stall-seconds", "120", "--stall-exit" }, &usage);
    try t.expectEqual(@as(u64, 120) * std.time.ns_per_s, on.stall_ns);
    try t.expect(on.stall_exit);
    for ([_][]const u8{ "", "-1", "bad", "0.5", "18446744074", "18446744073709551616" }) |value|
        try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--stall-seconds", value }, &usage));
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--stall-exit" }, &usage));
    var output: std.Io.Writer.Allocating = .init(a);
    defer output.deinit();
    try @import("metrics.zig").renderStalls(&output.writer, .{});
    try t.expectEqual(@as(usize, 0), output.written().len);
}
