//! A witness reader racing idle park and reload cycles sees each engine only with its own epoch, never a closed one.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const idle = @import("idle_unload.zig");
const witness = @import("witness.zig");
const parts = @import("witness_parts.zig");
const fixture = @import("witness_fixture.zig");
const Closer = @import("idle_unload_closer.zig").Closer;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Facts = api.execution_witness.Facts;

/// One engine generation: its facts name its number, and a read after its close is counted.
const Gen = struct {
    n: u64,
    owner: *Race,
    closed: std.atomic.Value(bool) = .init(false),

    fn engine(g: *Gen) api.Engine {
        return .{ .ctx = g, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Gen {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(ptr: *anyopaque) api.Info {
        return .{ .context_window = 4096, .witness = .{ .ctx = ptr, .read = read } };
    }
    fn read(ptr: *anyopaque) Facts {
        const g = cast(ptr);
        if (g.closed.load(.acquire)) _ = g.owner.closed_reads.fetchAdd(1, .acq_rel);
        var digest: [32]u8 = @splat(0);
        std.mem.writeInt(u64, digest[0..8], g.n, .little);
        return .{ .known = .{ .path = "test/race", .digest = digest } };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = true };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn close(ptr: *anyopaque) void {
        cast(ptr).closed.store(true, .release);
    }
};

const Race = struct {
    opened: u64 = 1,
    closer: Closer = undefined,
    closed_reads: std.atomic.Value(u32) = .init(0),
    mixed: std.atomic.Value(u32) = .init(0),
    known: std.atomic.Value(u32) = .init(0),
    away: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    fn reopen(r: *Race) idle.Reopen {
        return .{ .ctx = r, .open = open, .close_held = closeHeld, .replace = replace };
    }
    fn open(ctx: *anyopaque, a: Allocator, _: Allocator, _: std.Io, _: []const u8, _: []const u8, _: @import("cli.zig").Args, _: *[]const u8) anyerror!?api.Opened {
        const r: *Race = @ptrCast(@alignCast(ctx));
        r.opened += 1;
        const g = try a.create(Gen);
        g.* = .{ .n = r.opened, .owner = r };
        return .{ .engine = g.engine(), .close = Gen.close, .ctx = g };
    }
    fn closeHeld(ctx: *anyopaque) void {
        @as(*Race, @ptrCast(@alignCast(ctx))).closer.closeOnce();
    }
    fn replace(ctx: *anyopaque, opened: api.Opened, scratch: *std.heap.ArenaAllocator) void {
        @as(*Race, @ptrCast(@alignCast(ctx))).closer.adopt(opened, scratch);
    }

    fn reader(r: *Race, state: *witness.State, srv: *server.Server) void {
        while (!r.stop.load(.acquire)) {
            const snap = state.read(srv);
            switch (snap.result) {
                .known => |k| {
                    _ = r.known.fetchAdd(1, .acq_rel);
                    if (std.mem.readInt(u64, k.parts[0][0..8], .little) != snap.epoch) _ = r.mixed.fetchAdd(1, .acq_rel);
                    if (!std.mem.eql(u8, &k.witness, &witness.compose(snap.instance, snap.epoch, .{ .path = k.path, .digest = k.parts[0] }, k.parts[1..].*).witness))
                        _ = r.mixed.fetchAdd(1, .acq_rel);
                },
                .unknown => |u| switch (u.reason) {
                    .parked, .reloading => _ = r.away.fetchAdd(1, .acq_rel),
                    else => _ = r.mixed.fetchAdd(1, .acq_rel),
                },
            }
        }
    }
};

fn waitState(m: *idle.Manager, want: idle.State) !void {
    var waited: usize = 0;
    while (waited < 5000) : (waited += 1) {
        m.mutex.lockUncancelable(t.io);
        const now = m.state;
        m.mutex.unlock(t.io);
        if (now == want) return;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "execution witness: a reader through park and reload cycles never sees a mixed epoch" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    var race: Race = .{};
    var first: Gen = .{ .n = 1, .owner = &race };
    race.closer = .{ .opened = .{ .engine = first.engine(), .close = Gen.close, .ctx = &first } };
    defer race.closer.deinit();
    const srv = try server.Server.init(t.allocator, t.io, first.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0 }, null);
    defer srv.deinit();
    srv.load.attach(srv.engine);
    const text = try parts.Text.stamp(t.allocator, t.io, c.path, null);
    defer text.deinit();
    const binary = try parts.Binary.openRunning(t.allocator, t.io);
    defer binary.close(t.io);
    const state = try witness.State.init(t.allocator, t.io, srv, .{ .execution_witness = true }, binary, text);
    defer state.deinit();
    srv.witness = state;
    state.start();
    try t.expect(fixture.settled(state, srv).result == .known);
    try t.expectEqual(idle.Startup.enabled, idle.startServer(srv, .{ .idle_unload = 3600 }, race.reopen(), "d", "t"));
    defer idle.stopServer(srv);
    const m = srv.idle_manager.?;
    const reading = try std.Thread.spawn(.{}, Race.reader, .{ &race, state, srv });
    const cycles = 6;
    for (0..cycles) |_| {
        m.mutex.lockUncancelable(t.io);
        m.last_activity_ns = 0; // the window has passed: the timer parks the engine
        m.mutex.unlock(t.io);
        m.cond.broadcast(t.io);
        try waitState(m, .unloaded);
        std.Io.sleep(t.io, .fromMilliseconds(2), .awake) catch {};
        var reply: std.heap.ArenaAllocator = .init(t.allocator);
        defer reply.deinit();
        var problem: []const u8 = "";
        try idle.reload(m, srv, reply.allocator(), &problem);
        std.Io.sleep(t.io, .fromMilliseconds(2), .awake) catch {};
    }
    race.stop.store(true, .release);
    reading.join();
    try t.expectEqual(@as(u32, 0), race.mixed.load(.acquire));
    try t.expectEqual(@as(u32, 0), race.closed_reads.load(.acquire));
    try t.expect(race.known.load(.acquire) > 0 and race.away.load(.acquire) > 0);
    const last = state.read(srv);
    try t.expectEqual(@as(u64, cycles + 1), last.epoch);
    try t.expectEqual(@as(u64, cycles + 1), std.mem.readInt(u64, last.result.known.parts[0][0..8], .little));
}
