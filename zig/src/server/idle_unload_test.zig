//! The idle-unload test harness: a fake engine, a counting Reopen and a short-window manager.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const idle_unload = @import("idle_unload.zig");
const server_mod = @import("server.zig");
const Allocator = std.mem.Allocator;

/// A fake engine: refuses nothing, reports one lane, holds no weights, and panics on any call after its close.
const Fake = struct {
    closed: bool = false,
    /// A reader parked inside statusFn: the gate holds it while the test drives an unload attempt.
    hold_status: bool = false,
    status_entered: std.atomic.Value(bool) = .init(false),

    ticks: std.atomic.Value(u32) = .init(0),

    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{
            .info = infoFn,
            .submit = submitFn,
            .cancel = cancelFn,
            .status = statusFn,
            .memory = memoryFn,
            .keepalive = keepaliveFn,
        } };
    }

    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return .{ .ctx = ctx, .tick = tickFn };
    }

    fn tickFn(ctx: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.closed) @panic("keepalive tick after the engine's close"); // the lifetime failure park must prevent
        _ = f.ticks.fetchAdd(1, .monotonic);
    }

    fn infoFn(_: *anyopaque) api.Info {
        return .{ .lanes = 1, .name = "fake" };
    }

    fn submitFn(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }

    fn cancelFn(_: *anyopaque, _: api.Id) void {}

    fn statusFn(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.closed) @panic("engine read after its close"); // the lifetime failure the stub must prevent
        f.status_entered.store(true, .release);
        while (f.hold_status) std.Thread.yield() catch {};
        out.* = .{ .healthy = true };
    }

    fn memoryFn(ctx: *anyopaque, _: bool) ?api.Memory {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.closed) @panic("engine read after its close");
        return null;
    }

    fn close(f: *Fake) void {
        f.closed = true;
    }
};

/// The test shell: a real `Server` struct shell, a counting Reopen, and the manager under test.
pub const Harness = struct {
    gpa: Allocator = undefined,
    io: std.Io = undefined,
    man: ?*idle_unload.Manager = null,
    fake: ?*Fake = null,
    reloaded_fake: ?*Fake = null,
    srv: ?*server_mod.Server = null,
    arena: std.heap.ArenaAllocator = undefined,
    opens: u32 = 0,
    closes: u32 = 0,
    halts: u32 = 0,
    replaces: u32 = 0,
    fail_open: bool = false,
    keep_warm_s: i64 = 0,
    fail_closes: u32 = 0,
    opened: ?api.Opened = null,

    /// Builds a server shell and a manager with `idle_s`; the startup engine is a fresh fake.
    pub fn setup(self: *Harness, idle_s: i64) !void {
        const keep_warm_s = self.keep_warm_s; // read before the struct reset below
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .man = undefined, .fake = undefined, .srv = undefined };
        self.keep_warm_s = keep_warm_s;
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.fake = try a.create(Fake);
        self.fake.?.* = .{};
        // a Server shell, not Server.init: the manager reads engine, info, config, keepalive, recall
        const srv = try gpa.create(server_mod.Server);
        errdefer gpa.destroy(srv);
        const startup: api.Opened = .{ .engine = self.fake.?.engine(), .close = closeFake, .ctx = self.fake.? };
        srv.* = .{
            .gpa = gpa,
            .io = io,
            .engine = startup.engine,
            .info = startup.engine.info(),
            .text = undefined,
            .config = .{ .served_name = "fake", .model_ids = &.{}, .keep_warm_s = self.keep_warm_s },
            .keys = null,
            .metrics = .{ .gpa = gpa },
            .store = .{ .gpa = gpa },
            .arena = .init(gpa),
            .slide = .init(gpa, io, null),
            .teacher = .init(gpa),
        };
        var args: cli.Args = .{};
        args.idle_unload = idle_s;
        const re: idle_unload.Reopen = .{ .ctx = self, .open = open, .close_held = closeHeld, .halt_held = haltHeld, .replace = replace };
        self.man = idle_unload.start(gpa, io, args, re, "d", "t") orelse return error.TestUnexpectedResult;
        self.man.?.srv = srv;
        self.srv = srv;
        srv.idle_manager = self.man;
    }

    fn open(ctx: *anyopaque, a: Allocator, _: Allocator, _: std.Io, _: []const u8, _: []const u8, _: cli.Args, problem: *[]const u8) anyerror!?api.Opened {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        if (h.fail_open) {
            h.fail_closes += 1; // the caller closes a failed open's return; nothing to close here
            problem.* = "the fake open failed";
            return error.OpenFailed;
        }
        h.opens += 1;
        const fresh = try a.create(Fake);
        fresh.* = .{};
        h.reloaded_fake = fresh;
        h.opened = .{ .engine = fresh.engine(), .close = closeFake, .ctx = fresh };
        return h.opened;
    }

    fn closeHeld(ctx: *anyopaque) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.closes += 1;
        h.fake.?.closed = true; // the startup engine's close: later calls on it must be caught
    }

    fn closeFake(ctx: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.closed = true;
    }

    fn haltHeld(ctx: *anyopaque, reason: []const u8) void {
        _ = reason;
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.halts += 1;
    }

    fn replace(ctx: *anyopaque, opened: api.Opened) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.replaces += 1;
        h.opened = opened;
    }

    pub fn deinit(self: *Harness) void {
        if (self.man) |m| idle_unload.stop(m);
        if (self.srv) |s| {
            s.stopKeepalive(); // a test ending resident leaves no ticker ticking the freed fake
            s.arena.deinit();
            s.metrics.deinit();
            s.store.deinit();
            s.slide.deinit();
            s.teacher.deinit();
            self.gpa.destroy(s);
        }
        self.arena.deinit();
    }

    pub fn waitParked(self: *Harness, timeout_ms: u32) !void {
        var waited: u32 = 0;
        while (!self.man.?.isParked() and waited < timeout_ms) : (waited += 10) {
            std.Io.sleep(self.io, .fromMilliseconds(10), .awake) catch {};
        }
        try std.testing.expect(self.man.?.isParked());
    }

    /// Waits until the park fully finished (state `unloaded`): `parked` is already true while the engine closes.
    pub fn waitUnloaded(self: *Harness, timeout_ms: u32) !void {
        var waited: u32 = 0;
        while (self.man.?.state != .unloaded and waited < timeout_ms) : (waited += 10) {
            std.Io.sleep(self.io, .fromMilliseconds(10), .awake) catch {};
        }
        try std.testing.expectEqual(idle_unload.State.unloaded, self.man.?.state);
    }

    pub fn reload(self: *Harness, problem: *[]const u8) !void {
        try idle_unload.reload(self.man.?, self.srv.?, problem);
    }

    pub fn sleepMs(self: *Harness, ms: u64) void {
        std.Io.sleep(self.io, .fromMilliseconds(@intCast(ms)), .awake) catch {};
    }
};

test "the manager stays alive after setup: the server shell serves through the resident engine" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    var status: api.Status = .{};
    h.man.?.readStatus(h.srv.?, &status, &.{});
    try std.testing.expect(status.healthy); // the resident engine answered, no park yet
    try std.testing.expectEqual(@as(u32, 0), h.closes);
}

test "two complete cycles with the keepalive armed: every park stops the current keepalive" {
    var h: Harness = .{};
    defer h.deinit();
    h.keep_warm_s = 60;
    try h.setup(1);
    try h.waitUnloaded(3000); // the first park stopped the startup keepalive
    try std.testing.expect(h.srv.?.keepalive == null);
    var problem: []const u8 = "";
    try h.reload(&problem);
    try std.testing.expect(h.srv.?.keepalive != null); // the reload armed a fresh one
    try h.waitUnloaded(3000); // the second park must stop THAT one too
    try std.testing.expect(h.srv.?.keepalive == null);
    try h.reload(&problem);
    try std.testing.expectEqual(@as(u32, 2), h.opens);
}

test "a reader inside the engine holds the park off: the close waits for the read" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    h.fake.?.hold_status = true;
    const Reader = struct {
        fn run(w: *Harness, out: *api.Status) void {
            w.man.?.readStatus(w.srv.?, out, &.{});
        }
    };
    var read_status: api.Status = .{};
    const thread = try std.Thread.spawn(.{}, Reader.run, .{ &h, &read_status });
    var joined = false;
    defer if (!joined) {
        h.fake.?.hold_status = false; // a cancelled test must not leave the reader spinning
        thread.join();
    };
    while (!h.fake.?.status_entered.load(.acquire)) std.Io.sleep(h.io, .fromMilliseconds(5), .awake) catch {};
    h.sleepMs(1500); // a whole idle window with the reader holding the manager lock
    try std.testing.expectEqual(@as(u32, 0), h.closes); // the park cannot take the lock: the engine stays open
    h.fake.?.hold_status = false;
    thread.join();
    joined = true;
    try std.testing.expect(read_status.healthy); // the read finished on the live engine, no closed-engine read
    try h.waitUnloaded(5000); // the reader gone, the park proceeds and closes
    try std.testing.expectEqual(@as(u32, 1), h.closes);
}

test "a request arriving while parked wakes and reloads: the next reads hit the reopened engine" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000); // the route's begin() ran, the wake follows
    var problem: []const u8 = "";
    try h.reload(&problem); // what routes.post's centralized wake calls
    try std.testing.expectEqual(@as(u32, 1), h.opens);
    try std.testing.expect(!h.man.?.isParked());
}

test "the drain deadline halts the startup engine before a reload and the Closer's after" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try std.testing.expect(idle_unload.haltsStartupEngine(h.man.?));
    idle_unload.haltHeld(h.man.?, "test");
    try std.testing.expectEqual(@as(u32, 1), h.halts); // the startup engine's halt ran
    try h.waitUnloaded(3000);
    var problem: []const u8 = "";
    try h.reload(&problem);
    try std.testing.expect(!idle_unload.haltsStartupEngine(h.man.?)); // a reload's engine belongs to the Closer
    idle_unload.haltHeld(h.man.?, "test");
    try std.testing.expectEqual(@as(u32, 1), h.halts); // no second halt: the startup engine is gone
}
