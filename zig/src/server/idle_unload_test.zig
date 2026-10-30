//! Synchronized fake-engine owners and gates for idle-unload host tests.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const idle_unload = @import("idle_unload.zig");
const server_mod = @import("server.zig");
const Allocator = std.mem.Allocator;

const Fake = struct {
    owner: *Harness,
    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{
            .info = infoFn,
            .submit = submitFn,
            .cancel = cancelFn,
            .status = statusFn,
            .memory = memoryFn,
            .keepalive = target,
        } };
    }
    fn target(ctx: *anyopaque) ?api.keepalive.Target {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return .{ .ctx = f.owner, .tick = Harness.tick };
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
        if (f.owner.dead.load(.acquire)) _ = f.owner.unsafe_reads.fetchAdd(1, .acq_rel);
        f.owner.status_entered.set(f.owner.io);
        if (f.owner.gate_status.load(.acquire)) f.owner.status_release.waitUncancelable(f.owner.io);
        out.* = .{ .healthy = true };
    }
    fn memoryFn(ctx: *anyopaque, _: bool) ?api.Memory {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.owner.dead.load(.acquire)) _ = f.owner.unsafe_reads.fetchAdd(1, .acq_rel);
        return null;
    }
};

pub const Harness = struct {
    gpa: Allocator = std.testing.allocator,
    io: std.Io = std.testing.io,
    man: ?*idle_unload.Manager = null,
    fake: ?*Fake = null,
    srv: ?*server_mod.Server = null,
    arena: ?std.heap.ArenaAllocator = null,
    cycle: ?*std.heap.ArenaAllocator = null,
    opens: std.atomic.Value(u32) = .init(0),
    closes: std.atomic.Value(u32) = .init(0),
    replaces: std.atomic.Value(u32) = .init(0),
    halts: std.atomic.Value(u32) = .init(0),
    candidate_closes: std.atomic.Value(u32) = .init(0),
    unsafe_keepalive: std.atomic.Value(u32) = .init(0),
    unsafe_reads: std.atomic.Value(u32) = .init(0),
    unsafe_halts: std.atomic.Value(u32) = .init(0),
    gate_open: std.atomic.Value(bool) = .init(false),
    open_entered: std.Io.Event = .unset,
    open_release: std.Io.Event = .unset,
    gate_status: std.atomic.Value(bool) = .init(false),
    status_entered: std.Io.Event = .unset,
    status_release: std.Io.Event = .unset,
    fail_open: bool = false,
    null_open: bool = false,
    dead: std.atomic.Value(bool) = .init(false),
    ticks: std.atomic.Value(u32) = .init(0),
    unsafe_ticks: std.atomic.Value(u32) = .init(0),
    tick_seen: std.Io.Event = .unset,
    opened: ?api.Opened = null,
    opened_allocator: ?Allocator = null,
    gate_close: std.atomic.Value(bool) = .init(false),
    close_entered: std.Io.Event = .unset,
    close_release: std.Io.Event = .unset,

    pub fn setup(self: *Harness, idle_s: i64) !void {
        return self.setupWith(std.testing.allocator, idle_s);
    }

    pub fn setupWith(self: *Harness, gpa: Allocator, idle_s: i64) !void {
        self.* = .{ .gpa = gpa, .arena = .init(gpa) };
        errdefer self.deinit();
        self.fake = try self.arena.?.allocator().create(Fake);
        self.fake.?.* = .{ .owner = self };
        const srv = try gpa.create(server_mod.Server);
        srv.* = .{
            .gpa = gpa,
            .io = self.io,
            .engine = self.fake.?.engine(),
            .info = self.fake.?.engine().info(),
            .text = undefined,
            .config = .{ .served_name = "fake", .model_ids = &.{}, .keep_warm_s = 0 },
            .keys = null,
            .metrics = .{ .gpa = gpa },
            .store = .{ .gpa = gpa },
            .arena = .init(gpa),
            .slide = .init(gpa, self.io, null),
            .teacher = .init(gpa),
        };
        self.srv = srv;
        var args: cli.Args = .{};
        args.idle_unload = idle_s;
        if (idle_unload.startServer(srv, args, self.reopen(), "d", "t") != .enabled)
            return error.OutOfMemory;
        self.man = srv.idle_manager;
    }

    pub fn reopen(self: *Harness) idle_unload.Reopen {
        return .{ .ctx = self, .open = open, .close_held = closeHeld, .replace = replace, .halt_held = haltHeld };
    }

    fn open(ctx: *anyopaque, a: Allocator, _: Allocator, _: std.Io, _: []const u8, _: []const u8, _: cli.Args, problem: *[]const u8) anyerror!?api.Opened {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.open_entered.set(h.io);
        if (h.gate_open.load(.acquire)) h.open_release.waitUncancelable(h.io);
        const scratch = try a.alloc(u8, 4096);
        @memset(scratch, 7);
        if (h.fail_open or h.null_open) {
            problem.* = try a.dupe(u8, "the fake open failed");
            if (h.fail_open) return error.OpenFailed;
            return null;
        }
        h.opened_allocator = a;
        h.dead.store(false, .release);
        _ = h.opens.fetchAdd(1, .acq_rel);
        return .{ .engine = h.fake.?.engine(), .close = closeCandidate, .ctx = h };
    }

    fn closeCandidate(ctx: *anyopaque) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        _ = h.candidate_closes.fetchAdd(1, .acq_rel);
    }

    fn closeHeld(ctx: *anyopaque) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        if (h.srv.?.keepalive != null) _ = h.unsafe_keepalive.fetchAdd(1, .acq_rel);
        h.close_entered.set(h.io);
        if (h.gate_close.load(.acquire)) h.close_release.waitUncancelable(h.io);
        h.dead.store(true, .release);
        _ = h.closes.fetchAdd(1, .release);
    }

    fn replace(ctx: *anyopaque, opened: api.Opened, scratch: *std.heap.ArenaAllocator) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        if (h.cycle) |a| @import("idle_unload_closer.zig").destroyScratch(a);
        h.opened = opened;
        h.dead.store(false, .release);
        h.cycle = scratch;
        _ = h.replaces.fetchAdd(1, .release);
    }

    fn haltHeld(ctx: *anyopaque, _: []const u8) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        if (h.dead.load(.acquire)) _ = h.unsafe_halts.fetchAdd(1, .acq_rel);
        _ = h.halts.fetchAdd(1, .acq_rel);
    }

    fn tick(ctx: *anyopaque) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        if (h.dead.load(.acquire)) _ = h.unsafe_ticks.fetchAdd(1, .acq_rel);
        _ = h.ticks.fetchAdd(1, .release);
        h.tick_seen.set(h.io);
    }

    pub fn waitEvent(self: *Harness, event: *std.Io.Event) !void {
        var waited: u32 = 0;
        while (!event.isSet() and waited < 2000) : (waited += 10) self.sleepMs(10);
        try std.testing.expect(event.isSet());
    }

    pub fn deinit(self: *Harness) void {
        self.close_release.set(self.io);
        self.open_release.set(self.io);
        self.status_release.set(self.io);
        if (self.srv) |s| {
            idle_unload.stopServer(s);
            self.man = null;
            s.stopKeepalive();
            if (self.cycle) |a| @import("idle_unload_closer.zig").destroyScratch(a);
            self.cycle = null;
            s.arena.deinit();
            s.metrics.deinit();
            s.store.deinit();
            s.slide.deinit();
            s.teacher.deinit();
            self.gpa.destroy(s);
            self.srv = null;
        }
        if (self.arena) |*a| a.deinit();
        self.arena = null;
        self.fake = null;
    }

    pub fn state(self: *Harness) idle_unload.State {
        const m = self.man.?;
        m.mutex.lockUncancelable(self.io);
        defer m.mutex.unlock(self.io);
        return m.state;
    }

    pub fn expire(self: *Harness) void {
        const m = self.man.?;
        m.mutex.lockUncancelable(self.io);
        const at: i64 = @intCast(std.Io.Clock.awake.now(self.io).toNanoseconds());
        m.last_activity_ns = at -% (m.idle_s * std.time.ns_per_s) -% 1;
        m.mutex.unlock(self.io);
        m.cond.broadcast(self.io);
    }

    pub fn waitParked(self: *Harness, timeout_ms: u32) !void {
        var waited: u32 = 0;
        while (self.state() != .unloaded and waited < timeout_ms) : (waited += 10) self.sleepMs(10);
        if (self.state() != .unloaded) return error.CloseNotComplete;
    }

    /// Waits for completed park under the same lock the timer uses to publish the state.
    pub fn waitUnloaded(self: *Harness, timeout_ms: u32) !void {
        return self.waitParked(timeout_ms);
    }

    pub fn reload(self: *Harness, problem: *[]const u8) !void {
        var reply: std.heap.ArenaAllocator = .init(self.gpa);
        defer reply.deinit();
        idle_unload.reload(self.man.?, self.srv.?, reply.allocator(), problem) catch |err| {
            // Tests inspect only the result; retain no pointer into the temporary reply arena.
            problem.* = "fake reopen failed";
            return err;
        };
    }

    pub fn sleepMs(self: *Harness, ms: u64) void {
        std.Io.sleep(self.io, .fromMilliseconds(@intCast(ms)), .awake) catch {};
    }
};
