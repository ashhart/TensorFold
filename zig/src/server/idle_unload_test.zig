//! The idle-unload test harness: a fake engine, a counting Reopen and a short-window manager.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const idle_unload = @import("idle_unload.zig");
const server_mod = @import("server.zig");
const Allocator = std.mem.Allocator;

/// A fake engine: refuses nothing, reports one lane, holds no weights.
const Fake = struct {
    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{
            .info = infoFn,
            .submit = submitFn,
            .cancel = cancelFn,
            .status = statusFn,
            .memory = memoryFn,
        } };
    }

    fn infoFn(_: *anyopaque) api.Info {
        return .{ .lanes = 1, .name = "fake" };
    }

    fn submitFn(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }

    fn cancelFn(_: *anyopaque, _: api.Id) void {}

    fn statusFn(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = true };
    }

    fn memoryFn(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn noop(_: *anyopaque) void {}
};

/// The test shell: a real `Server` struct shell, a counting Reopen, and the manager under test.
pub const Harness = struct {
    gpa: Allocator = undefined,
    io: std.Io = undefined,
    man: ?*idle_unload.Manager = null,
    fake: ?*Fake = null,
    srv: ?*server_mod.Server = null,
    arena: std.heap.ArenaAllocator = undefined,
    opens: u32 = 0,
    closes: u32 = 0,
    replaces: u32 = 0,
    fail_open: bool = false,
    fail_closes: u32 = 0,
    opened: ?api.Opened = null,

    /// Builds a server shell and a manager with `idle_s`; the startup engine is a fresh fake.
    pub fn setup(self: *Harness, idle_s: i64) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .man = undefined, .fake = undefined, .srv = undefined };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.fake = try a.create(Fake);
        self.fake.?.* = .{};
        // a Server shell, not Server.init: the manager reads engine, info, config, keepalive, recall
        const srv = try gpa.create(server_mod.Server);
        errdefer gpa.destroy(srv);
        const startup: api.Opened = .{ .engine = self.fake.?.engine(), .close = Fake.noop, .ctx = self.fake.? };
        srv.* = .{
            .gpa = gpa,
            .io = io,
            .engine = startup.engine,
            .info = startup.engine.info(),
            .text = undefined,
            .config = .{ .served_name = "fake", .model_ids = &.{} },
            .keys = null,
            .metrics = .{ .gpa = gpa },
            .store = .{ .gpa = gpa },
            .arena = .init(gpa),
            .slide = .init(gpa, io, null),
            .teacher = .init(gpa),
        };
        var args: cli.Args = .{};
        args.idle_unload = idle_s;
        const re: idle_unload.Reopen = .{ .ctx = self, .open = open, .close_held = closeHeld, .replace = replace };
        self.man = idle_unload.start(gpa, io, args, re, "d", "t") orelse return error.TestUnexpectedResult;
        self.man.?.srv = srv;
        self.srv = srv;
        srv.idle_manager = self.man;
    }

    fn open(ctx: *anyopaque, _: Allocator, _: Allocator, _: std.Io, _: []const u8, _: []const u8, _: cli.Args, problem: *[]const u8) anyerror!?api.Opened {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        if (h.fail_open) {
            h.fail_closes += 1; // the caller closes a failed open's return; nothing to close here
            problem.* = "the fake open failed";
            return error.OpenFailed;
        }
        h.opens += 1;
        h.opened = .{ .engine = h.fake.?.engine(), .close = Fake.noop, .ctx = h.fake.? };
        return h.opened;
    }

    fn closeHeld(ctx: *anyopaque) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.closes += 1;
    }

    fn replace(ctx: *anyopaque, opened: api.Opened) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.replaces += 1;
        h.opened = opened;
    }

    pub fn deinit(self: *Harness) void {
        if (self.man) |m| idle_unload.stop(m);
        if (self.srv) |s| {
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

    /// Waits until the park fully finished (state `unloaded`): `parked` turns true while the startup
    /// engine is still closing, and a reload called in that window answers "still unloading" without an error.
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
