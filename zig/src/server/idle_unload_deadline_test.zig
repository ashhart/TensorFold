//! Drain deadlines exercise the real manager and current-owner closer without devices.
const std = @import("std");
const api = @import("engine_api");
const idle = @import("idle_unload.zig");
const Harness = @import("idle_unload_test.zig").Harness;
const Closer = @import("idle_unload_closer.zig").Closer;
const t = std.testing;

test "idle deadline: a parked startup engine is never halted" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.expire();
    try h.waitUnloaded(2000);
    h.man.?.begin();
    defer h.man.?.end();
    h.srv.?.generating.store(1, .seq_cst);
    idle.haltHeld(h.man.?, "restarting");
    try t.expectEqual(@as(u32, 0), h.halts.load(.acquire));
    try t.expectEqual(@as(u32, 0), h.unsafe_halts.load(.acquire));
}

test "idle deadline: the fallback closer ignores a closed owner and halts the replacement" {
    const Count = struct {
        closed: bool = false,
        halts: u32 = 0,
        unsafe: u32 = 0,
        fn close(ctx: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.closed = true;
        }
        fn halt(ctx: *anyopaque, _: []const u8) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (c.closed) c.unsafe += 1;
            c.halts += 1;
        }
    };
    var a: Count = .{};
    var b: Count = .{};
    var closer: Closer = .{ .opened = .{ .engine = undefined, .ctx = &a, .close = Count.close, .halt = Count.halt } };
    defer closer.deinit();
    closer.closeOnce();
    Closer.halt(&closer, "restarting");
    try t.expectEqual(@as(u32, 0), a.halts);
    try t.expectEqual(@as(u32, 0), a.unsafe);
    const owner = try t.allocator.create(std.heap.ArenaAllocator);
    owner.* = .init(t.allocator);
    closer.adopt(.{ .engine = undefined, .ctx = &b, .close = Count.close, .halt = Count.halt }, owner);
    Closer.halt(&closer, "restarting");
    try t.expectEqual(@as(u32, 0), a.halts);
    try t.expectEqual(@as(u32, 1), b.halts);
    closer.closeOnce();
    Closer.halt(&closer, "restarting");
    try t.expectEqual(@as(u32, 1), b.halts);
    try t.expectEqual(@as(u32, 0), b.unsafe);
}

pub fn loadingDrain(comptime drain: anytype) !void {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    const Worker = struct {
        fn reload(x: *Harness, failure: *?anyerror) void {
            defer x.srv.?.generating.store(0, .seq_cst);
            defer x.man.?.end();
            var problem: []const u8 = "";
            x.reload(&problem) catch |err| {
                failure.* = err;
            };
        }
        fn halt(x: *Harness, done: *std.atomic.Value(bool)) void {
            const Stop = @import("serve.zig").EngineStop;
            const stop: Stop = .{ .ctx = x, .close = noClose, .halt = fallback };
            drain(x.srv.?, x.io, stop, 0);
            done.store(true, .release);
        }
        fn noClose(_: *anyopaque) void {}
        fn fallback(ctx: *anyopaque, _: []const u8) void {
            const x: *Harness = @ptrCast(@alignCast(ctx));
            _ = x.unsafe_halts.fetchAdd(1, .acq_rel);
        }
    };
    for (0..2) |cycle| {
        h.expire();
        try h.waitUnloaded(2000);
        h.gate_open.store(true, .release);
        h.open_entered.reset();
        h.open_release.reset();
        h.srv.?.draining.store(false, .seq_cst);
        h.srv.?.generating.store(1, .seq_cst);
        h.man.?.begin();
        var failure: ?anyerror = null;
        var done: std.atomic.Value(bool) = .init(false);
        const request = try std.Thread.spawn(.{}, Worker.reload, .{ &h, &failure });
        defer request.join();
        defer h.open_release.set(h.io);
        try h.waitEvent(&h.open_entered);
        const deadline = try std.Thread.spawn(.{}, Worker.halt, .{ &h, &done });
        var joined = false;
        defer if (!joined) {
            h.open_release.set(h.io);
            deadline.join();
        };
        var waited: u32 = 0;
        while (h.man.?.cond.state.load(.acquire).waiters < 2 and waited < 2000) : (waited += 10) h.sleepMs(10);
        try t.expect(!done.load(.acquire));
        try t.expectEqual(@as(u32, @intCast(cycle)), h.halts.load(.acquire));
        try t.expectEqual(@as(u32, 0), h.unsafe_halts.load(.acquire));
        h.open_release.set(h.io);
        deadline.join();
        joined = true;
        try t.expectEqual(@as(u32, @intCast(cycle + 1)), h.halts.load(.acquire));
        try t.expectEqual(@as(u32, 0), h.unsafe_halts.load(.acquire));
        try t.expect(failure == null);
    }
}

test "idle reader: a real status lease blocks close and parked reads touch no engine" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    h.gate_status.store(true, .release);
    const Reader = struct {
        fn run(x: *Harness, out: *api.Status) void {
            x.man.?.readStatus(x.srv.?, out, &.{});
        }
    };
    var out: api.Status = .{};
    const reader = try std.Thread.spawn(.{}, Reader.run, .{ &h, &out });
    var joined = false;
    defer if (!joined) {
        h.status_release.set(h.io);
        reader.join();
    };
    try h.waitEvent(&h.status_entered);
    h.sleepMs(1200); // the idle window expires while the status reader holds the lifetime lock
    try t.expectEqual(@as(u32, 0), h.closes.load(.acquire));
    h.status_release.set(h.io);
    reader.join();
    joined = true;
    try t.expect(out.healthy);
    h.expire();
    try h.waitUnloaded(2000);
    h.man.?.readStatus(h.srv.?, &out, &.{});
    try t.expect(!out.healthy);
    try t.expectEqual(null, h.man.?.readMemory(h.srv.?, false));
    try t.expectEqual(@as(u32, 0), h.unsafe_reads.load(.acquire));
}

test "idle reader: a request arriving before the stub wins the pending park" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    const m = h.man.?;
    m.mutex.lockUncancelable(h.io);
    m.state = .unloading;
    m.mutex.unlock(h.io);
    m.begin();
    defer m.end();
    try t.expectEqual(idle.State.resident, h.state());
    try t.expect(!m.isParked());
    var out: api.Status = .{};
    m.readStatus(h.srv.?, &out, &.{});
    try t.expect(out.healthy);
    try t.expectEqual(@as(u32, 0), h.unsafe_reads.load(.acquire));
}
