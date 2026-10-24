//! Causal controls for the mechanical idle-unload carries; engine leases remain a separate design.
const std = @import("std");
const api = @import("engine_api");
const idle = @import("idle_unload.zig");
const Harness = @import("idle_unload_test.zig").Harness;
const t = std.testing;

test "idle carry: the published manager survives setup scope and complete teardown clears it once" {
    var h: Harness = .{};
    defer h.deinit();
    {
        try h.setup(3600);
    }
    const m = h.srv.?.idle_manager orelse return error.ManagerFreedDuringSetup;
    try t.expect(h.srv == m.srv);
    try t.expectEqual(idle.State.resident, h.state());
    idle.stopServer(h.srv.?);
    h.man = null;
    try t.expectEqual(null, h.srv.?.idle_manager);
    idle.stopServer(h.srv.?);
}

test "idle carry: default-off startup has no allocation-failure diagnostic outcome" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    idle.stopServer(h.srv.?);
    h.man = null;
    var args: @import("cli.zig").Args = .{};
    for ([_]i64{ 0, -1 }) |seconds| {
        args.idle_unload = seconds;
        try t.expectEqual(idle.Startup.disabled, idle.startServer(h.srv.?, args, h.reopen(), "d", "t"));
        try t.expectEqual(null, h.srv.?.idle_manager);
    }
}

test "idle carry: active work parks the timer on its condition and end re-arms the window" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    const m = h.man.?;
    m.begin();
    defer if (m.active > 0) m.end();
    var waited: u32 = 0;
    while (m.cond.state.load(.acquire).waiters == 0 and waited < 2000) : (waited += 10) h.sleepMs(10);
    try t.expect(m.cond.state.load(.acquire).waiters > 0);
    try t.expectEqual(@as(u32, 0), h.closes.load(.acquire));
    m.end();
    try t.expectEqual(idle.State.resident, h.state());
    h.expire();
    try h.waitParked(2000);
    try t.expectEqual(@as(u32, 1), h.closes.load(.acquire));
}

test "idle carry: parked publication is not mistaken for completed close" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.gate_close.store(true, .release);
    h.expire();
    try h.waitEvent(&h.close_entered);
    try t.expect(h.man.?.isParked());
    try t.expectEqual(idle.State.unloading, h.state());
    try t.expectEqual(@as(u32, 0), h.closes.load(.acquire));
    try t.expectError(error.CloseNotComplete, h.waitUnloaded(0));
    h.close_release.set(h.io);
    try h.waitUnloaded(2000);
    try t.expectEqual(@as(u32, 1), h.closes.load(.acquire));
}

test "idle carry: every one of three engine cycles stops its actual ticker before close" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.srv.?.config.keep_warm_s = 1;
    const target = h.srv.?.engine.keepaliveTarget().?;
    h.srv.?.keepalive = try api.keepalive.Keepalive.startEvery(t.allocator, h.io, target, std.time.ns_per_s, std.time.ns_per_ms);
    for (0..3) |cycle| {
        try h.waitEvent(&h.tick_seen);
        h.expire();
        try h.waitParked(2000);
        try t.expectEqual(null, h.srv.?.keepalive);
        try t.expectEqual(@as(u32, 0), h.unsafe_keepalive.load(.acquire));
        const ticks = h.ticks.load(.acquire);
        h.sleepMs(20);
        try t.expectEqual(ticks, h.ticks.load(.acquire));
        try t.expectEqual(@as(u32, 0), h.unsafe_ticks.load(.acquire));
        if (cycle != 2) {
            h.tick_seen.reset();
            var problem: []const u8 = "";
            try h.reload(&problem);
            try t.expect(h.srv.?.keepalive != null);
        }
    }
    try t.expectEqual(@as(u32, 3), h.closes.load(.acquire));
}

test "idle carry: deadline halt reaches each resident owner once and never a parked owner" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    const m = h.man.?;
    idle.haltHeld(m, "restarting");
    idle.haltHeld(m, "restarting");
    try t.expectEqual(@as(u32, 1), h.halts.load(.acquire));
    h.expire();
    try h.waitParked(2000);
    idle.haltHeld(m, "restarting");
    try t.expectEqual(@as(u32, 1), h.halts.load(.acquire));
    var problem: []const u8 = "";
    try h.reload(&problem);
    idle.haltHeld(m, "restarting");
    idle.haltHeld(m, "restarting");
    try t.expectEqual(@as(u32, 2), h.halts.load(.acquire));
}

fn live(a: *std.testing.FailingAllocator) usize {
    return a.allocated_bytes - a.freed_bytes;
}

test "idle carry: repeated null and throwing reopens release scratch and owned failure reasons" {
    var counting = t.FailingAllocator.init(t.allocator, .{});
    var h: Harness = .{};
    defer h.deinit();
    try h.setupWith(counting.allocator(), 3600);
    h.expire();
    try h.waitParked(2000);
    const baseline = live(&counting);
    for (0..12) |i| {
        h.fail_open = i % 2 == 0;
        h.null_open = !h.fail_open;
        var problem: []const u8 = "";
        if (h.fail_open) try t.expectError(error.EngineOpenFailed, h.reload(&problem)) else try t.expectError(error.ReloadFailed, h.reload(&problem));
        try t.expectEqual(baseline, live(&counting));
        try t.expectEqual(idle.State.unloaded, h.state());
    }
    h.fail_open = false;
    h.null_open = false;
    var retained: ?usize = null;
    for (0..4) |_| {
        var problem: []const u8 = "";
        try h.reload(&problem);
        try t.expect(live(&counting) > baseline);
        h.expire();
        try h.waitParked(2000);
        if (retained) |bytes| try t.expectEqual(bytes, live(&counting)) else retained = live(&counting);
    }
}

test "idle carry: a rejected recall candidate closes once and releases its cycle scratch" {
    var counting = t.FailingAllocator.init(t.allocator, .{});
    var h: Harness = .{};
    defer h.deinit();
    try h.setupWith(counting.allocator(), 3600);
    h.expire();
    try h.waitParked(2000);
    const Shared = @FieldType(api.recall_store.Store, "shared");
    const Record = @FieldType(Shared, "record");
    var dir: [0:0]u8 = .{};
    var name: [0]u8 = .{};
    var store: api.recall_store.Store = .{
        .gpa = t.allocator,
        .shared = .{ .io = h.io, .record = try Record.init(t.allocator, 1) },
        .dir = &dir,
        .name = &name,
        .identity = .{ .digest = @splat(0), .vocab = 1 },
        .cap_chunks = 2,
        .read_only = true,
        .writes = false,
    };
    defer store.shared.record.deinit();
    h.srv.?.recall = &store;
    defer h.srv.?.recall = null;
    const baseline = live(&counting);
    var problem: []const u8 = "";
    try t.expectError(error.ReloadFailed, h.reload(&problem));
    try t.expectEqual(@as(u32, 1), h.candidate_closes.load(.acquire));
    try t.expectEqual(baseline, live(&counting));
    try t.expectEqual(idle.State.unloaded, h.state());
}

test "idle carry: every partial fixture allocation can unwind and outer cleanup is idempotent" {
    try t.checkAllAllocationFailures(t.allocator, struct {
        fn run(a: std.mem.Allocator) !void {
            var h: Harness = .{};
            defer h.deinit();
            try h.setupWith(a, 3600);
            try t.expect(h.srv.?.idle_manager != null);
        }
    }.run, .{});
}

test "idle carry: failure-reason allocation failure leaves the manager retryable" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.expire();
    try h.waitParked(2000);
    h.null_open = true;
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    var problem: []const u8 = "";
    try t.expectError(error.OutOfMemory, idle.reload(h.man.?, h.srv.?, failing.allocator(), &problem));
    try t.expectEqual(idle.State.unloaded, h.state());
    h.null_open = false;
    try h.reload(&problem);
}

test "idle carry: the real process closer frees each cycle arena only after closing its engine" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    const Count = struct {
        a: *t.FailingAllocator,
        closes: usize = 0,
        halts: usize = 0,
        scratch_live_at_close: bool = false,
        fn close(ctx: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.closes += 1;
            c.scratch_live_at_close = live(c.a) > 0;
        }
        fn halt(ctx: *anyopaque, _: []const u8) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.halts += 1;
        }
    };
    var counting = t.FailingAllocator.init(t.allocator, .{});
    var count: Count = .{ .a = &counting };
    const opened: api.Opened = .{ .ctx = &count, .engine = h.srv.?.engine, .close = Count.close, .halt = Count.halt };
    const Closer = @import("idle_unload_closer.zig").Closer;
    var closer: Closer = .{ .opened = opened, .closed = true };
    defer closer.deinit();
    var retained: ?usize = null;
    for (0..3) |cycle| {
        const scratch = try scratchOwner(counting.allocator());
        closer.adopt(opened, scratch);
        Closer.halt(&closer, "restarting");
        closer.closeOnce();
        closer.closeOnce();
        Closer.halt(&closer, "restarting");
        try t.expect(count.scratch_live_at_close);
        try t.expectEqual(cycle + 1, count.closes);
        try t.expectEqual(cycle + 1, count.halts);
        if (retained) |bytes| try t.expectEqual(bytes, live(&counting)) else retained = live(&counting);
    }
    closer.deinit();
    closer.deinit();
    try t.expectEqual(@as(usize, 0), live(&counting));
}

fn scratchOwner(a: std.mem.Allocator) !*std.heap.ArenaAllocator {
    const owner = try a.create(std.heap.ArenaAllocator);
    owner.* = .init(a);
    errdefer @import("idle_unload_closer.zig").destroyScratch(owner);
    _ = try owner.allocator().alloc(u8, 4096);
    return owner;
}

test "idle carry: the reopened allocator keeps the same stable arena address after adoption" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.expire();
    try h.waitParked(2000);
    var problem: []const u8 = "";
    try h.reload(&problem);
    const captured = h.opened_allocator orelse return error.MissingOpenedAllocator;
    const owner = h.cycle orelse return error.MissingAdoptedArena;
    // Compare addresses only; the copied-owner control's old allocator is never dereferenced.
    try t.expect(captured.ptr == owner.allocator().ptr);
}

test "idle carry: arena-owner allocation refusal leaves the parked manager retryable" {
    var failing = t.FailingAllocator.init(t.allocator, .{});
    var h: Harness = .{};
    defer h.deinit();
    try h.setupWith(failing.allocator(), 3600);
    h.expire();
    try h.waitParked(2000);
    const before = live(&failing);
    failing.fail_index = failing.alloc_index;
    var problem: []const u8 = "";
    try t.expectError(error.OutOfMemory, idle.reload(h.man.?, h.srv.?, t.allocator, &problem));
    try t.expectEqual(before, live(&failing));
    try t.expectEqual(@as(u32, 0), h.opens.load(.acquire));
    try t.expectEqual(idle.State.unloaded, h.state());
    failing.fail_index = std.math.maxInt(usize);
    try h.reload(&problem);
}

test "idle carry: the merged wake helper keeps its failure reason in the caller arena" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.expire();
    try h.waitUnloaded(2000);
    var reply: std.heap.ArenaAllocator = .init(t.allocator);
    defer reply.deinit();
    h.fail_open = true;
    const problem = try idle.wakeReload(h.man.?, h.srv.?, reply.allocator());
    try t.expectEqualStrings("the fake open failed", problem);
    try t.expectEqual(idle.State.unloaded, h.state());
    h.fail_open = false;
    try t.expectEqualStrings("", try idle.wakeReload(h.man.?, h.srv.?, reply.allocator()));
    try t.expectEqualStrings("the fake open failed", problem);
    try t.expectEqual(idle.State.resident, h.state());
}
