//! The real relay and Handoff publish arrivals and terminal wakes through a host recording device.
const std = @import("std");
const mtl = @import("metal");
const relay = @import("src/families/flashnext/tp_service.zig");
const Handoff = @import("src/families/flashnext/tp_handoff.zig").Handoff;
const Job = @import("src/families/flashnext/tp_job.zig").Job;
const Bool = std.atomic.Value(bool);
const Word = std.atomic.Value(u64);
const epoch: u64 = 0x1_0000_0003;
const diagnostics = @import("pair_diagnostics");

const Mode = enum { late, stale, silent, send_failure, send_expiry };

fn expire(h: *Handoff) void {
    while (!h.lock.tryLock()) std.atomic.spinLoopHint();
    defer h.lock.unlock();
    h.deadline.pending.?.deadline = 0;
}

const Endpoint = struct {
    owner: *Fixture,
    sends: usize = 0,
    flushes: usize = 0,

    pub fn writeSignalFromUntil(e: *Endpoint, _: u32, _: usize, _: usize, _: usize, _: usize, _: u64, _: u64) !void {
        e.sends += 1;
        switch (e.owner.mode) {
            .send_failure => return error.TransferFailed,
            .send_expiry => expire(&e.owner.handoff),
            else => {},
        }
    }

    pub fn flushPeerUntil(e: *Endpoint, _: u32, _: u64) !void {
        e.flushes += 1;
    }

    pub fn setDeadline(_: *Endpoint, _: u64) void {}
};

const Host = struct {
    owner: *Fixture,
    handoff: *Handoff,
    stop: *Bool,
    failed: *Bool,
    queued: Word,
    submitted: Word,
    jobs: [1]Job,
    ep: *Endpoint,
    rd: *Endpoint,
    rank: u32 = 0,
    peer: u32 = 1,
    trace: bool = false,
    local: bool = false,
    stats: bool = false,
    stats_state: @import("src/families/flashnext/tp_stats.zig").State = .{},

    pub fn word64(t: *Host, off: usize) *u64 {
        std.debug.assert(off == 8);
        const f = t.owner;
        f.reads += 1;
        f.released_early = f.released_early or f.handoff.landed.value() != 0 or
            f.handoff.ready.slice(u64, 1)[0] != 0;
        if (f.mode == .send_expiry) {
            f.stop.store(true, .release);
        } else if (f.mode == .silent and f.reads == 1) {
            while (!f.handoff.lock.tryLock()) std.atomic.spinLoopHint();
            f.arrival_error = f.handoff.deadline.pending.?.expired;
            f.handoff.deadline.pending.?.deadline = 0;
            f.handoff.lock.unlock();
        } else if ((f.mode == .late or f.mode == .stale) and f.reads == 2) {
            @atomicStore(u64, &f.incoming, epoch, .release);
            f.stop.store(true, .release);
        }
        return &f.incoming;
    }

    pub fn sendNow(_: *Host, _: anytype, _: usize, _: u64) !void {
        return error.UnexpectedSend;
    }
};

const Fixture = struct {
    mode: Mode,
    failed: Bool = .init(false),
    stop: Bool = .init(false),
    handoff: Handoff,
    host: Host,
    endpoint: Endpoint,
    incoming: u64 = 0,
    reads: usize = 0,
    released_early: bool = false,
    arrival_error: ?anyerror = null,

    fn init(f: *Fixture, a: std.mem.Allocator, mode: Mode) !void {
        mtl.allocator = a;
        mtl.resetHandoffs();
        diagnostics.failure_name = "";
        diagnostics.failure_sequence = 0;
        f.* = .{ .mode = mode, .handoff = undefined, .host = undefined, .endpoint = undefined };
        f.handoff = try Handoff.init(.{}, &f.failed, &f.stop, 0);
        f.endpoint = .{ .owner = f };
        f.host = .{
            .owner = f,
            .handoff = &f.handoff,
            .stop = &f.stop,
            .failed = &f.failed,
            .queued = .init(epoch),
            .submitted = .init(epoch),
            .jobs = .{.{ .kind = .exchange, .flag = 8, .value = epoch }},
            .ep = &f.endpoint,
            .rd = &f.endpoint,
        };
        if (mode == .stale) f.incoming = epoch - 1;
        f.handoff.posted.set(epoch);
    }

    fn run(f: *Fixture) !void {
        if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
        const monitor: ?std.Thread = if (f.mode == .silent)
            try std.Thread.spawn(.{}, Handoff.watch, .{&f.handoff})
        else
            null;
        relay.serviceFrom(&f.host, epoch);
        f.stop.store(true, .release);
        if (monitor) |thread| thread.join();
    }

    fn terminal(f: *const Fixture, name: []const u8) !void {
        try std.testing.expect(f.failed.load(.acquire));
        try std.testing.expectEqual(std.math.maxInt(u64), f.handoff.landed.value());
        try std.testing.expectEqual(@as(u64, 0), f.handoff.ready.slice(u64, 1)[0]);
        try std.testing.expectEqual(@as(usize, 1), f.handoff.landed.setCount());
        try std.testing.expectEqualStrings(name, diagnostics.failure_name);
        try std.testing.expectEqual(epoch, diagnostics.failure_sequence);
    }
};

fn arrival(mode: Mode) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = undefined;
    try f.init(arena.allocator(), mode);
    defer f.handoff.deinit();
    try f.run();
    try std.testing.expectEqual(@as(usize, 2), f.reads);
    try std.testing.expect(!f.released_early);
    try std.testing.expect(!f.failed.load(.acquire));
    try std.testing.expectEqual(epoch, f.handoff.ready.slice(u64, 1)[0]);
    try std.testing.expectEqual(epoch, f.handoff.landed.value());
    try std.testing.expectEqual(@as(usize, 1), f.handoff.landed.setCount());
    try std.testing.expectEqual(@as(usize, 1), f.endpoint.sends);
    try std.testing.expectEqual(@as(usize, 1), f.endpoint.flushes);
}

test "relay late peer word releases once with the complete job epoch" {
    try arrival(.late);
}

test "relay stats record a full-width job through the timed publication path" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = undefined;
    try f.init(arena.allocator(), .late);
    defer f.handoff.deinit();
    f.host.stats = true;
    try f.run();
    const reply = f.host.stats_state.takeReply(@import("sys").c.mach_absolute_time());
    try std.testing.expectEqual(epoch, reply.first);
    try std.testing.expectEqual(epoch, reply.last);
    try std.testing.expectEqual(@as(u64, 1), reply.held.count);
    try std.testing.expectEqual(@as(u64, 1), reply.kinds[0].send.count);
    try std.testing.expectEqual(@as(u64, 1), reply.kinds[0].arrival.count);
    try std.testing.expectEqual(@as(u64, 0), reply.kinds[0].release_to_post.count);
    try std.testing.expectEqual(@as(u64, 0), f.host.stats_state.takeReply(@import("sys").c.mach_absolute_time()).held.count);
    try std.testing.expectEqual(epoch, f.handoff.ready.slice(u64, 1)[0]);
    try std.testing.expectEqual(@as(usize, 1), f.handoff.landed.setCount());
}

test "relay stale peer word does not release the next job" {
    try arrival(.stale);
}

test "relay silence releases a terminal epoch with PeerSilent before the GPU watchdog" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = undefined;
    try f.init(arena.allocator(), .silent);
    defer f.handoff.deinit();
    const before = try @import("fabric").words.checkedNowNs();
    try f.run();
    const elapsed = (try @import("fabric").words.checkedNowNs()) - before;
    try f.terminal("PeerSilent");
    try std.testing.expectEqual(error.PeerSilent, f.arrival_error.?);
    const budgets = @import("src/families/flashnext/tp_deadline.zig");
    try std.testing.expect(budgets.send_budget_ns + budgets.arrival_budget_ns < 5 * std.time.ns_per_s);
    try std.testing.expect(elapsed < 5 * std.time.ns_per_s);
}

test "relay failed send never starts an arrival phase or publishes an epoch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = undefined;
    try f.init(arena.allocator(), .send_failure);
    defer f.handoff.deinit();
    try f.run();
    try f.terminal("TransferFailed");
    try std.testing.expectEqual(@as(usize, 0), f.reads);
    try std.testing.expectEqual(@as(usize, 0), f.endpoint.flushes);
}

test "relay expired send cannot be replaced by the arrival deadline" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fixture = undefined;
    try f.init(arena.allocator(), .send_expiry);
    defer f.handoff.deinit();
    try f.run();
    try f.terminal("TpTransferTimeout");
    try std.testing.expectEqual(@as(usize, 0), f.reads);
    try std.testing.expectEqual(@as(usize, 1), f.endpoint.flushes);
}
