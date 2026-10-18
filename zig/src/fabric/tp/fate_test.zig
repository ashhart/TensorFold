//! The fate channel between two ranks over real TCP: a failure or a vanished rank ends both, a stop ends neither.
const std = @import("std");
const sock = @import("sock.zig");
const fate = @import("fate.zig");
const bootstrap = @import("bootstrap.zig");
const Config = @import("config.zig").Config;
const Fate = fate.Fate;

var exits: std.atomic.Value(u32) = .init(0);
var hooks: std.atomic.Value(u32) = .init(0);

fn recordExit(code: u8) void {
    std.debug.assert(code == fate.exit_code);
    _ = exits.fetchAdd(1, .acq_rel);
}

fn hook(_: *anyopaque, reason: []const u8) void {
    std.debug.assert(reason.len > 0);
    _ = hooks.fetchAdd(1, .acq_rel);
}

const grace = 50 * std.time.ns_per_ms;

/// Two connected ranks' fate channels (rank 0 first), started, with recorded exits and hooks.
fn pair(fates: *[2]Fate) !void {
    return pairAt(.{ &fates[0], &fates[1] });
}

fn pairAt(fates: [2]*Fate) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var fds0: [fate.max_ranks]sock.Fd = @splat(-1);
    var fds1: [fate.max_ranks]sock.Fd = @splat(-1);
    fds0[1] = b;
    fds1[0] = a;
    for (fates, [2][fate.max_ranks]sock.Fd{ fds0, fds1 }, 0..) |f, fds, r| {
        f.* = Fate.init(.{ .world = 2, .rank = @intCast(r), .grace_ns = grace, .report_ns = 20 * std.time.ns_per_ms }, fds);
        f.exit_fn = recordExit;
        f.usable = fakeUsable;
        f.onFail(.{ .ctx = f, .call = hook });
    }
    for (fates) |f| try f.start();
}

fn fakeUsable() u64 {
    return 123 << 20;
}

fn waitExits(want: u32) !u64 {
    const t0 = sock.nowNs();
    while (exits.load(.acquire) < want) {
        if (sock.nowNs() - t0 > 5 * std.time.ns_per_s) return error.Timeout;
        sock.sleepNs(std.time.ns_per_ms);
    }
    return sock.nowNs() - t0;
}

fn reset() void {
    exits.store(0, .release);
    hooks.store(0, .release);
}

test "rank 1's failure ends both ranks with its reason" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer for (&fates) |*f| f.close();
    sock.sleepNs(60 * std.time.ns_per_ms);
    const mem = fates[0].peerMem(1) orelse return error.NoReport;
    try std.testing.expectEqual(@as(u64, 123 << 20), mem.bytes);
    fates[1].fatal("prefill", error.OutOfMemory);
    const took = try waitExits(2);
    try std.testing.expect(took < grace + 500 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(u32, 2), hooks.load(.acquire));
    try std.testing.expect(std.mem.indexOf(u8, fates[0].reason(), "rank 1 failed in prefill: OutOfMemory") != null);
}

test "rank 0's failure ends rank 1" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer for (&fates) |*f| f.close();
    fates[0].fatal("window", error.Unexpected);
    _ = try waitExits(2);
    try std.testing.expect(std.mem.indexOf(u8, fates[1].reason(), "rank 0 failed in window") != null);
}

test "a rank that vanishes without a word ends the other" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer fates[0].close();
    // rank 1 killed: its socket closes with no FAIL frame (its own fate thread ends quietly first)
    fates[1].expectStop();
    fates[1].close();
    _ = try waitExits(1);
    try std.testing.expect(std.mem.indexOf(u8, fates[0].reason(), "rank 1 closed the fate channel") != null);
}

test "a normal stop ends nobody" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    fates[0].expectStop();
    fates[1].expectStop();
    fates[1].close();
    fates[0].close();
    sock.sleepNs(2 * grace);
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
}

test "a rank failing at boot ends both with its reason, and its channel is down before its owner frees it" {
    // a follower's boot failed after its session came up; its Fate is freed while the fate thread could still read it
    reset();
    // rank 1's Fate where a Model keeps it: on the heap, its pages unmapped when freed (a stale read faults)
    const pa = std.heap.page_allocator;
    var f0: Fate = undefined;
    const f1 = try pa.create(Fate);
    try pairAt(.{ &f0, f1 });
    defer f0.close();
    sock.sleepNs(60 * std.time.ns_per_ms);
    f1.failBoot("boot", error.RopeTooShort);
    try std.testing.expect(f1.thread == null);
    pa.destroy(f1);
    _ = try waitExits(2);
    // several report periods with rank 1's Fate gone: its thread would have reported from freed memory
    sock.sleepNs(5 * 20 * std.time.ns_per_ms);
    try std.testing.expect(std.mem.indexOf(u8, f0.reason(), "rank 1 failed in boot: RopeTooShort") != null);
}

test "rank 0's clean stop ends every rank with no fail-fast exit, whatever order the ranks close in" {
    // rank 0 ended its run and closed; the follower, still tearing down, must not see the hang-up as a failure
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    sock.sleepNs(60 * std.time.ns_per_ms);
    fates[0].announceStop();
    fates[0].close(); // rank 0 gone before rank 1 even started its own stop
    sock.sleepNs(2 * grace + 100 * std.time.ns_per_ms);
    fates[1].close();
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), hooks.load(.acquire));
}

// -- a server's drain on two ranks: the plan link and the fate channel in the order a family's stop uses them --

const planlink = @import("planlink.zig");

/// Plan words a family's rank 0 sends: a round's operation, and a stop word that ends a follower's loop.
const op_round: i64 = 1;
const op_stop: i64 = 4;

fn linkPair(links: *[2]planlink.PlanLink) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [planlink.max_ranks]sock.Fd = @splat(-1);
    var f1: [planlink.max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    links[0] = planlink.PlanLink.init(0, 2, f0);
    links[1] = planlink.PlanLink.init(1, 2, f1);
}

/// Rank 1's loop: each round as it comes, until the stop word, or until the link ends.
const Follower = struct {
    link: *planlink.PlanLink,
    rounds: u32 = 0,
    ended: enum { running, stopped, peer_closed, failed } = .running,

    fn run(f: *Follower) void {
        var buf: [16]i64 = undefined;
        while (true) {
            const msg = f.link.recv(&buf) catch |e| {
                f.ended = if (e == error.PeerClosed) .peer_closed else .failed;
                return;
            };
            if (msg.len == 0) continue;
            if (msg[0] == op_stop) {
                f.ended = .stopped;
                return;
            }
            if (msg[0] == op_round) f.rounds += 1;
        }
    }
};

test "a drain on rank 0, then its clean stop at a round boundary: rank 1 ends its loop there and nobody exits 70" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    var links: [2]planlink.PlanLink = undefined;
    try linkPair(&links);
    var follower: Follower = .{ .link = &links[1] };
    const t = try std.Thread.spawn(.{}, Follower.run, .{&follower});
    // serving, then draining (rank 0 refuses new work, rounds go on; the fate channel is untouched), past a grace
    var sent: u32 = 0;
    for (0..30) |_| {
        try links[0].send(&.{ op_round, sent });
        sent += 1;
        sock.sleepNs(5 * std.time.ns_per_ms);
    }
    try std.testing.expect(!fates[0].closing.load(.acquire));
    // the stop's order: the stop word first, so the follower ends its loop before the fate STOP and the hang-up
    try links[0].send(&.{op_stop});
    fates[0].announceStop();
    links[0].close();
    fates[0].close();
    t.join();
    // rank 1's loop returned: its model closes and the process exits 0
    fates[1].close();
    links[1].close();
    sock.sleepNs(2 * grace + 100 * std.time.ns_per_ms);
    try std.testing.expectEqual(.stopped, follower.ended);
    try std.testing.expectEqual(sent, follower.rounds);
    try std.testing.expectEqual(@as(u32, 0), exits.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), hooks.load(.acquire));
}

test "a drain on rank 0 does not disarm fail-fast: rank 1 killed mid-drain still ends rank 0 (exit 70)" {
    reset();
    var fates: [2]Fate = undefined;
    try pair(&fates);
    defer fates[0].close();
    var links: [2]planlink.PlanLink = undefined;
    try linkPair(&links);
    defer links[0].close();
    for (0..5) |i| try links[0].send(&.{ op_round, @intCast(i) });
    // rank 0 is draining (no stop announced); rank 1 is killed: its sockets close with no word
    links[1].close();
    fates[1].expectStop();
    fates[1].close();
    _ = try waitExits(1);
    try std.testing.expect(std.mem.indexOf(u8, fates[0].reason(), "rank 1 closed the fate channel") != null);
    try std.testing.expectEqual(@as(u32, 1), exits.load(.acquire));
}
