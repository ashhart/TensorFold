//! The RoCE protocol end to end on host threads: two ranks, TCP bootstrap, proxy, shm or verbs (TF_TP_ROCE_TEST_HCA).
const std = @import("std");
const roce = @import("roce.zig");
const rv = @import("roce_verbs.zig");
const bootstrap = @import("bootstrap.zig");
const sock = @import("sock.zig");
const reference = @import("reference.zig");
const Bootstrap = bootstrap.Bootstrap;

const Run = struct {
    port: u16,
    wire: roce.WireKind,
    devs: [2][]const u8 = .{ "", "" },
    ops: u32,
    timeout_ns: u64 = 10 * std.time.ns_per_s,
    /// Rank 1 stops after this many ops (rank 0 must then fail its next wait, by timeout or abort).
    stop_after: ?u32 = null,
    abort_after_ms: ?u64 = null,
    errs: [2]?anyerror = .{ null, null },
    done: [2]u32 = .{ 0, 0 },
    max_bytes: usize = 16 * 1024,
    /// Variable lengths (`Roce.allGatherV`'s contract): each op's lens[r] bytes of rank r's n, 0 and n included.
    var_lens: bool = false,
};

fn rank(run: *Run, me: u32, lfd: sock.Fd) void {
    run.errs[me] = body(run, me, lfd);
}

fn body(run: *Run, me: u32, lfd: sock.Fd) ?anyerror {
    var b = (if (me == 0) Bootstrap.acceptOn(lfd, 2, 5 * std.time.ns_per_s) else Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = run.port, .connect_timeout_ns = 5 * std.time.ns_per_s })) catch |e| return e;
    defer b.close();
    var spec: [1]rv.HcaSpec = .{.{}};
    if (run.wire == .verbs) _ = rv.parseSpecs(run.devs[me], @ptrCast(&spec)) catch |e| return e;
    const r = roce.Roce.init(null, &b, "", .{ .wire = run.wire, .hcas = if (run.wire == .verbs) &spec else &.{}, .max_bytes = run.max_bytes, .timeout_ns = run.timeout_ns }) catch |e| return e;
    defer r.deinit();
    var abort_thread: ?std.Thread = null;
    defer if (abort_thread) |t| t.join();
    var in: [64 * 1024]u8 = undefined;
    var peer: [64 * 1024]u8 = undefined;
    var out: [128 * 1024]u8 = undefined;
    for (0..run.ops) |k| {
        if (run.stop_after) |s| if (k == s) {
            // rank 1 skips op s; rank 0 waits in it until its timeout, or until the abort armed now
            if (me == 1) {
                sock.sleepNs(300 * std.time.ns_per_ms);
                return null;
            }
            if (run.abort_after_ms) |ms| abort_thread = std.Thread.spawn(.{}, abortLater, .{ r, ms }) catch null;
        };
        // sizes from 1 byte to the slot, odd ones included, both slots and wrap-arounds
        const n = 1 + (k * 7919) % run.max_bytes;
        reference.fill(in[0..n], .u8, me, k);
        reference.fill(peer[0..n], .u8, 1 - me, k);
        const lens: [2]u32 = if (k % 5 == 0) .{ 0, @intCast(n) } else .{ @intCast((k * 31) % (n + 1)), @intCast((k * 17 + 5) % (n + 1)) };
        const l: [2]usize = if (run.var_lens) .{ lens[0], lens[1] } else .{ n, n };
        r.hostGatherV(in[0..n], out[0 .. 2 * n], if (run.var_lens) lens else null) catch |e| return e;
        const r0 = if (me == 0) in[0..n] else peer[0..n];
        const r1 = if (me == 1) in[0..n] else peer[0..n];
        if (!std.mem.eql(u8, out[0..l[0]], r0[0..l[0]]) or !std.mem.eql(u8, out[n..][0..l[1]], r1[0..l[1]])) return error.WrongBytes;
        run.done[me] += 1;
    }
    r.check() catch |e| return e;
    // both ranks finish before either tears down (the peer may still read a flag this rank's wire writes)
    b.agree("done", "test end") catch |e| return e;
    return null;
}

fn abortLater(r: *roce.Roce, ms: u64) void {
    sock.sleepNs(ms * std.time.ns_per_ms);
    r.abort();
}

fn twoRanks(run: *Run) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    run.port = l.port;
    const t1 = try std.Thread.spawn(.{}, rank, .{ run, 1, l.fd });
    rank(run, 0, l.fd);
    t1.join();
}

test "the RoCE protocol over the shm wire: 2,000 gathers of every size, bit for bit" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 2000 };
    try twoRanks(&run);
    for (run.errs) |e| if (e) |err| return err;
    try std.testing.expectEqual([2]u32{ 2000, 2000 }, run.done);
}

test "the RoCE protocol with variable lengths over the shm wire: each rank posts its own padded count, empty shards included" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 2000, .var_lens = true };
    try twoRanks(&run);
    for (run.errs) |e| if (e) |err| return err;
    try std.testing.expectEqual([2]u32{ 2000, 2000 }, run.done);
}

test "a peer that stops makes the waiting rank time out, and abort ends a wait at once" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 50, .stop_after = 20, .timeout_ns = 100 * std.time.ns_per_ms };
    try twoRanks(&run);
    try std.testing.expectEqual(@as(?anyerror, error.Timeout), run.errs[0]);
    try std.testing.expectEqual(@as(u32, 20), run.done[0]);
    var run2: Run = .{ .port = 0, .wire = .shm, .ops = 50, .stop_after = 20, .abort_after_ms = 50 };
    try twoRanks(&run2);
    try std.testing.expectEqual(@as(?anyerror, error.Aborted), run2.errs[0]);
    try std.testing.expectEqual(@as(?anyerror, null), run2.errs[1]);
}

test "the RoCE protocol over verbs (TF_TP_ROCE_TEST_HCA=dev0,dev1: soft-RoCE or real HCAs)" {
    const raw = std.c.getenv("TF_TP_ROCE_TEST_HCA") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(raw), ',');
    var run: Run = .{ .port = 0, .wire = .verbs, .ops = 2000 };
    run.devs = .{ it.next() orelse return error.SkipZigTest, it.next() orelse return error.SkipZigTest };
    try twoRanks(&run);
    for (run.errs) |e| if (e) |err| return err;
    try std.testing.expectEqual([2]u32{ 2000, 2000 }, run.done);
}
