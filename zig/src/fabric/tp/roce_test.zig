//! The RoCE protocol on host threads: 2 to 16 ranks, a TCP bootstrap, the proxy, the shm wire (or TF_TP_ROCE_TEST_HCA).
const std = @import("std");
const roce = @import("roce.zig");
const rv = @import("roce_verbs.zig");
const shm = @import("roce_shm.zig");
const guard = @import("roce_guard.zig");
const bootstrap = @import("bootstrap.zig");
const sock = @import("sock.zig");
const reference = @import("reference.zig");
const Bootstrap = bootstrap.Bootstrap;

const max_world = 4;

const Run = struct {
    port: u16,
    world: u32 = 2,
    wire: roce.WireKind,
    devs: [max_world][]const u8 = .{ "", "", "", "" },
    ops: u32,
    timeout_ns: u64 = 10 * std.time.ns_per_s,
    /// The last rank stops after this many ops (the others must then fail their next wait, by timeout or abort).
    stop_after: ?u32 = null,
    abort_after_ms: ?u64 = null,
    errs: [max_world]?anyerror = @splat(null),
    done: [max_world]u32 = @splat(0),
    max_bytes: usize = 16 * 1024,
    /// Variable lengths (`Roce.allGatherV`'s contract): each op's lens[r] bytes of rank r's n, 0 and n included.
    var_lens: bool = false,
    /// Strides up to four slots through `hostGatherVChunked` (variable lengths, slot-sized ops).
    chunked: bool = false,
    /// The shm wire's stripes (2: the verbs wire's split over two HCAs).
    stripes: u32 = 1,
    /// Each rank's shm posts as the tap saw them: how many, and how many left their slot or the region.
    taps: [max_world]Tap = @splat(.{}),
};

/// Counts an shm wire's posts and any stripe outside its slot, its flag words or the region.
const Tap = struct {
    posts: u32 = 0,
    outside: u32 = 0,
    tap: shm.PostTap = undefined,

    fn seen(ctx: *anyopaque, l: guard.Layout, _: u32, slot: u32, padded: u32, s: []const guard.Stripe) void {
        const t: *Tap = @ptrCast(@alignCast(ctx));
        t.posts += 1;
        var sum: u64 = 0;
        for (s) |x| {
            sum += x.len;
            const in_slot = x.local >= l.send + slot * l.slot and x.local + x.len <= l.send + (slot + 1) * l.slot;
            const in_peer = x.remote >= l.recv and x.remote + x.len <= l.len;
            if (!in_slot or !in_peer or x.remote + x.len > l.len or x.flag < l.flags or x.flag + 4 > l.send) t.outside += 1;
        }
        if (sum != padded) t.outside += 1;
    }
};

/// Op k's lengths: the same on every rank (a function of k, n and the rank it describes).
fn opLens(k: usize, n: usize, world: u32, out: []u32) void {
    const mul = [max_world]usize{ 31, 17, 13, 11 };
    const add = [max_world]usize{ 0, 5, 3, 9 };
    for (0..world) |q| {
        out[q] = if (k % 5 == 0) (if (q == 0) 0 else @intCast(n)) else @intCast((k * mul[q] + add[q]) % (n + 1));
    }
}

fn rank(run: *Run, me: u32, lfd: sock.Fd) void {
    run.errs[me] = body(run, me, lfd);
}

fn body(run: *Run, me: u32, lfd: sock.Fd) ?anyerror {
    var b = (if (me == 0) Bootstrap.acceptOn(lfd, run.world, 5 * std.time.ns_per_s) else Bootstrap.open(.{ .world = run.world, .rank = me, .link_port = run.port, .connect_timeout_ns = 5 * std.time.ns_per_s })) catch |e| return e;
    defer b.close();
    var spec: [1]rv.HcaSpec = .{.{}};
    if (run.wire == .verbs) _ = rv.parseSpecs(run.devs[me], @ptrCast(&spec)) catch |e| return e;
    const mine = &run.taps[me];
    mine.tap = .{ .ctx = mine, .seen = Tap.seen };
    const r = roce.Roce.init(null, &b, "", .{ .wire = run.wire, .hcas = if (run.wire == .verbs) &spec else &.{}, .max_bytes = run.max_bytes, .timeout_ns = run.timeout_ns, .shm_stripes = run.stripes, .tap = &mine.tap }) catch |e| return e;
    defer r.deinit();
    var abort_thread: ?std.Thread = null;
    defer if (abort_thread) |t| t.join();
    const w = run.world;
    var in: [4 * 16 * 1024]u8 = undefined;
    var want: [4 * 16 * 1024]u8 = undefined;
    var out: [max_world * 4 * 16 * 1024]u8 = undefined;
    var lens: [max_world]u32 = undefined;
    for (0..run.ops) |k| {
        if (run.stop_after) |s| if (k == s) {
            // the last rank skips op s; the others wait in it until their timeout, or until the abort armed now
            if (me == w - 1) {
                sock.sleepNs(300 * std.time.ns_per_ms);
                return null;
            }
            if (run.abort_after_ms) |ms| abort_thread = std.Thread.spawn(.{}, abortLater, .{ r, ms }) catch null;
        };
        // sizes from 1 byte to the slot, odd ones included, both slots and wrap-arounds
        const n = 1 + (k * 7919) % (if (run.chunked) 4 * run.max_bytes else run.max_bytes);
        reference.fill(in[0..n], .u8, me, k);
        opLens(k, n, w, &lens);
        if (run.chunked) {
            r.hostGatherVChunked(in[0..n], out[0 .. w * n], lens[0..w]) catch |e| return e;
        } else r.hostGatherV(in[0..n], out[0 .. w * n], if (run.var_lens) lens[0..w] else null) catch |e| return e;
        for (0..w) |q| {
            const l: usize = if (run.var_lens or run.chunked) lens[q] else n;
            reference.fill(want[0..n], .u8, @intCast(q), k);
            if (!std.mem.eql(u8, out[q * n ..][0..l], want[0..l])) return error.WrongBytes;
        }
        run.done[me] += 1;
    }
    r.check() catch |e| return e;
    // every rank finishes before any tears down (a peer may still read a flag this rank's wire writes)
    b.agree("done", "test end") catch |e| return e;
    return null;
}

fn abortLater(r: *roce.Roce, ms: u64) void {
    sock.sleepNs(ms * std.time.ns_per_ms);
    r.abort();
}

fn ranks(run: *Run) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    run.port = l.port;
    var threads: [max_world]std.Thread = undefined;
    for (1..run.world) |i| threads[i] = try std.Thread.spawn(.{}, rank, .{ run, @as(u32, @intCast(i)), l.fd });
    rank(run, 0, l.fd);
    for (threads[1..run.world]) |t| t.join();
}

fn twoRanks(run: *Run) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    run.port = l.port;
    var threads: [max_world]std.Thread = undefined;
    for (1..run.world) |i| threads[i] = try std.Thread.spawn(.{}, rank, .{ run, @as(u32, @intCast(i)), l.fd });
    rank(run, 0, l.fd);
    for (threads[1..run.world]) |t| t.join();
}

fn expectAllDone(run: *const Run) !void {
    for (run.errs[0..run.world]) |e| if (e) |err| return err;
    for (run.done[0..run.world]) |d| try std.testing.expectEqual(run.ops, d);
}

test "the RoCE protocol over the shm wire: 2,000 gathers of every size, bit for bit" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 2000 };
    try ranks(&run);
    try expectAllDone(&run);
}

test "the RoCE protocol with variable lengths over the shm wire: each rank posts its own padded count, empty shards included" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 2000, .var_lens = true };
    try ranks(&run);
    try expectAllDone(&run);
}

test "the shm wire striped as over two HCAs: every post inside its slot and the region, bytes and lengths exact" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 2000, .var_lens = true, .stripes = 2 };
    try twoRanks(&run);
    for (run.errs) |e| if (e) |err| return err;
    for (0..2) |q| try std.testing.expectEqual(@as(u32, 2000), run.done[q]);
    for (run.taps[0..2]) |t| {
        try std.testing.expectEqual(@as(u32, 2000), t.posts);
        try std.testing.expectEqual(@as(u32, 0), t.outside);
    }
}

test "chunked gathers over the shm wire: strides up to four slots in slot-sized ops, tails empty on both ranks skipped" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 2000, .chunked = true, .stripes = 2 };
    try twoRanks(&run);
    for (run.errs) |e| if (e) |err| return err;
    for (0..2) |q| try std.testing.expectEqual(@as(u32, 2000), run.done[q]);
    for (run.taps[0..2]) |t| {
        try std.testing.expect(t.posts > 2000);
        try std.testing.expectEqual(@as(u32, 0), t.outside);
    }
}

test "a peer that stops makes the waiting rank time out, and abort ends a wait at once" {
    var run: Run = .{ .port = 0, .wire = .shm, .ops = 50, .stop_after = 20, .timeout_ns = 100 * std.time.ns_per_ms };
    try ranks(&run);
    try std.testing.expectEqual(@as(?anyerror, error.Timeout), run.errs[0]);
    try std.testing.expectEqual(@as(u32, 20), run.done[0]);
    var run2: Run = .{ .port = 0, .wire = .shm, .ops = 50, .stop_after = 20, .abort_after_ms = 50 };
    const kept = roce.Roce.kept.load(.monotonic);
    try ranks(&run2);
    // the aborted ranks keep their regions until exit, the one that finished frees its
    try std.testing.expectEqual(kept + 1, roce.Roce.kept.load(.monotonic));
    try std.testing.expectEqual(@as(?anyerror, error.Aborted), run2.errs[0]);
    try std.testing.expectEqual(@as(?anyerror, null), run2.errs[1]);
}

test "four ranks over the shm wire: 2,000 gathers of every size, each shard at its rank, bit for bit" {
    var run: Run = .{ .port = 0, .world = 4, .wire = .shm, .ops = 2000 };
    try ranks(&run);
    try expectAllDone(&run);
}

test "four ranks with variable lengths over the shm wire (allGatherV at TP=4, empty shards included)" {
    var run: Run = .{ .port = 0, .world = 4, .wire = .shm, .ops = 2000, .var_lens = true };
    try ranks(&run);
    try expectAllDone(&run);
}

test "four ranks: one rank that stops makes every other rank time out, and abort ends each wait at once" {
    // 500 ms: eight spinning threads (four ranks, four proxies) may share fewer cores; ops 0..19 must not time out
    var run: Run = .{ .port = 0, .world = 4, .wire = .shm, .ops = 50, .stop_after = 20, .timeout_ns = 500 * std.time.ns_per_ms };
    try ranks(&run);
    for (0..3) |q| {
        try std.testing.expectEqual(@as(?anyerror, error.Timeout), run.errs[q]);
        try std.testing.expectEqual(@as(u32, 20), run.done[q]);
    }
    try std.testing.expectEqual(@as(?anyerror, null), run.errs[3]);
    var run2: Run = .{ .port = 0, .world = 4, .wire = .shm, .ops = 50, .stop_after = 20, .abort_after_ms = 50 };
    try ranks(&run2);
    for (0..3) |q| try std.testing.expectEqual(@as(?anyerror, error.Aborted), run2.errs[q]);
    try std.testing.expectEqual(@as(?anyerror, null), run2.errs[3]);
}

test "three ranks over the shm wire (an odd world: the post and wait orders wrap)" {
    var run: Run = .{ .port = 0, .world = 3, .wire = .shm, .ops = 500, .var_lens = true };
    try ranks(&run);
    try expectAllDone(&run);
}

test "the RoCE protocol over verbs (TF_TP_ROCE_TEST_HCA=dev0,dev1[,dev2,dev3]: soft-RoCE or real HCAs, one a rank)" {
    const raw = std.c.getenv("TF_TP_ROCE_TEST_HCA") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(raw), ',');
    var run: Run = .{ .port = 0, .wire = .verbs, .ops = 2000, .world = 0 };
    while (it.next()) |dev| {
        if (run.world == max_world) return error.SkipZigTest;
        run.devs[run.world] = dev;
        run.world += 1;
    }
    if (run.world < 2) return error.SkipZigTest;
    try ranks(&run);
    try expectAllDone(&run);
}

test "the region layout: two ranks keep the old send offset; every source rank's slots and flags are disjoint" {
    const slot = 16 * 1024;
    const two = roce.Layout.of(2, slot);
    try std.testing.expectEqual(@as(u64, 8192), two.send); // ctrl 4 KiB, flags 1 KiB rounded to 4 KiB
    try std.testing.expectEqual(@as(u64, 8192 + 2 * slot), two.recv);
    try std.testing.expectEqual(two.recv + 4 * slot, two.len);
    const four = roce.Layout.of(4, slot);
    try std.testing.expectEqual(@as(u64, 4 * 2 * 2 * 128), roce.Layout.flagBytes(4));
    try std.testing.expectEqual(@as(u64, 8192), four.send);
    try std.testing.expectEqual(four.recv + 8 * slot, four.len);
    try std.testing.expectEqual(four.len, @as(u64, roce.Roce.regionBytes(4, slot)));
    // every (src, slot) receive slot and (src, slot, hca) flag of four ranks: in range, aligned, none twice
    var seen_recv: [8]usize = undefined;
    var seen_flag: [16]usize = undefined;
    var nr: usize = 0;
    var nf: usize = 0;
    for (0..4) |src| for (0..rv.slots) |s| {
        const at = four.recvAt(@intCast(src), @intCast(s));
        try std.testing.expect(at >= four.recv and at + slot <= four.len and at % 4096 == 0);
        for (seen_recv[0..nr]) |x| try std.testing.expect(x != at);
        seen_recv[nr] = at;
        nr += 1;
        for (0..rv.max_hcas) |h| {
            const f = four.flagAt(@intCast(src), @intCast(s), @intCast(h));
            try std.testing.expect(f >= roce.Layout.flags_at and f + 4 <= four.send and f % rv.flag_stride == 0);
            for (seen_flag[0..nf]) |x| try std.testing.expect(x != f);
            seen_flag[nf] = f;
            nf += 1;
        }
    };
    // mailbox.cu's roce_flag / roce_inbox: the same addresses in words and bytes
    try std.testing.expectEqual(roce.Layout.flags_at + ((3 * 2 + 1) * 2 + 1) * 128, four.flagAt(3, 1, 1));
    try std.testing.expectEqual(four.recv + (2 * 2 + 1) * slot, four.recvAt(2, 1));
    // the largest world fits the flags' page and a warp of pollers
    const most = roce.Layout.of(roce.max_world, slot);
    try std.testing.expect(most.flagAt(roce.max_world - 1, 1, rv.max_hcas - 1) + 4 <= most.send);
    try std.testing.expect((roce.max_world - 1) * rv.max_hcas <= 32);
}

test "the post order starts at the next rank: every rank sends to every other once, and no two ranks start at the same receiver" {
    for ([_]u32{ 2, 3, 4, 16 }) |w| {
        var firsts: [16]bool = @splat(false);
        for (0..w) |me_| {
            const me: u32 = @intCast(me_);
            var hit: [16]bool = @splat(false);
            for (0..w - 1) |k| {
                const p = rv.peerAt(me, w, @intCast(k));
                try std.testing.expect(p != me and p < w and !hit[p]);
                hit[p] = true;
            }
            const first = rv.peerAt(me, w, 0);
            try std.testing.expectEqual((me + 1) % w, first);
            try std.testing.expect(!firsts[first]);
            firsts[first] = true;
        }
    }
    // two ranks: the one peer, as before
    try std.testing.expectEqual(@as(u32, 1), rv.peerAt(0, 2, 0));
    try std.testing.expectEqual(@as(u32, 0), rv.peerAt(1, 2, 0));
}
