//! Actual multi-peer verbs posts on fake switched fabrics and rings, with byte and completion-order controls.
const std = @import("std");
const t = std.testing;
const Fabric = @import("roce_multi_fake.zig").Fabric;
const guard = @import("roce_multi_guard.zig");
const Roce = @import("roce_multi.zig").Roce;
const Record = @import("roce_multi_verbs.zig").Record;

fn check(f: *Fabric, seq: u32, n: usize) !void {
    const slot = seq & 1;
    for (0..f.world) |dst| for (0..f.world) |src| if (dst != src) {
        const expected = f.regions[src][f.layout.sendAt(slot)..][0..n];
        const got = f.regions[dst][f.layout.recvAt(@intCast(src), slot)..][0..n];
        if (!std.mem.eql(u8, expected, got)) return error.WrongBytes;
        for (0..f.wires[src].stripes) |h| {
            const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(f.regions[dst][f.layout.flagAt(@intCast(src), slot, @intCast(h))..].ptr));
            try t.expectEqual(seq, flag.load(.acquire));
        }
    };
}

test "multirank verbs: cyclic peer order, all-to-all bytes and both slots match rank-indexed gather at worlds 3, 4 and 8" {
    for ([_]u32{ 3, 4, 8 }) |world| for ([_]u32{ 1, 2 }) |nh| {
        var f: Fabric = undefined;
        try f.init(world, nh, 4096);
        defer f.deinit();
        for (1..18) |s| {
            const seq: u32 = @intCast(s);
            const n: u32 = ([_]u32{ 16, 48, 128, 1024, 4096 })[s % 5];
            for (0..world) |src| {
                const input = f.regions[src][f.layout.sendAt(seq & 1)..][0..n];
                for (input, 0..) |*v, i| v.* = @truncate(13 * src + 7 * s + i);
                f.peers_seen = 0;
                try f.wires[src].post(seq & 1, seq, n);
                for (0..world - 1) |k| try t.expectEqual(guard.peerAt(@intCast(src), world, @intCast(k)), f.peer_order[k]);
            }
            try check(&f, seq, n);
            for (0..world) |src| try f.drain(src);
            try t.expectEqual(@as(u64, 0), f.flight);
        }
        try t.expectEqual(@as(usize, 0), f.bad);
    };
}

test "multirank verbs: shared CQ reverse-peer completions subtract that QP's exact stripe bytes" {
    var f: Fabric = undefined;
    try f.init(3, 1, 1 << 20);
    defer f.deinit();
    var w = &f.wires[0];
    try w.post(1, 1, 1 << 20);
    try w.post(0, 2, 16);
    // Peer two has two queued completions. The second is tiny, while the global oldest is peer one's large stripe.
    try w.reap();
    try t.expectEqual(@as(u64, 16), f.last_bytes);
    try t.expectEqual(f.flight, w.inflight());
    try f.drain(0);
    try t.expectEqual(@as(u64, 0), w.inflight());
    try t.expect(f.peak <= 2 << 20);
}

test "multirank verbs: strict cap across every peer, unequal stripe sizes and many queue wraps" {
    for ([_]u32{ 3, 4 }) |world| for ([_]u32{ 1, 2 }) |nh| {
        var f: Fabric = undefined;
        try f.init(world, nh, 1 << 20);
        defer f.deinit();
        for (0..200) |i| {
            const seq: u32 = @intCast(i + 1);
            const n: u32 = ([_]u32{ 1 << 20, 16, (1 << 20) - 16, 48 })[i % 4];
            try f.wires[0].post(seq & 1, seq, n);
            try t.expectEqual(f.flight, f.wires[0].inflight());
            try t.expect(f.flight <= 2 << 20);
        }
        try f.drain(0);
        try t.expectEqual(@as(u64, 0), f.flight);
        try t.expect(f.peak <= 2 << 20);
    };
}

test "multirank verbs: wrong offset control reaches byte comparison and is rejected" {
    var f: Fabric = undefined;
    try f.init(3, 2, 4096);
    defer f.deinit();
    f.wrong_offset = true;
    for (0..3) |src| {
        @memset(f.regions[src][f.layout.sendAt(1)..][0..128], @intCast(src + 1));
        try f.wires[src].post(1, 1, 128);
    }
    try t.expectEqual(@as(usize, 0), f.bad);
    try t.expectError(error.WrongBytes, check(&f, 1, 128));
    for (0..3) |src| try f.drain(src);
}

test "multirank verbs: invalid geometry and direct reduction/exchange refuse before any verb or launch" {
    var f: Fabric = undefined;
    try f.init(3, 1, 4096);
    defer f.deinit();
    try t.expectError(error.OutOfRegion, f.wires[0].post(2, 1, 16));
    try t.expectError(error.OutOfRegion, f.wires[0].post(0, 1, 0));
    try t.expectError(error.WriteTooLarge, f.wires[0].post(0, 1, (1 << 20) + 16));
    try t.expectEqual(@as(u64, 0), f.order);
    var r: Roce = undefined;
    try t.expectError(error.NcclReductionRequired, r.sum(1, 2, 3, .f32, null));
    try t.expectError(error.NcclExchangeRequired, r.exchangeBytes(1, 2, 3, null));
    var plan: [2]guard.Stripe = undefined;
    for ([_]u32{ 0, 1, 2, 9 }) |world| try t.expectError(error.OutOfRegion, guard.stripes(guard.Layout.of(world, 4096), 0, 1, 1, 0, 16, &plan));
}

test "multirank verbs: wrong peer control is rejected by the byte gate" {
    var f: Fabric = undefined;
    try f.init(3, 1, 4096);
    defer f.deinit();
    f.wires[0].hcas[0].qp[1] = &f.qp[0][0][2];
    f.wires[0].peer_region[1] = f.wires[0].peer_region[2];
    f.wires[0].peer_rkey[1] = f.wires[0].peer_rkey[2];
    for (0..3) |src| {
        @memset(f.regions[src][f.layout.sendAt(1)..][0..128], @intCast(src + 1));
        try f.wires[src].post(1, 1, 128);
    }
    try t.expectEqual(@as(usize, 0), f.bad);
    try t.expectError(error.WrongBytes, check(&f, 1, 128));
    for (0..3) |src| try f.drain(src);
}

test "multirank verbs: a completion for another sequence is named before accounting is changed" {
    var f: Fabric = undefined;
    try f.init(3, 1, 4096);
    defer f.deinit();
    try f.wires[0].post(1, 1, 128);
    const bytes = f.wires[0].inflight();
    f.bad_completion = true;
    try t.expectError(error.Completion, f.wires[0].reap());
    try t.expectEqual(bytes, f.wires[0].inflight());
}

fn records(f: *const Fabric, out: []Record) void {
    for (out, 0..) |*r, i| r.* = f.wires[i].record();
}

fn expectNoRtr(f: *const Fabric) !void {
    for (f.dest[0..f.world]) |by_hca| for (by_hca) |by_peer| for (by_peer) |d| try t.expectEqual(@as(?u32, null), d);
}

test "multirank verbs: on a switch connect pairs HCA h with every peer's HCA h over two stripes, as before" {
    var f: Fabric = undefined;
    try f.init(4, 2, 4096);
    defer f.deinit();
    var recs: [4]Record = undefined;
    records(&f, &recs);
    for (0..4) |r| {
        f.wires[r].stripes = 0;
        f.wires[r].peer_rkey = @splat(@splat(0));
        try f.wires[r].connect(&recs);
    }
    for (0..4) |r| {
        try t.expectEqual(@as(u32, 2), f.wires[r].stripes);
        for (0..2) |h| for (0..4) |p| if (p != r) {
            try t.expectEqual(@as(u8, @intCast(h)), f.wires[r].via[p][h]);
            try t.expectEqual(@as(u32, @intCast(200 + 2 * p + h)), f.wires[r].peer_rkey[p][h]);
            try t.expectEqual(@as(?u32, f.qp[p][h][r].qp_num), f.dest[r][h][p]);
        };
    }
    try t.expectEqual(@as(usize, 0), f.bad);
}

test "multirank verbs: a switchless ring of three connects only each link's queue pairs and gathers over one stripe" {
    var f: Fabric = undefined;
    try f.init(3, 2, 4096);
    defer f.deinit();
    f.ring();
    var recs: [3]Record = undefined;
    records(&f, &recs);
    for (0..3) |r| try f.wires[r].connect(&recs);
    for (0..3) |r| {
        try t.expectEqual(@as(u32, 1), f.wires[r].stripes);
        for (0..2) |h| for (0..3) |p| if (p != r) {
            const want: ?u32 = if (f.lands[r][h][p]) |j| f.qp[p][j][r].qp_num else null;
            try t.expectEqual(want, f.dest[r][h][p]);
        };
    }
    for (1..12) |s| {
        const seq: u32 = @intCast(s);
        const n: u32 = ([_]u32{ 16, 48, 1024, 4096 })[s % 4];
        for (0..3) |src| {
            for (f.regions[src][f.layout.sendAt(seq & 1)..][0..n], 0..) |*v, i| v.* = @truncate(29 * src + 3 * s + i);
            try f.wires[src].post(seq & 1, seq, n);
        }
        try check(&f, seq, n);
        for (0..3) |src| try f.drain(src);
        try t.expectEqual(@as(u64, 0), f.flight);
    }
    try t.expectEqual(@as(usize, 0), f.bad);
}

test "multirank verbs: a ring of four, an HCA on nobody's /24 and unequal stripes refuse before any queue pair moves" {
    {
        // a ring of four: ranks 0 and 2 share no link
        var f: Fabric = undefined;
        try f.init(4, 2, 4096);
        defer f.deinit();
        f.ring();
        var recs: [4]Record = undefined;
        records(&f, &recs);
        for (0..4) |r| try t.expectError(error.BadRecord, f.wires[r].connect(&recs));
        try expectNoRtr(&f);
        try t.expectError(error.OutOfRegion, f.wires[0].post(1, 1, 16)); // never connected: no stripe to post
    }
    {
        // rank 0's HCA 1 on nobody's /24: it would idle; ranks 1 and 2 see one stripe to rank 0, two to each other
        var f: Fabric = undefined;
        try f.init(3, 2, 4096);
        defer f.deinit();
        f.wires[0].hcas[1].gid[14] = 7;
        var recs: [3]Record = undefined;
        records(&f, &recs);
        for (0..3) |r| try t.expectError(error.BadRecord, f.wires[r].connect(&recs));
        try expectNoRtr(&f);
    }
}

test "multirank verbs: malformed peer tables refuse before any connection or post" {
    var f: Fabric = undefined;
    try f.init(4, 2, 4096);
    defer f.deinit();
    var recs: [4]Record = undefined;
    records(&f, &recs);
    _ = try f.wires[0].validateRecords(&recs);
    const good = recs;
    try t.expectError(error.BadRecord, f.wires[0].validateRecords(recs[0..3]));
    recs[2].magic +%= 1;
    try t.expectError(error.BadRecord, f.wires[0].validateRecords(&recs));
    recs = good;
    recs[2].rank = 1;
    try t.expectError(error.BadRecord, f.wires[0].validateRecords(&recs));
    recs = good;
    recs[2].world = 3;
    try t.expectError(error.BadRecord, f.wires[0].validateRecords(&recs));
    recs = good;
    recs[2].region = std.math.maxInt(u64) - 1;
    try t.expectError(error.BadRecord, f.wires[0].validateRecords(&recs));
    recs = good;
    recs[2].qpn[0][0] = 0;
    try t.expectError(error.BadRecord, f.wires[0].validateRecords(&recs));
    try t.expectEqual(@as(u64, 0), f.order);
}

test "multirank setup: failed CUDA drain retains the partial device owner exactly once" {
    const cuda = @import("cuda");
    const device = @import("roce_multi_device.zig");
    const Driver = struct {
        var frees: usize = 0;
        fn sync() callconv(.c) cuda.abi.Result {
            return 700;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            frees += 1;
            return 0;
        }
    };
    var f: Fabric = undefined;
    try f.init(3, 1, 4096);
    defer f.deinit();
    var d: cuda.Driver = undefined;
    d.api.cuCtxSynchronize = Driver.sync;
    d.api.cuMemFree_v2 = Driver.free;
    var r: Roce = .{ .me = 0, .world = 3, .layout = f.layout, .slot_bytes = 4096, .timeout_ns = 1000, .region = f.regions[0], .mapping = f.regions[0], .wire = .{ .verbs = f.wires[0] }, .cpu = null, .d = &d, .state = .{ .d = &d, .ptr = 0x4000, .len = 16 } };
    const before = Roce.kept.load(.monotonic);
    defer Roce.kept.store(before, .monotonic);
    Driver.frees = 0;
    device.close(&r);
    try t.expectEqual(@as(usize, 0), Driver.frees);
    try t.expect(r.retained and r.state != null);
    try t.expectEqual(before + 1, Roce.kept.load(.monotonic));
    device.close(&r);
    try t.expectEqual(@as(usize, 0), Driver.frees);
    try t.expectEqual(before + 1, Roce.kept.load(.monotonic));
}
