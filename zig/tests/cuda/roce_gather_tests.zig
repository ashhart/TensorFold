//! The group's RoCE all-gather (Group.openRoce) against NCCL's byte for byte at any world (NCCL_COMM_ID): every size,
//! graph replays, 3,000 mixed-size calls, then the time a call of each path. `tf-cuda-test roce-gather <rank> [world]`.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const cap = 512 << 10;
/// Shards a rank: tiny, a few decode rows, the cap, and past it or not a multiple of 16 (those stay on NCCL).
const sizes = [_]usize{ 16, 48, 4096, 8192, 16384, 32768, 65536, 262144, cap, cap + 16, 100 };
const seeds = [_]u64{ 1, 2, 3 };
const graph_gathers = 4;
const graph_replays = 64;
const reps = 2000;
/// The mixed phase: decode-sized shards (a verify call's gathers are 16 KiB a rank at four ranks, the drafter's up to
/// 64 KiB) with a few tiny ones between, every call against NCCL's rows. The recipe's GO gate: zero mismatches.
const mixed_calls = 3000;
const mixed_sizes = [_]usize{ 16384, 32768, 65536, 16384, 32768, 65536, 4096, 16 };
/// The recipe's GO gate on time: a 32 KiB-a-rank gather in at most this many microseconds (printed, not failed: the
/// bytes decide pass or fail, the launcher decides on the time).
const go_us_32k: f64 = 25.0;

/// Rank `rank`'s shard for a case: bytes that differ by rank, size and seed.
fn shard(out: []u8, seed: u64, rank: usize) void {
    var rng = std.Random.DefaultPrng.init(seed *% 1000003 +% rank *% 7919 +% out.len);
    rng.random().bytes(out);
}

/// Every rank's shard of `n` bytes for `seed`, in rank order: what both all-gathers must produce.
fn expected(want: []u8, n: usize, seed: u64, world: usize) void {
    for (0..world) |r| shard(want[r * n ..][0..n], seed, r);
}

pub fn run(gpu: Gpu, rank: usize, world: usize) !void {
    const a = gpu.gpa;
    try check.expect(std.c.getenv("NCCL_COMM_ID") != null, "roce-gather: NCCL_COMM_ID names rank 0's address", .{});
    try check.expect(world >= 2 and rank < world, "roce-gather: rank {d} of world {d}", .{ rank, world });
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.tp_group.Group.open(gpu.d, rank, world);
    defer g.close();
    try g.openRoce(a, gpu.io, gpu.d, s, cap);
    try check.expect(g.fast != null, "roce-gather: the RoCE path did not come up on every rank (see the warning above)", .{});

    const most = sizes[std.mem.indexOfMax(usize, &sizes)];
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, most);
    defer mine.free();
    var by_group = try cuda.DeviceBuffer.alloc(gpu.d, world * most);
    defer by_group.free();
    var by_nccl = try cuda.DeviceBuffer.alloc(gpu.d, world * most);
    defer by_nccl.free();
    const host = try a.alloc(u8, most);
    defer a.free(host);
    const got = try a.alloc(u8, world * most);
    defer a.free(got);
    const want = try a.alloc(u8, world * most);
    defer a.free(want);

    var by_roce: usize = 0;
    for (sizes) |n| for (seeds) |seed| {
        shard(host[0..n], seed, rank);
        try mine.upload(0, host[0..n]);
        try g.allGather(s, mine.ptr, by_group.ptr, n);
        try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, by_nccl.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
        try s.synchronize();
        try g.check();
        expected(want, n, seed, world);
        try by_group.download(0, got[0 .. world * n]);
        try check.sameBytes("group all-gather against every rank's shard", got[0 .. world * n], want[0 .. world * n]);
        try by_nccl.download(0, got[0 .. world * n]);
        try check.sameBytes("NCCL all-gather against every rank's shard", got[0 .. world * n], want[0 .. world * n]);
        if (g.fast.?.fits(n)) by_roce += 1;
    };
    check.pass("roce-gather rank {d} of {d}: shards of {any} bytes (seeds 1-3) equal NCCL's all-gather; {d} of {d} cases by RoCE", .{ rank, world, sizes, by_roce, sizes.len * seeds.len });

    try replays(gpu, &g, s, rank, world);
    const mismatches = try mixed(gpu, &g, s, rank, world, mine, by_group, by_nccl, host, got, want);
    const us_32k = try timing(gpu, &g, s, rank, world);
    std.debug.print("RESULT roce-gather world {d} rank {d}: 32 KiB a rank RoCE {d:.1} us a call (GO at most {d:.1}), {d} mixed-size calls, {d} mismatch(es) (GO 0)\n", .{ world, rank, us_32k, go_us_32k, mixed_calls, mismatches });
    try check.expect(mismatches == 0, "roce-gather rank {d}: {d} of {d} mixed-size calls differed from NCCL's rows", .{ rank, mismatches, mixed_calls });
}

/// A graph of `graph_gathers` RoCE gathers replayed with new shards each time: each replay's rows are its own.
fn replays(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize, world: usize) !void {
    const a = gpu.gpa;
    const n = 8192;
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, graph_gathers * n);
    defer mine.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, graph_gathers * world * n);
    defer all.free();
    try cuda.graph.beginCapture(s, .thread_local);
    for (0..graph_gathers) |i| try g.allGather(s, mine.ptr + i * n, all.ptr + i * world * n, n);
    var captured = try cuda.graph.endCapture(s);
    defer captured.deinit();
    var exec = try captured.instantiate();
    defer exec.deinit();
    const host = try a.alloc(u8, graph_gathers * n);
    defer a.free(host);
    const got = try a.alloc(u8, graph_gathers * world * n);
    defer a.free(got);
    const want = try a.alloc(u8, graph_gathers * world * n);
    defer a.free(want);
    for (0..graph_replays) |r| {
        for (0..graph_gathers) |i| {
            const seed = 100 + r * graph_gathers + i;
            shard(host[i * n ..][0..n], seed, rank);
            expected(want[i * world * n ..][0 .. world * n], n, seed, world);
        }
        try mine.upload(0, host);
        try exec.launchOn(s);
        try s.synchronize();
        try g.check();
        try all.download(0, got);
        try check.sameBytes("a replayed graph's gathers", got, want);
    }
    check.pass("roce-gather rank {d} of {d}: a captured graph of {d} gathers of {d} bytes replayed {d} times, each replay's rows its own", .{ rank, world, graph_gathers, n, graph_replays });
}

/// `mixed_calls` gathers of cycling decode sizes, each by the group (RoCE) and by NCCL, every call's rows compared to
/// the expected shards and to each other; the count of calls that differed (the GO gate wants zero).
fn mixed(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize, world: usize, mine: cuda.DeviceBuffer, by_group: cuda.DeviceBuffer, by_nccl: cuda.DeviceBuffer, host: []u8, got: []u8, want: []u8) !usize {
    _ = gpu;
    var mismatches: usize = 0;
    var by_roce: usize = 0;
    for (0..mixed_calls) |i| {
        const n = mixed_sizes[i % mixed_sizes.len];
        const seed: u64 = 10_000 + i;
        shard(host[0..n], seed, rank);
        try mine.upload(0, host[0..n]);
        try g.allGather(s, mine.ptr, by_group.ptr, n);
        try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, by_nccl.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
        try s.synchronize();
        try g.check();
        expected(want, n, seed, world);
        try by_group.download(0, got[0 .. world * n]);
        const group_ok = std.mem.eql(u8, got[0 .. world * n], want[0 .. world * n]);
        try by_nccl.download(0, got[0 .. world * n]);
        const nccl_ok = std.mem.eql(u8, got[0 .. world * n], want[0 .. world * n]);
        if (!group_ok or !nccl_ok) {
            mismatches += 1;
            if (mismatches <= 8) std.debug.print("FAIL roce-gather rank {d}: call {d} ({d} bytes a rank): group {s}, NCCL {s}\n", .{ rank, i, n, if (group_ok) "ok" else "differs", if (nccl_ok) "ok" else "differs" });
        }
        if (g.fast.?.fits(n)) by_roce += 1;
    }
    check.pass("roce-gather rank {d} of {d}: {d} mixed-size calls ({any} bytes a rank, cycling), {d} by RoCE, {d} mismatch(es) against NCCL's rows", .{ rank, world, mixed_calls, mixed_sizes, by_roce, mismatches });
    return mismatches;
}

/// Median microseconds a call of `reps` back-to-back gathers, RoCE against NCCL, at decode-sized shards; the RoCE
/// figure at 32 KiB a rank.
fn timing(gpu: Gpu, g: *cuda.tp_group.Group, s: cuda.Stream, rank: usize, world: usize) !f64 {
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    var mine = try cuda.DeviceBuffer.alloc(gpu.d, 65536);
    defer mine.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, world * 65536);
    defer all.free();
    var at_32k: f64 = 0;
    for ([_]usize{ 4096, 8192, 16384, 32768, 65536 }) |n| {
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            for (0..2) |round| { // the first round warms
                try t0.record(s);
                for (0..reps) |_| {
                    if (which == 0) try g.allGather(s, mine.ptr, all.ptr, n) else try g.lib.check(g.lib.api.ncclAllGather(mine.ptr, all.ptr, n, .u8, g.comm, s.handle), "ncclAllGather");
                }
                try t1.record(s);
                try t1.synchronize();
                if (round == 1) us[which] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) * 1000.0 / reps;
            }
        }
        try g.check();
        if (n == 32768) at_32k = us[0];
        check.pass("roce-gather rank {d} of {d}: {d} bytes a rank, {d} calls: RoCE {d:.1} us a call, NCCL {d:.1} us", .{ rank, world, n, reps, us[0], us[1] });
    }
    return at_32k;
}
