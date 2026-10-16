//! The rank-order reduce-scatter on real ranks: each rank's rows against an all-gather plus a host rank-order sum.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const rs = cuda.rank_scatter;
const width = 4096;
const rows_cases = [_]usize{ 1, 4, 64, 2047, 2048 };
const reps = 20;

/// Rank `rank`'s partial rows for a case: normal values scaled over nine binades, so the adds round.
fn partial(gpa: std.mem.Allocator, seed: u64, rank: usize, rows: usize) ![]f32 {
    var rng = std.Random.DefaultPrng.init(seed *% 1000003 +% rank *% 7919 +% rows);
    const out = try gpa.alloc(f32, rows * width);
    for (out, 0..) |*v, i| {
        const binade: i32 = @as(i32, @intCast(i % 9)) - 4;
        v.* = rng.random().floatNorm(f32) * std.math.ldexp(@as(f32, 1.0), binade);
    }
    return out;
}

/// A communicator: one rank alone, or `world` ranks on NCCL_COMM_ID or on rank 0's unique id in `id_path` (as hex).
fn comm(lib: *const cuda.nccl.Library, gpu: Gpu, rank: usize, world: usize, id_path: ?[]const u8) !cuda.nccl.Comm {
    var c: cuda.nccl.Comm = null;
    if (world == 1) {
        const devices = [1]c_int{gpu.ctx.device};
        try lib.check(lib.api.ncclCommInitAll(@ptrCast(&c), 1, &devices), "ncclCommInitAll");
        return c;
    }
    var id: cuda.nccl.UniqueId = undefined;
    if (id_path) |path| {
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            const hex = std.fmt.bytesToHex(id.internal, .lower);
            try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = &hex });
        } else {
            var hex: [256]u8 = undefined;
            const got = try std.Io.Dir.cwd().readFile(gpu.io, path, &hex);
            try check.expect(got.len == 256, "unique id file {s}: {d} hex digits", .{ path, got.len });
            _ = try std.fmt.hexToBytes(&id.internal, &hex);
        }
    } else {
        try check.expect(std.c.getenv("NCCL_COMM_ID") != null, "rank-scatter: NCCL_COMM_ID or an id file", .{});
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
    return c;
}

pub fn run(gpu: Gpu, rank: usize, world: usize, id_path: ?[]const u8) !void {
    const a = gpu.gpa;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const c = try comm(&lib, gpu, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(c);
    var k = try rs.RankScatter.load(gpu.d);
    defer k.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    for (rows_cases) |rows| for ([_]u64{ 1, 2, 3 }) |seed| {
        const mine = rs.share(rows, world, rank);
        const own = mine[1] - mine[0];
        const host = try partial(a, seed, rank, rows);
        defer a.free(host);
        var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer dp.free();
        var scratch = try cuda.DeviceBuffer.alloc(gpu.d, @max(rs.scratchBytes(rows, world, width), 4));
        defer scratch.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, @max(own * width * 4, 4));
        defer out.free();
        var gathered = try cuda.DeviceBuffer.alloc(gpu.d, world * rows * width * 4);
        defer gathered.free();
        try k.run(&lib, c, world, rank, s, dp.ptr, rows, width, scratch.ptr, out.ptr);
        try lib.check(lib.api.ncclAllGather(dp.ptr, gathered.ptr, rows * width, .f32, c, s.handle), "ncclAllGather");
        try s.synchronize();
        const g = try check.download(gpu, gathered);
        defer a.free(g);
        const gf = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(g)));
        const want = try a.alloc(f32, own * width);
        defer a.free(want);
        for (want, 0..) |*w, i| {
            const at = mine[0] * width + i;
            var acc = gf[at];
            for (1..world) |r| acc += gf[r * rows * width + at];
            w.* = acc;
        }
        if (own > 0) {
            const got = try check.download(gpu, out);
            defer a.free(got);
            try check.sameBytes("rank-order reduce-scatter", got[0 .. own * width * 4], std.mem.sliceAsBytes(want));
        }
    };
    const what = "rank-scatter rank {d} of {d}: rows {any} x {d} (seeds 1-3), every slice byte-equal to an all-gather";
    check.pass(what ++ " plus a rank-order sum", .{ rank, world, rows_cases, width });
    try timing(gpu, &lib, c, &k, s, rank, world);
}

/// Median ms of the reduce-scatter (sends, receives and the sum) and of an all-gather of the same partials, 2,048 rows.
fn timing(
    gpu: Gpu,
    lib: *const cuda.nccl.Library,
    c: cuda.nccl.Comm,
    k: *const rs.RankScatter,
    s: cuda.Stream,
    rank: usize,
    world: usize,
) !void {
    const rows = 2048;
    const host = try partial(gpu.gpa, 9, rank, rows);
    defer gpu.gpa.free(host);
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
    defer dp.free();
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, rs.scratchBytes(rows, world, width));
    defer scratch.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * width * 4);
    defer out.free();
    var gathered = try cuda.DeviceBuffer.alloc(gpu.d, world * rows * width * 4);
    defer gathered.free();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    var ms: [2][reps]f32 = undefined;
    for (0..reps + 3) |i| for (0..2) |which| {
        try t0.record(s);
        if (which == 0) {
            try k.run(lib, c, world, rank, s, dp.ptr, rows, width, scratch.ptr, out.ptr);
        } else {
            const all = lib.api.ncclAllGather(dp.ptr, gathered.ptr, rows * width, .f32, c, s.handle);
            try lib.check(all, "ncclAllGather");
        }
        try t1.record(s);
        try t1.synchronize();
        if (i >= 3) ms[which][i - 3] = try cuda.Event.elapsedMs(t0, t1);
    };
    for (&ms) |*m| std.mem.sort(f32, m, {}, std.sort.asc(f32));
    const what = "rank-scatter rank {d} of {d}, {d} x {d} fp32: reduce-scatter {d:.3} ms, all-gather alone {d:.3} ms";
    check.pass(what ++ " (medians of {d})", .{ rank, world, rows, width, ms[0][reps / 2], ms[1][reps / 2], reps });
}

/// tf_rank_sum alone on one GPU (no NCCL), for 2 to 5 ranks as each rank: every element the rank-order sum's bits.
pub fn sum(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var k = try rs.RankScatter.load(gpu.d);
    defer k.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const rows = 300;
    const n = rows * width - 3; // past 4,096 blocks of 256 (the loop wraps) and not a multiple of 256
    for (2..6) |world| for (0..world) |me| {
        const terms = try a.alloc([]f32, world);
        for (terms, 0..) |*t, r| t.* = try partial(a, 5, r, rows);
        const slots = try a.alloc(f32, world * n);
        @memset(slots, std.math.nan(f32)); // a rank's own slot is never read: NaN would show
        for (terms, 0..) |t, r| if (r != me) @memcpy(slots[r * n ..][0..n], t[0..n]);
        var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(slots));
        defer dp.free();
        var own = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(terms[me][0..n]));
        defer own.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, n * 4);
        defer out.free();
        try k.rankSum(s, dp.ptr, own.ptr, world, me, n, out.ptr);
        try s.synchronize();
        const want = try a.alloc(f32, n);
        for (want, 0..) |*w, i| {
            var acc = terms[0][i];
            for (1..world) |r| acc += terms[r][i];
            w.* = acc;
        }
        const got = try check.download(gpu, out);
        defer gpu.gpa.free(got);
        try check.sameBytes("tf_rank_sum against the rank-order sum", got, std.mem.sliceAsBytes(want));
    };
    check.pass("rank-sum: 2 to 5 ranks, as each rank, {d} elements: every element the rank-order sum's bits", .{n});
}
