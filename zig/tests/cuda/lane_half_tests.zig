//! The lane's half tiles (16x32, MXFP8 and BF16): the 64-column tile's bytes, FP64-bounded, timed on DRAM weights.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Mode = cuda.qmmf.Mode;
const Shape = struct { mode: Mode, m: usize, n: usize, k: usize };

/// Flash Next decode projections whose 64-column grids leave GB10's SMs idle, the 64-column edge, wider row counts.
const shapes = [_]Shape{
    .{ .mode = .bf16, .m = 1, .n = 324, .k = 10240 },  .{ .mode = .bf16, .m = 1, .n = 320, .k = 10240 },
    .{ .mode = .bf16, .m = 1, .n = 512, .k = 2560 },   .{ .mode = .mxfp8, .m = 1, .n = 640, .k = 2560 },
    .{ .mode = .mxfp8, .m = 1, .n = 2560, .k = 640 },  .{ .mode = .bf16, .m = 7, .n = 324, .k = 10240 },
    .{ .mode = .mxfp8, .m = 16, .n = 640, .k = 2560 }, .{ .mode = .bf16, .m = 3, .n = 96, .k = 512 },
    .{ .mode = .mxfp8, .m = 5, .n = 32, .k = 128 },    .{ .mode = .bf16, .m = 1, .n = 10240, .k = 320 },
    .{ .mode = .mxfp8, .m = 1, .n = 2560, .k = 6144 }, .{ .mode = .mxfp8, .m = 1, .n = 6144, .k = 2560 },
};

const Built = struct { w: cuda.qmmf.Weight, deq: []f64 };

/// A random weight in `mode`, packed for the lane, and its exact float64 values [n][k].
fn build(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, sh: Shape) !Built {
    const deq = try a.alloc(f64, sh.n * sh.k);
    if (sh.mode == .bf16) {
        const w = r.bfs(sh.n * sh.k, 0.05);
        for (deq, w) |*o, v| o.* = rig.bf(v);
        const tiled = try a.alloc(u16, cuda.nvfp4.padded(sh.n) * sh.k);
        cuda.qmmf.packBf16(tiled, w, sh.n, sh.k);
        return .{ .w = cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(tiled)), sh.n, sh.k), .deq = deq };
    }
    const codes = try a.alloc(u8, sh.n * sh.k);
    for (codes) |*c| c.* = rng.int(u8) & 0xF7; // no e4m3 NaN (0x7F, 0xFF)
    const scales = try a.alloc(u8, sh.n * sh.k / 32);
    for (scales) |*s| s.* = 120 + rng.uintLessThan(u8, 8);
    for (deq, 0..) |*o, i| o.* = @import("core").quant.fp8.e4m3(codes[i]) * std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, scales[i / 32]) - 127)));
    const pc = try a.alloc(u8, cuda.mxfp8.padded(sh.n) * sh.k);
    cuda.mxfp8.packCodes(pc, codes, sh.n, sh.k);
    const ps = try a.alloc(u8, cuda.mxfp8.padded(sh.n) * (sh.k / 32));
    cuda.mxfp8.packScales(ps, scales, sh.n, sh.k);
    return .{ .w = cuda.mxfp8.weight(try r.dev(pc), try r.dev(ps), sh.n, sh.k), .deq = deq };
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x4a1f);
    var r = try rig.Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.capability() / 10));
    defer lane.unload();
    const keep = lane.half_below;
    var halves: usize = 0;
    for (shapes) |sh| {
        const built = try build(&r, a, prng.random(), sh);
        const weight = built.w;
        const xs = r.bfs(sh.m * sh.k, 1.0);
        const x = try r.dev(std.mem.sliceAsBytes(xs));
        const part = try r.out(@max(16, lane.partBytes(sh.m, weight)));
        const p = cuda.qmmf.plan(sh.m, sh.n, sh.k, lane.clusters);
        const takes = p.bm == 16 and !p.fused and ((sh.n + 63) / 64) * p.sk < keep;
        if (takes) halves += 1;
        var times: [2]f64 = .{ 0, 0 };
        var f32_rows: []f32 = &.{};
        inline for (.{ cuda.qmmf.Out.f32, cuda.qmmf.Out.bf16 }) |out| {
            const bytes = sh.m * sh.n * (if (out == .f32) @as(usize, 4) else 2);
            const ys = [2]u64{ try r.out(bytes), try r.out(bytes) };
            for (0..2) |on| {
                lane.half_below = if (on == 1) keep else 0;
                try lane.matmulOut(r.s, x, sh.k, sh.m, weight, ys[on], out, part);
                try r.s.synchronize();
            }
            const got = try r.back(u8, ys[1], bytes);
            const want = try r.back(u8, ys[0], bytes);
            try check.sameBytes("half tiles equal the 64-column tile", got, want);
            if (out == .f32) f32_rows = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(got)));
        }
        // FP64: every fp32 output within the lane's sum bound
        for (0..sh.m) |row| for (0..sh.n) |j| {
            var want: f64 = 0;
            var mag: f64 = 0;
            for (0..sh.k) |i| {
                const t = rig.bf(xs[row * sh.k + i]) * built.deq[j * sh.k + i];
                want += t;
                mag += @abs(t);
            }
            try r.within("half-tile lane fp32 vs FP64", f32_rows[row * sh.n + j], want, rig.sumTol(mag, sh.k) + 1e-6);
        };
        // time on cold weights: copies past the L2, cycled
        const wbytes: usize = if (sh.mode == .bf16) sh.n * sh.k * 2 else sh.n * sh.k + sh.n * sh.k / 32;
        const copies = @min(64, @max(2, (128 << 20) / wbytes));
        var cold: [64]cuda.qmmf.Weight = undefined;
        for (0..copies) |c| cold[c] = (try build(&r, a, prng.random(), sh)).w;
        const y = try r.out(sh.m * sh.n * 4);
        for (0..2) |on| {
            lane.half_below = if (on == 1) keep else 0;
            const reps = 300;
            const t0 = check.now(gpu.io);
            for (0..reps) |i| try lane.matmulOut(r.s, x, sh.k, sh.m, cold[i % copies], y, .f32, part);
            try r.s.synchronize();
            times[on] = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e3 / reps;
        }
        std.debug.print("lane-half {t} m={d} n={d} k={d} (sk {d}{s}): 64-col {d:.1} us ({d:.0} GB/s), half {d:.1} us ({d:.0} GB/s), bytes equal\n", .{ sh.mode, sh.m, sh.n, sh.k, p.sk, if (takes) ", takes half tiles" else ", stays 64-col", times[0], @as(f64, @floatFromInt(wbytes)) / 1e3 / times[0], times[1], @as(f64, @floatFromInt(wbytes)) / 1e3 / times[1] });
    }
    lane.half_below = keep;
    try check.expect(halves >= 5, "the decode shapes take half tiles ({d})", .{halves});
    r.report("lane half tiles: MXFP8 and BF16, f32 and bf16 out, bit-equal to the 64-column tile, FP64-bounded");
}
