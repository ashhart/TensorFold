//! The box's bf16 tensor-unit GEMM at prompt scale: MLX's `gemm_nax` body from the tree, dispatched with the
//! engine's own conventions. The reference rate every "at the peak" claim about the prompt GEMMs needs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");

const ENTRY = "custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
/// What the engine dispatches at rows = 3619 (M not 64-aligned, N = 384 128-aligned): the bounded variant.
const ENTRY_BOUND = "custom_kernel_tf_gemm_nax_bf16_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

const Shape = struct { m: u32, k: u32, n: u32 };

fn bench(device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, s: Shape, reps: usize, trials: usize) !f64 {
    const a = try device.buffer(@as(usize, s.m) * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer a.deinit();
    const b = try device.buffer(@as(usize, s.n) * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer b.deinit();
    const d = try device.buffer(@as(usize, s.m) * s.n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer d.deinit();
    // in-range bf16 values, so nothing overflows or denormalises
    for (a.slice(u16, @as(usize, s.m) * s.k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64.0)) >> 16);
    for (b.slice(u16, @as(usize, s.n) * s.k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 64.0)) >> 16);

    const tn = (s.n + 127) / 128;
    const tm = (s.m + 63) / 64;
    var p: [16]i32 = @splat(0);
    const vals = [_]u32{ s.m, s.n, s.k, s.k, s.k, s.n, tn, tm, 2, s.k / 256, 0, 0, 0, 0, 0 };
    for (vals, 0..) |x, i| p[i] = @intCast(x);

    var samples: [16]f64 = undefined;
    const nt = @max(1, @min(trials, samples.len));
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| {
            e.setPipeline(pipe);
            e.setBuffer(a, 0, 0);
            e.setBuffer(b, 0, 1);
            e.setBytes(std.mem.asBytes(&p), 2);
            e.setBuffer(d, 0, 3);
            e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  {d}x{d}x{d}: failed: {s}\n", .{ s.m, s.n, s.k, msg });
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    }
    return median(samples[0..nt]);
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    const io = init.io;
    dense_rows = benchDenseRows();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    if (std.c.getenv("TF_BENCH_FP8") != null) {
        try fp8DenseBench(device, queue, gpa);
        return;
    }
    if (std.c.getenv("TF_BENCH_FP8_REAL") != null) {
        try fp8RealBench(device, queue, gpa, io);
        return;
    }
    if (std.c.getenv("TF_BENCH_MMA8") != null) {
        try mxDenseBench(device, queue, gpa);
        return;
    }
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    var gemm_src: ?[]const u8 = null;
    inline for (ks.prefill) |f| {
        if (comptime std.mem.eql(u8, f.name, "gemm_nax")) gemm_src = f.text;
    }
    const nax_text = try std.mem.replaceOwned(u8, gpa, gemm_src orelse return error.NoGemmSource, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_text);
    const text: []const u8 = if (device.tensorUnits()) nax_text else blk: {
        break :blk try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", nax_text });
    };
    defer if (!device.tensorUnits()) gpa.free(text);
    const lib = try mtl.Library.fromSource(device, text, options);
    const pipe = try mtl.Pipeline.init(device, lib, ENTRY, false);
    const shapes = [_]Shape{
        .{ .m = 3584, .k = 5120, .n = 8192 }, // wo_b at prompt rows
        .{ .m = 3584, .k = 8192, .n = 5120 },
        .{ .m = 3584, .k = 5120, .n = 2304 }, // routed gate at prompt rows
        .{ .m = 3584, .k = 5120, .n = 384 }, // the router gate
    };
    std.debug.print("gemm bench: MLX gemm_nax (bf16, tb), tensor units {}\n", .{device.tensorUnits()});
    std.debug.print("  {s:>18} {s:>9} {s:>9} {s:>9}\n", .{ "M x N x K", "ms", "TMAC/s", "GB/s" });
    // The dispatch convention at the engine's own case: M = 3619 (not 64-aligned) and N = 384, the bounded variant.
    {
        const s = Shape{ .m = 3619, .k = 5120, .n = 384 };
        const bound = try mtl.Pipeline.init(device, lib, ENTRY_BOUND, false);
        const ms = try bench(device, queue, bound, s, 1, 1);
        _ = ms;
        const a = try device.buffer(@as(usize, s.m) * s.k * 2, mtl.ResourceOptions.shared);
        defer a.deinit();
        const b = try device.buffer(@as(usize, s.n) * s.k * 2, mtl.ResourceOptions.shared);
        defer b.deinit();
        const dbytes = @as(usize, s.m) * s.n * 2;
        const pad: usize = 1 << 20;
        const d = try device.buffer(dbytes + pad, mtl.ResourceOptions.shared);
        defer d.deinit();
        @memset(d.slice(u8, dbytes + pad), 0xA5);
        for (a.slice(u16, @as(usize, s.m) * s.k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64.0)) >> 16);
        for (b.slice(u16, @as(usize, s.n) * s.k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 64.0)) >> 16);
        const tn = (s.n + 127) / 128;
        const tm = (s.m + 63) / 64;
        var q: [16]i32 = @splat(0);
        const vals = [_]u32{ s.m, s.n, s.k, s.k, s.k, s.n, tn, tm, 2, s.k / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |x, i| q[i] = @intCast(x);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(bound);
        e.setBuffer(a, 0, 0);
        e.setBuffer(b, 0, 1);
        e.setBytes(std.mem.asBytes(&q), 2);
        e.setBuffer(d, 0, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
        e.end();
        cb.commit();
        cb.wait();
        const got = d.slice(u16, @as(usize, s.m) * s.n);
        const av = a.slice(u16, @as(usize, s.m) * s.k);
        const bv = b.slice(u16, @as(usize, s.n) * s.k);
        const f = struct {
            fn v(x: u16) f32 {
                return @bitCast(@as(u32, x) << 16);
            }
        }.v;
        var worst: f32 = 0;
        var worst_at: [2]usize = .{ 0, 0 };
        for ([_][2]usize{ .{ 0, 0 }, .{ 0, 5 }, .{ 63, 383 }, .{ 64, 0 }, .{ 3618, 383 }, .{ 2048, 100 } }) |rc| {
            var acc: f32 = 0;
            for (0..s.k) |k| acc += f(av[rc[0] * s.k + k]) * f(bv[rc[1] * s.k + k]);
            const err = @abs(acc - f(got[rc[0] * s.n + rc[1]]));
            if (err > worst) {
                worst = err;
                worst_at = rc;
            }
        }
        std.debug.print("  bounded check 3619x384x5120: worst |fp32 - gemm_nax| = {d:.5} at row {d} col {d}\n", .{ worst, worst_at[0], worst_at[1] });
        // Does the bounded dispatch write past rows*384 bf16? The engine's private/scratch destinations are exactly
        // that size; a canary here separates an out-of-bounds write from a plumbing fault.
        const bytes = d.slice(u8, dbytes + pad);
        var first_dirty: usize = 0;
        var dirty: usize = 0;
        for (bytes[dbytes..], 0..) |x, i| if (x != 0xA5) {
            if (dirty == 0) first_dirty = i;
            dirty += 1;
        };
        std.debug.print("  bounded canary: {d} bytes written past rows*384 bf16 (first at +{d})\n", .{ dirty, first_dirty });
    }
    // One small case against an fp32 host reference: the dispatch convention itself, not just its speed.
    {
        const s = Shape{ .m = 64, .k = 5120, .n = 384 };
        const ms = try bench(device, queue, pipe, s, 1, 1);
        _ = ms;
        const a = try device.buffer(@as(usize, s.m) * s.k * 2, mtl.ResourceOptions.shared);
        defer a.deinit();
        const b = try device.buffer(@as(usize, s.n) * s.k * 2, mtl.ResourceOptions.shared);
        defer b.deinit();
        const d = try device.buffer(@as(usize, s.m) * s.n * 2, mtl.ResourceOptions.shared);
        defer d.deinit();
        for (a.slice(u16, @as(usize, s.m) * s.k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64.0)) >> 16);
        for (b.slice(u16, @as(usize, s.n) * s.k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 64.0)) >> 16);
        const tn = (s.n + 127) / 128;
        const tm = (s.m + 63) / 64;
        var q: [16]i32 = @splat(0);
        const vals = [_]u32{ s.m, s.n, s.k, s.k, s.k, s.n, tn, tm, 2, s.k / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |x, i| q[i] = @intCast(x);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(a, 0, 0);
        e.setBuffer(b, 0, 1);
        e.setBytes(std.mem.asBytes(&q), 2);
        e.setBuffer(d, 0, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
        e.end();
        cb.commit();
        cb.wait();
        const got = d.slice(u16, @as(usize, s.m) * s.n);
        const av = a.slice(u16, @as(usize, s.m) * s.k);
        const bv = b.slice(u16, @as(usize, s.n) * s.k);
        const f = struct {
            fn v(x: u16) f32 {
                return @bitCast(@as(u32, x) << 16);
            }
        }.v;
        var worst: f32 = 0;
        for ([_][2]usize{ .{ 0, 0 }, .{ 0, 5 }, .{ 1, 0 }, .{ 63, 383 }, .{ 32, 100 } }) |rc| {
            var acc: f32 = 0;
            for (0..s.k) |k| acc += f(av[rc[0] * s.k + k]) * f(bv[rc[1] * s.k + k]);
            const err = @abs(acc - f(got[rc[0] * s.n + rc[1]]));
            worst = @max(worst, err);
        }
        std.debug.print("  convention check 64x384x5120: worst |fp32 - gemm_nax| = {d:.5}\n", .{worst});
    }
    for (shapes) |s| {
        _ = try bench(device, queue, pipe, s, 1, 1);
        const ms = try bench(device, queue, pipe, s, 2, 5);
        const macs = @as(f64, @floatFromInt(s.m)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k));
        const bytes = (@as(f64, @floatFromInt(s.m)) * s.k + @as(f64, @floatFromInt(s.n)) * s.k + @as(f64, @floatFromInt(s.m)) * s.n) * 2.0;
        std.debug.print("  {d:>6}x{d:>6}x{d:>5} {d:>9.3} {d:>9.1} {d:>9.1}\n", .{ s.m, s.n, s.k, ms, macs / (ms * 1e-3) / 1e12, bytes / (ms * 1e-3) / 1e9 });
    }
    const only_routed = std.c.getenv("TF_BENCH_ROUTED_ONLY") != null;
    const verify_only = std.c.getenv("TF_BENCH_VERIFY_ONLY") != null;
    const cmp_only = std.c.getenv("TF_BENCH_CMP") != null;
    if (std.c.getenv("TF_BENCH_READ") != null) {
        try routedReadBench(device, queue, gpa);
        return;
    }
    if (std.c.getenv("TF_BENCH_ROUTE_DIAG") != null) try routeDiag(device, queue, gpa);
    if (!only_routed and !verify_only and !cmp_only) try oneRowBench(device, queue, gpa);
    // The lane-packing run is the one-row bench alone (the routed, multiset and prefill arms are priced elsewhere
    // and their fixtures are heavy).
    if (std.c.getenv("TF_BENCH_LANE") != null) return;
    if (std.c.getenv("TF_BENCH_WIDTH") != null) {
        try widthSweep(device, queue, gpa);
        return;
    }
    if (std.c.getenv("TF_BENCH_AFFINE_PREFILL") != null) {
        try affinePrefillPrice(device, queue, gpa);
        return;
    }
    if (!verify_only and !cmp_only) try routedBench(device, queue, gpa);
    if (!cmp_only and std.c.getenv("TF_BENCH_NO_VERIFY") == null) try verifyMultisetBench(device, queue, gpa);
    if (!only_routed and std.c.getenv("TF_BENCH_ONE_ROW_ONLY") == null) try exl3Bench(device, queue, gpa);
}

// -- EXL3 prompt body at the routed stage shape ---------------------------------------------------------------
// The stage shape: 384 expert tiles x 64 rows (the grouped prompt padding), N = 2304, K = 5120, the same
// tf_exl3_matmul_prefill_nax3 (RG=4, CG=2) the engine dispatches for a grouped prompt chunk. This is the
// 18.7 TMAC/s term in the prefill ledger; the double-buffered candidate is nax7 below it.

const Exl3Args = extern struct {
    k: u32,
    n: u32,
    rows: u32,
    halves: u32,
    codebook: u32,
    su_signs: u32,
    sv_signs: u32,
    has_bias: u32,
    output_f32: u32,
    input_ids: u32,
    output_ids: u32,
    routed: u32,
    expert_id: u32,
    repeat: u32,
    indexed: u32 = 0,
    experts: u32 = 0,
    splits: u32 = 1,
    x_stride: u64,
    y_stride: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
};

const Exl3Expert = extern struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    bias: u64,
    tile_k_stride: u64,
    tile_n_stride: u64,
    halves: u32,
    codebook: u32,
    suh_signs: u32,
    svh_signs: u32,
    has_bias: u32,
    reserved: u32 = 0,
};

const exl3_tiles = 384;
const exl3_rows = exl3_tiles * 64;
const exl3_n = 2304;
const exl3_k = 5120;
const exl3_halves = 6;
const exl3_tile_bytes = exl3_halves * 8 * 2;
const exl3_n_stride = exl3_tile_bytes;
const exl3_k_stride = (exl3_n / 16) * exl3_n_stride;
const exl3_expert_bytes = (exl3_k / 16) * exl3_k_stride;

fn timeExl3(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, table: mtl.Buffer, cols: usize, threads: usize, trials: usize) !f64 {
    const tn = (exl3_n + cols - 1) / cols;
    const tm = exl3_rows / 64;
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.setBuffer(table, 0, 7);
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  exl3: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// The same timing helper with the ROW COUNT as a parameter -- the grouped plan's shape (the engine dispatches 685
// tiles of sixty-four rows at the prompt's shape) is not the bench's default fixture (384 tiles).
fn timeRows(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, table: mtl.Buffer, cols: usize, threads: usize, rows: usize, trials: usize) !f64 {
    const tn = (exl3_n + cols - 1) / cols;
    const tm = rows / 64;
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.setBuffer(table, 0, 7);
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  plan exl3: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// The same dispatch with the tile's ROW COUNT as a parameter. The sixty-four-row helper hard-coded it, so the
// hundred-and-twenty-eight-row plans dispatched twice the tiles they need (half of them empty, and the empty ones
// cost the same launch overhead the arm is measuring).
fn timeRowsT(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, table: mtl.Buffer, cols: usize, threads: usize, rows: usize, rpt: usize, trials: usize) !f64 {
    const tn = (exl3_n + cols - 1) / cols;
    const tm = (rows + rpt - 1) / rpt;
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.setBuffer(table, 0, 7);
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  plan exl3: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// The engine's grouped plan at a given tile granularity: each expert's block padded to a multiple of the tile's
// rows, the blocks laid out consecutively (the GPU's own grouping rule). Returns the rows the plan uses and writes
// each expert's first row.
fn planOffsets(counts: []const u32, rpt: usize, offsets: []u32) usize {
    var used: usize = 0;
    for (counts, 0..) |c, i| {
        offsets[i] = @intCast(used);
        used += ((@as(usize, c) + rpt - 1) / rpt) * rpt;
    }
    return used;
}

// The live rows' hash, expert by expert (each expert's live rows are contiguous in the plan), so two plans at
// different granularities are compared ROW FOR ROW: the values must be the same bits.
fn hashLiveRows(sums: mtl.Buffer, floats: usize, counts: []const u32, offsets: []const u32, n: usize) u64 {
    var h: u64 = 0;
    const s = sums.slice(f32, floats);
    for (counts, 0..) |c, i| {
        if (c == 0) continue;
        const base: usize = offsets[i];
        const bytes = std.mem.sliceAsBytes(s[base * n ..][0 .. @as(usize, c) * n]);
        h = h *% 0x9E3779B97F4A7C15 +% std.hash.Wyhash.hash(0, bytes);
    }
    return h;
}

// The two plans' live rows, compared row for row (the first difference and where it is).
fn compareLiveRows(a_sums: mtl.Buffer, a_floats: usize, a_counts: []const u32, a_offsets: []const u32, b_sums: mtl.Buffer, b_floats: usize, b_offsets: []const u32, n: usize) void {
    const sa = a_sums.slice(f32, a_floats);
    const sb = b_sums.slice(f32, b_floats);
    var diffs: usize = 0;
    for (a_counts, 0..) |c, i| {
        if (c == 0) continue;
        const ba: usize = a_offsets[i];
        const bb: usize = b_offsets[i];
        for (0..c) |j| {
            const ra = sa[(ba + j) * n ..][0..n];
            const rb = sb[(bb + j) * n ..][0..n];
            for (ra, rb) |x, y| {
                if (x != y) {
                    if (diffs < 5) std.debug.print("    difference: expert {d} row {d} col {d}: {d} vs {d}\n", .{ i, j, 0, x, y });
                    diffs += 1;
                    break;
                }
            }
        }
    }
    std.debug.print("  the two plans' live rows: {d} differing rows\n", .{diffs});
}

// The grouped plan's own histogram at the prompt's shape -- the engine's per-expert row counts, read back from the
// GPU by its own instrument. The plan pads EACH expert's block to a multiple of the tile's sixty-four rows and lays
// the blocks out consecutively, so the tiles that do work are the sum of the per-expert ceilings and the plan's
// tail is unused.
const plan_counts = [_]u32{
    13,  7,   219, 2,   122, 13,  1,   9,   76,  10,  61,  37,  95,  0,   1,   7,
    17,  53,  6,   1,   21,  2,   4,   29,  16,  0,   7,   20,  163, 69,  36,  3,
    29,  50,  15,  857, 15,  0,   21,  109, 193, 71,  22,  56,  73,  31,  7,   13,
    128, 121, 0,   1,   7,   20,  16,  148, 29,  12,  1,   36,  2,   147, 32,  5,
    126, 5,   76,  117, 223, 294, 24,  31,  23,  113, 20,  7,   69,  84,  4,   0,
    7,   10,  49,  4,   101, 128, 119, 116, 6,   42,  7,   3,   37,  0,   12,  59,
    32,  11,  27,  1,   2,   21,  4,   0,   22,  105, 2,   49,  56,  12,  66,  14,
    230, 2,   10,  147, 400, 15,  41,  54,  17,  34,  0,   7,   38,  9,   53,  133,
    100, 4,   0,   17,  117, 25,  93,  28,  38,  1,   15,  2,   0,   80,  221, 7,
    1,   53,  48,  35,  0,   59,  0,   45,  37,  81,  5,   50,  95,  22,  0,   74,
    6,   21,  355, 0,   0,   28,  74,  34,  9,   23,  140, 27,  61,  18,  33,  2,
    44,  41,  5,   12,  50,  102, 63,  136, 53,  61,  13,  0,   75,  6,   16,  8,
    5,   21,  26,  3,   70,  133, 150, 10,  52,  69,  0,   181, 3,   12,  11,  21,
    126, 54,  0,   14,  4,   16,  8,   4,   41,  71,  1,   37,  2,   1,   49,  11,
    4,   74,  0,   30,  57,  152, 79,  21,  65,  0,   5,   4,   83,  838, 163, 12,
    20,  55,  653, 6,   19,  7,   2,   10,  88,  0,   10,  18,  6,   17,  20,  15,
    92,  49,  2,   121, 69,  47,  54,  34,  93,  1,   32,  13,  42,  118, 55,  9,
    23,  10,  111, 39,  48,  99,  38,  10,  16,  14,  105, 81,  0,   47,  105, 79,
    6,   73,  118, 5,   0,   11,  85,  8,   153, 60,  7,   254, 18,  58,  125, 11,
    28,  33,  68,  100, 44,  1,   40,  33,  1,   6,   85,  19,  14,  62,  122, 34,
    19,  28,  81,  165, 3,   78,  9,   40,  23,  1,   30,  1,   90,  71,  8,   44,
    10,  78,  142, 1,   78,  59,  26,  17,  5,   44,  13,  158, 20,  49,  92,  12,
    37,  101, 21,  84,  8,   24,  4,   16,  2,   40,  5,   0,   16,  3,   30,  66,
    39,  197, 43,  49,  41,  0,   10,  70,  4,   1,   252, 94,  13,  59,  2,   9,
};

// Default 3619 (the record's long-standing fixture).
fn benchDenseRows() usize {
    if (std.c.getenv("TF_BENCH_DENSE_ROWS")) |v| {
        if (std.fmt.parseInt(usize, std.mem.span(v), 10)) |n| {
            if (n >= 128 and n <= 8192) return n;
        } else |_| {}
    }
    return 3619;
}
var dense_rows: usize = 3619;
const dense_n = 8192;
const dense_k = 5120;

fn timeShape(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, table: ?mtl.Buffer, m: usize, n: usize, cols: usize, threads: usize, tg_rows: usize, trials: usize) !f64 {
    const tn = (n + cols - 1) / cols;
    const tm = (m + tg_rows - 1) / tg_rows;
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        if (table) |tbl| e.setBuffer(tbl, 0, 7); // The 128-column body's codebook table
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  shape exl3: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

fn timeGrouped128(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, trials: usize) !f64 {
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.dispatchGroups(.{ .width = 36, .height = 384, .depth = 1 }, .{ .width = 256 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  grouped128: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// The tap projection's width-8 funnel.
fn benchTapFunnel(device: mtl.Device, queue: mtl.Queue, p31: mtl.Pipeline, p51: mtl.Pipeline, m: usize, n: usize, k: usize, halves: u32) !void {
    const tile_bytes = @as(usize, halves) * 8 * 2;
    const n_stride = tile_bytes;
    const trellis_bytes = (k / 16) * (n / 16) * tile_bytes;
    const tr = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer tr.deinit();
    var rng: u64 = 0x4330_0000 ^ @as(u64, n) ^ @as(u64, k) ^ @as(u64, halves);
    for (tr.slice(u16, trellis_bytes / 2)) |*v| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        v.* = @truncate(rng);
    }
    const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    @memset(x.slice(u16, m * k), 0x3c00);
    const sums = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    @memset(sums.slice(u32, m * n), 0);
    const args = Exl3Args{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(m),
        .halves = halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 0,
        .experts = 0,
        .splits = 1,
        .x_stride = @intCast(k),
        .y_stride = @intCast(n),
        .tile_k_stride = (n / 16) * n_stride,
        .tile_n_stride = n_stride,
    };
    const macs = @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    std.debug.print("tap projection {d}x{d}x{d} halves {d}, dense 128x64 tile, 128 rows a threadgroup:\n", .{ m, n, k, halves });
    _ = try timeShape(queue, p31, &args, x, tr, sums, null, m, n, 64, 256, 128, 1);
    const ms31 = try timeShape(queue, p31, &args, x, tr, sums, null, m, n, 64, 256, 128, 5);
    const h31 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
    std.debug.print("  {s:>7} {d:>8.3} ms {d:>6.1} TMAC/s\n", .{ "nax31", ms31, macs / (ms31 * 1e-3) / 1e12 });
    _ = try timeShape(queue, p51, &args, x, tr, sums, null, m, n, 64, 256, 128, 1);
    const ms51 = try timeShape(queue, p51, &args, x, tr, sums, null, m, n, 64, 256, 128, 5);
    const h51 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
    std.debug.print("  {s:>7} {d:>8.3} ms {d:>6.1} TMAC/s ({d:>5.1}%) {s}\n", .{ "nax51", ms51, macs / (ms51 * 1e-3) / 1e12, 100.0 * (ms51 - ms31) / ms31, if (h31 == h51) "IDENTICAL" else "DIFFER" });
}

fn benchDenseShape(device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, m: usize, n: usize, k: usize) !void {
    return benchDenseShapeAt(device, queue, pipe, m, n, k, 128, 256);
}

fn benchDenseShapeAt(device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, m: usize, n: usize, k: usize, tg_rows: usize, threads: usize) !void {
    const trellis_bytes = (k / 16) * (n / 16) * exl3_tile_bytes;
    const tr = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer tr.deinit();
    var rng: u64 = 0xD1CEB00C ^ @as(u64, n);
    for (tr.slice(u16, trellis_bytes / 2)) |*v| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        v.* = @truncate(rng);
    }
    const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    @memset(x.slice(u16, m * k), 0x3c00);
    const sums = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    @memset(sums.slice(u32, m * n), 0);
    const args = Exl3Args{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(m),
        .halves = exl3_halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 0,
        .experts = 0,
        .splits = 1,
        .x_stride = @intCast(k),
        .y_stride = @intCast(n),
        .tile_k_stride = (n / 16) * exl3_n_stride,
        .tile_n_stride = exl3_n_stride,
    };
    _ = try timeShape(queue, pipe, &args, x, tr, sums, null, m, n, 64, threads, tg_rows, 1);
    const ms = try timeShape(queue, pipe, &args, x, tr, sums, null, m, n, 64, threads, tg_rows, 5);
    const macs = @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    std.debug.print("  {d:>6}x{d:>6}x{d:>5} {d:>9.3} {d:>9.1}\n", .{ m, n, k, ms, macs / (ms * 1e-3) / 1e12 });
}

// Nax31 against the 128-column double-buffered bodies at one prompt shape.
fn benchDenseCompare128(device: mtl.Device, queue: mtl.Queue, p31: mtl.Pipeline, p79: ?mtl.Pipeline, p80: ?mtl.Pipeline, p82: ?mtl.Pipeline, p83: ?mtl.Pipeline, table: mtl.Buffer, m: usize, n: usize, k: usize) !void {
    const trellis_bytes = (k / 16) * (n / 16) * exl3_tile_bytes;
    const tr = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer tr.deinit();
    var rng: u64 = 0x4310_0000 ^ @as(u64, n) ^ @as(u64, k);
    for (tr.slice(u16, trellis_bytes / 2)) |*v| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        v.* = @truncate(rng);
    }
    const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    @memset(x.slice(u16, m * k), 0x3c00);
    const sums = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    @memset(sums.slice(u32, m * n), 0);
    const args = Exl3Args{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(m),
        .halves = exl3_halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 0,
        .experts = 0,
        .splits = 1,
        .x_stride = @intCast(k),
        .y_stride = @intCast(n),
        .tile_k_stride = (n / 16) * exl3_n_stride,
        .tile_n_stride = exl3_n_stride,
    };
    const macs = @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    _ = try timeShape(queue, p31, &args, x, tr, sums, null, m, n, 64, 256, 128, 1);
    const ms31 = try timeShape(queue, p31, &args, x, tr, sums, null, m, n, 64, 256, 128, 5);
    const h31 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
    std.debug.print("  {d:>6}x{d:>6}x{d:>5} nax31 {d:>8.3} ms {d:>6.1} TMAC/s", .{ m, n, k, ms31, macs / (ms31 * 1e-3) / 1e12 });
    if (p79) |q| {
        _ = try timeShape(queue, q, &args, x, tr, sums, table, m, n, 128, 512, 128, 1);
        const ms = try timeShape(queue, q, &args, x, tr, sums, table, m, n, 128, 512, 128, 5);
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
        std.debug.print(" | nax79 {d:>8.3} ms {d:>6.1} ({d:>5.1}%) {s}", .{ ms, macs / (ms * 1e-3) / 1e12, 100.0 * (ms - ms31) / ms31, if (h == h31) "IDENTICAL" else "DIFFER" });
    }
    if (p80) |q| {
        _ = try timeShape(queue, q, &args, x, tr, sums, null, m, n, 128, 512, 128, 1);
        const ms = try timeShape(queue, q, &args, x, tr, sums, null, m, n, 128, 512, 128, 5);
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
        std.debug.print(" | nax80 {d:>8.3} ms {d:>6.1} ({d:>5.1}%) {s}", .{ ms, macs / (ms * 1e-3) / 1e12, 100.0 * (ms - ms31) / ms31, if (h == h31) "IDENTICAL" else "DIFFER" });
    }
    if (p82) |q| {
        _ = try timeShape(queue, q, &args, x, tr, sums, null, m, n, 128, 256, 64, 1);
        const ms = try timeShape(queue, q, &args, x, tr, sums, null, m, n, 128, 256, 64, 5);
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
        std.debug.print(" | nax82 {d:>8.3} ms {d:>6.1} ({d:>5.1}%) {s}", .{ ms, macs / (ms * 1e-3) / 1e12, 100.0 * (ms - ms31) / ms31, if (h == h31) "IDENTICAL" else "DIFFER" });
    }
    if (p83) |q| {
        _ = try timeShape(queue, q, &args, x, tr, sums, null, m, n, 128, 256, 128, 1);
        const ms = try timeShape(queue, q, &args, x, tr, sums, null, m, n, 128, 256, 128, 5);
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, m * n)));
        std.debug.print(" | nax83 {d:>8.3} ms {d:>6.1} ({d:>5.1}%) {s}", .{ ms, macs / (ms * 1e-3) / 1e12, 100.0 * (ms - ms31) / ms31, if (h == h31) "IDENTICAL" else "DIFFER" });
    }
    std.debug.print("\n", .{});
}

// Is the dense bench's rate cache-inflated? The per-dispatch time is the number to compare with the in-situ stage.
fn timeShapeCold(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trs: []const mtl.Buffer, sums: mtl.Buffer, m: usize, n: usize, cols: usize, threads: usize, tg_rows: usize, distinct: bool) !f64 {
    const tn = (n + cols - 1) / cols;
    const tm = (m + tg_rows - 1) / tg_rows;
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (trs, 0..) |tr, i| {
        const b = if (distinct) tr else trs[0];
        if (!distinct and i > 0) {}
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(b, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("  cold: failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(trs.len));
}

fn benchDenseCold(device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, m: usize, n: usize, k: usize, copies: usize) !void {
    const trellis_bytes = (k / 16) * (n / 16) * exl3_tile_bytes;
    var bufs: [16]mtl.Buffer = undefined;
    var made: usize = 0;
    errdefer for (bufs[0..made]) |b| b.deinit();
    while (made < copies and made < 16) : (made += 1) {
        bufs[made] = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var rng: u64 = 0xD1CEB00C ^ @as(u64, n) ^ @as(u64, made) * 0x9E3779B1;
        for (bufs[made].slice(u16, trellis_bytes / 2)) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = @truncate(rng);
        }
    }
    defer for (bufs[0..made]) |b| b.deinit();
    const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    @memset(x.slice(u16, m * k), 0x3c00);
    const sums = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    @memset(sums.slice(u32, m * n), 0);
    const args = Exl3Args{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(m),
        .halves = exl3_halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 0,
        .experts = 0,
        .splits = 1,
        .x_stride = @intCast(k),
        .y_stride = @intCast(n),
        .tile_k_stride = (n / 16) * exl3_n_stride,
        .tile_n_stride = exl3_n_stride,
    };
    _ = try timeShapeCold(queue, pipe, &args, x, bufs[0..made], sums, m, n, 64, 256, 128, false);
    const hot = try timeShapeCold(queue, pipe, &args, x, bufs[0..made], sums, m, n, 64, 256, 128, false);
    const cold = try timeShapeCold(queue, pipe, &args, x, bufs[0..made], sums, m, n, 64, 256, 128, true);
    const macs = @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    std.debug.print("  {d:>6}x{d:>6}x{d:>5} hot {d:>8.3} ms {d:>7.1} TMAC/s | cold x{0} {d:>8.3} ms {d:>7.1} TMAC/s\n", .{ m, n, k, hot, macs / (hot * 1e-3) / 1e12, cold, macs / (cold * 1e-3) / 1e12 });
}

// The routed prompt stream's one-cold-pass READ arm at the live plan's shape. 388 row tiles x 36 column tiles, 384
// distinct expert allocations, no decode / no activation / no MMA. The arms vary the stream's prefetch width and
// the grid's scheduling; the shipped body's full time is the control (the record's 384-tile nax59). Bench only: no
// engine path selects any of these kernels.
fn timeRead(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sink: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, gw: usize, gh: usize, threads: usize, trials: usize) !f64 {
    var samples: [8]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sink, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.dispatchGroups(.{ .width = gw, .height = gh, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  read: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

fn routedReadBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    const src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const arms = [_]struct { name: []const u8, pipe: mtl.Pipeline, gw: usize, gh: usize, threads: usize }{
        .{ .name = "read90 coop 1-block", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax90", false), .gw = 36, .gh = 388, .threads = 128 },
        .{ .name = "read91 vec16 1-block", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax91", false), .gw = 36, .gh = 388, .threads = 128 },
        .{ .name = "read92 shipped 2w", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax92", false), .gw = 36, .gh = 388, .threads = 128 },
        .{ .name = "read93 slab seq", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax93", false), .gw = 1, .gh = 388, .threads = 128 },
        .{ .name = "read94 grid transposed", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax94", false), .gw = 388, .gh = 36, .threads = 128 },
        .{ .name = "read95 8warps splitk", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax95", false), .gw = 36, .gh = 388, .threads = 256 },
        .{ .name = "read96 vec16 chunk 4w", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax96", false), .gw = 36, .gh = 388, .threads = 128 },
        .{ .name = "read97 vec16 chunk 8w", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax97", false), .gw = 36, .gh = 388, .threads = 256 },
        .{ .name = "read98 slab 16w x9", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax98", false), .gw = 1, .gh = 388, .threads = 512 },
        .{ .name = "read99 linear ceiling", .pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_stream_read_nax99", false), .gw = 36, .gh = 388, .threads = 128 },
    };
    defer for (arms) |a| a.pipe.deinit();

    var experts: [exl3_tiles]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (experts[0..made]) |bb| bb.deinit();
    while (made < exl3_tiles) : (made += 1) {
        experts[made] = try device.buffer(exl3_expert_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var rng: u64 = 0x9E3779B97F4A7C15 ^ @as(u64, made);
        for (experts[made].slice(u16, exl3_expert_bytes / 2)) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = @truncate(rng);
        }
    }
    const tiles: usize = 388;
    const plan_rows: usize = tiles * 64;
    const routes = try device.buffer(plan_rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer routes.deinit();
    for (routes.slice(u32, plan_rows), 0..) |*r, i| r.* = @intCast((i / 64) % exl3_tiles);
    const descs = try device.buffer(exl3_tiles * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer descs.deinit();
    const descs_hot = try device.buffer(exl3_tiles * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer descs_hot.deinit();
    {
        const d: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
        const dh: [*]Exl3Expert = @ptrCast(@alignCast(descs_hot.contents()));
        for (0..exl3_tiles) |i| {
            d[i] = .{
                .trellis = experts[i].gpuAddress(),
                .suh = 0,
                .svh = 0,
                .bias = 0,
                .tile_k_stride = exl3_k_stride,
                .tile_n_stride = exl3_n_stride,
                .halves = exl3_halves,
                .codebook = 2,
                .suh_signs = 0,
                .svh_signs = 0,
                .has_bias = 0,
            };
            dh[i] = d[i];
            dh[i].trellis = experts[0].gpuAddress();
        }
    }
    // The ENGINE's layout is one trellis buffer with the experts back to back (exl3_layout.zig), not 384 separate
    // allocations. Copy the fixture into that layout so the read arms can price it.
    const slab = try device.buffer(exl3_tiles * exl3_expert_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer slab.deinit();
    for (0..exl3_tiles) |i| {
        @memcpy(slab.slice(u8, exl3_tiles * exl3_expert_bytes)[i * exl3_expert_bytes .. (i + 1) * exl3_expert_bytes], experts[i].slice(u8, exl3_expert_bytes));
    }
    const descs_slab = try device.buffer(exl3_tiles * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer descs_slab.deinit();
    {
        const d: [*]Exl3Expert = @ptrCast(@alignCast(descs_slab.contents()));
        const src_desc: [*]const Exl3Expert = @ptrCast(@alignCast(descs.contents()));
        for (0..exl3_tiles) |i| {
            d[i] = src_desc[i];
            d[i].trellis = slab.gpuAddress() + i * exl3_expert_bytes;
        }
    }
    const x = try device.buffer(256, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    const sink = try device.buffer(388 * 8192 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sink.deinit();
    @memset(sink.slice(u32, 388 * 8192), 0);
    var args = Exl3Args{
        .k = exl3_k,
        .n = exl3_n,
        .rows = @intCast(plan_rows),
        .halves = exl3_halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 2,
        .experts = exl3_tiles,
        .splits = 1,
        .x_stride = exl3_k,
        .y_stride = exl3_n,
        .tile_k_stride = exl3_k_stride,
        .tile_n_stride = exl3_n_stride,
    };
    const bytes = @as(f64, @floatFromInt(tiles)) * 36.0 * (64.0 * 5120.0 * 3.0 / 8.0);
    // The same-run FULL control: the engine's body (nax59) on the same 388-tile plan and the same contiguous
    // expert layout, so the read arms and the tile are priced against one another in one run.
    const p59 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax59", false);
    defer p59.deinit();
    const fx = try device.buffer(plan_rows * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer fx.deinit();
    @memset(fx.slice(u16, plan_rows * exl3_k), 0x3c00);
    const fsums = try device.buffer(plan_rows * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer fsums.deinit();
    @memset(fsums.slice(u32, plan_rows * exl3_n), 0);
    const fms = try timeRows(queue, p59, &args, fx, experts[0], fsums, routes, descs_slab, x, 64, 128, plan_rows, 5);
    const fmacs = @as(f64, @floatFromInt(plan_rows)) * @as(f64, exl3_n) * @as(f64, exl3_k);
    std.debug.print("  {s:>22} FULL nax59 median5 {d:>7.3} ms  {d:>6.1} TMAC/s  ({d:.1} GB/s effective)\n", .{ "full 388-tile", fms, fmacs / (fms * 1e-3) / 1e12, bytes / (fms * 1e-3) / 1e9 });
    const big = try device.buffer(@as(usize, @intFromFloat(bytes)), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer big.deinit();
    {
        var rng: u64 = 0x51A7E5C3D2B19F47;
        for (big.slice(u64, @as(usize, @intFromFloat(bytes)) / 8)) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = rng;
        }
    }
    std.debug.print("routed stream read: {d} tiles x 36 columns, {d} live rows, {d:.3} GB a pass, 384 distinct experts\n", .{ tiles, plan_rows, bytes / 1e9 });
    for (arms) |a| {
        const trellis = if (std.mem.eql(u8, a.name, "read99 linear ceiling")) big else experts[0];
        const warm = try timeRead(queue, a.pipe, &args, x, trellis, sink, routes, descs, a.gw, a.gh, a.threads, 1);
        const cold = try timeRead(queue, a.pipe, &args, x, trellis, sink, routes, descs, a.gw, a.gh, a.threads, 1);
        const ms = try timeRead(queue, a.pipe, &args, x, trellis, sink, routes, descs, a.gw, a.gh, a.threads, 5);
        std.debug.print("  {s:>22} first {d:>7.3} ms  second {d:>7.3}  median5 {d:>7.3} ms  {d:>7.1} GB/s (cold {d:>6.1})\n", .{ a.name, warm, cold, ms, bytes / (ms * 1e-3) / 1e9, bytes / (cold * 1e-3) / 1e9 });
    }
    // The same arms on the ENGINE's contiguous layout.
    for (arms[6..9]) |a| {
        const ms = try timeRead(queue, a.pipe, &args, x, experts[0], sink, routes, descs_slab, a.gw, a.gh, a.threads, 5);
        std.debug.print("  {s:>22} ENGINE-SLAB median5 {d:>7.3} ms  {d:>7.1} GB/s\n", .{ a.name, ms, bytes / (ms * 1e-3) / 1e9 });
    }
    for (arms[0..3]) |a| {
        const ms = try timeRead(queue, a.pipe, &args, x, experts[0], sink, routes, descs_hot, a.gw, a.gh, a.threads, 5);
        std.debug.print("  {s:>22} HOT median5 {d:>7.3} ms  {d:>7.1} GB/s\n", .{ a.name, ms, bytes / (ms * 1e-3) / 1e9 });
    }
}

fn timeDense(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, cols: usize, threads: usize, trials: usize) !f64 {
    const tn = (dense_n + cols - 1) / cols;
    const tm = (dense_rows + 127) / 128;
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  dense exl3: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// -- one-row dense decode: the shipped 16-column table body against the 128-column bench arm -------------------
// wq_b / wo_a / wo_b at the decode's one row. `splits` is the engine's own (16 narrow, 8 wide n), so the number
// includes the partials round trip the shipped stage pays; the wide arm takes the same split order.
// The engine's own width.halves, read from the one-row body log: 10 for wq_b, 12 for the rest. The earlier
// one-row arms ran at halves = 6, which is not what the pack stores.
const OneRow = struct { m: u32, k: u32, n: u32, halves: u32 };
const one_row_shapes = [_]OneRow{
    .{ .m = 1, .k = 1280, .n = 32768, .halves = 10 }, // wq_b, short k
    .{ .m = 1, .k = 4096, .n = 1024, .halves = 12 }, // wo_a, narrow n
    .{ .m = 1, .k = 5120, .n = 8192, .halves = 12 }, // wo_b, long k
    // The shapes the engine's one-row dispatch names, at the widths it names them (the serve log's "one-row body"
    // lines), so the arms are priced on the engine's own cases.
    .{ .m = 1, .k = 8192, .n = 5120, .halves = 10 },
    .{ .m = 1, .k = 5120, .n = 2304, .halves = 10 },
    .{ .m = 1, .k = 2304, .n = 5120, .halves = 10 },
    .{ .m = 1, .k = 6144, .n = 25600, .halves = 10 },
    .{ .m = 1, .k = 5120, .n = 1280, .halves = 12 },
    .{ .m = 1, .k = 5120, .n = 512, .halves = 12 },
    // The rest of the dense set the prefill dispatch log names (rows = 3619, all take the one-row path at decode).
    .{ .m = 1, .k = 8192, .n = 5120, .halves = 12 },
    .{ .m = 1, .k = 5120, .n = 2304, .halves = 12 },
    .{ .m = 1, .k = 5120, .n = 1280, .halves = 12 },
    .{ .m = 1, .k = 5120, .n = 512, .halves = 12 },
    // The output head, the one dense shape the 128-column body never took: n = 129,280, k = 5,120.
    .{ .m = 1, .k = 5120, .n = 129280, .halves = 12 },
    // The last two dense shapes the one-row log names (both wide n, so direct should hold).
    .{ .m = 1, .k = 6144, .n = 25600, .halves = 12 },
    .{ .m = 1, .k = 2304, .n = 5120, .halves = 12 },
};

const RowDispatch = struct { width: usize, height: usize, threads: usize, depth: usize = 1 };

fn timeRow(queue: mtl.Queue, pipe: mtl.Pipeline, reduce: ?mtl.Pipeline, args: Exl3Args, x: mtl.Buffer, trs: []const mtl.Buffer, sums: mtl.Buffer, partials: mtl.Buffer, table: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, d: RowDispatch, n_out: usize, trials: usize, reps: usize) !f64 {
    var samples: [16]f64 = undefined;
    const nt = @max(1, @min(trials, samples.len));
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |r| {
            e.setPipeline(pipe);
            e.setBuffer(x, 0, 0);
            e.setBuffer(trs[(t * reps + r) % trs.len], 0, 1);
            e.setBuffer(sums, 0, 2);
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(routes, 0, 4); // unread on the dense unrouted path
            e.setBuffer(descs, 0, 5);
            e.setBuffer(partials, 0, 6);
            e.setBuffer(table, 0, 7);
            e.dispatchGroups(.{ .width = d.width, .height = d.height, .depth = d.depth }, .{ .width = d.threads });
            if (args.splits > 1) {
                e.setPipeline(reduce.?);
                e.setBuffer(partials, 0, 0);
                e.setBuffer(sums, 0, 1);
                const ra = extern struct { splits: u32, n: u32 }{ .splits = args.splits, .n = @intCast(n_out) };
                e.setBytes(std.mem.asBytes(&ra), 2);
                e.dispatchThreads(.{ .width = n_out }, .{ .width = 256 });
            }
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  one-row: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    }
    return median(samples[0..nt]);
}

// The register-only body (the comparator's structure, already the routed decode's) on the DENSE one-row fixture.
// The body reads its weights through descriptors[expert], so a dense matrix needs a descriptor of its own; the
// pointer is rewritten a trial at a time so the eight trellis copies rotate and the weights are not resident when
// they come back (the same cold protocol as the other arms).
fn timeRowDesc(queue: mtl.Queue, pipe: mtl.Pipeline, args: Exl3Args, x: mtl.Buffer, trs: []const mtl.Buffer, sums: mtl.Buffer, partials: mtl.Buffer, table: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, d: RowDispatch, trials: usize, tile_bytes: u64, halves: u32) !f64 {
    var samples: [16]f64 = undefined;
    const nt = @max(1, @min(trials, samples.len));
    const dp: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
    for (0..nt) |t| {
        const src = trs[t % trs.len];
        dp[0] = .{ .trellis = src.gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = args.tile_k_stride, .tile_n_stride = tile_bytes, .halves = halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(src, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.setBuffer(partials, 0, 6);
        e.setBuffer(table, 0, 7);
        e.dispatchGroups(.{ .width = d.width, .height = d.height, .depth = d.depth }, .{ .width = d.threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  one-row fma: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// THE COMPARATOR'S LANE PACKING, PRICED AS A BENCH ARM. The source spot-check found the one structural difference
// left at the dense one-row shapes: the comparator gives a sixteen-column tile ONE simdgroup whose thirty-two lanes
// split the whole K walk, while the engine gives it TF_FMA_WARPS = 16 warps whose partials meet through threadgroup
// memory at the end. This helper selects the three shapes the decode ledger names (wq_b, wo_a, wo_b) so a lane-only
// run prices exactly those.
fn lanePackShape(s: OneRow) bool {
    return (s.n == 32768 and s.k == 1280) or (s.n == 8192 and s.k == 5120) or (s.n == 1024 and s.k == 4096);
}

// THE DENSE PACK'S WIDTH, PRICED AT THE DECODE'S OWN SHAPES. The engine's dense pack is 5-bit (halves 10: wq_b,
// wo_b, wo_a, shared) and 6-bit (halves 12: wq_a, wkv); the experts are 3-bit (halves 6). This sweeps the same
// three one-row shapes over halves 6/8/10/12 on the shipped dense kernel at the engine's sixteen warps, with the
// same cold-copy protocol, so any repack's bytes term is read directly from the time and the GB/s.
fn widthSweep(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(src);
    const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n{s}", .{src});
    defer gpa.free(wsrc);
    const lib = try mtl.Library.fromSource(device, wsrc, options);
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_decode_fma_dense", false);
    defer pipe.deinit();
    const words_src = try std.mem.concat(gpa, u8, &.{
        \\kernel void tf_exl3_diag_words(const device ushort* t [[buffer(0)]], device uint* out [[buffer(1)]],
        \\                             uint gid [[thread_position_in_grid]]) {
        \\  out[gid] = (uint)t[2 * gid] | ((uint)t[2 * gid + 1] << 16);
        \\}
    });
    defer gpa.free(words_src);
    const words_lib = try mtl.Library.fromSource(device, words_src, mtl.CompileOptions.mlx());
    defer words_lib.deinit();
    const words_pipe = try mtl.Pipeline.init(device, words_lib, "tf_exl3_diag_words", false);
    defer words_pipe.deinit();
    const scratch = try device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer scratch.deinit();
    const table = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer table.deinit();
    for (table.slice(u16, 3 * 65536), 0..) |*v, i| {
        if (i >= 2 * 65536 and i < 3 * 65536) {
            const state: u16 = @intCast(i - 2 * 65536);
            v.* = @bitCast(@as(f16, @floatCast(exl3TableValue(state))));
        } else v.* = 0x3c00 | @as(u16, @intCast(i & 0xff));
    }
    const shapes = [_]struct { n: u32, k: u32, pack: u32 }{
        .{ .n = 32768, .k = 1280, .pack = 10 }, // wq_b
        .{ .n = 8192, .k = 5120, .pack = 12 }, // wo_b
        .{ .n = 1024, .k = 4096, .pack = 12 }, // wo_a (one group)
    };
    std.debug.print("dense width sweep: the shipped dense kernel at sixteen warps, one row, the decode's shapes\n", .{});
    std.debug.print("  {s:>16} {s:>6} {s:>4} {s:>9} {s:>9} {s:>9} {s:>10}\n", .{ "shape", "halves", "bits", "ms", "weights/s", "GB/s", "vs pack" });
    for (shapes) |s| {
        const x = try device.buffer(s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        @memset(x.slice(u16, s.k), 0x3c00);
        const sums = try device.buffer(s.n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sums.deinit();
        const hs = [_]u32{ 6, 8, 10, 12 };
        var sets: [4][8]mtl.Buffer = undefined;
        for (hs, 0..) |h, hi| {
            const tile_bytes = h * 16;
            const trellis_bytes: u32 = (s.k / 16) * (s.n / 16) * tile_bytes;
            for (0..8) |ci| {
                sets[hi][ci] = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                const seed = 0x4421D7 ^ @as(u64, s.n) ^ @as(u64, s.k) ^ @as(u64, h) ^ @as(u64, ci);
                if (!fillTrellisVisible(queue, words_pipe, sets[hi][ci], trellis_bytes, seed, scratch))
                    std.debug.print("  width sweep: fixture {d} of {d}x{d} h{d} never became visible\n", .{ ci, s.n, s.k, h });
            }
        }
        defer for (sets) |set| {
            for (set) |b| b.deinit();
        };
        var t_by: [4]f64 = .{ 0, 0, 0, 0 };
        // Two passes like the one-row bench: pass 0 streams every buffer once (the clock ramp), pass 1 is the
        // number. Both passes rotate the eight copies, so every timed dispatch reads a buffer the SLC has not
        // just held -- the cold protocol the record's one-row arms use.
        for (0..2) |pass| {
            for (hs, 0..) |h, hi| {
                const tile_bytes = h * 16;
                const args = Exl3Args{
                    .k = s.k,
                    .n = s.n,
                    .rows = 1,
                    .halves = h,
                    .codebook = 2,
                    .su_signs = 0,
                    .sv_signs = 0,
                    .has_bias = 0,
                    .output_f32 = 1,
                    .input_ids = 0,
                    .output_ids = 0,
                    .routed = 0,
                    .expert_id = 0,
                    .repeat = 1,
                    .indexed = 0,
                    .experts = 0,
                    .splits = 1,
                    .x_stride = s.k,
                    .y_stride = s.n,
                    .tile_k_stride = (s.n / 16) * tile_bytes,
                    .tile_n_stride = tile_bytes,
                };
                @memset(sums.slice(u32, s.n), 0);
                const t = timeRow(queue, pipe, null, args, x, sets[hi][0..], sums, sums, table, x, x, .{ .width = s.n / 16, .height = 1, .threads = 512 }, s.n, if (pass == 0) 1 else 7, 1) catch 0;
                if (pass == 1) t_by[hi] = t;
            }
        }
        var t_pack: f64 = 0;
        for (hs, 0..) |h, hi| if (h == s.pack) {
            t_pack = t_by[hi];
        };
        for (hs, 0..) |h, hi| {
            const t = t_by[hi];
            if (t <= 0) continue;
            const trellis_bytes: u32 = (s.k / 16) * (s.n / 16) * h * 16;
            const gbs = @as(f64, @floatFromInt(trellis_bytes)) / 1e9;
            const weights = @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k));
            std.debug.print("  {d:>7}x{d:>6} {d:>6} {d:>4} {d:>9.3} {d:>9.0} {d:>8.1} {d:>9.0}%\n", .{
                s.n,                                           s.k, h, h / 2, t, weights / (t * 1e-3), gbs / (t * 1e-3),
                if (t_pack > 0) 100.0 * t / t_pack else 100.0,
            });
        }
    }
}

fn oneRowBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, ks.core_exl3_projection, options);
    defer lib.deinit();
    const shipped = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_tab", false);
    defer shipped.deinit();
    const wide = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_tab", false) catch |err| {
        std.debug.print("one-row bench: wide pipeline failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer wide.deinit();
    // The 32-k continuation of the wide body (two 16-k tiles decoded before one barrier).
    const wide2 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide2_tab", false) catch |err| {
        std.debug.print("one-row bench: wide2 pipeline failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer wide2.deinit();
    // Diagnostic: the wide body with the decode removed (timing only; values meaningless).
    const noload = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_noload_tab", false) catch null;
    defer if (noload) |p| p.deinit();
    // The separation experiment's three arms at the dense one-row shapes: the codebook read from a 1024-entry
    // threadgroup LUT (the dynamic selection left in), the static state selection with the table, and the static
    // selection with the LUT.
    const wide_lut = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_lut", false) catch null;
    defer if (wide_lut) |p| p.deinit();
    const wide_static = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_tab", false) catch null;
    defer if (wide_static) |p| p.deinit();
    const wide_static_lut = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_lut", false) catch null;
    defer if (wide_static_lut) |p| p.deinit();
    // The static one-row body with a deeper weight prefetch (PF blocks of stream words in flight a lane): the
    // narrow one-row shapes have one wave of threadgroups, so a block's DRAM latency needs more than one block's
    // compute to hide behind.
    const wide_static_lut_pf2 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_lut_pf2", false) catch null;
    defer if (wide_static_lut_pf2) |p| p.deinit();
    const wide_static_lut_pf4 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_wide_static_lut_pf4", false) catch null;
    defer if (wide_static_lut_pf4) |p| p.deinit();
    // The direct-register one-row body (a lane owns a column; no threadgroup tile, no barrier).
    const direct = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_direct_tab", false) catch |err| {
        std.debug.print("one-row bench: direct pipeline failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer direct.deinit();
    // The direct-register one-row body with a wider load width: the trellis stream word as one 32-bit load and the
    // activation row as four half4 loads (the same bytes in the same order, so the output is bit-identical).
    const directv = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_direct_tab_v", false) catch null;
    defer if (directv) |p| p.deinit();
    // The direct body with its stream words named instead of an array indexed by the runtime cursor: the array
    // could not live in registers, so the narrow body used thread-local memory.
    const directr = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_direct_tab_r", false) catch null;
    defer if (directr) |p| p.deinit();
    // The direct body loading only the words the map names (max(rel) + 2) instead of eight.
    const directnw_src = std.fmt.allocPrint(gpa, "#define TF_DIRECT_NW 1\n{s}", .{ks.core_exl3_projection}) catch null;
    defer if (directnw_src) |s| gpa.free(s);
    const directnw_lib = if (directnw_src) |s| (mtl.Library.fromSource(device, s, mtl.CompileOptions.mlx()) catch null) else null;
    defer if (directnw_lib) |l| l.deinit();
    const directnw = if (directnw_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_direct_tab", false) catch null) else null;
    defer if (directnw) |p| p.deinit();
    const reduce = mtl.Pipeline.init(device, lib, "tf_exl3_splitk_reduce", false) catch null;
    defer if (reduce) |p| p.deinit();
    // The routed tensor-unit decode body, to read the MMA class at the dense one-row shapes: a synthetic single
    // descriptor makes the dense trellis an "expert", which is exactly the reduction the band would have to cover.
    const nax_src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_src);
    const nax_lib = mtl.Library.fromSource(device, nax_src, mtl.CompileOptions.mlx()) catch null;
    defer if (nax_lib) |l| l.deinit();
    const mma = if (nax_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma) |p| p.deinit();
    // The register-direct MMA's decode-free floor (a bench-only arm; no engine path selects it).
    const regfloor = if (nax_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matvec_nax_regfloor", false) catch null) else null;
    defer if (regfloor) |p| p.deinit();
    // The register-only body on the dense one-row fixture. It is the routed decode's body (the comparator's
    // structure: one warp a sixteen-column tile, the decoded weights in registers, no threadgroup tile, no barrier
    // in the k loop, no split-K and no reduce dispatch), priced here on the DENSE shapes -- the shared expert's
    // gate/up and down and the attention's projections -- where the engine today runs the 128-column
    // threadgroup-tile body at half the rate the routed stage reaches on the same shape class.
    const fma_dense_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n{s}", .{nax_src});
    defer gpa.free(fma_dense_src);
    const fma_dense_lib = mtl.Library.fromSource(device, fma_dense_src, mtl.CompileOptions.mlx()) catch null;
    defer if (fma_dense_lib) |l| l.deinit();
    const fma_dense = if (fma_dense_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
    defer if (fma_dense) |p| p.deinit();
    // THE DECODE'S SHARE AT THE DECODE'S OWN SHAPES. Two more libraries of the same source: TF_DIAG_DECODE = 1
    // keeps the state's extraction and the trellis read and drops the VALUE path (the hash, the byte sum, the index
    // and the LUT load); = 2 drops the whole decode, so the trellis reads go too. The arm shape is the fma/4 arm's
    // own, so the three are apples-to-apples.
    const diag1_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n#define TF_DIAG_DECODE 1\n{s}", .{nax_src});
    defer gpa.free(diag1_src);
    const diag1_lib = mtl.Library.fromSource(device, diag1_src, mtl.CompileOptions.mlx()) catch null;
    defer if (diag1_lib) |l| l.deinit();
    const diag1 = if (diag1_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
    defer if (diag1) |p| p.deinit();
    const diag2_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n#define TF_DIAG_DECODE 2\n{s}", .{nax_src});
    defer gpa.free(diag2_src);
    const diag2_lib = mtl.Library.fromSource(device, diag2_src, mtl.CompileOptions.mlx()) catch null;
    defer if (diag2_lib) |l| l.deinit();
    const diag2 = if (diag2_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
    defer if (diag2) |p| p.deinit();
    // THREE MORE LEVELS OF THE SAME DIAGNOSTIC LADDER, on the same arm shape (the fma/4 arm's own source, the same
    // dispatch, the same cold protocol). = 3 DOUBLES the per-weight FP op: a removal arm would let the compiler
    // delete the decode (the weight would be dead), so the price of the arithmetic is read as the MARGINAL cost of
    // one more FMA -- the record's doubling-arm discipline. = 4 drops the ACTIVATION READ (the x row is not read;
    // the FMA stays because the weight is still needed). = 5 drops the CROSS-WARP REDUCTION (the warps' partials
    // through the threadgroup memory).
    const diag3_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n#define TF_DIAG_DECODE 3\n{s}", .{nax_src});
    defer gpa.free(diag3_src);
    const diag3_lib = mtl.Library.fromSource(device, diag3_src, mtl.CompileOptions.mlx()) catch null;
    defer if (diag3_lib) |l| l.deinit();
    const diag3 = if (diag3_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
    defer if (diag3) |p| p.deinit();
    const diag4_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n#define TF_DIAG_DECODE 4\n{s}", .{nax_src});
    defer gpa.free(diag4_src);
    const diag4_lib = mtl.Library.fromSource(device, diag4_src, mtl.CompileOptions.mlx()) catch null;
    defer if (diag4_lib) |l| l.deinit();
    const diag4 = if (diag4_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
    defer if (diag4) |p| p.deinit();
    const diag5_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n#define TF_DIAG_DECODE 5\n{s}", .{nax_src});
    defer gpa.free(diag5_src);
    const diag5_lib = mtl.Library.fromSource(device, diag5_src, mtl.CompileOptions.mlx()) catch null;
    defer if (diag5_lib) |l| l.deinit();
    const diag5 = if (diag5_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
    defer if (diag5) |p| p.deinit();
    // THE PREFETCH DEPTH OF THE REGISTER-ONLY BODY'S K-WALK. The shipped body keeps ONE block of stream words in
    // flight, so a block's DRAM latency is exposed once a block unless the block's own work (eight states) hides it
    // -- and it cannot. TF_FMA_PF blocks in flight hides it behind PF blocks of work. The values, their order and
    // the arithmetic are untouched, so every arm is BIT-IDENTICAL to the shipped body (the bench's hash says so).
    const PfArm = struct { name: []const u8, pf: u32, pipe: ?mtl.Pipeline, lib: ?mtl.Library };
    var pf_arms: [3]PfArm = undefined;
    var npf: usize = 0;
    defer for (pf_arms[0..npf]) |arm| {
        if (arm.pipe) |p| p.deinit();
        if (arm.lib) |l| l.deinit();
    };
    {
        for ([_]u32{ 2, 4, 8 }) |pf| {
            const psrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n#define TF_FMA_PF {d}\n{s}", .{ pf, nax_src });
            defer gpa.free(psrc);
            const plib = mtl.Library.fromSource(device, psrc, mtl.CompileOptions.mlx()) catch null;
            const ppipe = if (plib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = switch (pf) {
                2 => "fma/4 pf2",
                4 => "fma/4 pf4",
                else => "fma/4 pf8",
            };
            pf_arms[npf] = .{ .name = nm, .pf = pf, .pipe = ppipe, .lib = plib };
            npf += 1;
        }
    }
    // THE ENGINE'S OWN CONFIGURATION AND THE COMPARATOR'S READ. The engine compiles the register-only body with
    // SIXTEEN warps a threadgroup (the bench's fma arm is at four), and the body reads its stream words as TWO
    // 2-byte loads a word, where the prefill tiles (and the comparator) read one 4-byte load a word. Both are the
    // same bytes, the same values and the same order, so every arm here is BIT-IDENTICAL to the shipped body.
    const WideArm = struct { name: []const u8, warps: u32, pipe: ?mtl.Pipeline, lib: ?mtl.Library };
    var wide_arms: [8]WideArm = undefined;
    var nwide: usize = 0;
    defer for (wide_arms[0..nwide]) |arm| {
        if (arm.pipe) |p| p.deinit();
        if (arm.lib) |l| l.deinit();
    };
    {
        const specs = [_]struct { w: u32, wide: bool, nm: []const u8 }{
            .{ .w = 16, .wide = false, .nm = "fma/16" },
            .{ .w = 16, .wide = true, .nm = "fma/16 ww" },
            .{ .w = 4, .wide = true, .nm = "fma/4 ww" },
            .{ .w = 8, .wide = true, .nm = "fma/8 ww" },
            .{ .w = 16, .wide = false, .nm = "fma/16 arith" },
            .{ .w = 4, .wide = false, .nm = "fma/4 arith" },
        };
        for (specs) |sp| {
            const wsrc = if (sp.wide)
                try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_WIDE_WORD 1\n{s}", .{ sp.w, nax_src })
            else if (std.mem.endsWith(u8, sp.nm, "arith"))
                try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_VALUE_ARITH 1\n{s}", .{ sp.w, nax_src })
            else
                try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n{s}", .{ sp.w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            const wpipe = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            wide_arms[nwide] = .{ .name = sp.nm, .warps = sp.w, .pipe = wpipe, .lib = wlib };
            nwide += 1;
        }
    }
    // The engine's OWN dense kernel (tf_exl3_matmul_decode_fma_dense): the same body, but the trellis is read from
    // the buffer and the strides, the width and the codebook from the args (a dense matrix has no expert
    // descriptor).
    const fma_dense_kernel = if (fma_dense_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_dense", false) catch null) else null;
    defer if (fma_dense_kernel) |p| p.deinit();
    const DenseWarp = struct { warps: u32, pipe: ?mtl.Pipeline, lib: ?mtl.Library };
    var dense_warps: [4]DenseWarp = undefined;
    var ndense: usize = 0;
    defer for (dense_warps[0..ndense]) |dw| {
        if (dw.pipe) |p| p.deinit();
        if (dw.lib) |l| l.deinit();
    };
    {
        for ([_]u32{ 4, 8, 16, 32 }) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_dense", false) catch null) else null;
            dense_warps[ndense] = .{ .warps = w, .pipe = p, .lib = wlib };
            ndense += 1;
        }
    }
    // THE LANE PACKING (the comparator's structure) AT THE DENSE ONE-ROW SHAPES. The body and the value path are
    // the shipped dense kernel's; only TF_FMA_WARPS and the load form move, so every arm is the same arithmetic in
    // a different lane packing. `w1` is the comparator's: one simdgroup a sixteen-column tile, its thirty-two lanes
    // walking the whole k, no cross-warp reduction. The hash column prices whether an arm is bit-identical to the
    // w16 control (a changed reduction order may differ; a changed load form may not).
    const LaneArm = struct { name: []const u8, warps: u32, pipe: ?mtl.Pipeline, lib: ?mtl.Library };
    var lane_arms: [8]LaneArm = undefined;
    var nlane: usize = 0;
    defer for (lane_arms[0..nlane]) |arm| {
        if (arm.pipe) |p| p.deinit();
        if (arm.lib) |l| l.deinit();
    };
    {
        const specs = [_]struct { nm: []const u8, defs: []const u8, w: u32 }{
            .{ .nm = "w16 engine", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n", .w = 16 },
            .{ .nm = "w8", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 8\n", .w = 8 },
            .{ .nm = "w4", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n", .w = 4 },
            .{ .nm = "w2", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 2\n", .w = 2 },
            .{ .nm = "w1 cmp", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 1\n", .w = 1 },
            .{ .nm = "w1 cmp pf4", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 1\n#define TF_FMA_PF 4\n", .w = 1 },
            .{ .nm = "w1 cmp pf8", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 1\n#define TF_FMA_PF 8\n", .w = 1 },
            .{ .nm = "w1 cmp ww", .defs = "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 1\n#define TF_FMA_WIDE_WORD 1\n", .w = 1 },
        };
        for (specs) |sp| {
            const wsrc = try std.fmt.allocPrint(gpa, "{s}{s}", .{ sp.defs, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_dense", false) catch null) else null;
            lane_arms[nlane] = .{ .name = sp.nm, .warps = sp.w, .pipe = p, .lib = wlib };
            nlane += 1;
        }
    }
    // THE SIXTEEN-STATE LANE PACKING (TF_FMA_S16). A lane owns sixteen consecutive states (two columns by eight
    // k-rows) so one 16-k block's four stream words decode them all; lanes 0..15 walk one block and 16..31 the next
    // of the same sixteen-column tile, so the dispatch geometry is the shipped body's. It is NOT bit-identical (the
    // reduction order changes), so a landing owes the FP64 reference.
    const S16Arm = struct { name: []const u8, warps: u32, pipe: ?mtl.Pipeline, lib: ?mtl.Library };
    var s16_arms: [6]S16Arm = undefined;
    var ns16: usize = 0;
    defer for (s16_arms[0..ns16]) |arm| {
        if (arm.pipe) |p| p.deinit();
        if (arm.lib) |l| l.deinit();
    };
    {
        const specs = [_]struct { nm: []const u8, w: u32, wide: bool }{
            .{ .nm = "s16 w1", .w = 1, .wide = false },
            .{ .nm = "s16 w2", .w = 2, .wide = false },
            .{ .nm = "s16 w4", .w = 4, .wide = false },
            .{ .nm = "s16 w8", .w = 8, .wide = false },
            .{ .nm = "s16 w16", .w = 16, .wide = false },
            .{ .nm = "s16 w16 ww", .w = 16, .wide = true },
        };
        for (specs) |sp| {
            const wsrc = if (sp.wide)
                try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_S16 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_WIDE_WORD 1\n{s}", .{ sp.w, nax_src })
            else
                try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_S16 1\n#define TF_FMA_WARPS {d}\n{s}", .{ sp.w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch |e| blk: {
                std.debug.print("  s16: library failed: {s}\n", .{@errorName(e)});
                break :blk null;
            };
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_s16", false) catch |e| blk: {
                std.debug.print("  s16: pipeline failed: {s}\n", .{@errorName(e)});
                break :blk null;
            }) else null;
            s16_arms[ns16] = .{ .name = sp.nm, .warps = sp.w, .pipe = p, .lib = wlib };
            ns16 += 1;
        }
    }
    const lane_only = std.c.getenv("TF_BENCH_LANE") != null;
    const routes = try device.buffer(4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer routes.deinit();
    routes.slice(u32, 1)[0] = 0;
    const descs = try device.buffer(@sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer descs.deinit();

    // The one-row fixture's visibility. A large shared buffer the CPU has just written can read as zeros on the
    // GPU's first touch (The finding: the CPU's dirty lines are not yet in memory), so every copy is verified from
    // the GPU and refilled until it lands. Without this the arms' value-level checks are made against a buffer the
    // kernel never saw -- and every arm then reads the same zeros, so every pair of arms agrees for the wrong
    // reason.
    const words_src = try std.mem.concat(gpa, u8, &.{
        \\kernel void tf_exl3_diag_words(const device ushort* t [[buffer(0)]], device uint* out [[buffer(1)]],
        \\                             uint gid [[thread_position_in_grid]]) {
        \\  out[gid] = (uint)t[2 * gid] | ((uint)t[2 * gid + 1] << 16);
        \\}
    });
    defer gpa.free(words_src);
    const words_lib = try mtl.Library.fromSource(device, words_src, mtl.CompileOptions.mlx());
    defer words_lib.deinit();
    const words_pipe = try mtl.Pipeline.init(device, words_lib, "tf_exl3_diag_words", false);
    defer words_pipe.deinit();
    const fixture_scratch = try device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer fixture_scratch.deinit();
    const split_sweep = [_]u32{ 1, 2, 4, 8, 16, 32 };
    std.debug.print("one-row dense decode bench: shipped 16-col table body against the 128-col wide arm\n", .{});
    std.debug.print("  {s:>12} {s:>7} {s:>5} {s:>9} {s:>9} {s:>8} {s:>9}\n", .{ "N x K", "body", "splits", "ms", "weights/s", "GB/s", "hash" });
    const table = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer table.deinit();
    // The codebook-2 region holds the format's own mul1 values, so the LUT arm (which builds the same values
    // arithmetically) and the table arm are bit-identical by construction and checkable by hash.
    for (table.slice(u16, 3 * 65536), 0..) |*v, i| {
        if (i >= 2 * 65536 and i < 3 * 65536) {
            const state: u16 = @intCast(i - 2 * 65536);
            v.* = @bitCast(@as(f16, @floatCast(exl3TableValue(state))));
        } else v.* = 0x3c00 | @as(u16, @intCast(i & 0xff));
    }
    for (one_row_shapes) |s| {
        if (lane_only and !lanePackShape(s)) continue;
        const tile_bytes = s.halves * 8 * 2;
        const trellis_bytes = (s.k / 16) * (s.n / 16) * tile_bytes;
        // Eight copies, rotated per dispatch: each sample streams a buffer the SLC has not just held, which is
        // what a token does (every layer's weights are read once).
        var trs: [8]mtl.Buffer = undefined;
        var made: usize = 0;
        while (made < trs.len) : (made += 1) {
            trs[made] = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            const seed = 0xD1CEB00C ^ @as(u64, s.n) ^ @as(u64, s.k) ^ @as(u64, made);
            if (!fillTrellisVisible(queue, words_pipe, trs[made], @intCast(trellis_bytes), seed, fixture_scratch))
                std.debug.print("  one-row: fixture copy {d} of {d}x{d} never became visible\n", .{ made, s.n, s.k });
        }
        defer for (trs) |b| b.deinit();
        const x = try device.buffer(s.m * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        @memset(x.slice(u16, s.m * s.k), 0x3c00);
        const sums = try device.buffer(s.m * s.n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sums.deinit();
        const partials = try device.buffer(32 * s.m * s.n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer partials.deinit();
        const macs = @as(f64, @floatFromInt(s.m)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k));
        const gbs = @as(f64, @floatFromInt(trellis_bytes)) / 1e9;
        var base_hash: ?u64 = null;
        for (0..2) |pass| {
            for (split_sweep) |splits| {
                const args = Exl3Args{
                    .k = @intCast(s.k),
                    .n = @intCast(s.n),
                    .rows = @intCast(s.m),
                    .halves = s.halves,
                    .codebook = 2,
                    .su_signs = 0,
                    .sv_signs = 0,
                    .has_bias = 0,
                    .output_f32 = 1,
                    .input_ids = 0,
                    .output_ids = 0,
                    .routed = 0,
                    .expert_id = 0,
                    .repeat = 1,
                    .indexed = 0,
                    .experts = 0,
                    .splits = splits,
                    .x_stride = @intCast(s.k),
                    .y_stride = @intCast(s.n),
                    .tile_k_stride = (s.n / 16) * tile_bytes,
                    .tile_n_stride = tile_bytes,
                };
                const n_out = s.m * s.n;
                @memset(sums.slice(u32, n_out), 0);
                @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                _ = timeRow(queue, shipped, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 16, .height = splits, .threads = 32 }, n_out, 1, 1) catch 0;
                _ = timeRow(queue, wide, reduce, args, x, trs[1..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 1, 1) catch 0;
                const t_ship = timeRow(queue, shipped, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 16, .height = splits, .threads = 32 }, n_out, 7, 1) catch 0;
                const h_ship = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                @memset(sums.slice(u32, n_out), 0);
                @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                const t_wide = timeRow(queue, wide, reduce, args, x, trs[4..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                const h_wide = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                if (base_hash == null) base_hash = h_wide;
                @memset(sums.slice(u32, n_out), 0);
                @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                const t_wide2 = timeRow(queue, wide2, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                const h_wide2 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                // The separation arms. The LUT arm reads the codebook from a 1024-entry threadgroup table (the
                // dynamic selection left in), the static arm resolves the state's word pair and bit offset at
                // compile time (the table left in), and the static+LUT arm has both.
                var t_wlut: f64 = 0;
                var h_wlut: u64 = 0;
                var t_wstat: f64 = 0;
                var h_wstat: u64 = 0;
                var t_wstatlut: f64 = 0;
                var h_wstatlut: u64 = 0;
                var t_pf2: f64 = 0;
                var h_pf2: u64 = 0;
                var t_pf4: f64 = 0;
                var h_pf4: u64 = 0;
                if (pass == 1) {
                    if (wide_lut) |wl| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_wlut = timeRow(queue, wl, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_wlut = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    if (wide_static) |ws| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_wstat = timeRow(queue, ws, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_wstat = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    if (wide_static_lut) |wsl| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_wstatlut = timeRow(queue, wsl, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_wstatlut = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    if (wide_static_lut_pf2) |w2| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_pf2 = timeRow(queue, w2, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_pf2 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    if (wide_static_lut_pf4) |w4| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_pf4 = timeRow(queue, w4, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_pf4 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                }
                var t_direct: f64 = 0;
                var t_direct128: f64 = 0;
                var t_direct64: f64 = 0;
                var h_direct64: u64 = 0;
                var t_direct32: f64 = 0;
                var t_directv: f64 = 0;
                var t_directv128: f64 = 0;
                var t_directv64: f64 = 0;
                var t_directr: f64 = 0;
                var t_directnw: f64 = 0;
                var h_directnw: u64 = 0;
                var h_directr: u64 = 0;
                var h_direct: u64 = 0;
                var h_directv: u64 = 0;
                if (pass == 1) {
                    @memset(sums.slice(u32, n_out), 0);
                    @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                    t_direct = timeRow(queue, direct, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 256, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                    h_direct = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    @memset(sums.slice(u32, n_out), 0);
                    @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                    t_direct128 = timeRow(queue, direct, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 128, .height = splits, .threads = 128 }, n_out, 7, 1) catch 0;
                    if (s.n % 64 == 0) {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_direct64 = timeRow(queue, direct, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 64, .height = splits, .threads = 64 }, n_out, 7, 1) catch 0;
                        h_direct64 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    if (s.n % 32 == 0) {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_direct32 = timeRow(queue, direct, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 32, .height = splits, .threads = 32 }, n_out, 7, 1) catch 0;
                    }
                    // The register-named body at the engine's geometry.
                    if (directr) |dr| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_directr = timeRow(queue, dr, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 256, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_directr = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    // The nw body at the engine's geometry.
                    if (directnw) |dnw| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_directnw = timeRow(queue, dnw, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 256, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_directnw = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    }
                    // The wider-load body at the same three geometries.
                    if (directv) |dv| {
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_directv = timeRow(queue, dv, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 256, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                        h_directv = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                        @memset(sums.slice(u32, n_out), 0);
                        @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                        t_directv128 = timeRow(queue, dv, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 128, .height = splits, .threads = 128 }, n_out, 7, 1) catch 0;
                        if (s.n % 64 == 0) {
                            @memset(sums.slice(u32, n_out), 0);
                            @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                            t_directv64 = timeRow(queue, dv, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 64, .height = splits, .threads = 64 }, n_out, 7, 1) catch 0;
                        }
                    }
                }
                // The register-only body on this dense shape, against the same cold protocol (the eight trellis
                // copies rotate a trial at a time).
                var t_fma: f64 = 0;
                var t_regfloor: f64 = 0;
                var h_fma: u64 = 0;
                var t_diag1: f64 = 0;
                var t_diag2: f64 = 0;
                var t_diag3: f64 = 0;
                var t_diag4: f64 = 0;
                var t_diag5: f64 = 0;
                var t_pf: [9]f64 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var h_pf: [9]u64 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var t_ww: [8]f64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
                var h_ww: [8]u64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
                if (pass == 1 and fma_dense != null) {
                    var fa = args;
                    fa.indexed = 1;
                    fa.experts = 1;
                    fa.routed = 1;
                    fa.splits = 1;
                    @memset(sums.slice(u32, n_out), 0);
                    t_fma = timeRowDesc(queue, fma_dense.?, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    h_fma = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    if (regfloor) |rf| {
                        @memset(sums.slice(u32, n_out), 0);
                        t_regfloor = timeRowDesc(queue, rf, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = (s.n + 127) / 128, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    }
                    // The two diagnostics, on the same args and the same dispatch as the fma arm.
                    if (diag1) |d1| {
                        @memset(sums.slice(u32, n_out), 0);
                        t_diag1 = timeRowDesc(queue, d1, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    }
                    if (diag2) |d2| {
                        @memset(sums.slice(u32, n_out), 0);
                        t_diag2 = timeRowDesc(queue, d2, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    }
                    // The three new levels, on the same args and the same dispatch as the fma arm.
                    if (diag3) |d3| {
                        @memset(sums.slice(u32, n_out), 0);
                        t_diag3 = timeRowDesc(queue, d3, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    }
                    if (diag4) |d4| {
                        @memset(sums.slice(u32, n_out), 0);
                        t_diag4 = timeRowDesc(queue, d4, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    }
                    if (diag5) |d5| {
                        @memset(sums.slice(u32, n_out), 0);
                        t_diag5 = timeRowDesc(queue, d5, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                    }
                    // The engine's own configuration (sixteen warps) and the wide-word read, on the same args and
                    // the same dispatch as the fma arm (the thread count follows the warp count).
                    for (wide_arms[0..nwide], 0..) |arm, ai| {
                        if (arm.pipe) |wp| {
                            @memset(sums.slice(u32, n_out), 0);
                            t_ww[ai] = timeRowDesc(queue, wp, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 32 * arm.warps }, 7, tile_bytes, s.halves) catch 0;
                            h_ww[ai] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                        }
                    }
                    // The prefetch-depth arms, on the same args and the same dispatch as the fma arm.
                    for (pf_arms[0..npf]) |arm| {
                        if (arm.pipe) |pp| {
                            @memset(sums.slice(u32, n_out), 0);
                            t_pf[arm.pf] = timeRowDesc(queue, pp, fa, x, trs[0..], sums, partials, table, routes, descs, .{ .width = s.n / 16, .height = 1, .threads = 128 }, 7, tile_bytes, s.halves) catch 0;
                            h_pf[arm.pf] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                        }
                    }
                }
                var t_noload: f64 = 0;
                if (pass == 1) if (noload) |nl| {
                    @memset(sums.slice(u32, n_out), 0);
                    @memset(partials.slice(u32, @as(usize, splits) * n_out), 0);
                    t_noload = timeRow(queue, nl, reduce, args, x, trs[0..], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                };
                if (pass == 1 and s.m == 1 and mma != null and splits == 8) {
                    const d: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
                    d[0] = .{ .trellis = trs[0].gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (s.n / 16) * tile_bytes, .tile_n_stride = tile_bytes, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
                    var ia = args;
                    ia.indexed = 1;
                    ia.experts = 1;
                    @memset(sums.slice(u32, n_out), 0);
                    @memset(partials.slice(u32, 8 * n_out), 0);
                    const t_mma = timeRow(queue, mma.?, reduce, ia, x, trs[0..1], sums, partials, table, routes, descs, .{ .width = s.n / 32, .height = 1, .threads = 64, .depth = 8 }, n_out, 7, 1) catch 0;
                    @memset(sums.slice(u32, n_out), 0);
                    const t_wh = timeRow(queue, wide, reduce, args, x, trs[0..1], sums, partials, table, x, x, .{ .width = (s.n + 127) / 128, .height = splits, .threads = 256 }, n_out, 7, 1) catch 0;
                    std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}\n", .{ s.n, s.k, "mma/8", 8, t_mma, macs / (t_mma * 1e-3), gbs / (t_mma * 1e-3) });
                    std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (wide, one hot buffer)\n", .{ s.n, s.k, "wide/h", 8, t_wh, macs / (t_wh * 1e-3), gbs / (t_wh * 1e-3) });
                }
                if (pass == 1) {
                    std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}\n", .{ s.n, s.k, "shipped", splits, t_ship, macs / (t_ship * 1e-3), gbs / (t_ship * 1e-3), h_ship & 0xffffffff });
                    std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}\n", .{ s.n, s.k, "wide", splits, t_wide, macs / (t_wide * 1e-3), gbs / (t_wide * 1e-3), h_wide & 0xffffffff, if (h_ship == h_wide) "IDENTICAL" else "DIFFERS" });
                    std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}\n", .{ s.n, s.k, "wide2/32k", splits, t_wide2, macs / (t_wide2 * 1e-3), gbs / (t_wide2 * 1e-3), h_wide2 & 0xffffffff, if (h_ship == h_wide2) "IDENTICAL" else "DIFFERS" });
                    // The separation arms, each against the shipped body's hash.
                    if (pass == 1 and t_wlut > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (lut, {d:.1} % of wide)\n", .{ s.n, s.k, "wide/lut", splits, t_wlut, macs / (t_wlut * 1e-3), gbs / (t_wlut * 1e-3), h_wlut & 0xffffffff, if (h_ship == h_wlut) "IDENTICAL" else "DIFFERS", 100.0 * t_wlut / t_wide });
                    if (pass == 1 and t_wstat > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (static, {d:.1} % of wide)\n", .{ s.n, s.k, "wide/st", splits, t_wstat, macs / (t_wstat * 1e-3), gbs / (t_wstat * 1e-3), h_wstat & 0xffffffff, if (h_ship == h_wstat) "IDENTICAL" else "DIFFERS", 100.0 * t_wstat / t_wide });
                    if (pass == 1 and t_wstatlut > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (static+lut, {d:.1} % of wide)\n", .{ s.n, s.k, "wide/st+lu", splits, t_wstatlut, macs / (t_wstatlut * 1e-3), gbs / (t_wstatlut * 1e-3), h_wstatlut & 0xffffffff, if (h_ship == h_wstatlut) "IDENTICAL" else "DIFFERS", 100.0 * t_wstatlut / t_wide });
                    if (pass == 1 and t_pf2 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (static+lut, prefetch 2, {d:.1} % of static+lut)\n", .{ s.n, s.k, "st+lu/pf2", splits, t_pf2, macs / (t_pf2 * 1e-3), gbs / (t_pf2 * 1e-3), h_pf2 & 0xffffffff, if (h_ship == h_pf2) "IDENTICAL" else "DIFFERS", 100.0 * t_pf2 / t_wstatlut });
                    if (pass == 1 and t_pf4 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (static+lut, prefetch 4, {d:.1} % of static+lut)\n", .{ s.n, s.k, "st+lu/pf4", splits, t_pf4, macs / (t_pf4 * 1e-3), gbs / (t_pf4 * 1e-3), h_pf4 & 0xffffffff, if (h_ship == h_pf4) "IDENTICAL" else "DIFFERS", 100.0 * t_pf4 / t_wstatlut });
                    if (pass == 1 and t_direct > 0) {
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}\n", .{ s.n, s.k, "direct", splits, t_direct, macs / (t_direct * 1e-3), gbs / (t_direct * 1e-3), h_direct & 0xffffffff, if (h_ship == h_direct) "IDENTICAL" else "DIFFERS" });
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (direct, 128 threads)\n", .{ s.n, s.k, "direct/128", splits, t_direct128, macs / (t_direct128 * 1e-3), gbs / (t_direct128 * 1e-3) });
                        if (t_directr > 0) std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  (named words, 256 threads) {s}\n", .{ s.n, s.k, "directr", splits, t_directr, macs / (t_directr * 1e-3), gbs / (t_directr * 1e-3), h_directr & 0xffffffff, if (h_ship == h_directr) "IDENTICAL" else "DIFFERS" });
                        if (t_directnw > 0) std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  (nw words, 256 threads) {s}\n", .{ s.n, s.k, "directnw", splits, t_directnw, macs / (t_directnw * 1e-3), gbs / (t_directnw * 1e-3), h_directnw & 0xffffffff, if (h_ship == h_directnw) "IDENTICAL" else "DIFFERS" });
                        if (t_direct64 > 0) std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  (direct, 64 threads) {s}\n", .{ s.n, s.k, "direct/64", splits, t_direct64, macs / (t_direct64 * 1e-3), gbs / (t_direct64 * 1e-3), h_direct64 & 0xffffffff, if (h_ship == h_direct64) "IDENTICAL" else "DIFFERS" });
                        if (t_direct32 > 0) std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (direct, 32 threads)\n", .{ s.n, s.k, "direct/32", splits, t_direct32, macs / (t_direct32 * 1e-3), gbs / (t_direct32 * 1e-3) });
                        if (t_directv > 0) {
                            std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  (wide loads, 256 threads) {s}\n", .{ s.n, s.k, "directv", splits, t_directv, macs / (t_directv * 1e-3), gbs / (t_directv * 1e-3), h_directv & 0xffffffff, if (h_ship == h_directv) "IDENTICAL" else "DIFFERS" });
                            std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (wide loads, 128 threads)\n", .{ s.n, s.k, "directv/128", splits, t_directv128, macs / (t_directv128 * 1e-3), gbs / (t_directv128 * 1e-3) });
                            if (t_directv64 > 0) std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (wide loads, 64 threads)\n", .{ s.n, s.k, "directv/64", splits, t_directv64, macs / (t_directv64 * 1e-3), gbs / (t_directv64 * 1e-3) });
                        }
                    }
                    if (pass == 1 and t_fma > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (register-only, {d:.1} % of static+lut)\n", .{ s.n, s.k, "fma/4", 1, t_fma, macs / (t_fma * 1e-3), gbs / (t_fma * 1e-3), h_fma & 0xffffffff, if (h_ship == h_fma) "IDENTICAL" else "DIFFERS", 100.0 * t_fma / t_wstatlut });
                    if (pass == 1 and t_regfloor > 0 and t_fma > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (register-direct MMA floor: B fragment constant, no B staging/decode, {d:.1} % of the fma/4 body)\n", .{ s.n, s.k, "regfloor", 1, t_regfloor, macs / (t_regfloor * 1e-3), gbs / (t_regfloor * 1e-3), 100.0 * t_regfloor / t_fma });
                    if (pass == 1 and t_diag1 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (VALUE path gone: the value = the state's low byte)\n", .{ s.n, s.k, "fma/4 d1", 1, t_diag1, macs / (t_diag1 * 1e-3), gbs / (t_diag1 * 1e-3) });
                    if (pass == 1 and t_diag2 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (the WHOLE decode gone: the weight = 1)\n", .{ s.n, s.k, "fma/4 d2", 1, t_diag2, macs / (t_diag2 * 1e-3), gbs / (t_diag2 * 1e-3) });
                    if (pass == 1 and t_diag3 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (the per-weight FP op DOUBLED: the arithmetic's marginal)\n", .{ s.n, s.k, "fma/4 d3", 1, t_diag3, macs / (t_diag3 * 1e-3), gbs / (t_diag3 * 1e-3) });
                    if (pass == 1 and t_diag4 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (the ACTIVATION READ gone: the x row is not read)\n", .{ s.n, s.k, "fma/4 d4", 1, t_diag4, macs / (t_diag4 * 1e-3), gbs / (t_diag4 * 1e-3) });
                    if (pass == 1 and t_diag5 > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (the CROSS-WARP REDUCTION gone: the warps' partials)\n", .{ s.n, s.k, "fma/4 d5", 1, t_diag5, macs / (t_diag5 * 1e-3), gbs / (t_diag5 * 1e-3) });
                    // The engine's own warp count and the comparator's 4-byte stream word.
                    for (wide_arms[0..nwide], 0..) |arm, ai| {
                        const t = t_ww[ai];
                        if (pass == 1 and t > 0)
                            std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  ({d:.1} % of the fma/4 arm)\n", .{ s.n, s.k, arm.name, 1, t, macs / (t * 1e-3), gbs / (t * 1e-3), h_ww[ai] & 0xffffffff, if (h_fma == h_ww[ai]) "IDENTICAL" else "DIFFERS", 100.0 * t / t_fma });
                    }
                    // The prefetch depth, on the same dispatch. The hash says whether the arm is bit-identical to
                    // the shipped body (it must be: the ring only moves the loads in time).
                    for (pf_arms[0..npf]) |arm| {
                        const t = t_pf[arm.pf];
                        if (pass == 1 and t > 0)
                            std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  (prefetch {d} blocks in flight, {d:.1} % of the shipped body)\n", .{ s.n, s.k, arm.name, 1, t, macs / (t * 1e-3), gbs / (t * 1e-3), h_pf[arm.pf] & 0xffffffff, if (h_fma == h_pf[arm.pf]) "IDENTICAL" else "DIFFERS", arm.pf, 100.0 * t / t_fma });
                    }
                    if (pass == 1 and t_noload > 0)
                        std.debug.print("  {d:>7}x{d:>5} {s:>7} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1}  (decode removed, {d:.0} % of wide)\n", .{ s.n, s.k, "noload", splits, t_noload, macs / (t_noload * 1e-3), gbs / (t_noload * 1e-3), 100.0 * (t_wide - t_noload) / t_wide });
                }
                // The changed class is the k-sum of the dense one-row projection: the shipped body folds every k
                // term into one fp32 accumulator in ascending k order (and pays a split-K partials round trip), the
                // register-only body takes one accumulator a state and closes them in a tree.
                if (std.c.getenv("TF_BENCH_DENSE_FP64") != null and pass == 1 and splits == 8 and s.m == 1 and
                    wide_static_lut != null and fma_dense_kernel != null)
                {
                    const rn_out = s.n;
                    const host_a = try gpa.alloc(f32, rn_out);
                    defer gpa.free(host_a);
                    const host_b = try gpa.alloc(f32, rn_out);
                    defer gpa.free(host_b);
                    const host_d = try gpa.alloc(f32, rn_out);
                    defer gpa.free(host_d);
                    const host_ref = try gpa.alloc(f64, rn_out);
                    defer gpa.free(host_ref);
                    // A varied activation row: the shipped fixture's all-ones row would make every product equal
                    // and hide the cancellation the two orders differ in.
                    const xr = try device.buffer(s.m * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer xr.deinit();
                    for (xr.slice(u16, s.m * s.k), 0..) |*v, i| {
                        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 32.0;
                        v.* = @bitCast(@as(f16, @floatCast(f)));
                    }
                    // The engine's own split count for this shape (16 narrow, 8 wide n).
                    var sargs = args;
                    sargs.splits = if (s.n >= 16384) @as(u32, 8) else @as(u32, 16);
                    // The shipped dense body: the 128-column threadgroup tile with the static state selection and
                    // the codebook LUT. One dispatch over trs[0], the copy the truth is taken over.
                    @memset(sums.slice(u32, rn_out), 0);
                    @memset(partials.slice(u32, @as(usize, sargs.splits) * rn_out), 0);
                    _ = try timeRow(queue, wide_static_lut.?, reduce, sargs, xr, trs[0..], sums, partials, table, xr, xr, .{ .width = (s.n + 127) / 128, .height = sargs.splits, .threads = 256 }, rn_out, 1, 1);
                    @memcpy(host_a, sums.slice(f32, rn_out));
                    // The engine's other dense body: the direct-register one-row body (the head's shape).
                    var direct_ok = false;
                    if (s.n >= 4096 and s.n % 256 == 0 and s.halves <= 14) {
                        @memset(sums.slice(u32, rn_out), 0);
                        @memset(partials.slice(u32, @as(usize, sargs.splits) * rn_out), 0);
                        _ = try timeRow(queue, direct, reduce, sargs, xr, trs[0..], sums, partials, table, xr, xr, .{ .width = s.n / 256, .height = sargs.splits, .threads = 256 }, rn_out, 1, 1);
                        @memcpy(host_d, sums.slice(f32, rn_out));
                        direct_ok = true;
                    }
                    // The changed path: the engine's own dense register-only kernel. One k walk, one term at a
                    // time, no split-K and no partials round trip.
                    var fargs = args;
                    fargs.splits = 1;
                    @memset(sums.slice(u32, rn_out), 0);
                    _ = try timeRow(queue, fma_dense_kernel.?, reduce, fargs, xr, trs[0..], sums, partials, table, xr, xr, .{ .width = s.n / 16, .height = 1, .threads = 128 }, rn_out, 1, 1);
                    @memcpy(host_b, sums.slice(f32, rn_out));
                    // The FP64 truth: the same trellis bytes, the same products, summed in fp64.
                    exl3F64Row(trs[0].slice(u16, trellis_bytes / 2), xr.slice(u16, s.k), s.n, s.k, s.halves, 0, host_ref);
                    var sq_ref: f64 = 0;
                    var max_ref: f64 = 0;
                    for (host_ref) |v| {
                        sq_ref += v * v;
                        max_ref = @max(max_ref, @abs(v));
                    }
                    const Stat = struct { worst: f64 = 0, sq: f64 = 0, maxv: f64 = 0 };
                    var sa = Stat{};
                    var sb = Stat{};
                    var sd = Stat{};
                    var differing: usize = 0;
                    for (host_ref, 0..) |ref, j| {
                        const va = @as(f64, host_a[j]);
                        const vb = @as(f64, host_b[j]);
                        const da = @abs(va - ref);
                        const db = @abs(vb - ref);
                        sa.worst = @max(sa.worst, da);
                        sa.sq += da * da;
                        sa.maxv = @max(sa.maxv, @abs(va));
                        sb.worst = @max(sb.worst, db);
                        sb.sq += db * db;
                        sb.maxv = @max(sb.maxv, @abs(vb));
                        if (@as(u32, @bitCast(@as(f32, @floatCast(host_a[j])))) !=
                            @as(u32, @bitCast(@as(f32, @floatCast(host_b[j]))))) differing += 1;
                        if (direct_ok) {
                            const vd = @as(f64, host_d[j]);
                            const dd = @abs(vd - ref);
                            sd.worst = @max(sd.worst, dd);
                            sd.sq += dd * dd;
                            sd.maxv = @max(sd.maxv, @abs(vd));
                        }
                    }
                    const nrms_a = @sqrt(sa.sq / @max(sq_ref, 1e-30));
                    const nrms_b = @sqrt(sb.sq / @max(sq_ref, 1e-30));
                    const nrms_d = if (direct_ok) @sqrt(sd.sq / @max(sq_ref, 1e-30)) else 0;
                    // The engine dispatches the 128-column wide body for n <= 32768 and the direct body above it.
                    const shipped_is_wide = s.n <= 32768;
                    const nrms_shipped = if (shipped_is_wide) nrms_a else nrms_d;
                    std.debug.print("  {d:>7}x{d:>5} h{d} DENSE-FP64 shipped({s}) worst {e:.3} nrms {e:.3} | register-only worst {e:.3} nrms {e:.3} | bitdiff {d}/{d} -> {s}\n", .{
                        s.n,      s.k,    s.halves,  if (shipped_is_wide) "wide" else "direct", sa.worst,                                            nrms_a,
                        sb.worst, nrms_b, differing, rn_out,                                    if (nrms_b <= nrms_shipped) "NO-WORSE" else "WORSE",
                    });
                    std.debug.print("  {d:>7}x{d:>5} h{d} DENSE-FP64 detail |max| wide {d:.2} register-only {d:.2} direct {d:.2} ref {d:.2} | direct nrms {e:.3}\n", .{
                        s.n, s.k, s.halves, sa.maxv, sb.maxv, if (direct_ok) sd.maxv else 0, max_ref, nrms_d,
                    });
                    for (dense_warps[0..ndense]) |dw| {
                        if (dw.pipe) |dp| {
                            @memset(sums.slice(u32, rn_out), 0);
                            _ = try timeRow(queue, dp, reduce, fargs, xr, trs[0..], sums, partials, table, xr, xr, .{ .width = s.n / 16, .height = 1, .threads = 32 * dw.warps }, rn_out, 1, 1);
                            @memcpy(host_b, sums.slice(f32, rn_out));
                            var wb: f64 = 0;
                            var sbw: f64 = 0;
                            for (host_ref, 0..) |ref, j| {
                                const vb = @as(f64, host_b[j]);
                                const db = @abs(vb - ref);
                                wb = @max(wb, db);
                                sbw += db * db;
                            }
                            const nrms_dw = @sqrt(sbw / @max(sq_ref, 1e-30));
                            std.debug.print("  {d:>7}x{d:>5} h{d} DENSE-FP64 w{d}: worst {e:.3} nrms {e:.3} -> {s}\n", .{
                                s.n,                                                                                              s.k, s.halves, dw.warps, wb, nrms_dw,
                                if (nrms_dw <= nrms_shipped) "NO-WORSE than the shipped path" else "worse than the shipped path",
                            });
                        }
                    }
                }
            }
            if (pass == 1 and lane_only) {
                const n_out = s.m * s.n;
                const la = Exl3Args{
                    .k = @intCast(s.k),
                    .n = @intCast(s.n),
                    .rows = @intCast(s.m),
                    .halves = s.halves,
                    .codebook = 2,
                    .su_signs = 0,
                    .sv_signs = 0,
                    .has_bias = 0,
                    .output_f32 = 1,
                    .input_ids = 0,
                    .output_ids = 0,
                    .routed = 0,
                    .expert_id = 0,
                    .repeat = 1,
                    .indexed = 0,
                    .experts = 0,
                    .splits = 1,
                    .x_stride = @intCast(s.k),
                    .y_stride = @intCast(s.n),
                    .tile_k_stride = (s.n / 16) * tile_bytes,
                    .tile_n_stride = tile_bytes,
                };
                std.debug.print("  {d:>7}x{d:>5} h{d} LANE PACKING: one simdgroup a 16-column tile, the whole k walk (the comparator's structure)\n", .{ s.n, s.k, s.halves });
                std.debug.print("  {s:>12} {s:>5} {s:>9} {s:>9} {s:>8} {s:>9}\n", .{ "body", "warps", "ms", "weights/s", "GB/s", "hash" });
                var h_ref: u64 = 0;
                var t_ref: f64 = 0;
                for (lane_arms[0..nlane], 0..) |arm, ai| {
                    const p = arm.pipe orelse continue;
                    @memset(sums.slice(u32, n_out), 0);
                    const t = timeRow(queue, p, null, la, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 16, .height = 1, .threads = 32 * arm.warps }, n_out, 7, 1) catch 0;
                    if (t <= 0) continue;
                    const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    if (ai == 0) {
                        h_ref = h;
                        t_ref = t;
                    }
                    std.debug.print("  {s:>12} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  ({d:.1} % of w16, {d:.1} us a tile)\n", .{
                        arm.name,                                                               arm.warps,                                 t,                                           macs / (t * 1e-3), gbs / (t * 1e-3), h & 0xffffffff,
                        if (ai == 0) "control" else if (h_ref == h) "IDENTICAL" else "DIFFERS", if (ai == 0) 100.0 else 100.0 * t / t_ref, t * 1e3 / @as(f64, @floatFromInt(s.n / 16)),
                    });
                }
                // The sixteen-state body, timed on the same cold fixture and dispatch geometry.
                std.debug.print("  {d:>7}x{d:>5} h{d} SIXTEEN-STATE LANE PACKING: four words for sixteen states, two lanes a column\n", .{ s.n, s.k, s.halves });
                std.debug.print("  {s:>12} {s:>5} {s:>9} {s:>9} {s:>8} {s:>9}\n", .{ "body", "warps", "ms", "weights/s", "GB/s", "hash" });
                for (s16_arms[0..ns16]) |arm| {
                    const p = arm.pipe orelse continue;
                    @memset(sums.slice(u32, n_out), 0);
                    const t = timeRow(queue, p, null, la, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 16, .height = 1, .threads = 32 * arm.warps }, n_out, 7, 1) catch 0;
                    if (t <= 0) continue;
                    const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    std.debug.print("  {s:>12} {d:>5} {d:>9.3} {d:>9.0} {d:>8.1} {x:>8}  {s}  ({d:.1} % of w16, {d:.1} us a tile)\n", .{
                        arm.name,                                   arm.warps,                                 t,                                           macs / (t * 1e-3), gbs / (t * 1e-3), h & 0xffffffff,
                        if (h_ref == h) "IDENTICAL" else "DIFFERS", if (t_ref > 0) 100.0 * t / t_ref else 0.0, t * 1e3 / @as(f64, @floatFromInt(s.n / 16)),
                    });
                }
                // A value-level cross-check of the s16 body against the shipped w16 on this fixture.
                if (lane_arms[0].pipe != null and s16_arms[ns16 - 2].pipe != null) {
                    const hp = try gpa.alloc(f32, n_out);
                    defer gpa.free(hp);
                    const hq = try gpa.alloc(f32, n_out);
                    defer gpa.free(hq);
                    @memset(sums.slice(u32, n_out), 0);
                    _ = timeRow(queue, lane_arms[0].pipe.?, null, la, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 16, .height = 1, .threads = 512 }, n_out, 1, 1) catch 0;
                    @memcpy(hp, sums.slice(f32, n_out));
                    @memset(sums.slice(u32, n_out), 0);
                    _ = timeRow(queue, s16_arms[ns16 - 2].pipe.?, null, la, x, trs[0..], sums, partials, table, x, x, .{ .width = s.n / 16, .height = 1, .threads = 512 }, n_out, 1, 1) catch 0;
                    @memcpy(hq, sums.slice(f32, n_out));
                    var mx: f64 = 0;
                    var cnt: usize = 0;
                    var nz: usize = 0;
                    for (hp, 0..) |v, i| {
                        const d = @abs(@as(f64, v) - @as(f64, hq[i]));
                        if (d > mx) mx = d;
                        if (d > 1e-4) cnt += 1;
                        if (v != 0 or hq[i] != 0) nz += 1;
                    }
                    std.debug.print("  s16 CROSS-CHECK {d}x{d} h{d}: max|w16-s16| {e:.3}, {d}/{d} differ >1e-4, nonzero {d}; w16[0..3] {e:.4} {e:.4} {e:.4} | s16 {e:.4} {e:.4} {e:.4}\n", .{
                        s.n,   s.k,   s.halves, mx,    cnt,   n_out, nz,
                        hp[0], hp[1], hp[2],    hq[0], hq[1], hq[2],
                    });
                }
            }
        }
    }
}

// Its weights come through descriptors[expert], one row an expert, so the dense direct body does not apply to it.
// The pack's expert matrices are three bits (the trellis tensors are [K/16, N/16, 48] int16, so width.halves = 6)
// at the shapes the pack stores: gate/up K = 5120, N = 2304; down K = 2304, N = 5120. Six rows a token.
const Routed = struct { n: u32, k: u32, halves: u32, rows: u32, name: []const u8 };
const routed_shapes = [_]Routed{
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 6, .name = "gate/up" },
    .{ .n = 5120, .k = 2304, .halves = 6, .rows = 6, .name = "down" },
    // The routed calls the engine's own dispatch names (the serve log's "routed decode body" lines): the top-12
    // experts' gate/up, and the shared expert's own projection at the dense width.
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 12, .name = "gate/up 12" },
    .{ .n = 1024, .k = 4096, .halves = 12, .rows = 8, .name = "shared" },
    .{ .n = 2304, .k = 5120, .halves = 12, .rows = 12, .name = "gate/up h12" },
    // The drafter's own routed shapes (halves = 8, two rows -- the depth-1 window's packed pair).
    .{ .n = 2304, .k = 5120, .halves = 8, .rows = 2, .name = "drafter gate/up h8" },
    .{ .n = 5120, .k = 2304, .halves = 8, .rows = 2, .name = "drafter down h8" },
    // THE VERIFY'S OWN DISPATCH SHAPE. The routed dispatch's height IS the slot count, so the four-row verify's
    // dispatch is 24 slots (four rows x six picks) -- the shape the drafted path's routed stage actually runs.
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 24, .name = "gate/up 24" },
    // THE VERIFY'S OWN DISPATCH SHAPES, CORRECTED. The engine's fused gate/up call's count is `rows * 2 * picks +
    // (merge_up ? 2 * rows: 0)` (deepseek_v41/model.zig:2549-2550) -- TWO slots a pick (the gate and the up) -- so
    // the four-row verify's gate/up dispatch is 4*2*6 + 2*4 = 56 slots, and the down's is `picks*r + r` = 24 + 4 =
    // 28. The prior session's "24" shape counted only the picks, not the gate/up pair: it is the DOWN's routed
    // count, and it is NOT the verify's gate/up.
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 56, .name = "gate/up 56" },
    .{ .n = 5120, .k = 2304, .halves = 6, .rows = 28, .name = "down 28" },
    // THE STANDING NOVEL BAND-S OWN TWO-ROW DISPATCHES (depth 1: 2 rows x 6 picks -> 28 gate/up slots and 12 down
    // slots). Without these the fma arms never see the shape the drafted path runs.
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 28, .name = "gate/up 28v" },
    .{ .n = 5120, .k = 2304, .halves = 6, .rows = 12, .name = "down 12v" },
};
const routed_copies = 8;

// One routed dispatch, timed: every expert points at a fresh copy of its trellis, so the weights are not
// resident when they come back (the routed stage reads each expert once a layer).
fn timeRouted(queue: mtl.Queue, pipe: mtl.Pipeline, reduce: ?mtl.Pipeline, args: Exl3Args, x: mtl.Buffer, sums: mtl.Buffer, partials: mtl.Buffer, table: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, trs: []const mtl.Buffer, experts: usize, d: RowDispatch, n_out: usize, trials: usize) !f64 {
    var samples: [16]f64 = undefined;
    const nt = @max(1, @min(trials, samples.len));
    const dp: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
    for (0..nt) |t| {
        for (0..experts) |e| {
            const src = trs[(t * experts + e) % trs.len];
            dp[e] = .{ .trellis = src.gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = dp[e].tile_k_stride, .tile_n_stride = dp[e].tile_n_stride, .halves = dp[e].halves, .codebook = dp[e].codebook, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
        }
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trs[0], 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(routes, 0, 4);
        e.setBuffer(descs, 0, 5);
        e.setBuffer(partials, 0, 6);
        e.setBuffer(table, 0, 7);
        e.dispatchGroups(.{ .width = d.width, .height = d.height, .depth = d.depth }, .{ .width = d.threads });
        if (args.splits > 1) {
            e.setPipeline(reduce.?);
            e.setBuffer(partials, 0, 0);
            e.setBuffer(sums, 0, 1);
            const ra = extern struct { splits: u32, n: u32 }{ .splits = args.splits, .n = @intCast(n_out) };
            e.setBytes(std.mem.asBytes(&ra), 2);
            e.dispatchThreads(.{ .width = n_out }, .{ .width = 256 });
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  routed: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// One dispatch of the fair affine tile body (or its scalar reference); descriptors are not rewritten.
fn runAffine(queue: mtl.Queue, pipe: mtl.Pipeline, args: Exl3Args, x: mtl.Buffer, codes: mtl.Buffer, sums: mtl.Buffer, scales: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, w: u32, h: u32, threads: u32) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(codes, 0, 1);
    e.setBuffer(sums, 0, 2);
    e.setBytes(std.mem.asBytes(&args), 3);
    e.setBuffer(routes, 0, 4);
    e.setBuffer(descs, 0, 5);
    e.setBuffer(sums, 0, 6);
    e.setBuffer(scales, 0, 7);
    e.dispatchGroups(.{ .width = w, .height = h, .depth = 1 }, .{ .width = threads });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("  affine tile3: failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
}

// The routed fixture's diagnostic ------------------------------------------------------------------ The routed
// bench's value-level checks are degenerate (The column probe: every column of a row reads the same value, whatever
// the trellis holds), so the fixture, not the bodies, is what needs the answer. This dumps what the GPU sees of the
// descriptor and of the words the routed body reads, beside the host's own reading of the same buffers, so the
// divergence is local to a line. The fixture's visibility. A large shared buffer the CPU has just written can still
// read as zeros on the GPU's first touch -- the CPU's dirty lines are not yet in memory -- so every fixture write
// is verified from the GPU and repeated until it lands. Without this the bench's value-level checks are made
// against a buffer the kernel never saw.
/// The fixture's fill is index-addressed, so the host and the GPU can write the SAME pattern independently (the GPU
/// fill is what makes every page visible to the kernel).
fn fillWord(seed: u64, i: u64) u16 {
    var x: u64 = seed +% i *% 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    x ^= x >> 31;
    return @truncate(x);
}

fn fillTrellis(buf: mtl.Buffer, bytes: u32, seed: u64) void {
    for (buf.slice(u16, bytes / 2), 0..) |*v, i| v.* = fillWord(seed, @intCast(i));
}

fn fillTrellisGpu(queue: mtl.Queue, pipe: mtl.Pipeline, buf: mtl.Buffer, bytes: u32, seed: u64) void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(buf, 0, 0);
    e.setBytes(std.mem.asBytes(&seed), 1);
    e.dispatchThreads(.{ .width = bytes / 2 }, .{ .width = 256 });
    e.end();
    cb.commit();
    cb.wait();
}

/// CPU fill + GPU fill (the same pattern) + a GPU-side read-back at the first, middle and last tile.
fn fillTrellisGpuVisible(queue: mtl.Queue, pfill: mtl.Pipeline, pw: mtl.Pipeline, buf: mtl.Buffer, bytes: u32, seed: u64, scratch: mtl.Buffer) bool {
    var attempts: usize = 0;
    while (attempts < 8) : (attempts += 1) {
        fillTrellis(buf, bytes, seed);
        fillTrellisGpu(queue, pfill, buf, bytes, seed);
        var ok = true;
        for ([_]u32{ 0, bytes / 2, bytes - 96 }) |off| {
            trellisWords(queue, pw, buf, off, scratch);
            const w = scratch.slice(u32, 8);
            const host = hostWordsAt(buf, off);
            for (0..8) |i| {
                if (host[i] != w[i]) ok = false;
            }
        }
        if (ok) return true;
    }
    return false;
}

/// The GPU's reading of the trellis's first eight words at `offset`, against the host's.
fn trellisWords(queue: mtl.Queue, pipe: mtl.Pipeline, buf: mtl.Buffer, offset: u32, scratch: mtl.Buffer) void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(buf, offset, 0);
    e.setBuffer(scratch, 0, 1);
    e.dispatchThreads(.{ .width = 8 }, .{ .width = 1 });
    e.end();
    cb.commit();
    cb.wait();
}

/// The host's first eight stream words at `offset` bytes into the buffer.
fn hostWordsAt(buf: mtl.Buffer, offset: u32) [8]u32 {
    const p: [*]const u16 = @ptrCast(@alignCast(buf.contents()));
    const base = offset / 2;
    var out: [8]u32 = undefined;
    for (0..8) |i| out[i] = @as(u32, p[base + 2 * i]) | (@as(u32, p[base + 2 * i + 1]) << 16);
    return out;
}

/// Fill the trellis and verify from the GPU that the words landed, at the first, the middle and the last tile.
fn fillTrellisVisible(queue: mtl.Queue, pipe: mtl.Pipeline, buf: mtl.Buffer, bytes: u32, seed: u64, scratch: mtl.Buffer) bool {
    var attempts: usize = 0;
    while (attempts < 16) : (attempts += 1) {
        fillTrellis(buf, bytes, seed);
        var ok = true;
        for ([_]u32{ 0, bytes / 2, bytes - 96 }) |off| {
            trellisWords(queue, pipe, buf, off, scratch);
            const w = scratch.slice(u32, 8);
            const host = hostWordsAt(buf, off);
            for (0..8) |i| {
                if (host[i] != w[i]) ok = false;
            }
        }
        if (ok) return true;
    }
    return false;
}

fn exl3MapFor(p: u32, halves: u32) u32 {
    const ring: u32 = 128 * halves;
    const odd: u32 = if (halves & 1 != 0) (if ((p + 1) & 1 != 0) 1 else 0) else 0;
    const end: u32 = ((p + 1) * halves - odd) / 2;
    var start: u32 = end +% ring -% 16;
    if (start >= ring) start -%= ring;
    const wi = start >> 5;
    const off = start & 31;
    const next = if (wi + 1 == 4 * halves) 0 else wi + 1;
    return wi | (off << 6) | (next << 11);
}

fn exl3WordHost(tile: []const u16, i: u32) u32 {
    return @as(u32, tile[2 * i]) | (@as(u32, tile[2 * i + 1]) << 16);
}

fn routeDiag(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    const n: u32 = 2304;
    const k: u32 = 5120;
    const halves: u32 = 6;
    const rows: u32 = 6;
    const tile_bytes: u32 = halves * 8 * 2;
    const trellis_bytes: u32 = (k / 16) * (n / 16) * tile_bytes;
    const nax_src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_src);
    const diag_src = try std.mem.concat(gpa, u8, &.{
        nax_src,
        \\kernel void tf_exl3_diag_desc(const device TfExl3Expert* d [[buffer(0)]], device uint* out [[buffer(1)]],
        \\                              uint gid [[thread_position_in_grid]]) {
        \\  if (gid != 0) return;
        \\  const ulong tp = (ulong)d[0].trellis;
        \\  out[0] = (uint)(tp & 0xffffffffu);
        \\  out[1] = (uint)((tp >> 32) & 0xffffffffu);
        \\  out[2] = (uint)(d[0].tile_k_stride & 0xffffffffu);
        \\  out[3] = (uint)((d[0].tile_k_stride >> 32) & 0xffffffffu);
        \\  out[4] = (uint)(d[0].tile_n_stride & 0xffffffffu);
        \\  out[5] = d[0].halves;
        \\  out[6] = d[0].codebook;
        \\  out[7] = (uint)((d[0].tile_n_stride >> 32) & 0xffffffffu);
        \\}
        \\
        \\kernel void tf_exl3_diag_words(const device ushort* t [[buffer(0)]], device uint* out [[buffer(1)]],
        \\                             uint gid [[thread_position_in_grid]]) {
        \\  out[gid] = (uint)t[2 * gid] | ((uint)t[2 * gid + 1] << 16);
        \\}
        \\
        \\kernel void tf_exl3_diag_states(const device TfExl3Expert* d [[buffer(0)]], device uint* out [[buffer(1)]],
        \\                                uint gid [[thread_position_in_grid]]) {
        \\  const uint lane = gid & 31u;
        \\  const uint blk = (gid >> 5) & 7u;
        \\  const uint nb = (gid >> 8) & 7u;
        \\  const uint halves = d[0].halves;
        \\  const uint words = 4 * halves;
        \\  const uint ring = 128 * halves;
        \\  const uint wi0 = tf_exl3_map_for(lane * 8u, halves) & 63u;
        \\  const uint cur = (wi0 << 5) | ((tf_exl3_map_for(lane * 8u, halves) >> 6) & 31u);
        \\  uint a1 = wi0 + 1u; if (a1 >= words) a1 -= words;
        \\  const device uchar* data = d[0].trellis;
        \\  const ulong ks = d[0].tile_k_stride;
        \\  const ulong ns = d[0].tile_n_stride;
        \\  const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(blk) * ks + ulong(nb) * ns);
        \\  for (uint k = 0; k < 8; ++k) {
        \\    const uint c8 = (cur + k * (halves >> 1)) % ring;
        \\    const uint off = c8 & 31u;
        \\    const uint sh = (32u - off) & 31u;
        \\    const uint rel = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
        \\    const uint lo = rel == 0 ? tf_exl3_word(t0, wi0) : tf_exl3_word(t0, a1);
        \\    const uint hi = rel == 0 ? tf_exl3_word(t0, a1) : 0u;
        \\    const uint carry = (hi >> sh) & (0u - uint(off != 0));
        \\    out[gid * 12u + k] = (uint)ushort(((lo << off) | carry) >> 16);
        \\  }
        \\  out[gid * 12u + 8u] = tf_exl3_word(t0, wi0);
        \\  out[gid * 12u + 9u] = tf_exl3_word(t0, a1);
        \\  out[gid * 12u + 10u] = (uint)(ulong(blk) * ks + ulong(nb) * ns);
        \\  out[gid * 12u + 11u] = (uint)((ulong)data & 0xffffffffu);
        \\}
    });
    defer gpa.free(diag_src);
    const lib = mtl.Library.fromSource(device, diag_src, mtl.CompileOptions.mlx()) catch |err| {
        std.debug.print("route diag: library failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer lib.deinit();
    const pdesc = try mtl.Pipeline.init(device, lib, "tf_exl3_diag_desc", false);
    defer pdesc.deinit();
    const pwords = try mtl.Pipeline.init(device, lib, "tf_exl3_diag_words", false);
    defer pwords.deinit();
    const pstates = try mtl.Pipeline.init(device, lib, "tf_exl3_diag_states", false);
    defer pstates.deinit();
    // The engine's own register-only body, so the diagnostic prices the same body the engine dispatches.
    const fma_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n{s}", .{nax_src});
    defer gpa.free(fma_src);
    const flib = mtl.Library.fromSource(device, fma_src, mtl.CompileOptions.mlx()) catch null;
    defer if (flib) |l| l.deinit();
    const pfma = if (flib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma", false) catch null) else null;
    defer if (pfma) |p| p.deinit();

    // The same fixture the routed bench builds: one trellis per expert, six rows, the descriptor per expert.
    var trs: [6]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (trs[0..made]) |b| b.deinit();
    while (made < trs.len) : (made += 1) {
        trs[made] = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var rng: u64 = 0xD1CEB00C ^ @as(u64, n) ^ @as(u64, k) ^ @as(u64, made);
        for (trs[made].slice(u16, trellis_bytes / 2)) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = @truncate(rng);
        }
    }
    const descs = try device.buffer(rows * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer descs.deinit();
    {
        const dp: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
        for (0..rows) |e| dp[e] = .{ .trellis = trs[e].gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (n / 16) * tile_bytes, .tile_n_stride = tile_bytes, .halves = halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
    }
    const x = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    for (x.slice(u16, rows * k), 0..) |*v, i| v.* = 0x3400 | @as(u16, @intCast(i & 0x3ff));
    const routes = try device.buffer(rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer routes.deinit();
    for (routes.slice(u32, rows), 0..) |*r, i| r.* = @intCast(i);
    const table = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer table.deinit();
    {
        const scale: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
        const bias: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));
        for (0..65536) |state| {
            const xx: u32 = @truncate(@as(u64, state) * 0x83DCD12D);
            const h: f64 = @floatFromInt(1024 + (xx & 255) + ((xx >> 8) & 255) + ((xx >> 16) & 255) + (xx >> 24));
            const v: f16 = @floatCast(h * scale + bias);
            table.slice(u16, 3 * 65536)[2 * 65536 + state] = @bitCast(v);
        }
    }
    const sums = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    const partials = try device.buffer(32 * rows * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer partials.deinit();
    const dout = try device.buffer(1024 * 1024, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer dout.deinit();

    const nstates: u32 = 32 * 8 * 8;
    const args = Exl3Args{
        .k = k,
        .n = n,
        .rows = rows,
        .halves = halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 1,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 1,
        .experts = rows,
        .splits = 1,
        .x_stride = k,
        .y_stride = n,
        .tile_k_stride = (n / 16) * tile_bytes,
        .tile_n_stride = tile_bytes,
    };

    // The routed bench's own column probe, reproduced here: expert 0's trellis zeroed but for its first tile. If
    // this is what makes a row's columns constant, the probe (not the fixture) is the degeneracy.
    {
        const bytes = trs[0].slice(u8, trellis_bytes);
        @memset(bytes, 0);
        for (trs[0].slice(u16, tile_bytes / 2), 0..) |*v, i| v.* = @truncate(@as(u64, i) * 2654435761);
        if (pfma) |p| {
            @memset(sums.slice(u32, rows * n), 0);
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(p);
            e.setBuffer(x, 0, 0);
            e.setBuffer(trs[0], 0, 1);
            e.setBuffer(sums, 0, 2);
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(routes, 0, 4);
            e.setBuffer(descs, 0, 5);
            e.setBuffer(partials, 0, 6);
            e.setBuffer(table, 0, 7);
            e.dispatchGroups(.{ .width = n / 16, .height = rows, .depth = 1 }, .{ .width = 128 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| std.debug.print("route diag: fma(zeroed) dispatch failed: {s}\n", .{msg});
        }
        const col = sums.slice(f32, rows * n);
        std.debug.print("\nroute diag: expert 0's trellis zeroed (the routed bench's column probe, reproduced)\n", .{});
        std.debug.print("  host words after the memset: {x:0>8} {x:0>8} {x:0>8}\n", .{
            exl3WordHost(trs[0].slice(u16, 8), 0), exl3WordHost(trs[0].slice(u16, 8), 1), exl3WordHost(trs[0].slice(u16, 8), 2),
        });
        std.debug.print("  sums row 0: cols 0..5 = {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} | cols 16..19 = {d:.4} {d:.4} {d:.4} {d:.4} | col 100 = {d:.4} | row 1 col 0 = {d:.4}\n", .{
            col[0], col[1], col[2], col[3], col[4], col[5], col[16], col[17], col[18], col[19], col[100], col[n],
        });
        std.debug.print("  sums row 1: cols 0..5 = {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}\n", .{
            col[n + 0], col[n + 1], col[n + 2], col[n + 3], col[n + 4], col[n + 5],
        });
        // Put the random data back for the passes below.
        {
            var rng: u64 = 0xD1CEB00C ^ @as(u64, n) ^ @as(u64, k);
            for (trs[0].slice(u16, trellis_bytes / 2)) |*v| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                v.* = @truncate(rng);
            }
        }
    }

    // Three passes: the first dispatch of the trellis may read stale pages (the bench's own routed probe found
    // exactly that), so the same reads are repeated and the pass number is printed with each.
    for (0..3) |pass| {
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pdesc);
            e.setBuffer(descs, 0, 0);
            e.setBuffer(dout, 0, 1);
            e.dispatchThreads(.{ .width = 1 }, .{ .width = 1 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| std.debug.print("route diag: desc dispatch failed: {s}\n", .{msg});
        }
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pwords);
            e.setBuffer(trs[0], 0, 0);
            e.setBuffer(dout, 0, 1);
            e.dispatchThreads(.{ .width = 8 }, .{ .width = 1 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| std.debug.print("route diag: words dispatch failed: {s}\n", .{msg});
        }
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pstates);
            e.setBuffer(descs, 0, 0);
            e.setBuffer(dout, 0, 1);
            e.dispatchThreads(.{ .width = nstates }, .{ .width = 1 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| std.debug.print("route diag: states dispatch failed: {s}\n", .{msg});
        }
        // The register-only body on the same fixture: the routed bench's column probe, reproduced.
        if (pfma) |p| {
            @memset(sums.slice(u32, rows * n), 0);
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(p);
            e.setBuffer(x, 0, 0);
            e.setBuffer(trs[0], 0, 1);
            e.setBuffer(sums, 0, 2);
            e.setBytes(std.mem.asBytes(&args), 3);
            e.setBuffer(routes, 0, 4);
            e.setBuffer(descs, 0, 5);
            e.setBuffer(partials, 0, 6);
            e.setBuffer(table, 0, 7);
            e.dispatchGroups(.{ .width = n / 16, .height = rows, .depth = 1 }, .{ .width = 128 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| std.debug.print("route diag: fma dispatch failed: {s}\n", .{msg});
        }
        const d0 = dout.slice(u32, 8);
        const w = dout.slice(u32, 8);
        const g = dout.slice(u32, @as(usize, nstates) * 12);
        const col = sums.slice(f32, rows * n);
        std.debug.print("\nroute diag pass {d}\n", .{pass});
        std.debug.print("  descriptor: trellis host {x} gpu {x}{x} | k_stride {d}/{d}{d} | n_stride {d}/{d}{d} | halves {d}/{d} codebook {d}/{d}\n", .{
            trs[0].gpuAddress(), d0[1], d0[0], (n / 16) * tile_bytes, d0[3], d0[2], tile_bytes, d0[7], d0[4], halves, d0[5], @as(u32, 2), d0[6],
        });
        std.debug.print("  first tile words (host vs gpu, direct binding): {x:0>8} {x:0>8} | {x:0>8} {x:0>8} | {x:0>8} {x:0>8}\n", .{
            exl3WordHost(trs[0].slice(u16, 8), 0), w[0], exl3WordHost(trs[0].slice(u16, 8), 1), w[1], exl3WordHost(trs[0].slice(u16, 8), 2), w[2],
        });
        std.debug.print("  states lane 0 blk 0 nb 0 (host vs gpu): {x:0>8} {x:0>8} | {x:0>8} {x:0>8} | {x:0>8} {x:0>8}\n", .{
            g[0], g[0], g[1], g[1], g[2], g[2],
        });
        std.debug.print("  fma sums row 0: cols 0..5 = {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} | cols 16..19 = {d:.4} {d:.4} {d:.4} {d:.4} | col 100 = {d:.4} | row 1 col 0 = {d:.4}\n", .{
            col[0], col[1], col[2], col[3], col[4], col[5], col[16], col[17], col[18], col[19], col[100], col[n],
        });
        // The host's own reading of the same eight states, beside the GPU's.
        const words: u32 = 4 * halves;
        const ring: u32 = 128 * halves;
        const map = exl3MapFor(0, halves);
        const wi0 = map & 63;
        const cur = (wi0 << 5) | ((map >> 6) & 31);
        const a1h: u32 = if (wi0 + 1 >= words) wi0 + 1 - words else wi0 + 1;
        const ht = trs[0].slice(u16, 8);
        std.debug.print("  host states lane 0 blk 0 nb 0: ", .{});
        for (0..8) |kk| {
            const c8 = (cur + @as(u32, @intCast(kk)) * (halves >> 1)) % ring;
            const off = c8 & 31;
            const sh: u5 = @truncate((32 - off) & 31);
            const rel = if ((c8 >> 5) >= wi0) (c8 >> 5) - wi0 else (c8 >> 5) + words - wi0;
            const lo = if (rel == 0) exl3WordHost(ht, wi0) else exl3WordHost(ht, a1h);
            const hi = if (rel == 0) exl3WordHost(ht, a1h) else 0;
            const carry = (hi >> sh) & (if (off != 0) @as(u32, 0xFFFFFFFF) else @as(u32, 0));
            std.debug.print("{x:0>8} ", .{((lo << @as(u5, @truncate(off))) | carry) >> 16});
        }
        std.debug.print("\n", .{});
    }
}

// The fp64 reference for the routed decode's reduction class ------------------------------- The changed class is
// the k-sum: the tensor-unit body adds sixteen terms at a time, the register-only body one at a time. GOAL-UNLOCK's
// band: the changed path's agreement with the fp64 truth is no worse than the shipped path's own.
fn exl3TableValue(state: u16) f64 {
    // The mul1 codebook, exactly as the engine's table holds it: the byte sums collapse into one 16-bit
    // lane, the +1024 bias is added, and the value is the half.
    const x: u32 = @as(u32, state) *% 0x83DCD12D;
    const bs: u32 = (x & 0x00FF00FF) +% ((x >> 8) & 0x00FF00FF);
    const hs: u16 = @truncate((bs *% 0x00010001 +% 0x64000000) >> 16);
    const scale: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
    const bias: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));
    // The kernel reinterprets the lane as a half (the +1024 bias and the byte sums ride in the mantissa).
    const h: f64 = @floatCast(@as(f16, @bitCast(hs)));
    const v: f16 = @floatCast(h * scale + bias);
    return @floatCast(v);
}

fn exl3StateFor(tile: []const u16, p: u32, halves: u32) u16 {
    const map = exl3MapFor(p, halves);
    const wi = map & 63;
    const off: u5 = @truncate((map >> 6) & 31);
    const next = map >> 11;
    const lo = exl3WordHost(tile, wi);
    const hi = exl3WordHost(tile, next);
    const sh: u5 = @truncate((32 - @as(u32, off)) & 31);
    const carry = (hi >> sh) & (if (off != 0) @as(u32, 0xFFFFFFFF) else @as(u32, 0));
    return @truncate(((lo << off) | carry) >> 16);
}

/// One row's sums in fp64, over the trellis the bench writes: the same products the bodies compute.
fn exl3F64Row(trellis: []const u16, x: []const u16, n: u32, k: u32, halves: u32, row: usize, out: []f64) void {
    const tile_bytes = halves * 8 * 2;
    const kstride = (n / 16) * tile_bytes;
    const nstride = tile_bytes;
    const blocks = k / 16;
    for (0..n) |j| out[j] = 0.0;
    for (0..blocks) |kb| {
        for (0..n / 16) |nb| {
            const base = (kb * kstride + nb * nstride) / 2;
            const tile = trellis[base..];
            for (0..256) |p| {
                const l = p / 8;
                const e = p % 8;
                const row_in_tile = 2 * (l % 4) + (e & 1) + 8 * ((e >> 1) & 1);
                const col_in_tile = l / 4 + 8 * (e >> 2);
                const state = exl3StateFor(tile, @intCast(p), halves);
                const w = exl3TableValue(state);
                const kk = kb * 16 + row_in_tile;
                const xv: f64 = @floatCast(@as(f16, @bitCast(x[row * k + kk])));
                out[nb * 16 + col_in_tile] += xv * w;
            }
        }
    }
}

// THE AFFINE 4-BIT ROUTED DECODE, PRICED AT OUR OWN ROUTED SHAPES AND IN THE SAME RUN AS THE EXL3 BODIES. The
// comparator's decode is affine (one simdgroup a column); our pack is a codebook. The affine body already exists in
// this tree -- tile_qmv's indexed twin, ported whole from the sibling branch (Apache-2.0, THIRD_PARTY_NOTICES) --
// with 4/6/8-bit codes and group-64 bf16 scales and biases. It is timed on synthetic codes: the access pattern is
// the same whatever the codes hold, and the question here is the body's cost, not a repack's quality (that owes the
// band later).
fn timeAffine(queue: mtl.Queue, pipe: mtl.Pipeline, args: anytype, ws: []const mtl.Buffer, sc: mtl.Buffer, bi: mtl.Buffer, x: mtl.Buffer, y: mtl.Buffer, ids: mtl.Buffer, n: u32, slots: u32, trials: usize) !f64 {
    var samples: [16]f64 = undefined;
    const nt = @max(1, @min(trials, samples.len));
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(ws[t % ws.len], 0, 1);
        e.setBuffer(sc, 0, 2);
        e.setBuffer(bi, 0, 3);
        e.setBytes(std.mem.asBytes(&args), 4);
        e.setBuffer(y, 0, 5);
        e.setBuffer(ids, 0, 6);
        e.setThreadgroupMemory(10240, 0);
        e.dispatchGroups(.{ .width = n / 8, .height = slots }, .{ .width = 64 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  affine: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// THE 3-BIT AFFINE PREFILL ROUTE PRICED AT THE ROUTED TILE'S SHAPE. The affine_mm gather is the tensor-unit path
// with NO per-weight extraction (a 3-bit code and a group-64 scale), and it is already in this tree (ported from
// the sibling, Apache-2.0). Priced here on the plan's own geometry: 384 experts x one 64-row tile each, n = 2304, k
// = 5120. Synthetic codes and scales: the access pattern is the body's, the values are not the question (a repack
// owes the band).
fn affinePrefillPrice(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    const experts: u32 = 384;
    const n: u32 = 2304;
    const k: u32 = 5120;
    const bm: u32 = 64;
    const rows: u32 = experts * bm;
    const g: u32 = k / 64;
    const body = try std.mem.replaceOwned(u8, gpa, ks.core_affine_mm, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(body);
    const src = try std.fmt.allocPrint(gpa, "#define TF_BITS 3\n#define TF_GROUP 64\n#define TF_OUT_T bfloat\n{s}", .{body});
    defer gpa.free(src);
    const lib = mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx()) catch {
        std.debug.print("affine prefill: the library did not build\n", .{});
        return;
    };
    defer lib.deinit();
    const sums_p = mtl.Pipeline.init(device, lib, "tf_affine_row_sums", false) catch return;
    defer sums_p.deinit();
    const gp = mtl.Pipeline.init(device, lib, "tf_affine_gather_64", false) catch return;
    defer gp.deinit();
    const wrow_bytes: usize = @as(usize, k) * 3 / 8;
    const wbytes: usize = @as(usize, experts) * n * wrow_bytes;
    const sbytes: usize = @as(usize, experts) * n * g * 2;
    const xbytes: usize = @as(usize, rows) * k * 2;
    const xsbytes: usize = @as(usize, rows) * g * 4;
    const ybytes: usize = @as(usize, rows) * n * 2;
    const w = try device.buffer(wbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer w.deinit();
    const s = try device.buffer(sbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer s.deinit();
    const bi = try device.buffer(sbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer bi.deinit();
    const x = try device.buffer(xbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    const xs = try device.buffer(xsbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer xs.deinit();
    const y = try device.buffer(ybytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer y.deinit();
    const offs = try device.buffer((@as(usize, experts) + 1) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer offs.deinit();
    std.debug.print("\naffine prefill price: {d} experts x {d} rows, n {d}, k {d}, 3-bit group-64 (3.5 bits a weight)\n", .{ experts, bm, n, k });
    // The fixture: codes and scales are patterns; the reads are the body's own.
    var i: usize = 0;
    while (i < wbytes) : (i += 8) {
        const v: u64 = 0x5A5A5A5A5A5A5A5A ^ (@as(u64, i) *% 0x9E3779B97F4A7C15);
        @memcpy(w.slice(u8, 8)[0..8], std.mem.asBytes(&v));
    }
    for (s.slice(u16, sbytes / 2), 0..) |*v, j| v.* = @truncate(@as(u32, 0x3A00) +% (fillWord(11, j) & 127));
    for (bi.slice(u16, sbytes / 2), 0..) |*v, j| v.* = @truncate(@as(u32, 0xB600) +% (fillWord(13, j) & 127));
    for (x.slice(u16, xbytes / 2), 0..) |*v, j| v.* = 0x3400 | @as(u16, @intCast(j & 0x3ff));
    for (xs.slice(f32, xsbytes / 4), 0..) |*v, j| v.* = @floatFromInt(@as(u32, @intCast(j & 255)));
    {
        const op = offs.slice(i32, experts + 1);
        for (op, 0..) |*v, j| v.* = @intCast(j * bm);
    }
    std.debug.print("  fixture {d:.2} GB (w {d:.2}, scales+biases {d:.2}, x {d:.2})\n", .{ @as(f64, @floatFromInt(wbytes + 2 * sbytes + xbytes + xsbytes + ybytes)) / 1e9, @as(f64, @floatFromInt(wbytes)) / 1e9, @as(f64, @floatFromInt(2 * sbytes)) / 1e9, @as(f64, @floatFromInt(xbytes)) / 1e9 });
    const args = extern struct { rows: i32, n: i32, k: i32, experts: i32 }{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .experts = @intCast(experts) };
    var samples: [5]f64 = undefined;
    for (&samples) |*t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        // The rows' group sums.
        e.setPipeline(sums_p);
        e.setBuffer(x, 0, 0);
        e.setBytes(std.mem.asBytes(&args), 5);
        e.setBuffer(xs, 0, 6);
        e.dispatchThreads(.{ .width = g, .height = rows }, .{ .width = @min(g, 64) });
        e.setPipeline(gp);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(s, 0, 2);
        e.setBuffer(bi, 0, 3);
        e.setBuffer(xs, 0, 4);
        e.setBytes(std.mem.asBytes(&args), 5);
        e.setBuffer(y, 0, 6);
        e.setBuffer(offs, 0, 7);
        e.dispatchGroups(.{ .width = n / 64, .height = experts, .depth = 1 }, .{ .width = 32, .height = 2, .depth = 2 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  affine prefill: failed: {s}\n", .{msg});
            return;
        }
        t.* = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    const ms = median(&samples);
    const macs = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    const bytes = @as(f64, @floatFromInt(wbytes + 2 * sbytes));
    std.debug.print("  gather_64 3-bit: {d:.3} ms  {d:.1} TMAC/s  {d:.1} GB/s  ({d:.2} GB of weights a pass)\n", .{
        ms, macs / (ms * 1e-3) / 1e12, bytes / (ms * 1e-3) / 1e9, bytes / 1e9,
    });
    std.debug.print("  (the EXL3 plan-shape tile: 537 tiles, 34,368 rows)\n", .{});
}

fn routedBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, ks.core_exl3_projection, options);
    defer lib.deinit();
    // The direct-register descriptor body: the table path (what the engine would dispatch) and the arithmetic
    // path (the shipped MMA body's value path, so the two read the same values).
    const direct = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_direct_desc", false);
    defer direct.deinit();
    const arith_src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection, "#include <metal_stdlib>", "#define TF_DIRECT_DESC_ARITH 1\n#include <metal_stdlib>");
    defer gpa.free(arith_src);
    const alib = try mtl.Library.fromSource(device, arith_src, options);
    defer alib.deinit();
    const direct_arith = try mtl.Pipeline.init(device, alib, "tf_exl3_matmul_direct_desc", false);
    defer direct_arith.deinit();
    const reduce = mtl.Pipeline.init(device, lib, "tf_exl3_splitk_reduce", false) catch null;
    defer if (reduce) |p| p.deinit();
    const nax_src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_src);
    const nax_lib = mtl.Library.fromSource(device, nax_src, mtl.CompileOptions.mlx()) catch null;
    defer if (nax_lib) |l| l.deinit();
    const mma = if (nax_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma) |p| p.deinit();
    // The one-simdgroup-per-column routed decode (the comparator's plan) on our tile pack.
    const col_nax = if (nax_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matvec_col_nax", false) catch null) else null;
    defer if (col_nax) |p| p.deinit();
    // The affine 4-bit routed decode (the comparator's value path) on the tree's own tile_qmv body.
    const a4_lib = mtl.Library.fromSource(device, ks.core_tile_qmm, mtl.CompileOptions.mlx()) catch null;
    defer if (a4_lib) |l| l.deinit();
    const affine4 = if (a4_lib) |l| (mtl.Pipeline.init(device, l, "tf_tile_qmv_indexed_q4", false) catch null) else null;
    defer if (affine4) |p| p.deinit();
    // The 3-bit affine TILE body (sixteen warps, our geometry) -- what a repack would dispatch.
    const t3_src = try std.fmt.allocPrint(gpa, "#define TF_AFF_WARPS 16\n{s}", .{nax_src});
    defer gpa.free(t3_src);
    const t3_lib = mtl.Library.fromSource(device, t3_src, mtl.CompileOptions.mlx()) catch null;
    defer if (t3_lib) |l| l.deinit();
    const aff_tile3 = if (t3_lib) |l| (mtl.Pipeline.init(device, l, "tf_affine_tile3_nax", false) catch null) else null;
    defer if (aff_tile3) |p| p.deinit();
    // The 3-bit affine body on a COLUMN-SEQUENTIAL pack (the form the repack would build).
    const col3 = if (nax_lib) |l| (mtl.Pipeline.init(device, l, "tf_affine_col3_nax", false) catch null) else null;
    defer if (col3) |p| p.deinit();
    const compact_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_COMPACT 1\n{s}", .{nax_src});
    defer gpa.free(compact_src);
    const compact_lib = mtl.Library.fromSource(device, compact_src, mtl.CompileOptions.mlx()) catch null;
    defer if (compact_lib) |l| l.deinit();
    const mma_c = if (compact_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma_c) |p| p.deinit();
    // The same compact tile with the codebook read from the resident table: the engine binds the table for every
    // stage and the table holds the format's own decoder's values, so this arm is the bit-exact value path.
    const tab_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_DECODE_COMPACT 1\n{s}", .{nax_src});
    defer gpa.free(tab_src);
    const tab_lib = mtl.Library.fromSource(device, tab_src, mtl.CompileOptions.mlx()) catch null;
    defer if (tab_lib) |l| l.deinit();
    const mma_t = if (tab_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma_t) |p| p.deinit();
    // The two-word funnel read (TF_DECODE_FUNNEL), the routed pack's halves = 6. The four-row and compact bodies,
    // arithmetic and table, so the port is priced on the engine's own arm and hash-checked against the shipped
    // four-word body on the same buffers.
    const funnel_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_FUNNEL_H8 1\n{s}", .{nax_src});
    defer gpa.free(funnel_src);
    const funnel_lib = mtl.Library.fromSource(device, funnel_src, mtl.CompileOptions.mlx()) catch null;
    defer if (funnel_lib) |l| l.deinit();
    const mma_f = if (funnel_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma_f) |p| p.deinit();
    const funnel_c_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_FUNNEL_H8 1\n#define TF_DECODE_COMPACT 1\n{s}", .{nax_src});
    defer gpa.free(funnel_c_src);
    const funnel_c_lib = mtl.Library.fromSource(device, funnel_c_src, mtl.CompileOptions.mlx()) catch null;
    defer if (funnel_c_lib) |l| l.deinit();
    const mma_cf = if (funnel_c_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma_cf) |p| p.deinit();
    const funnel_t_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_FUNNEL_H8 1\n#define TF_DECODE_TABLE 1\n#define TF_DECODE_COMPACT 1\n{s}", .{nax_src});
    defer gpa.free(funnel_t_src);
    const funnel_t_lib = mtl.Library.fromSource(device, funnel_t_src, mtl.CompileOptions.mlx()) catch null;
    defer if (funnel_t_lib) |l| l.deinit();
    const mma_tf = if (funnel_t_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma_tf) |p| p.deinit();
    // The two-word read at halves = 8 with the cursor's own extraction (TF_DECODE_TWO).
    const two_src = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TWO 1\n#define TF_DECODE_TABLE 1\n#define TF_DECODE_COMPACT 1\n{s}", .{nax_src});
    defer gpa.free(two_src);
    const two_lib = mtl.Library.fromSource(device, two_src, mtl.CompileOptions.mlx()) catch null;
    defer if (two_lib) |l| l.deinit();
    const mma_t2 = if (two_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_nax", false) catch null) else null;
    defer if (mma_t2) |p| p.deinit();
    // THE FAIR 3-BIT AFFINE TILE BODY, rebuilt on its own 96-byte tile and its own cold fixture (The tile3 arm
    // handed it a 288-byte stride and the EXL3 trellis copies, so it was never a price). One pipeline a warp count;
    // `tf_affine_tile3_ref` is the scalar value check on the same bytes.
    var t3_warps_pipes: [5]?mtl.Pipeline = .{ null, null, null, null, null };
    var t3_libs: [5]?mtl.Library = .{ null, null, null, null, null };
    const t3_warps = [_]u32{ 1, 2, 4, 8, 16 };
    for (t3_warps, 0..) |w, wi| {
        const src = try std.fmt.allocPrint(gpa, "#define TF_AFF_WARPS {d}\n{s}", .{ w, nax_src });
        defer gpa.free(src);
        const l = mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx()) catch null;
        if (l) |ll| {
            t3_libs[wi] = ll;
            t3_warps_pipes[wi] = mtl.Pipeline.init(device, ll, "tf_affine_tile3b_nax", false) catch null;
        }
    }
    var t3_bias_pipes: [2]?mtl.Pipeline = .{ null, null };
    const t3_bias_warps = [_]u32{ 8, 16 };
    for (t3_bias_warps, 0..) |_, wi| {
        if (t3_libs[wi + 3]) |l| {
            t3_bias_pipes[wi] = mtl.Pipeline.init(device, l, "tf_affine_tile3s_nax", false) catch null;
        }
    }
    defer for (t3_bias_pipes) |maybe| if (maybe) |p| p.deinit();
    const t3_ref = if (t3_libs[0]) |l| (mtl.Pipeline.init(device, l, "tf_affine_tile3_ref", false) catch null) else null;
    defer if (t3_ref) |p| p.deinit();
    defer for (t3_warps_pipes) |maybe| if (maybe) |p| p.deinit();
    defer for (t3_libs) |maybe| if (maybe) |l| l.deinit();

    // The comparator's decode structure -- one warp a sixteen-column tile, one k walk, no split-K, no partials, no
    // reduce dispatch, the decoded weights in registers (no threadgroup round trip). The warps a threadgroup are
    // TF_FMA_WARPS: each warp walks its own slice of the k blocks and the partials meet once, at the end of the
    // walk, so the parallelism of a split-K dispatch without its round trip.
    const FmaArm = struct { name: []const u8, warps: u32, pipe: ?mtl.Pipeline };
    var fma_arms: [48]FmaArm = undefined;
    var nfma: usize = 0;
    var fma_libs: [48]?mtl.Library = undefined;
    var nfma_libs: usize = 0;
    // The libraries outlive the pipelines: the pipelines are released first, then the libraries.
    defer for (fma_libs[0..nfma_libs]) |maybe| if (maybe) |l| l.deinit();
    defer for (fma_arms[0..nfma]) |arm| if (arm.pipe) |q| q.deinit();
    {
        const warps = [_]u32{ 1, 2, 4, 8, 16, 32 };
        for (warps) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            if (wlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma", false) catch null) else null;
            const nm = switch (w) {
                1 => "fma w1",
                2 => "fma w2",
                4 => "fma w4",
                8 => "fma w8",
                16 => "fma w16",
                else => "fma w32",
            };
            fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
            nfma += 1;
        }
        // The four-word read at the wider groups (the funnel's two-way select is a branch in the unrolled loop).
        for ([_]u32{ 4, 8, 16 }) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            if (wlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma", false) catch null) else null;
            const nm = switch (w) {
                4 => "fma w4 4wd",
                8 => "fma w8 4wd",
                else => "fma w16 4wd",
            };
            fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
            nfma += 1;
        }
        // The value path arms at one warp: the codebook table against the ALU.
        const tsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n{s}", .{nax_src});
        defer gpa.free(tsrc);
        const tlib = mtl.Library.fromSource(device, tsrc, mtl.CompileOptions.mlx()) catch null;
        if (tlib) |l| {
            fma_libs[nfma_libs] = l;
            nfma_libs += 1;
        }
        const tp = if (tlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma", false) catch null) else null;
        fma_arms[nfma] = .{ .name = "fma w1 tab", .warps = 1, .pipe = tp };
        nfma += 1;
        const alib2 = mtl.Library.fromSource(device, nax_src, mtl.CompileOptions.mlx()) catch null;
        if (alib2) |l| {
            fma_libs[nfma_libs] = l;
            nfma_libs += 1;
        }
        const ap = if (alib2) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma", false) catch null) else null;
        fma_arms[nfma] = .{ .name = "fma w1 alu", .warps = 1, .pipe = ap };
        nfma += 1;
        // The static state selection in the register-only body (the engine's w4/table arm): the lane's eight
        // windows at compile-time shifts, so the `rel[k]` cursor leaves the inner loop.
        const ssrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 4\n{s}", .{nax_src});
        defer gpa.free(ssrc);
        const slib = mtl.Library.fromSource(device, ssrc, mtl.CompileOptions.mlx()) catch null;
        if (slib) |l| {
            fma_libs[nfma_libs] = l;
            nfma_libs += 1;
        }
        const sp = if (slib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static", false) catch null) else null;
        fma_arms[nfma] = .{ .name = "fma w4 stat", .warps = 4, .pipe = sp };
        nfma += 1;
        // The same arm with the codebook read from a 1024-entry threadgroup LUT.
        const sl2 = if (slib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
        fma_arms[nfma] = .{ .name = "fma w4 st+lu", .warps = 4, .pipe = sl2 };
        nfma += 1;
        // THE ENGINE'S OWN ROUTED CONFIGURATION (sixteen warps, the same body form) AND THE COMPARATOR'S 4-BYTE
        // STREAM WORD (the prefill tiles' read, The form). Both are the same bytes, the same values and the same
        // order: every arm is BIT-IDENTICAL to the shipped body.
        for ([_]u32{ 4, 16 }) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            if (wlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = if (w == 4) "fma w4 st+lu w16" else "fma w16 st+lu";
            fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
            nfma += 1;
        }
        // THE TWO-WORD STATIC BODY (TF_FMA_TWO) at halves = 6 -- the routed stage's width. The shipped static body
        // reads three stream words a lane a block; at halves = 6 two suffice by construction (The proof: every
        // lane's run starts at off0 in {3,11,19,27} and spans 37 bits), so the third word is dropped.
        for ([_]u32{ 4, 16 }) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_TWO 1\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            if (wlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = if (w == 4) "2wd st+lu w4" else "2wd st+lu w16";
            fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
            nfma += 1;
        }
        // THE VALUE PATH'S FORM AT THE ROUTED SHAPES.
        for ([_]u32{ 4, 16 }) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_VALUE_ARITH 1\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            if (wlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = if (w == 4) "st+lu arith w4" else "st+lu arith w16";
            fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
            nfma += 1;
        }
        // THE PAIR BODY AT THE ROUTED SHAPES. The engine already HAS the comparator's pair form
        // (tf_exl3_matmul_decode_fma_static_pair_lut: it scans the routes for the next slot naming the same expert,
        // the even rank leads, and ONE weight decode serves the pair -- the engine's own comment says the pair's
        // rows are bit-identical to the single-row body's). Priced it in the PLAIN path (one row: a mate never
        // exists) and read a LOSS; the VERIFY is a different population (56 and 28 slots, and the shared expert's
        // weights are decoded once a row). This arm prices the pair at the routed shapes. The bench's model gives
        // every slot its own expert, so the pair finds NO mate here: the arm prices the pair's OVERHEAD (the mate
        // scan over a.rows and the doubled body) ALONE, which is the upper bound on the pair's cost -- the saving
        // (the duplicate decodes) can only be smaller.
        {
            const qsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n{s}", .{nax_src});
            defer gpa.free(qsrc);
            const qlib = mtl.Library.fromSource(device, qsrc, mtl.CompileOptions.mlx()) catch null;
            if (qlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const q = if (qlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_pair_lut", false) catch null) else null;
            fma_arms[nfma] = .{ .name = "pair st+lu w16", .warps = 16, .pipe = q };
            nfma += 1;
        }
        // THE PREFETCH DEPTH AT THE ROUTED SHAPES. The ring only moves the loads in time: the values, their order
        // and the arithmetic are untouched, so every arm is BIT-IDENTICAL to the shipped body (the bench's hash
        // says so).
        for ([_]u32{ 16, 4 }) |w| {
            for ([_]u32{ 2, 4, 8 }) |pf| {
                const psrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_PF {d}\n{s}", .{ w, pf, nax_src });
                defer gpa.free(psrc);
                const plib = mtl.Library.fromSource(device, psrc, mtl.CompileOptions.mlx()) catch null;
                if (plib) |l| {
                    fma_libs[nfma_libs] = l;
                    nfma_libs += 1;
                }
                const p = if (plib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
                const nm = try std.fmt.allocPrint(gpa, "pf{d} st+lu w{d}", .{ pf, w });
                fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
                nfma += 1;
            }
        }
        // THE ROUTED STAGE'S OWN ROOFLINE -- the same separation ladder as the dense one-row bench, at the ENGINE'S
        // dispatch shape (the static+LUT body at sixteen warps, six experts a dispatch). TF_DIAG_DECODE = 1 keeps
        // the state's extraction and the trellis read and drops the VALUE path; = 2 drops the whole decode (the
        // trellis reads go with it); = 3 doubles the per-weight FP op.
        for ([_]u32{ 1, 2, 3 }) |d| {
            const dsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n#define TF_DIAG_DECODE {d}\n{s}", .{ d, nax_src });
            defer gpa.free(dsrc);
            const dlib = mtl.Library.fromSource(device, dsrc, mtl.CompileOptions.mlx()) catch null;
            if (dlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (dlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = switch (d) {
                1 => "d1 st+lu w16",
                2 => "d2 st+lu w16",
                else => "d3 st+lu w16",
            };
            fma_arms[nfma] = .{ .name = nm, .warps = 16, .pipe = p };
            nfma += 1;
        }
        // THE VALUE PATH'S SPLIT AT THE ROUTED SHAPES. d6 keeps the LOAD and drops the index arithmetic; d7 keeps
        // the index arithmetic and drops the LOAD. Neither is bit-exact (the value is a timing stand-in) and both
        // are bench-only.
        for ([_]u32{ 6, 7 }) |d| {
            const dsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n#define TF_DIAG_DECODE {d}\n{s}", .{ d, nax_src });
            defer gpa.free(dsrc);
            const dlib = mtl.Library.fromSource(device, dsrc, mtl.CompileOptions.mlx()) catch null;
            if (dlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (dlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = if (d == 6) "d6 st+lu w16" else "d7 st+lu w16";
            fma_arms[nfma] = .{ .name = nm, .warps = 16, .pipe = p };
            nfma += 1;
        }
        // The ladder's last two levels at the routed shapes: d4 (the ACTIVATION READ gone) and d5 (the CROSS-WARP
        // REDUCTION gone -- the routed body reduces sixteen warps' partials through threadgroup memory, so its
        // reduction is not the dense body's).
        for ([_]u32{ 4, 5 }) |d| {
            const dsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n#define TF_DIAG_DECODE {d}\n{s}", .{ d, nax_src });
            defer gpa.free(dsrc);
            const dlib = mtl.Library.fromSource(device, dsrc, mtl.CompileOptions.mlx()) catch null;
            if (dlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (dlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = if (d == 4) "d4 st+lu w16" else "d5 st+lu w16";
            fma_arms[nfma] = .{ .name = nm, .warps = 16, .pipe = p };
            nfma += 1;
        }
        // THE VALUE PATH'S FORM AT THE ENGINE'S OWN WARP COUNT -- the DEVICE TABLE (the 65536-entry half table, the
        // static body's own form) against the threadgroup LUT and the arithmetic. The record has the w4 pair; this
        // is the w16 pair.
        {
            const ssrc16 = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n{s}", .{nax_src});
            defer gpa.free(ssrc16);
            const slib16 = mtl.Library.fromSource(device, ssrc16, mtl.CompileOptions.mlx()) catch null;
            if (slib16) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const sp16 = if (slib16) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static", false) catch null) else null;
            fma_arms[nfma] = .{ .name = "fma w16 stat", .warps = 16, .pipe = sp16 };
            nfma += 1;
        }
        for ([_]u32{ 4, 16 }) |w| {
            const wsrc = try std.fmt.allocPrint(gpa, "#define TF_DECODE_FUNNEL 1\n#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS {d}\n#define TF_FMA_WIDE_WORD 1\n{s}", .{ w, nax_src });
            defer gpa.free(wsrc);
            const wlib = mtl.Library.fromSource(device, wsrc, mtl.CompileOptions.mlx()) catch null;
            if (wlib) |l| {
                fma_libs[nfma_libs] = l;
                nfma_libs += 1;
            }
            const p = if (wlib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_matmul_decode_fma_static_lut", false) catch null) else null;
            const nm = if (w == 4) "fma w4 ww" else "fma w16 ww";
            fma_arms[nfma] = .{ .name = nm, .warps = w, .pipe = p };
            nfma += 1;
        }
    }

    // A one-word probe of the trellis as the GPU sees it (the fixture's visibility is the question the routed
    // bench's own column probe raised).
    const words_src = try std.mem.concat(gpa, u8, &.{
        nax_src,
        \\kernel void tf_exl3_probe_words(const device ushort* t [[buffer(0)]], device uint* out [[buffer(1)]],
        \\                             uint gid [[thread_position_in_grid]]) {
        \\  out[gid] = (uint)t[2 * gid] | ((uint)t[2 * gid + 1] << 16);
        \\}
        \\kernel void tf_exl3_fill_trellis(device ushort* t [[buffer(0)]], constant ulong& seed [[buffer(1)]],
        \\                                uint gid [[thread_position_in_grid]]) {
        \\  ulong x = seed + (ulong)gid * 0x9E3779B97F4A7C15UL;
        \\  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        \\  x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        \\  x ^= x >> 31;
        \\  t[gid] = (ushort)x;
        \\}
        \\kernel void tf_affine_fill_scale(device ushort* t [[buffer(0)]], constant ulong& seed [[buffer(1)]],
        \\                                  uint gid [[thread_position_in_grid]]) {
        \\  ulong x = seed + (ulong)gid * 0x9E3779B97F4A7C15UL;
        \\  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        \\  x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        \\  x ^= x >> 31;
        \\  t[gid] = (ushort)(0x3B00u | (uint)(x & 63u));
        \\}
    });
    defer gpa.free(words_src);
    const words_lib = mtl.Library.fromSource(device, words_src, mtl.CompileOptions.mlx()) catch null;
    defer if (words_lib) |l| l.deinit();
    const pwords = if (words_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_probe_words", false) catch null) else null;
    defer if (pwords) |p| p.deinit();
    const pfill = if (words_lib) |l| (mtl.Pipeline.init(device, l, "tf_exl3_fill_trellis", false) catch null) else null;
    defer if (pfill) |p| p.deinit();
    const pscale = if (words_lib) |l| (mtl.Pipeline.init(device, l, "tf_affine_fill_scale", false) catch null) else null;
    defer if (pscale) |p| p.deinit();
    const palt = if (words_lib) |l| (mtl.Pipeline.init(device, l, "tf_affine_fill_alt", false) catch null) else null;
    defer if (palt) |p| p.deinit();
    const pone = if (words_lib) |l| (mtl.Pipeline.init(device, l, "tf_affine_fill_one", false) catch null) else null;
    defer if (pone) |p| p.deinit();

    // The body value certificate on a tiny shape (n 64, k 64, one row) where every byte is known and every buffer
    // is small: the warp body and the scalar reference against a host FP64 decode of the same 3-bit codes, with all
    // scales one. A mismatch here is a port bug, not a numerics class.
    if (t3_ref != null and t3_warps_pipes[0] != null and pfill != null and pwords != null and pone != null) {
        const tn: usize = 64;
        const tk: usize = 64;
        const cb64: usize = tk * tn * 3 / 8;
        const codes = try device.buffer(cb64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer codes.deinit();
        const scratchv = try device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer scratchv.deinit();
        _ = fillTrellisGpuVisible(queue, pfill.?, pwords.?, codes, @intCast(cb64), 0x4480 ^ @as(u64, tn), scratchv);
        const scl = try device.buffer(tn * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer scl.deinit();
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pone.?);
            e.setBuffer(scl, 0, 0);
            e.dispatchThreads(.{ .width = @intCast(tn) }, .{ .width = 64 });
            e.end();
            cb.commit();
            cb.wait();
        }
        const x64 = try device.buffer(tk * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x64.deinit();
        for (x64.slice(u16, tk), 0..) |*v, i| v.* = 0x3400 | @as(u16, @intCast(i & 0x3ff));
        const r64 = try device.buffer(tn * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer r64.deinit();
        const w64 = try device.buffer(tn * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer w64.deinit();
        const d64 = try device.buffer(@sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer d64.deinit();
        {
            const dp: [*]Exl3Expert = @ptrCast(@alignCast(d64.contents()));
            dp[0] = .{ .trellis = codes.gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (tn / 16) * 96, .tile_n_stride = 96, .halves = 6, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
        }
        const rt64 = try device.buffer(4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer rt64.deinit();
        rt64.slice(u32, 1)[0] = 0;
        const args64 = Exl3Args{
            .k = tk,
            .n = tn,
            .rows = 1,
            .halves = 6,
            .codebook = 2,
            .su_signs = 0,
            .sv_signs = 0,
            .has_bias = 0,
            .output_f32 = 1,
            .input_ids = 0,
            .output_ids = 0,
            .routed = 1,
            .expert_id = 0,
            .repeat = 1,
            .indexed = 1,
            .experts = 1,
            .splits = 1,
            .x_stride = tk,
            .y_stride = tn,
            .tile_k_stride = (tn / 16) * 96,
            .tile_n_stride = 96,
        };
        try runAffine(queue, t3_ref.?, args64, x64, codes, r64, scl, rt64, d64, 1, 1, 64);
        try runAffine(queue, t3_warps_pipes[0].?, args64, x64, codes, w64, scl, rt64, d64, tn / 16, 1, 32);
        const rv64 = r64.slice(f32, tn);
        const wv64 = w64.slice(f32, tn);
        const cseed64: u64 = 0x4480 ^ @as(u64, tn);
        var max_ref: f64 = 0;
        var max_warp: f64 = 0;
        var sq: f64 = 0;
        var sb: f64 = 0;
        const hx = x64.slice(u16, tk);
        for (0..tn) |col| {
            var acc: f64 = 0;
            for (0..tk) |k_| {
                const b = k_ / 16;
                const kk = k_ % 16;
                const q = (kk >> 1) & 3;
                const j = (kk & 1) + 2 * ((kk >> 3) & 1) + 4 * ((col >> 3) & 1);
                const idx = 32 * (col & 7) + 8 * q + j;
                const bit = 3 * idx;
                const bp: usize = b * ((tn / 16) * 96) + (col / 16) * 96 + (bit >> 3);
                const two: u32 = @as(u32, fillWord(cseed64, @intCast(bp / 2))) | (@as(u32, fillWord(cseed64, @intCast(bp / 2 + 1))) << 16);
                const shift: u5 = @intCast((bp & 1) * 8 + (bit & 7));
                const code = (two >> shift) & 7;
                const xh: f16 = @bitCast(hx[k_]);
                acc += @as(f64, @floatCast(xh)) * @as(f64, @floatFromInt(code));
            }
            max_ref = @max(max_ref, @abs(acc - @as(f64, rv64[col])));
            max_warp = @max(max_warp, @abs(acc - @as(f64, wv64[col])));
            sq += (@as(f64, rv64[col]) - @as(f64, wv64[col])) * (@as(f64, rv64[col]) - @as(f64, wv64[col]));
            sb += acc * acc;
        }
        std.debug.print("  tiny value check 64x64 scale1: worst host-ref {e:.4}, host-warp {e:.4}, ref-vs-warp NRMS {e:.4}\n", .{ max_ref, max_warp, @sqrt(sq / sb) });
    }
    const table = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer table.deinit();
    for (table.slice(u16, 3 * 65536), 0..) |*v, i| v.* = 0x3c00 | @as(u16, @intCast(i & 0xff));
    // The mul1 codebook the engine's table holds (exl3_format.zig's codebookValue): the table path must be the
    // arithmetic path's values, entry for entry, or the arm is not the same reduction.
    {
        const scale: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
        const bias: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));
        for (0..65536) |state| {
            const x: u32 = @truncate(@as(u64, state) * 0x83DCD12D);
            const h: f64 = @floatFromInt(1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24));
            const v: f16 = @floatCast(h * scale + bias);
            table.slice(u16, 3 * 65536)[2 * 65536 + state] = @bitCast(v);
        }
    }

    std.debug.print("\nrouted decode bench: the shipped MMA body against the direct-register descriptor body\n", .{});
    std.debug.print("  {s:>10} {s:>14} {s:>5} {s:>9} {s:>9} {s:>8} {s:>10}\n", .{ "shape", "body", "splits", "ms", "TMAC/s", "GB/s", "hash" });
    for (routed_shapes) |s| {
        const tile_bytes = s.halves * 8 * 2;
        const trellis_bytes = (s.k / 16) * (s.n / 16) * tile_bytes;
        const rows: usize = s.rows;
        // The verify-s own dispatch shapes need one fresh trellis copy a row (56 rows), so the pool grows from 48
        // to 80 copies -- the copies keep the experts- weights from being resident.
        var trs: [routed_copies * 10]mtl.Buffer = undefined;
        var made: usize = 0;
        defer for (trs[0..made]) |b| b.deinit();
        while (made < trs.len) : (made += 1) {
            trs[made] = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        }
        // Every fixture write is verified from the GPU before any value-level check reads it.
        const scratch = try device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer scratch.deinit();
        var invisible: usize = 0;
        if (pwords != null and pfill != null) {
            for (trs[0..made], 0..) |b, i| {
                if (!fillTrellisGpuVisible(queue, pfill.?, pwords.?, b, trellis_bytes, 0xD1CEB00C ^ @as(u64, s.n) ^ @as(u64, s.k) ^ @as(u64, i), scratch)) invisible += 1;
            }
        } else {
            for (trs[0..made], 0..) |b, i| fillTrellis(b, trellis_bytes, 0xD1CEB00C ^ @as(u64, s.n) ^ @as(u64, s.k) ^ @as(u64, i));
        }
        if (invisible > 0) std.debug.print("  {s:>10} WARNING: {d} of {d} trellis copies did not become visible to the GPU\n", .{ s.name, invisible, made });
        const x = try device.buffer(rows * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        for (x.slice(u16, rows * s.k), 0..) |*v, i| v.* = 0x3400 | @as(u16, @intCast(i & 0x3ff));
        const sums = try device.buffer(rows * s.n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sums.deinit();
        const partials = try device.buffer(32 * rows * s.n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer partials.deinit();
        const routes = try device.buffer(rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer routes.deinit();
        for (routes.slice(u32, rows), 0..) |*r, i| r.* = @intCast(i);
        const descs = try device.buffer(rows * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer descs.deinit();
        {
            const dp: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
            for (0..rows) |e| dp[e] = .{ .trellis = trs[e].gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (s.n / 16) * tile_bytes, .tile_n_stride = tile_bytes, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
        }
        const n_out = rows * s.n;
        const macs = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) * 2.0;
        const gbs = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(trellis_bytes));

        // The numerics check: the direct body against the MMA body on the same buffers, one dispatch each.
        if (mma_c != null) {
            const host_a = try gpa.alloc(f32, n_out);
            defer gpa.free(host_a);
            const host_b = try gpa.alloc(f32, n_out);
            defer gpa.free(host_b);
            const args = Exl3Args{
                .k = s.k,
                .n = s.n,
                .rows = s.rows,
                .halves = s.halves,
                .codebook = 2,
                .su_signs = 0,
                .sv_signs = 0,
                .has_bias = 0,
                .output_f32 = 1,
                .input_ids = 0,
                .output_ids = 0,
                .routed = 1,
                .expert_id = 0,
                .repeat = 1,
                .indexed = 1,
                .experts = s.rows,
                .splits = 8,
                .x_stride = s.k,
                .y_stride = s.n,
                .tile_k_stride = (s.n / 16) * tile_bytes,
                .tile_n_stride = tile_bytes,
            };
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, mma_c.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = 8, .threads = 32 }, n_out, 1);
            @memcpy(host_a, sums.slice(f32, n_out));
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, direct_arith, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = (s.n + 255) / 256, .height = rows, .depth = 8, .threads = 256 }, n_out, 1);
            @memcpy(host_b, sums.slice(f32, n_out));
            var worst: f64 = 0;
            var scale: f64 = 0;
            var differing: usize = 0;
            for (host_a, host_b) |va, vb| {
                const d = @abs(va - vb);
                if (d > 0) differing += 1;
                worst = @max(worst, d);
                scale = @max(scale, @abs(va));
            }
            std.debug.print("  {s:>10} vs direct: worst |diff| = {d:.6} on |max| = {d:.3} ({d} of {d} differ)\n", .{ s.name, worst, scale, differing, n_out });
        }
        // The two-word funnel body against the shipped four-word body, byte for byte, on the same buffers. The
        // funnel is bit-identical by construction (the rel = 1 states' next word lies below the 16-bit state at
        // halves = 6), so any differing byte is a port bug, not a numerics class.
        if (mma_t != null and mma_tf != null) {
            const host_a = try gpa.alloc(f32, n_out);
            defer gpa.free(host_a);
            const host_b = try gpa.alloc(f32, n_out);
            defer gpa.free(host_b);
            const args = Exl3Args{
                .k = s.k,
                .n = s.n,
                .rows = s.rows,
                .halves = s.halves,
                .codebook = 2,
                .su_signs = 0,
                .sv_signs = 0,
                .has_bias = 0,
                .output_f32 = 1,
                .input_ids = 0,
                .output_ids = 0,
                .routed = 1,
                .expert_id = 0,
                .repeat = 1,
                .indexed = 1,
                .experts = s.rows,
                .splits = 8,
                .x_stride = s.k,
                .y_stride = s.n,
                .tile_k_stride = (s.n / 16) * tile_bytes,
                .tile_n_stride = tile_bytes,
            };
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, mma_t.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = 8, .threads = 32 }, n_out, 1);
            @memcpy(host_a, sums.slice(f32, n_out));
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, mma_tf.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = 8, .threads = 32 }, n_out, 1);
            @memcpy(host_b, sums.slice(f32, n_out));
            var differing: usize = 0;
            for (host_a, host_b) |va, vb| if (va != vb) {
                differing += 1;
            };
            std.debug.print("  {s:>10} funnel vs shipped: {d} of {d} differ ({s})\n", .{ s.name, differing, n_out, if (differing == 0) "BIT-IDENTICAL" else "DIFFERS" });
        }
        // The register-only body (the comparator's structure) against the shipped tensor-unit body on the same
        // buffers. The reduction order is the comparator's, so the two are not bit-identical: the check is the size
        // of the difference against the size of the sums.
        if (mma_tf != null and nfma > 0 and fma_arms[0].pipe != null) {
            const host_a = try gpa.alloc(f32, n_out);
            defer gpa.free(host_a);
            const host_b = try gpa.alloc(f32, n_out);
            defer gpa.free(host_b);
            const args = Exl3Args{
                .k = s.k,
                .n = s.n,
                .rows = s.rows,
                .halves = s.halves,
                .codebook = 2,
                .su_signs = 0,
                .sv_signs = 0,
                .has_bias = 0,
                .output_f32 = 1,
                .input_ids = 0,
                .output_ids = 0,
                .routed = 1,
                .expert_id = 0,
                .repeat = 1,
                .indexed = 1,
                .experts = s.rows,
                .splits = 8,
                .x_stride = s.k,
                .y_stride = s.n,
                .tile_k_stride = (s.n / 16) * tile_bytes,
                .tile_n_stride = tile_bytes,
            };
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, mma_tf.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = 8, .threads = 32 }, n_out, 1);
            @memcpy(host_a, sums.slice(f32, n_out));
            var nargs = args;
            nargs.splits = 1;
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, fma_arms[0].pipe.?, reduce, nargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 }, n_out, 1);
            @memcpy(host_b, sums.slice(f32, n_out));
            var worst: f64 = 0;
            var scale: f64 = 0;
            var sq: f64 = 0;
            var sqref: f64 = 0;
            var differing: usize = 0;
            for (host_a, host_b) |va, vb| {
                const d = @abs(va - vb);
                if (d > 0) differing += 1;
                worst = @max(worst, d);
                scale = @max(scale, @abs(va));
                sq += d * d;
                sqref += va * va;
            }
            std.debug.print("  {s:>10} fma vs shipped: worst |diff| = {d:.6} on |max| = {d:.3} (NRMS {e:.2}), {d} of {d} differ\n", .{ s.name, worst, scale, @sqrt(sq / @max(sqref, 1e-30)), differing, n_out });
        }

        // The column plan's value-level agreement with the register-only body. The map inverse is the arm's only
        // new arithmetic, so a wrong (row-in-tile, column) -> stream-position map would show as a large difference;
        // the reduction order differs, so the check is the NRMS, not bit-equality.
        if (col_nax != null and nfma > 0 and fma_arms[0].pipe != null) {
            const host_a = try gpa.alloc(f32, n_out);
            defer gpa.free(host_a);
            const host_b = try gpa.alloc(f32, n_out);
            defer gpa.free(host_b);
            const cargs = Exl3Args{
                .k = s.k,
                .n = s.n,
                .rows = s.rows,
                .halves = s.halves,
                .codebook = 2,
                .su_signs = 0,
                .sv_signs = 0,
                .has_bias = 0,
                .output_f32 = 1,
                .input_ids = 0,
                .output_ids = 0,
                .routed = 1,
                .expert_id = 0,
                .repeat = 1,
                .indexed = 1,
                .experts = s.rows,
                .splits = 1,
                .x_stride = s.k,
                .y_stride = s.n,
                .tile_k_stride = (s.n / 16) * tile_bytes,
                .tile_n_stride = tile_bytes,
            };
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, fma_arms[0].pipe.?, reduce, cargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 }, n_out, 1);
            @memcpy(host_a, sums.slice(f32, n_out));
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, col_nax.?, reduce, cargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = (s.n + 3) / 4, .height = rows, .depth = 1, .threads = 128 }, n_out, 1);
            @memcpy(host_b, sums.slice(f32, n_out));
            var cq: f64 = 0;
            var cre: f64 = 0;
            for (host_a, host_b) |va, vb| {
                const d = @as(f64, va) - @as(f64, vb);
                cq += d * d;
                cre += @as(f64, va) * @as(f64, va);
            }
            std.debug.print("  {s:>10} col plan vs fma w1: NRMS {e:.3}\n", .{ s.name, @sqrt(cq / @max(cre, 1e-30)) });
        }

        // The results are collected per arm and printed after.
        const Arm = struct { name: []const u8, splits: u32, ms: f64 = 0, hash: u64 = 0 };
        var arms: [64]Arm = undefined;
        var narms: usize = 0;
        for (0..2) |pass| {
            narms = 0;
            for ([_]u32{ 8, 16, 32, 1 }) |splits| {
                const args = Exl3Args{
                    .k = s.k,
                    .n = s.n,
                    .rows = s.rows,
                    .halves = s.halves,
                    .codebook = 2,
                    .su_signs = 0,
                    .sv_signs = 0,
                    .has_bias = 0,
                    .output_f32 = 1,
                    .input_ids = 0,
                    .output_ids = 0,
                    .routed = 1,
                    .expert_id = 0,
                    .repeat = 1,
                    .indexed = 1,
                    .experts = s.rows,
                    .splits = splits,
                    .x_stride = s.k,
                    .y_stride = s.n,
                    .tile_k_stride = (s.n / 16) * tile_bytes,
                    .tile_n_stride = tile_bytes,
                };
                const depth: usize = if (splits > 1) splits else 1;
                if (mma != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t = try timeRouted(queue, mma.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = (rows + 3) / 4, .depth = depth, .threads = 128 }, n_out, 15);
                    const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma 4row", .splits = splits, .ms = t, .hash = h };
                    narms += 1;
                }
                if (mma_c != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t = try timeRouted(queue, mma_c.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = depth, .threads = 32 }, n_out, 15);
                    const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma compact", .splits = splits, .ms = t, .hash = h };
                    narms += 1;
                }
                if (mma_t != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t_tb = try timeRouted(queue, mma_t.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = depth, .threads = 32 }, n_out, 15);
                    const h_tb = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma compact+tab", .splits = splits, .ms = t_tb, .hash = h_tb };
                    narms += 1;
                }
                if (mma_f != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t_f = try timeRouted(queue, mma_f.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = (rows + 3) / 4, .depth = depth, .threads = 128 }, n_out, 15);
                    const h_f = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma 4row funnel", .splits = splits, .ms = t_f, .hash = h_f };
                    narms += 1;
                }
                if (mma_cf != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t_cf = try timeRouted(queue, mma_cf.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = depth, .threads = 32 }, n_out, 15);
                    const h_cf = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma compact funnel", .splits = splits, .ms = t_cf, .hash = h_cf };
                    narms += 1;
                }
                if (mma_tf != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t_tf = try timeRouted(queue, mma_tf.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = depth, .threads = 32 }, n_out, 15);
                    const h_tf = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma compact+tab funnel", .splits = splits, .ms = t_tf, .hash = h_tf };
                    narms += 1;
                }
                if (mma_t2 != null) {
                    @memset(sums.slice(u32, n_out), 0);
                    const t_t2 = try timeRouted(queue, mma_t2.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = depth, .threads = 32 }, n_out, 15);
                    const h_t2 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "mma compact+tab 2wd", .splits = splits, .ms = t_t2, .hash = h_t2 };
                    narms += 1;
                }
                @memset(sums.slice(u32, n_out), 0);
                const t_a = try timeRouted(queue, direct_arith, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = (s.n + 255) / 256, .height = rows, .depth = depth, .threads = 256 }, n_out, 15);
                const h_a = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                arms[narms] = .{ .name = "direct arith", .splits = splits, .ms = t_a, .hash = h_a };
                narms += 1;
                @memset(sums.slice(u32, n_out), 0);
                const t_128 = try timeRouted(queue, direct, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = (s.n + 127) / 128, .height = rows, .depth = depth, .threads = 128 }, n_out, 15);
                const h_128 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                arms[narms] = .{ .name = "direct tbl/128", .splits = splits, .ms = t_128, .hash = h_128 };
                narms += 1;
                @memset(sums.slice(u32, n_out), 0);
                const t_t = try timeRouted(queue, direct, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = (s.n + 255) / 256, .height = rows, .depth = depth, .threads = 256 }, n_out, 15);
                const h_t = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                arms[narms] = .{ .name = "direct table", .splits = splits, .ms = t_t, .hash = h_t };
                narms += 1;
            }
            // The comparator's structure -- one warp a sixteen-column tile, one k walk, no split-K, no partials and
            // no reduce dispatch -- timed as one dispatch of the routed stage for the same six experts.
            {
                const fargs = Exl3Args{
                    .k = s.k,
                    .n = s.n,
                    .rows = s.rows,
                    .halves = s.halves,
                    .codebook = 2,
                    .su_signs = 0,
                    .sv_signs = 0,
                    .has_bias = 0,
                    .output_f32 = 1,
                    .input_ids = 0,
                    .output_ids = 0,
                    .routed = 1,
                    .expert_id = 0,
                    .repeat = 1,
                    .indexed = 1,
                    .experts = s.rows,
                    .splits = 1,
                    .x_stride = s.k,
                    .y_stride = s.n,
                    .tile_k_stride = (s.n / 16) * tile_bytes,
                    .tile_n_stride = tile_bytes,
                };
                for (fma_arms[0..nfma]) |arm| {
                    if (arm.pipe) |p| {
                        @memset(sums.slice(u32, n_out), 0);
                        const t = try timeRouted(queue, p, reduce, fargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 * arm.warps }, n_out, 15);
                        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                        arms[narms] = .{ .name = arm.name, .splits = 1, .ms = t, .hash = h };
                        narms += 1;
                    }
                }
                // One simdgroup a column, the lanes walking k (the comparator's plan); one warp a column, four
                // warps a threadgroup.
                if (col_nax) |cp| {
                    @memset(sums.slice(u32, n_out), 0);
                    const t = try timeRouted(queue, cp, reduce, fargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = (s.n + 3) / 4, .height = rows, .depth = 1, .threads = 128 }, n_out, 15);
                    const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    arms[narms] = .{ .name = "col plan", .splits = 1, .ms = t, .hash = h };
                    narms += 1;
                }
                // The affine 4-bit routed decode (the comparator's value path), same shape, same run.
                if (affine4) |ap| {
                    const ng = s.k / 64;
                    const wbytes: usize = @as(usize, rows) * s.n * s.k / 2;
                    const sbytes: usize = @as(usize, rows) * s.n * ng * 2;
                    var wbufs: [4]mtl.Buffer = undefined;
                    for (&wbufs, 0..) |*wb, ci| {
                        wb.* = try device.buffer(wbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                        for (wb.slice(u8, wbytes), 0..) |*v, i| v.* = @truncate(fillWord(0xA4B4C4D4 +% @as(u64, ci) * 7919, i));
                    }
                    defer for (wbufs) |wb| wb.deinit();
                    const scbuf = try device.buffer(sbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer scbuf.deinit();
                    const bibuf = try device.buffer(sbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer bibuf.deinit();
                    for (scbuf.slice(u16, sbytes / 2), 0..) |*v, i| v.* = @truncate(@as(u32, 0x3B00) +% (fillWord(5, i) & 63));
                    for (bibuf.slice(u16, sbytes / 2), 0..) |*v, i| v.* = @truncate(@as(u32, 0xB800) +% (fillWord(7, i) & 63));
                    const ax = try device.buffer(rows * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer ax.deinit();
                    for (ax.slice(u16, rows * s.k), 0..) |*v, i| v.* = 0x3400 | @as(u16, @intCast(i & 0x3ff));
                    const ay = try device.buffer(rows * s.n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer ay.deinit();
                    const aids = try device.buffer(rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer aids.deinit();
                    for (aids.slice(u32, rows), 0..) |*v, i| v.* = @intCast(i);
                    const A4Args = extern struct { k: i32, n: i32, r: i32, one: f32 };
                    const a4args = A4Args{ .k = @intCast(s.k), .n = @intCast(s.n), .r = 1, .one = 1.0 };
                    const t4 = try timeAffine(queue, ap, a4args, wbufs[0..], scbuf, bibuf, ax, ay, aids, s.n, @intCast(rows), 15);
                    const a4bytes = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) * 0.5625;
                    const a4macs = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) * 2.0;
                    std.debug.print("  {s:>10} {s:>14} {d:>5} {d:>9.3} {d:>9.1} {d:>8.1}  (affine 4-bit, {d:.2} GB of codes+scales, 4.5 bits a weight)\n", .{ s.name, "affine4", 1, t4, a4macs / (t4 * 1e-3) / 1e12, a4bytes / (t4 * 1e-3) / 1e9, a4bytes / 1e9 });
                }
                // The 3-bit affine TILE body (what a repack would dispatch), same shape, same run.
                if (aff_tile3 != null) {
                    const tb = s.halves * 8 * s.halves; // 96 bytes a 16x16 tile (256 codes, 3 bits each)
                    const cbytes: usize = @as(usize, rows) * s.k * s.n * 3 / 8;
                    const sbytes3: usize = @as(usize, rows) * s.n * (s.k / 64) * 2;
                    const codes = try device.buffer(cbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer codes.deinit();
                    for (codes.slice(u64, cbytes / 8), 0..) |*v, i| v.* = fillWord(0x3A3A3A3A, i) | (@as(u64, fillWord(9, i)) << 48);
                    const scales3 = try device.buffer(sbytes3, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer scales3.deinit();
                    for (scales3.slice(u16, sbytes3 / 2), 0..) |*v, i| v.* = @truncate(@as(u32, 0x3B00) +% (fillWord(5, i) & 63));
                    const t3args = Exl3Args{
                        .k = s.k,
                        .n = s.n,
                        .rows = s.rows,
                        .halves = s.halves,
                        .codebook = 2,
                        .su_signs = 0,
                        .sv_signs = 0,
                        .has_bias = 0,
                        .output_f32 = 1,
                        .input_ids = 0,
                        .output_ids = 0,
                        .routed = 1,
                        .expert_id = 0,
                        .repeat = 1,
                        .indexed = 1,
                        .experts = s.rows,
                        .splits = 1,
                        .x_stride = s.k,
                        .y_stride = s.n,
                        .tile_k_stride = (s.n / 16) * tb,
                        .tile_n_stride = tb,
                    };
                    const dp3: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
                    for (0..rows) |e| dp3[e] = .{ .trellis = codes.gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (s.n / 16) * tb, .tile_n_stride = tb, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
                    @memset(sums.slice(u32, n_out), 0);
                    const t3 = try timeRouted(queue, aff_tile3.?, reduce, t3args, x, sums, partials, scales3, routes, descs, trs[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 * 16 }, n_out, 15);
                    const h3 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    const b3 = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) * 3.25 / 8.0;
                    std.debug.print("  {s:>10} {s:>14} {d:>5} {d:>9.3} {d:>9.1} {d:>8.1} {x:>10}  (affine 3-bit tile, 3.25 bits a weight -- INSTRUMENT ONLY, NOT A PRICE: 4-5x the twin at the same shape, unexplained)\n", .{ s.name, "aff tile3 BROKEN", 1, t3, macs / (t3 * 1e-3) / 1e12, b3 / (t3 * 1e-3) / 1e9, h3 & 0xffffffff });
                }
                // The 3-bit affine body on a column-sequential pack, same shape, same run.
                if (col3 != null) {
                    const cb3: usize = @as(usize, rows) * s.n * ((s.k / 32) * 12);
                    const sb3: usize = @as(usize, rows) * s.n * (s.k / 64) * 2;
                    const codes3 = try device.buffer(cb3, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer codes3.deinit();
                    for (codes3.slice(u64, cb3 / 8), 0..) |*v, i| v.* = fillWord(0x5C5C5C5C, i) | (@as(u64, fillWord(0xB7, i)) << 48);
                    const scales3b = try device.buffer(sb3, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer scales3b.deinit();
                    for (scales3b.slice(u16, sb3 / 2), 0..) |*v, i| v.* = @truncate(@as(u32, 0x3B00) +% (fillWord(17, i) & 63));
                    const c3args = Exl3Args{
                        .k = s.k,
                        .n = s.n,
                        .rows = s.rows,
                        .halves = s.halves,
                        .codebook = 2,
                        .su_signs = 0,
                        .sv_signs = 0,
                        .has_bias = 0,
                        .output_f32 = 1,
                        .input_ids = 0,
                        .output_ids = 0,
                        .routed = 1,
                        .expert_id = 0,
                        .repeat = 1,
                        .indexed = 1,
                        .experts = s.rows,
                        .splits = 1,
                        .x_stride = s.k,
                        .y_stride = s.n,
                        .tile_k_stride = 0,
                        .tile_n_stride = 0,
                    };
                    const dpc: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
                    for (0..rows) |e| dpc[e] = .{ .trellis = codes3.gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = 0, .tile_n_stride = 0, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
                    @memset(sums.slice(u32, n_out), 0);
                    const tc3 = try timeRouted(queue, col3.?, reduce, c3args, x, sums, partials, scales3b, routes, descs, trs[0..], rows, .{ .width = (s.n + 3) / 4, .height = rows, .depth = 1, .threads = 128 }, n_out, 15);
                    const hc3 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                    const bc3 = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) * 3.25 / 8.0;
                    std.debug.print("  {s:>10} {s:>14} {d:>5} {d:>9.3} {d:>9.1} {d:>8.1} {x:>10}  (affine 3-bit COLUMN-SEQUENTIAL, 3.25 bits a weight)\n", .{ s.name, "aff col3", 1, tc3, macs / (tc3 * 1e-3) / 1e12, bc3 / (tc3 * 1e-3) / 1e9, hc3 & 0xffffffff });
                }
                // THE FAIR 3-BIT AFFINE TILE BODY -- its own 96-byte tile, its own per-expert cold copies, and the
                // warp count swept in the same run (the control is the shipped fma w16 arm above).
                if (pass == 1 and t3_warps_pipes[0] != null) {
                    const tb3: usize = 96;
                    const cbytes3: usize = @as(usize, s.k) * s.n * 3 / 8;
                    const sbytes3: usize = @as(usize, s.rows) * s.n * (s.k / 64) * 2;
                    var cps: [routed_copies * 10]mtl.Buffer = undefined;
                    var cmade: usize = 0;
                    defer for (cps[0..cmade]) |b| b.deinit();
                    while (cmade < cps.len) : (cmade += 1) {
                        cps[cmade] = try device.buffer(cbytes3, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    }
                    var invis3: usize = 0;
                    for (cps[0..cmade], 0..) |b, ci| {
                        if (!fillTrellisGpuVisible(queue, pfill.?, pwords.?, b, @intCast(cbytes3), 0xAFF3C0DE ^ @as(u64, s.n) ^ (@as(u64, ci) << 17), scratch)) invis3 += 1;
                    }
                    if (invis3 > 0) std.debug.print("  {s:>10} WARNING: {d} affine code copies did not become visible\n", .{ s.name, invis3 });
                    const scl3 = try device.buffer(sbytes3, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer scl3.deinit();
                    if (pscale) |ps| {
                        const cb = queue.commandBuffer();
                        const e = cb.compute(.serial);
                        e.setPipeline(ps);
                        e.setBuffer(scl3, 0, 0);
                        const seed: u64 = 0x5CA1E5 ^ @as(u64, s.n);
                        e.setBytes(std.mem.asBytes(&seed), 1);
                        e.dispatchThreads(.{ .width = @intCast(sbytes3 / 2) }, .{ .width = 256 });
                        e.end();
                        cb.commit();
                        cb.wait();
                        if (cb.failure()) |msg| std.debug.print("  affine scales: failed: {s}\n", .{msg});
                    }
                    const adescs = try device.buffer(rows * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer adescs.deinit();
                    {
                        const dp3: [*]Exl3Expert = @ptrCast(@alignCast(adescs.contents()));
                        for (0..rows) |e| dp3[e] = .{ .trellis = cps[e % cmade].gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (s.n / 16) * tb3, .tile_n_stride = tb3, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
                    }
                    const a3args = Exl3Args{
                        .k = s.k,
                        .n = s.n,
                        .rows = s.rows,
                        .halves = s.halves,
                        .codebook = 2,
                        .su_signs = 0,
                        .sv_signs = 0,
                        .has_bias = 0,
                        .output_f32 = 1,
                        .input_ids = 0,
                        .output_ids = 0,
                        .routed = 1,
                        .expert_id = 0,
                        .repeat = 1,
                        .indexed = 1,
                        .experts = s.rows,
                        .splits = 1,
                        .x_stride = s.k,
                        .y_stride = s.n,
                        .tile_k_stride = (s.n / 16) * tb3,
                        .tile_n_stride = tb3,
                    };
                    const b3 = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(s.n)) * @as(f64, @floatFromInt(s.k)) * 3.25 / 8.0;
                    for (t3_warps_pipes, 0..) |maybe, wi| {
                        if (maybe) |p| {
                            @memset(sums.slice(u32, n_out), 0);
                            const t3b = try timeRouted(queue, p, null, a3args, x, sums, partials, scl3, routes, adescs, cps[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 * t3_warps[wi] }, n_out, 15);
                            const h3b = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                            var nm: [24]u8 = undefined;
                            const nm3 = try std.fmt.bufPrint(&nm, "aff t3 fair w{d}", .{t3_warps[wi]});
                            std.debug.print("  {s:>10} {s:>14} {d:>5} {d:>9.3} {d:>9.1} {d:>8.1} {x:>10}  (3-bit affine tile, 96 B, group-64 scales)\n", .{ s.name, nm3, 1, t3b, macs / (t3b * 1e-3) / 1e12, b3 / (t3b * 1e-3) / 1e9, h3b & 0xffffffff });
                        }
                    }

                    const sclb = try device.buffer(2 * sbytes3, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer sclb.deinit();
                    if (pscale) |ps| {
                        const cb = queue.commandBuffer();
                        const e = cb.compute(.serial);
                        e.setPipeline(ps);
                        e.setBuffer(sclb, 0, 0);
                        const seed: u64 = 0x5CA1E5 ^ @as(u64, s.n) ^ 0x8000;
                        e.setBytes(std.mem.asBytes(&seed), 1);
                        e.dispatchThreads(.{ .width = @intCast(sbytes3) }, .{ .width = 256 });
                        e.end();
                        cb.commit();
                        cb.wait();
                    }
                    for (t3_bias_pipes, 0..) |maybe, wi| {
                        if (maybe) |p| {
                            @memset(sums.slice(u32, n_out), 0);
                            const tb = try timeRouted(queue, p, null, a3args, x, sums, partials, sclb, routes, adescs, cps[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 * t3_bias_warps[wi] }, n_out, 15);
                            const hb = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
                            var nmb: [26]u8 = undefined;
                            const nmb3 = try std.fmt.bufPrint(&nmb, "aff t3 bias w{d}", .{t3_bias_warps[wi]});
                            std.debug.print("  {s:>10} {s:>14} {d:>5} {d:>9.3} {d:>9.1} {d:>8.1} {x:>10}  (3-bit affine tile + bias, 96 B, group-64)\n", .{ s.name, nmb3, 1, tb, macs / (tb * 1e-3) / 1e12, b3 / (tb * 1e-3) / 1e9, hb & 0xffffffff });
                        }
                    }
                }
            }
            if (pass == 1) {
                for (arms[0..narms]) |a| {
                    std.debug.print("  {s:>10} {s:>14} {d:>5} {d:>9.3} {d:>9.1} {d:>8.1} {x:>10}\n", .{ s.name, a.name, a.splits, a.ms, macs / (a.ms * 1e-3) / 1e12, gbs / (a.ms * 1e-3) / 1e9, a.hash & 0xffffffff });
                }
            }
        }
        // What the GPU sees of the trellis the bench wrote (the fixture's visibility).
        if (pwords) |pw| {
            const pwout = try device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer pwout.deinit();
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pw);
            e.setBuffer(trs[0], 0, 0);
            e.setBuffer(pwout, 0, 1);
            e.dispatchThreads(.{ .width = 8 }, .{ .width = 1 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| std.debug.print("  {s:>10} probe dispatch failed: {s}\n", .{ s.name, msg });
            const w = pwout.slice(u32, 8);
            const ht = trs[0].slice(u16, 8);
            std.debug.print("  {s:>10} trellis words (host | gpu): {x:0>8}|{x:0>8} {x:0>8}|{x:0>8} {x:0>8}|{x:0>8}\n", .{
                s.name,
                exl3WordHost(ht, 0),
                w[0],
                exl3WordHost(ht, 1),
                w[1],
                exl3WordHost(ht, 2),
                w[2],
            });
        }

        if (std.c.getenv("TF_BENCH_ROUTE_FP64") != null and mma_tf != null and nfma > 0 and fma_arms[0].pipe != null) {
            const host_a = try gpa.alloc(f32, n_out);
            defer gpa.free(host_a);
            const host_b = try gpa.alloc(f32, n_out);
            defer gpa.free(host_b);
            const host_ref = try gpa.alloc(f64, s.n);
            defer gpa.free(host_ref);
            const args = Exl3Args{
                .k = s.k,
                .n = s.n,
                .rows = s.rows,
                .halves = s.halves,
                .codebook = 2,
                .su_signs = 0,
                .sv_signs = 0,
                .has_bias = 0,
                .output_f32 = 1,
                .input_ids = 0,
                .output_ids = 0,
                .routed = 1,
                .expert_id = 0,
                .repeat = 1,
                .indexed = 1,
                .experts = s.rows,
                .splits = 8,
                .x_stride = s.k,
                .y_stride = s.n,
                .tile_k_stride = (s.n / 16) * tile_bytes,
                .tile_n_stride = tile_bytes,
            };
            // The shipped body: the engine's routed default (compact tile, the table, the funnel read).
            @memset(sums.slice(u32, n_out), 0);
            _ = try timeRouted(queue, mma_tf.?, reduce, args, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 32, .height = rows, .depth = 8, .threads = 32 }, n_out, 1);
            @memcpy(host_a, sums.slice(f32, n_out));
            // The register-only body: one k walk, one term at a time.
            var nargs = args;
            nargs.splits = 1;
            // The fp64 truth, row by row, over the same trellis each row reads.
            const xh = x.slice(u16, rows * s.k);
            var worst_a: f64 = 0;
            var worst_b: f64 = 0;
            var sq_a: f64 = 0;
            var sq_b: f64 = 0;
            var sq_ref: f64 = 0;
            var max_a: f64 = 0;
            var max_b: f64 = 0;
            for (0..rows) |r| {
                // Row r's expert is r (routes[r] = r), so row r reads expert r's trellis.
                exl3F64Row(trs[r].slice(u16, trellis_bytes / 2), xh, s.n, s.k, s.halves, r, host_ref);
                for (0..s.n) |j| {
                    const ref = host_ref[j];
                    const va = host_a[r * s.n + j];
                    const vb = host_b[r * s.n + j];
                    const da = @abs(@as(f64, va) - ref);
                    const db = @abs(@as(f64, vb) - ref);
                    worst_a = @max(worst_a, da);
                    worst_b = @max(worst_b, db);
                    max_a = @max(max_a, @abs(@as(f64, va)));
                    max_b = @max(max_b, @abs(@as(f64, vb)));
                    sq_a += da * da;
                    sq_b += db * db;
                    sq_ref += ref * ref;
                }
            }
            {
                // The reference's own states and weights for the first tile's first lane, beside the GPU's.
                const tt = trs[0].slice(u16, trellis_bytes / 2);
                std.debug.print("  {s:>10} ref states p0..7 tile(0,0): ", .{s.name});
                for (0..8) |pp| {
                    const st = exl3StateFor(tt, @intCast(pp), s.halves);
                    std.debug.print("{x:0>4}({d:.3}) ", .{ st, exl3TableValue(st) });
                }
                std.debug.print("\n", .{});
                exl3F64Row(trs[0].slice(u16, trellis_bytes / 2), x.slice(u16, rows * s.k), s.n, s.k, s.halves, 0, host_ref);
                std.debug.print("  {s:>10} row 0 cols 0..5  fp64 {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}\n", .{ s.name, host_ref[0], host_ref[1], host_ref[2], host_ref[3], host_ref[4], host_ref[5] });
                std.debug.print("  {s:>10} row 0 cols 0..5  gpu  {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}\n", .{ s.name, host_a[0], host_a[1], host_a[2], host_a[3], host_a[4], host_a[5] });
                std.debug.print("  {s:>10} row 0 cols 0..5  fma  {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}\n", .{ s.name, host_b[0], host_b[1], host_b[2], host_b[3], host_b[4], host_b[5] });
            }
            std.debug.print("  {s:>10} FP64 RECEIPT: shipped (tensor unit, 16 terms a tree) worst {d:.6} NRMS {e:.3} | |max| {d:.1}\n", .{
                s.name, worst_a, @sqrt(sq_a / @max(sq_ref, 1e-30)), max_a,
            });
            for (fma_arms[0..nfma]) |arm| {
                if (arm.pipe == null) continue;
                if (arm.warps != 1 and arm.warps != 4 and arm.warps != 8 and arm.warps != 16 and arm.warps != 32) continue;
                // The fixture's visibility is re-verified PER ARM.
                if (pwords) |pw| {
                    for (trs[0..rows], 0..) |b, i| {
                        const seed = 0xD1CEB00C ^ @as(u64, s.n) ^ @as(u64, s.k) ^ @as(u64, i);
                        if (pfill) |pf| {
                            if (!fillTrellisGpuVisible(queue, pf, pw, b, trellis_bytes, seed, scratch))
                                std.debug.print("  {s:>10} WARNING: arm {s} trellis copy {d} did not become visible\n", .{ s.name, arm.name, i });
                        } else {
                            fillTrellis(b, trellis_bytes, seed);
                        }
                    }
                }
                @memset(sums.slice(u32, n_out), 0);
                _ = try timeRouted(queue, arm.pipe.?, reduce, nargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 * arm.warps }, n_out, 1);
                @memcpy(host_b, sums.slice(f32, n_out));
                var wb: f64 = 0;
                var sb: f64 = 0;
                var mb: f64 = 0;
                for (0..rows) |r| {
                    exl3F64Row(trs[r].slice(u16, trellis_bytes / 2), xh, s.n, s.k, s.halves, r, host_ref);
                    for (0..s.n) |j| {
                        const ref = host_ref[j];
                        const vb = host_b[r * s.n + j];
                        const db = @abs(@as(f64, vb) - ref);
                        wb = @max(wb, db);
                        mb = @max(mb, @abs(@as(f64, vb)));
                        sb += db * db;
                    }
                }
                const nrms_b = @sqrt(sb / @max(sq_ref, 1e-30));
                const nrms_a = @sqrt(sq_a / @max(sq_ref, 1e-30));
                std.debug.print("  {s:>10} FP64 RECEIPT: {s:>10} worst {d:.6} NRMS {e:.3} | |max| {d:.1} hash {x} -> {s}\n", .{
                    s.name,                                                                                    arm.name, wb, nrms_b, mb, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out))),
                    if (nrms_b <= nrms_a) "NO WORSE than the shipped path" else "worse than the shipped path",
                });
            }
        }

        // Does the routed body read the tile of its OWN column? Zero every tile of the first expert's trellis but
        // the first one, dispatch, and read the row back: the columns that first tile covers must differ from the
        // rest.
        if (fma_arms[0].pipe != null) {
            const bytes = trs[0].slice(u8, trellis_bytes);
            @memset(bytes, 0);
            for (trs[0].slice(u16, tile_bytes / 2), 0..) |*v, i| v.* = @truncate(@as(u64, i) * 2654435761);
            const fargs = Exl3Args{
                .k = s.k,
                .n = s.n,
                .rows = s.rows,
                .halves = s.halves,
                .codebook = 2,
                .su_signs = 0,
                .sv_signs = 0,
                .has_bias = 0,
                .output_f32 = 1,
                .input_ids = 0,
                .output_ids = 0,
                .routed = 1,
                .expert_id = 0,
                .repeat = 1,
                .indexed = 1,
                .experts = s.rows,
                .splits = 1,
                .x_stride = s.k,
                .y_stride = s.n,
                .tile_k_stride = (s.n / 16) * tile_bytes,
                .tile_n_stride = tile_bytes,
            };
            @memset(sums.slice(u32, n_out), 0);
            _ = timeRouted(queue, fma_arms[0].pipe.?, reduce, fargs, x, sums, partials, table, routes, descs, trs[0..], rows, .{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 }, n_out, 1) catch 0;
            const col: []const f32 = sums.slice(f32, n_out);
            std.debug.print("  {s:>10} COLUMN PROBE (only tile 0 of expert 0 has data): row0 cols 0..5 = {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} | cols 16..19 = {d:.2} {d:.2} {d:.2} {d:.2} | col 100 = {d:.2} | row1 col0 = {d:.2}\n", .{
                s.name, col[0], col[1], col[2], col[3], col[4], col[5], col[16], col[17], col[18], col[19], col[100], col[s.n],
            });
        }
    }
}

// The engine's verify dispatch is not that population: one indexed call whose route table names the ROUTED picks
// (four rows x six picks x two matrices) AND the SHARED expert's two matrices four times each (the fused gate/up
// tail fids[48..55], route.metal). Here the routes carry that multiset -- the duplicates point at the SAME
// descriptor and hence the SAME trellis -- so the pair kernel finds its mates, and the shipped and pair bodies are
// timed on the same buffers, in the same run, with the same-binary control.
const VerifyShape = struct { n: u32, k: u32, halves: u32, rows: u32, name: []const u8, down: bool };

const verify_shapes = [_]VerifyShape{
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 56, .name = "gate/up 56 verify", .down = false },
    .{ .n = 5120, .k = 2304, .halves = 6, .rows = 24, .name = "down 24 verify", .down = true },
    // THE STANDING NOVEL PATH: the depth-1 window is two token rows, so its fused gate/up dispatch is
    // 2*2*6 + 2*2 = 28 slots and its down dispatch is 2*6 = 12.
    .{ .n = 2304, .k = 5120, .halves = 6, .rows = 28, .name = "gate/up 28 verify", .down = false },
    .{ .n = 5120, .k = 2304, .halves = 6, .rows = 12, .name = "down 12 verify", .down = true },
};

/// The four verify rows' six picks each from a 24-expert pool: 23 distinct of the 24 routed slots (the
/// record's ~23.4), so one routed collision pair exists.
fn verifyPick(row: usize, p: usize) u32 {
    if (row == 3 and p == 5) return 0;
    return @intCast((row * 6 + p) % 24);
}

fn timeVerifySet(queue: mtl.Queue, pipe: mtl.Pipeline, args: Exl3Args, x: mtl.Buffer, sums: mtl.Buffer, dummy: mtl.Buffer, table: mtl.Buffer, routes: mtl.Buffer, descs: mtl.Buffer, d: RowDispatch) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(table, 0, 1);
    e.setBuffer(sums, 0, 2);
    e.setBytes(std.mem.asBytes(&args), 3);
    e.setBuffer(routes, 0, 4);
    e.setBuffer(descs, 0, 5);
    e.setBuffer(dummy, 0, 6);
    e.setBuffer(table, 0, 7);
    e.dispatchGroups(.{ .width = d.width, .height = d.height, .depth = d.depth }, .{ .width = d.threads });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("  verify multiset: failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1e3;
}

/// The shared expert's gate/up as TWO dense multi-row calls (one matrix each) in one command buffer. The dense body
/// decodes the matrix once for all the window's rows; the indexed call's shared slots decode it once a row.
fn timeDensePair(queue: mtl.Queue, pipe: mtl.Pipeline, args: Exl3Args, x: mtl.Buffer, xoff: usize, xoff_up: usize, sums: mtl.Buffer, sums_off: usize, dummy: mtl.Buffer, table: mtl.Buffer, tr_gate: mtl.Buffer, tr_up: mtl.Buffer, d: RowDispatch) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(x, xoff, 0);
    e.setBuffer(tr_gate, 0, 1);
    e.setBuffer(sums, sums_off, 2);
    e.setBytes(std.mem.asBytes(&args), 3);
    e.setBuffer(dummy, 0, 4);
    e.setBuffer(dummy, 0, 5);
    e.setBuffer(dummy, 0, 6);
    e.setBuffer(table, 0, 7);
    e.dispatchGroups(.{ .width = d.width, .height = d.height, .depth = d.depth }, .{ .width = d.threads });
    e.barrier();
    e.setBuffer(x, xoff_up, 0);
    e.setBuffer(tr_up, 0, 1);
    e.setBuffer(sums, sums_off + @as(usize, args.rows) * @as(usize, args.n) * 4, 2);
    e.dispatchGroups(.{ .width = d.width, .height = d.height, .depth = d.depth }, .{ .width = d.threads });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("  shared split: failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1e3;
}

fn verifyMultisetBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const nax_src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_src);
    const both = try std.fmt.allocPrint(gpa, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n{s}", .{nax_src});
    defer gpa.free(both);
    const lib = try mtl.Library.fromSource(device, both, options);
    defer lib.deinit();
    const ship = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_decode_fma_static_lut", false);
    defer ship.deinit();
    const pair = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_decode_fma_static_pair_lut", false);
    defer pair.deinit();
    // The multi-row dense body for the shared-expert split's two matrices.
    const dense2 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_decode_fma_dense_multi2", false) catch null;
    defer if (dense2) |q| q.deinit();
    const dense4 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_decode_fma_dense_multi4", false) catch null;
    defer if (dense4) |q| q.deinit();
    const table = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer table.deinit();
    for (table.slice(u16, 3 * 65536), 0..) |*v, i| v.* = 0x3c00 | @as(u16, @intCast(i & 0xff));
    {
        const scale: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
        const bias: f64 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));
        for (0..65536) |state| {
            const x: u32 = @truncate(@as(u64, state) * 0x83DCD12D);
            const h: f64 = @floatFromInt(1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24));
            const v: f16 = @floatCast(h * scale + bias);
            table.slice(u16, 3 * 65536)[2 * 65536 + state] = @bitCast(v);
        }
    }

    std.debug.print("\nverify multiset bench (): the pair kernel on the verify's own routes\n", .{});
    std.debug.print("  {s:>18} {s:>6} {s:>6} {s:>6} {s:>9} {s:>9} {s:>9} {s:>8} {s:>10}\n", .{ "shape", "slots", "dist", "saved", "ship ms", "ident ms", "pair ms", "pair", "hash" });
    for (verify_shapes) |s| {
        const tile_bytes = s.halves * 8 * 2;
        const trellis_bytes = (s.k / 16) * (s.n / 16) * tile_bytes;
        const rows: usize = s.rows;
        const n_ids: u32 = @intCast(@max(if (s.down) @as(usize, 25) else 50, rows));
        const routes_h = try gpa.alloc(u32, rows);
        defer gpa.free(routes_h);
        const picks: usize = 6;
        const r: usize = if (s.down) rows / picks else rows / (2 * picks + 2);
        if (!s.down) {
            for (0..r) |row| for (0..picks) |p| {
                routes_h[row * picks + p] = 2 * verifyPick(row, p);
            };
            for (0..r) |row| for (0..picks) |p| {
                routes_h[(r + row) * picks + p] = 2 * verifyPick(row, p) + 1;
            };
            const tail = 2 * r * picks;
            for (0..r) |row| routes_h[tail + row] = 2 * 24;
            for (0..r) |row| routes_h[tail + r + row] = 2 * 24 + 1;
        } else {
            for (0..r) |row| for (0..picks) |p| {
                routes_h[row * picks + p] = verifyPick(row, p);
            };
        }
        // The distinct descriptors the routes name, and the decodes the pair's rank rule can save.
        var seen: [64]bool = undefined;
        @memset(&seen, false);
        var distinct: usize = 0;
        for (routes_h) |v| {
            if (!seen[v]) {
                seen[v] = true;
                distinct += 1;
            }
        }
        var saved: usize = 0;
        for (0..64) |v| {
            const vv: u32 = @intCast(v);
            var c: usize = 0;
            for (routes_h) |w| {
                if (w == vv) c += 1;
            }
            saved += c / 2;
        }
        var trs = try gpa.alloc(mtl.Buffer, 2 * n_ids);
        defer {
            for (trs) |b| b.deinit();
            gpa.free(trs);
        }
        for (trs, 0..) |*b, i| {
            b.* = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            fillTrellis(b.*, trellis_bytes, 0x9E3779B9 ^ @as(u64, s.n) ^ @as(u64, i));
        }
        const descs = try gpa.alloc(mtl.Buffer, 2);
        defer for (descs) |b| b.deinit();
        descs[0] = try device.buffer(n_ids * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        descs[1] = try device.buffer(n_ids * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        for (0..2) |set| {
            const dp: [*]Exl3Expert = @ptrCast(@alignCast(descs[set].contents()));
            for (0..n_ids) |id| dp[id] = .{ .trellis = trs[set * n_ids + id].gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (s.n / 16) * tile_bytes, .tile_n_stride = tile_bytes, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
        }
        // The same-run identity control: every slot its own descriptor (the routed bench's no-reuse model),
        // timed interleaved with the multiset arm so a clock drift lands on both.
        const descs_i = try device.buffer(n_ids * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer descs_i.deinit();
        {
            const dp: [*]Exl3Expert = @ptrCast(@alignCast(descs_i.contents()));
            for (0..n_ids) |id| dp[id] = .{ .trellis = trs[id].gpuAddress(), .suh = 0, .svh = 0, .bias = 0, .tile_k_stride = (s.n / 16) * tile_bytes, .tile_n_stride = tile_bytes, .halves = s.halves, .codebook = 2, .suh_signs = 0, .svh_signs = 0, .has_bias = 0 };
        }
        const routes_i = try device.buffer(rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer routes_i.deinit();
        for (routes_i.slice(u32, rows), 0..) |*v, i| v.* = @intCast(i);
        const x = try device.buffer(rows * s.k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        for (x.slice(u16, rows * s.k), 0..) |*v, i| v.* = 0x3400 | @as(u16, @intCast(i & 0x3ff));
        const sums = try device.buffer(rows * s.n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sums.deinit();
        const dummy = try device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer dummy.deinit();
        const routes = try device.buffer(rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer routes.deinit();
        @memcpy(routes.slice(u32, rows), routes_h);
        const n_out = rows * s.n;
        const args = Exl3Args{
            .k = s.k,
            .n = s.n,
            .rows = s.rows,
            .halves = s.halves,
            .codebook = 2,
            .su_signs = 0,
            .sv_signs = 0,
            .has_bias = 0,
            .output_f32 = 1,
            .input_ids = 0,
            .output_ids = 0,
            .routed = 1,
            .expert_id = 0,
            .repeat = 1,
            .indexed = 1,
            .experts = n_ids,
            .splits = 1,
            .x_stride = s.k,
            .y_stride = s.n,
            .tile_k_stride = (s.n / 16) * tile_bytes,
            .tile_n_stride = tile_bytes,
        };
        const d = RowDispatch{ .width = s.n / 16, .height = rows, .depth = 1, .threads = 32 * 16 };
        // The numerics check: the pair's rows must be bit-identical to the shipped body's (its own comment).
        const host_a = try gpa.alloc(f32, n_out);
        defer gpa.free(host_a);
        const host_b = try gpa.alloc(f32, n_out);
        defer gpa.free(host_b);
        @memset(sums.slice(u32, n_out), 0);
        _ = try timeVerifySet(queue, ship, args, x, sums, dummy, table, routes, descs[0], d);
        @memcpy(host_a, sums.slice(f32, n_out));
        @memset(sums.slice(u32, n_out), 0);
        _ = try timeVerifySet(queue, pair, args, x, sums, dummy, table, routes, descs[0], d);
        @memcpy(host_b, sums.slice(f32, n_out));
        var differing: usize = 0;
        var worst: f64 = 0;
        for (host_a, host_b) |va, vb| {
            if (va != vb) {
                differing += 1;
                worst = @max(worst, @abs(@as(f64, va) - @as(f64, vb)));
            }
        }
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, n_out)));
        // The timing: the two arms interleaved, the two trellis sets alternating, so a clock drift lands on both.
        var ts: [9]f64 = undefined;
        var tp: [9]f64 = undefined;
        for (0..9) |t| {
            ts[t] = try timeVerifySet(queue, ship, args, x, sums, dummy, table, routes, descs[t % 2], d);
            tp[t] = try timeVerifySet(queue, pair, args, x, sums, dummy, table, routes, descs[t % 2], d);
        }
        var ti: [5]f64 = undefined;
        var tm: [5]f64 = undefined;
        for (0..5) |t| {
            tm[t] = try timeVerifySet(queue, ship, args, x, sums, dummy, table, routes, descs[t % 2], d);
            ti[t] = try timeVerifySet(queue, ship, args, x, sums, dummy, table, routes_i, descs_i, d);
        }
        const t_ship = median(&ts);
        const t_pair = median(&tp);
        const t_ident = median(&ti);
        std.debug.print("  {s:>18} {d:>6} {d:>6} {d:>6} {d:>9.3} {d:>9.3} {d:>9.3} {d:>7.1}% {d:>10}  ({d} of {d} differ, worst {e:.2})\n", .{
            s.name, rows, distinct, saved, t_ship, t_ident, t_pair, 100.0 * (t_pair - t_ship) / t_ship, h & 0xffffffff, differing, n_out, worst,
        });
        // THE SHARED EXPERT'S SLOTS OUT OF THE FUSED INDEXED CALL, priced in the SAME interleaved run as the
        // shipped call (The watch-out: an arm timed in another block is another cache regime). The fused call's
        // tail names each shared matrix once a token row; the split runs the routed-only call plus two dense
        // multi-row calls that decode each matrix ONCE for the window.
        if (!s.down) {
            const r_tok: usize = rows / (2 * picks + 2);
            const routed_slots: usize = 2 * r_tok * picks;
            var ar = args;
            ar.rows = @intCast(routed_slots);
            const dr = RowDispatch{ .width = s.n / 16, .height = routed_slots, .depth = 1, .threads = 32 * 16 };
            var ad = args;
            ad.indexed = 0;
            ad.routed = 0;
            ad.experts = 1;
            ad.rows = @intCast(r_tok);
            const dd = RowDispatch{ .width = s.n / 16, .height = 1, .depth = 1, .threads = 32 * 16 };
            const sd: usize = 2 * 24;
            const pd = (if (r_tok == 4) dense4 else dense2).?;
            var tfv: [9]f64 = undefined;
            var tq: [9]f64 = undefined;
            var td: [9]f64 = undefined;
            for (0..9) |t| {
                tfv[t] = try timeVerifySet(queue, ship, args, x, sums, dummy, table, routes, descs[t % 2], d);
                tq[t] = try timeVerifySet(queue, ship, ar, x, sums, dummy, table, routes, descs[t % 2], dr);
                td[t] = try timeDensePair(queue, pd, ad, x, routed_slots * @as(usize, s.k) * 2, (routed_slots + r_tok) * @as(usize, s.k) * 2, sums, 0, dummy, table, trs[sd], trs[sd + 1], dd);
            }
            const t_full = median(&tfv);
            const t_routed = median(&tq);
            const t_dense = median(&td);
            std.debug.print("  {s:>18} full {d:>7.3} | routed {d:>6.3} + dense {d:>6.3} = {d:>6.3}  ({d:>6.1}%)\n", .{
                s.name, t_full, t_routed, t_dense, t_routed + t_dense, 100.0 * (t_routed + t_dense - t_full) / t_full,
            });
        }
    }
}

fn exl3Bench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    const src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(src);
    var owned: ?[]u8 = null;
    var text: []const u8 = src;
    if (!device.tensorUnits()) {
        owned = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", src });
        text = owned.?;
    }
    defer if (owned) |o| gpa.free(o);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p3 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax3", false);
    defer p3.deinit();
    const p5 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax5", false);
    defer p5.deinit();
    const p37 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax37", false) catch null;
    defer if (p37) |q| q.deinit();
    // The occupancy-keeping form (nax61), its control (nax63) and the activation ablation (nax62).
    const p61 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax61", false) catch null;
    defer if (p61) |q| q.deinit();
    const p62 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax62", false) catch null;
    defer if (p62) |q| q.deinit();
    const p63 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax63", false) catch null;
    defer if (p63) |q| q.deinit();
    // The SHIPPED body (nax48) is the control the form must beat.
    const p48 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax48", false) catch null;
    defer if (p48) |q| q.deinit();
    // The ENGINE's routed body (nax59, the funnel path as its only path -- The landing) and the stream-read arms
    // the bench never timed (nax57, the 32-bit word; nax53, the cooperative read).
    const p59 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax59", false) catch null;
    defer if (p59) |q| q.deinit();
    const p57 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax57", false) catch null;
    defer if (p57) |q| q.deinit();
    const p53 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax53", false) catch null;
    defer if (p53) |q| q.deinit();
    // The store path's modes on the engine's body.
    const p69 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax69", false) catch null;
    defer if (p69) |q| q.deinit();
    const p70 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax70", false) catch null;
    defer if (p70) |q| q.deinit();
    const p71 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax71", false) catch null;
    defer if (p71) |q| q.deinit();
    const p72 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax72", false) catch null;
    defer if (p72) |q| q.deinit();
    const p73 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax73", false) catch null;
    // The 128-row x 64-column tile (RG=8, CG=1) -- the tile-height lever, at the plan's own shape.
    const p49 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax49", false) catch null;
    defer if (p49) |q| q.deinit();
    // THE EXPERT'S SLAB DECODED ONCE FOR THE TILE'S ROWS -- nax76 (128 rows, two row groups a warp, four warps) and
    // nax77 (256 rows, four row groups a warp, four warps): the padding's term, addressed.
    const p76 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax76", false) catch null;
    defer if (p76) |q| q.deinit();
    const p77 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax77", false) catch null;
    defer if (p77) |q| q.deinit();
    defer if (p73) |q| q.deinit();
    // The routed tile's store path, priced.
    const p64 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax64", false) catch null;
    defer if (p64) |q| q.deinit();
    const p65 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax65", false) catch null;
    defer if (p65) |q| q.deinit();
    const p66 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax66", false) catch null;
    defer if (p66) |q| q.deinit();
    const p67 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax67", false) catch null;
    defer if (p67) |q| q.deinit();
    const p68 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax68", false) catch null;
    defer if (p68) |q| q.deinit();
    const p7 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax7", false) catch null;
    defer if (p7) |q| q.deinit();
    const p8 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax8", false) catch null;
    defer if (p8) |q| q.deinit();
    const p9 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax9", false) catch null;
    defer if (p9) |q| q.deinit();
    const p10 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax10", false) catch null;
    defer if (p10) |q| q.deinit();
    const p11 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax11", false) catch null;
    defer if (p11) |q| q.deinit();
    const p12 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax12", false) catch null;
    defer if (p12) |q| q.deinit();
    const p13 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax13", false) catch null;
    defer if (p13) |q| q.deinit();
    const p14 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax14", false) catch null;
    defer if (p14) |q| q.deinit();
    const p15 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax15", false) catch null;
    defer if (p15) |q| q.deinit();
    const p16 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax16", false) catch null;
    defer if (p16) |q| q.deinit();
    const p17 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax17", false) catch null;
    defer if (p17) |q| q.deinit();
    const p18 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax18", false) catch null;
    defer if (p18) |q| q.deinit();
    const p19 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax19", false) catch null;
    defer if (p19) |q| q.deinit();
    const p20 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax20", false) catch null;
    defer if (p20) |q| q.deinit();
    const p21 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax21", false) catch null;
    defer if (p21) |q| q.deinit();
    // The funnel read in the prompt body (the comparator's structure) and the no-value diagnostic.
    const p40 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax40", false) catch null;
    defer if (p40) |q| q.deinit();
    const p41 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax41", false) catch null;
    defer if (p41) |q| q.deinit();
    // The funnel's extraction in 32-bit words (the comparator's dig_dec_job structure).
    const p50 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax50", false) catch null;
    defer if (p50) |q| q.deinit();
    // The cooperative read (one lane a word) and the shuffle.
    const p52 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax52", false) catch null;
    defer if (p52) |q| q.deinit();
    // The funnel's value path -- the resident table, the threadgroup LUT, the cheap-value diagnostic -- and the
    // double-buffered body with the funnel.
    const p43 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax43", false) catch null;
    defer if (p43) |q| q.deinit();
    const p44 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax44", false) catch null;
    defer if (p44) |q| q.deinit();
    const p46 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax46", false) catch null;
    defer if (p46) |q| q.deinit();
    const p47 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax47", false) catch null;
    defer if (p47) |q| q.deinit();
    const p22 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax22", false) catch null;
    defer if (p22) |q| q.deinit();
    const p23 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax23", false) catch null;
    defer if (p23) |q| q.deinit();
    const p24 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax24", false) catch null;
    defer if (p24) |q| q.deinit();
    const p25 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax25", false) catch null;
    defer if (p25) |q| q.deinit();
    const p2 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax2", false) catch null;
    defer if (p2) |q| q.deinit();
    const p26 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax26", false) catch null;
    defer if (p26) |q| q.deinit();
    const p27 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax27", false) catch null;
    defer if (p27) |q| q.deinit();
    const p28 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax28", false) catch null;
    defer if (p28) |q| q.deinit();
    const p29 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax29", false) catch null;
    defer if (p29) |q| q.deinit();
    const p30 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax30", false) catch null;
    defer if (p30) |q| q.deinit();
    const p31 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax31", false) catch null;
    defer if (p31) |q| q.deinit();
    // The double-buffered 128-column dense body (table and ALU).
    const p79 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax79", false) catch null;
    defer if (p79) |q| q.deinit();
    const p80 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax80", false) catch null;
    defer if (p80) |q| q.deinit();
    const p82 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax82", false) catch null;
    defer if (p82) |q| q.deinit();
    const p83 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax83", false) catch null;
    defer if (p83) |q| q.deinit();
    // The comparator's routed prefill geometry (BM 128, BN 64, BK 64, 8 SG x 32x32).
    const p84 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax84", false) catch null;
    defer if (p84) |q| q.deinit();
    // The dense 128x64 tile with a simdgroup that owns thirty-two rows (the comparator's NAXTile shape): the lane
    // keeps eight states and the weight fragments are read once for two row groups.
    const p56 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax56", false) catch null;
    defer if (p56) |q| q.deinit();
    const p32 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax32", false) catch null;
    defer if (p32) |q| q.deinit();
    const p33 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax33", false) catch null;
    defer if (p33) |q| q.deinit();
    const p34 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax34", false) catch null;
    defer if (p34) |q| q.deinit();
    const p35 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax35", false) catch null;
    defer if (p35) |q| q.deinit();
    const p36 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax36", false) catch null;
    defer if (p36) |q| q.deinit();
    // The dense funnel body (nax51), timed against nax31 at the tap's shape and width.
    if (std.c.getenv("TF_BENCH_TAP") != null) {
        const p51 = mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax51", false) catch null;
        defer if (p51) |q| q.deinit();
        if (p31 == null or p51 == null) return error.NoTapPipeline;
        try benchTapFunnel(device, queue, p31.?, p51.?, 128, 5120, 15360, 8);
        try benchTapFunnel(device, queue, p31.?, p51.?, 128, 5120, 15360, 6);
        return;
    }

    // 384 experts' trellis, one buffer each so the descriptor's device pointer is the expert's own allocation.
    var experts: [exl3_tiles]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (experts[0..made]) |b| b.deinit();
    while (made < exl3_tiles) : (made += 1) {
        experts[made] = try device.buffer(exl3_expert_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        // Realistic stream: a full-range pseudo-random word per slot, not one repeated state (a repeated state
        // makes the codebook table cache-hot and prices the table wrong).
        var rng: u64 = 0x9E3779B97F4A7C15 ^ @as(u64, made);
        for (experts[made].slice(u16, exl3_expert_bytes / 2)) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = @truncate(rng);
        }
    }
    const x = try device.buffer(@as(usize, exl3_rows) * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    @memset(x.slice(u16, @as(usize, exl3_rows) * exl3_k), 0x3c00); // bf16 1.0
    const sums = try device.buffer(@as(usize, exl3_rows) * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    @memset(sums.slice(u32, @as(usize, exl3_rows) * exl3_n), 0);
    const routes = try device.buffer(@as(usize, exl3_rows) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer routes.deinit();
    for (routes.slice(u32, exl3_rows), 0..) |*r, i| r.* = @intCast(i / 64);
    const descs = try device.buffer(exl3_tiles * @sizeOf(Exl3Expert), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer descs.deinit();
    const table = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer table.deinit();
    {
        const tw: [*]u16 = @ptrCast(@alignCast(table.contents()));
        for (0..3) |cb| for (0..65536) |st| {
            const e: tf.exl3_format.Codebook = @fromBackingInt(@intCast(cb));
            tw[cb * 65536 + st] = @bitCast(tf.exl3_format.codebookValue(e, @intCast(st)));
        };
    }
    {
        const d: [*]Exl3Expert = @ptrCast(@alignCast(descs.contents()));
        for (0..exl3_tiles) |i| d[i] = .{
            .trellis = experts[i].gpuAddress(),
            .suh = 0,
            .svh = 0,
            .bias = 0,
            .tile_k_stride = exl3_k_stride,
            .tile_n_stride = exl3_n_stride,
            .halves = exl3_halves,
            .codebook = 2,
            .suh_signs = 0,
            .svh_signs = 0,
            .has_bias = 0,
        };
    }
    var args = Exl3Args{
        .k = exl3_k,
        .n = exl3_n,
        .rows = exl3_rows,
        .halves = exl3_halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 2,
        .experts = exl3_tiles,
        .splits = 1,
        .x_stride = exl3_k,
        .y_stride = exl3_n,
        .tile_k_stride = exl3_k_stride,
        .tile_n_stride = exl3_n_stride,
    };
    const macs = @as(f64, exl3_rows) * @as(f64, exl3_n) * @as(f64, exl3_k);
    std.debug.print("exl3 bench: {d} tiles x 64 rows, N {d}, K {d}, halves {d}, indexed grouped\n", .{ exl3_tiles, exl3_n, exl3_k, exl3_halves });
    std.debug.print("  {s:>8} {s:>9} {s:>9}\n", .{ "kernel", "ms", "TMAC/s" });
    // The 64-row plan on the shipped body (nax59) is the bit-identity reference; nax76 is the same-run control (the
    // record's 128-row form).
    if (std.c.getenv("TF_BENCH_CMP") != null) {
        if (p84 == null or p59 == null) return error.NoCmpPipeline;
        var off64: [exl3_tiles]u32 = undefined;
        const rows64 = planOffsets(&plan_counts, 64, &off64);
        var off128: [exl3_tiles]u32 = undefined;
        const rows128 = planOffsets(&plan_counts, 128, &off128);
        var tiles128: usize = 0;
        for (plan_counts) |c| tiles128 += (c + 127) / 128;
        var live: usize = 0;
        for (plan_counts) |c| live += c;
        const xb64 = try device.buffer(rows64 * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer xb64.deinit();
        @memset(xb64.slice(u16, rows64 * exl3_k), 0x3c00);
        const sb64 = try device.buffer(rows64 * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sb64.deinit();
        @memset(sb64.slice(u32, rows64 * exl3_n), 0);
        const rt64 = try device.buffer(rows64 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer rt64.deinit();
        {
            const rr = rt64.slice(u32, rows64);
            @memset(rr, 0xffffffff);
            for (plan_counts, 0..) |c, i| {
                const base: usize = off64[i];
                for (0..c) |j| rr[base + j] = @intCast(i);
            }
        }
        const xb128 = try device.buffer(rows128 * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer xb128.deinit();
        @memset(xb128.slice(u16, rows128 * exl3_k), 0x3c00);
        const sb128 = try device.buffer(rows128 * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sb128.deinit();
        @memset(sb128.slice(u32, rows128 * exl3_n), 0);
        const rt128 = try device.buffer(rows128 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer rt128.deinit();
        {
            const rr = rt128.slice(u32, rows128);
            @memset(rr, 0xffffffff);
            for (plan_counts, 0..) |c, i| {
                const base: usize = off128[i];
                for (0..c) |j| rr[base + j] = @intCast(i);
            }
        }
        // If this differs the body is wrong; if it matches, the plan comparison's only suspect is its reference.
        {
            var dargs = args;
            dargs.rows = 128;
            dargs.indexed = 0;
            dargs.expert_id = 0;
            dargs.experts = 0;
            @memset(sums.slice(u32, 128 * exl3_n), 0);
            _ = try timeShape(queue, p31.?, &dargs, x, experts[0], sums, table, 128, exl3_n, 64, 256, 128, 1);
            const dms31 = try timeShape(queue, p31.?, &dargs, x, experts[0], sums, table, 128, exl3_n, 64, 256, 128, 5);
            const dh31 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, 128 * exl3_n)));
            const dv31 = sums.slice(f32, 128 * exl3_n)[0];
            @memset(sums.slice(u32, 128 * exl3_n), 0);
            _ = try timeShape(queue, p84.?, &dargs, x, experts[0], sums, table, 128, exl3_n, 64, 256, 128, 1);
            const dms84 = try timeShape(queue, p84.?, &dargs, x, experts[0], sums, table, 128, exl3_n, 64, 256, 128, 5);
            const dh84 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, 128 * exl3_n)));
            const dv84 = sums.slice(f32, 128 * exl3_n)[0];
            std.debug.print("  dense control 128x2304x5120: nax31 {d:.3} ms {x:0>16} v0 {d:.7} | nax84 {d:.3} ms {x:0>16} v0 {d:.7} {s}\n", .{ dms31, dh31, dv31, dms84, dh84, dv84, if (dh31 == dh84) "IDENTICAL" else "DIFFER" });
        }
        var args64 = args;
        args64.rows = @intCast(rows64);
        var args128 = args;
        args128.rows = @intCast(rows128);
        const lm = @as(f64, @floatFromInt(live)) * @as(f64, exl3_n) * @as(f64, exl3_k);
        const pm128 = @as(f64, @floatFromInt(rows128)) * @as(f64, exl3_n) * @as(f64, exl3_k);
        std.debug.print("\ncomparator routed prefill geometry: N {d} K {d}, live rows {d}; 64-row plan {d} rows; 128-row plan {d} rows ({d} tiles)\n", .{ exl3_n, exl3_k, live, rows64, rows128, tiles128 });
        _ = try timeRows(queue, p59.?, &args64, xb64, experts[0], sb64, rt64, descs, table, 64, 128, rows64, 1);
        const ms64 = try timeRows(queue, p59.?, &args64, xb64, experts[0], sb64, rt64, descs, table, 64, 128, rows64, 5);
        const h64 = hashLiveRows(sb64, rows64 * exl3_n, &plan_counts, &off64, exl3_n);
        std.debug.print("  {s:>8} {d:>9.3} ms {d:>6.1} TMAC/s live ({d:>6.1} padded)  {x:>16}\n", .{ "nax59/64", ms64, lm / (ms64 * 1e-3) / 1e12, pm128 / (ms64 * 1e-3) / 1e12, h64 });
        _ = try timeRowsT(queue, p84.?, &args128, xb128, experts[0], sb128, rt128, descs, table, 64, 256, rows128, 128, 1);
        const ms84 = try timeRowsT(queue, p84.?, &args128, xb128, experts[0], sb128, rt128, descs, table, 64, 256, rows128, 128, 5);
        const h84 = hashLiveRows(sb128, rows128 * exl3_n, &plan_counts, &off128, exl3_n);
        std.debug.print("  {s:>8} {d:>9.3} ms {d:>6.1} TMAC/s live ({d:>6.1} padded)  {x:>16}  {s} vs nax59, {s}{d:.1} % of the 64-row plan\n", .{ "nax84", ms84, lm / (ms84 * 1e-3) / 1e12, pm128 / (ms84 * 1e-3) / 1e12, h84, if (h84 == h64) "IDENTICAL" else "DIFFER", if (ms84 >= ms64) "+" else "", 100.0 * (ms84 - ms64) / ms64 });
        std.debug.print("  the 128-row plan (nax84) against the 64-row plan (nax59), row for row:\n", .{});
        compareLiveRows(sb64, rows64 * exl3_n, &plan_counts, &off64, sb128, rows128 * exl3_n, &off128, exl3_n);
        if (p76) |pipe6| {
            _ = try timeRowsT(queue, pipe6, &args128, xb128, experts[0], sb128, rt128, descs, table, 64, 128, rows128, 128, 1);
            const ms76 = try timeRowsT(queue, pipe6, &args128, xb128, experts[0], sb128, rt128, descs, table, 64, 128, rows128, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} ms {d:>6.1} TMAC/s live  (same-run control: the record's 128-row form; nax84 vs it {s}{d:.1} %)\n", .{ "nax76", ms76, lm / (ms76 * 1e-3) / 1e12, if (ms84 >= ms76) "+" else "", 100.0 * (ms84 - ms76) / ms76 });
        }
        return;
    }

    // warm the pipelines once, then median of five.
    _ = try timeExl3(queue, p3, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
    const ms3 = try timeExl3(queue, p3, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
    std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}\n", .{ "nax3", ms3, macs / (ms3 * 1e-3) / 1e12 });
    const hash3 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
    var hash5: u64 = 0;
    {
        _ = try timeExl3(queue, p5, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms5 = try timeExl3(queue, p5, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (engine's CG=1 baseline)\n", .{ "nax5", ms5, macs / (ms5 * 1e-3) / 1e12 });
        hash5 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax3 {x:0>16} nax5 {x:0>16} {s}\n", .{ hash3, hash5, if (hash3 == hash5) "IDENTICAL" else "DIFFER" });
    }
    // The routed tile's activation term -- the shipped body, the occupancy-keeping form against its control, and
    // the ablation that prices the activation's per-block read.
    if (p48) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms48 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the SHIPPED body, nax48)\n", .{ "nax48", ms48, macs / (ms48 * 1e-3) / 1e12 });
    }
    // The routed tile's store path, priced. The SHIPPED body is the control (nax48).
    var hash48: u64 = 0;
    if (p48) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms48b = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        hash48 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (control: the SHIPPED body, nax48)\n", .{ "nax48", ms48b, macs / (ms48b * 1e-3) / 1e12 });
    }
    // THE PLAN'S OWN SHAPE. The engine's grouped plan at the prompt's shape: 384 experts' blocks, each padded to a
    // multiple of the tile's sixty-four rows, the blocks laid out consecutively, the plan's tail unused. The weight
    // decode is PER TILE, so the plan's padding is a direct multiplier on the routed tile's biggest term -- and the
    // plan's tile count is the one number the record has never had. Three arms: A: the plan's shape (the dispatched
    // tiles, the padding, the live rows) -- the engine's own geometry; B: the same live rows packed into the
    // minimum number of tiles (the padding's price = A - B); C: the plan's shape with the activation's read hoisted
    // out of the k loop (the ablation) -- the activation's share AT THE PLAN'S SHAPE.
    {
        const live_rows: usize = blk: {
            var s2: usize = 0;
            for (plan_counts) |c| s2 += c;
            break :blk s2;
        };
        // The GPU's grouping: each expert's block starts at its offset and is padded to a multiple of the tile.
        var offsets: [exl3_tiles]u32 = undefined;
        var used_rows: usize = 0;
        for (plan_counts, 0..) |c, i| {
            offsets[i] = @intCast(used_rows);
            used_rows += ((c + 63) / 64) * 64;
        }
        const plan_rows: usize = used_rows; // the rows the plan actually uses (the tail beyond it is unused)
        const pad_rows: usize = 685 * 64; // the plan's dispatched rows (the host's formula: 685 tiles)
        var live_tiles: usize = 0;
        for (plan_counts) |c| live_tiles += (c + 63) / 64;
        std.debug.print("\nplan shape: live_rows {d} used_rows {d} (padded tiles {d}) live_tiles {d} packed_tiles {d}\n", .{ live_rows, plan_rows, pad_rows / 64, live_tiles, (live_rows + 63) / 64 });
        // The routes: the expert id on a live row, the padding sentinel on the rest (the engine's own sentinel).
        const px = try device.buffer(plan_rows * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer px.deinit();
        @memset(px.slice(u16, plan_rows * exl3_k), 0x3c00);
        const psums = try device.buffer(plan_rows * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer psums.deinit();
        @memset(psums.slice(u32, plan_rows * exl3_n), 0);
        const proutes = try device.buffer(plan_rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer proutes.deinit();
        {
            const rr = proutes.slice(u32, plan_rows);
            @memset(rr, 0xffffffff);
            for (plan_counts, 0..) |c, i| {
                const base: usize = offsets[i];
                for (0..c) |j| rr[base + j] = @intCast(i);
            }
        }
        var pargs = args;
        pargs.rows = @intCast(plan_rows);
        // The packed arm: the same live rows, no padding (the minimum number of tiles).
        const kx = try device.buffer(live_rows * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer kx.deinit();
        @memset(kx.slice(u16, live_rows * exl3_k), 0x3c00);
        const ksums = try device.buffer(live_rows * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer ksums.deinit();
        @memset(ksums.slice(u32, live_rows * exl3_n), 0);
        const kroutes = try device.buffer(live_rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer kroutes.deinit();
        {
            const rr = kroutes.slice(u32, live_rows);
            for (rr, 0..) |*r, i| r.* = @intCast(i / 64);
        }
        var kargs = args;
        kargs.rows = @intCast(live_rows);
        const macs_live = @as(f64, @floatFromInt(live_rows)) * @as(f64, exl3_n) * @as(f64, exl3_k);
        if (p59) |pipe| {
            _ = try timeRows(queue, pipe, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 1);
            const msA = try timeRows(queue, pipe, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (A: THE PLAN'S SHAPE -- {d} tiles, {d} live rows of {d})\n", .{ "nax59", msA, macs_live / (msA * 1e-3) / 1e12, live_tiles, live_rows, plan_rows });
            const hashA = hashLiveRows(psums, plan_rows * exl3_n, &plan_counts, &offsets, exl3_n);
            std.debug.print("  {s:>8} {x:>16}  (the 64-row plan-s live rows after nax59)\n", .{ "HASH59", hashA });
            {
                const xv = px.slice(u16, plan_rows * exl3_k);
                std.debug.print("  the plan-s activation buffer: u16[0]={x} u16[1]={x} u16[5120]={x} u16[327680]={x} u16[655360]={x}\n", .{ xv[0], xv[1], xv[5120], xv[64 * 5120], xv[128 * 5120] });
            }
            const msB = try timeRows(queue, pipe, &kargs, kx, experts[0], ksums, kroutes, descs, table, 64, 128, live_rows, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (B: the same live rows PACKED, {d} tiles -- the padding's price is A - B = {d:.3} ms, {d:.1} %)\n", .{ "nax59", msB, macs_live / (msB * 1e-3) / 1e12, (live_rows + 63) / 64, msA - msB, 100.0 * (msA - msB) / msA });
            if (p62) |pipe2| {
                _ = try timeRows(queue, pipe2, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 1);
                const msC = try timeRows(queue, pipe2, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (C: the ABLATION -- the activation's read at the plan's shape = {d:.3} ms = {d:.1} % of A)\n", .{ "nax62", msC, macs_live / (msC * 1e-3) / 1e12, msA - msC, 100.0 * (msA - msC) / msA });
            }
            if (p73) |pipe3| {
                _ = try timeRows(queue, pipe3, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 1);
                const msD = try timeRows(queue, pipe3, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (D: the extraction dropped at the plan's shape = {d:.3} ms = {d:.1} % of A)\n", .{ "nax73", msD, macs_live / (msD * 1e-3) / 1e12, msA - msD, 100.0 * (msA - msD) / msA });
            }
            // THE TILE-HEIGHT LEVER AT THE PLAN'S SHAPE. The body is nax49 (RG=8, CG=1): the same funnel +
            // padded-rows-skipped family as nax48, with the tile one hundred and twenty-eight rows tall.
            if (p49) |pipe4| {
                var off128: [exl3_tiles]u32 = undefined;
                var used128: usize = 0;
                var tiles128: usize = 0;
                for (plan_counts, 0..) |c, i| {
                    off128[i] = @intCast(used128);
                    used128 += ((c + 127) / 128) * 128;
                    tiles128 += (c + 127) / 128;
                }
                const rows128: usize = used128;
                const x128p = try device.buffer(rows128 * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer x128p.deinit();
                @memset(x128p.slice(u16, rows128 * exl3_k), 0x3c00);
                const sums128p = try device.buffer(rows128 * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer sums128p.deinit();
                @memset(sums128p.slice(u32, rows128 * exl3_n), 0);
                const routes128p = try device.buffer(rows128 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer routes128p.deinit();
                {
                    const rr = routes128p.slice(u32, rows128);
                    @memset(rr, 0xffffffff);
                    for (plan_counts, 0..) |c, i| {
                        const base: usize = off128[i];
                        for (0..c) |j| rr[base + j] = @intCast(i);
                    }
                }
                var args128p = args;
                args128p.rows = @intCast(rows128);
                std.debug.print("  {s:>8} {d:>9} {d:>9}  (the 128-row plan: {d} working tiles, {d} rows)\n", .{ "PLAN128", @as(u64, 0), @as(u64, 0), tiles128, rows128 });
                _ = try timeRows(queue, pipe4, &args128p, x128p, experts[0], sums128p, routes128p, descs, table, 64, 256, rows128, 1);
                const msE = try timeRows(queue, pipe4, &args128p, x128p, experts[0], sums128p, routes128p, descs, table, 64, 256, rows128, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (E: the 128-row tile at the 128-row plan -- against the 64-row plan's {d:.3} ms: {d:.1} %)\n", .{ "nax49", msE, macs_live / (msE * 1e-3) / 1e12, msA, 100.0 * (msE - msA) / msA });
                if (p48) |pipe5| {
                    _ = try timeRows(queue, pipe5, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 1);
                    const msF = try timeRows(queue, pipe5, &pargs, px, experts[0], psums, proutes, descs, table, 64, 128, plan_rows, 5);
                    std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (F: the SAME BODY FAMILY at 64 rows (nax48) -- the tile height-s price = {d:.1} %)\n", .{ "nax48", msF, macs_live / (msF * 1e-3) / 1e12, 100.0 * (msE - msF) / msF });
                }
                // Does the EXISTING 128-row tile (nax49) give the same rows as the 64-row plan? The question the
                // new forms raise: is a plan's row offset part of the result?
                {
                    const h49 = hashLiveRows(sums128p, rows128 * exl3_n, &plan_counts, &off128, exl3_n);
                    std.debug.print("  {s:>8} {x:>16}  (the 128-row plan-s live rows after nax49 -- the EXISTING 128-row tile)\n", .{ "HASH49", h49 });
                    compareLiveRows(psums, plan_rows * exl3_n, &plan_counts, &offsets, sums128p, rows128 * exl3_n, &off128, exl3_n);
                }
                // Arm E re-run with the CORRECT dispatch (the sixty-four-row helper dispatched twice the tiles the
                // 128-row plan needs, so half of them were empty).
                if (p49) |pipe4b| {
                    _ = try timeRowsT(queue, pipe4b, &args128p, x128p, experts[0], sums128p, routes128p, descs, table, 64, 256, rows128, 128, 1);
                    const msE2 = try timeRowsT(queue, pipe4b, &args128p, x128p, experts[0], sums128p, routes128p, descs, table, 64, 256, rows128, 128, 5);
                    std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (E2: nax49, the 128-row tile, the CORRECT dispatch ({d} tiles) -- against the 64-row plan-s {d:.3} ms: {d:.1} %)\n", .{ "nax49", msE2, macs_live / (msE2 * 1e-3) / 1e12, tiles128, msA, 100.0 * (msE2 - msA) / msA });
                }
            }
            // THE EXPERT'S SLAB DECODED ONCE FOR THE TILE-S ROWS -- the padding-s term, addressed. The ideal (one
            // decode an expert) is 359 tiles; the packed arm (B, 308 tiles) is the ceiling.
            {
                const hash64 = hashLiveRows(psums, plan_rows * exl3_n, &plan_counts, &offsets, exl3_n);
                std.debug.print("  {s:>8} {x:>16}  (the 64-row plan-s LIVE ROWS (the exactness reference for the new forms))\n", .{ "HASH64", hash64 });
                // G: the 128-row plan with nax76.
                var off128: [exl3_tiles]u32 = undefined;
                const rows128b = planOffsets(&plan_counts, 128, &off128);
                var tiles128b: usize = 0;
                for (plan_counts) |c| tiles128b += (c + 127) / 128;
                const x128b = try device.buffer(rows128b * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer x128b.deinit();
                @memset(x128b.slice(u16, rows128b * exl3_k), 0x3c00);
                const sums128b = try device.buffer(rows128b * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer sums128b.deinit();
                @memset(sums128b.slice(u32, rows128b * exl3_n), 0);
                const routes128b = try device.buffer(rows128b * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer routes128b.deinit();
                {
                    const rr = routes128b.slice(u32, rows128b);
                    @memset(rr, 0xffffffff);
                    for (plan_counts, 0..) |c, i| {
                        const base: usize = off128[i];
                        for (0..c) |j| rr[base + j] = @intCast(i);
                    }
                }
                var args128b = pargs;
                args128b.rows = @intCast(rows128b);
                if (p76) |pipe6| {
                    _ = try timeRowsT(queue, pipe6, &args128b, x128b, experts[0], sums128b, routes128b, descs, table, 64, 128, rows128b, 128, 1);
                    const msG = try timeRowsT(queue, pipe6, &args128b, x128b, experts[0], sums128b, routes128b, descs, table, 64, 128, rows128b, 128, 5);
                    const hashG = hashLiveRows(sums128b, rows128b * exl3_n, &plan_counts, &off128, exl3_n);
                    std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (G: nax76 -- the slab decoded once for 128 rows, {d} tiles (24 % less decode); against the 64-row plan-s {d:.3} ms: {d:.1} %; the live rows {x:0>16} {s})\n", .{ "nax76", msG, macs_live / (msG * 1e-3) / 1e12, tiles128b, msA, 100.0 * (msG - msA) / msA, hashG, if (hashG == hash64) "IDENTICAL" else "DIFFER" });
                    std.debug.print("  the 128-row plan (nax76) against the 64-row plan (nax48), row for row:\n", .{});
                    compareLiveRows(psums, plan_rows * exl3_n, &plan_counts, &offsets, sums128b, rows128b * exl3_n, &off128, exl3_n);
                    std.debug.print("  the 64-row plan (nax48), expert 0 and expert 1:\n", .{});
                    std.debug.print("  the 128-row plan (nax76), expert 0 and expert 1:\n", .{});
                }
                // H: the 256-row plan with nax77.
                var off256: [exl3_tiles]u32 = undefined;
                const rows256 = planOffsets(&plan_counts, 256, &off256);
                var tiles256: usize = 0;
                for (plan_counts) |c| tiles256 += (c + 255) / 256;
                const x256 = try device.buffer(rows256 * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer x256.deinit();
                @memset(x256.slice(u16, rows256 * exl3_k), 0x3c00);
                const sums256 = try device.buffer(rows256 * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer sums256.deinit();
                @memset(sums256.slice(u32, rows256 * exl3_n), 0);
                const routes256 = try device.buffer(rows256 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer routes256.deinit();
                {
                    const rr = routes256.slice(u32, rows256);
                    @memset(rr, 0xffffffff);
                    for (plan_counts, 0..) |c, i| {
                        const base: usize = off256[i];
                        for (0..c) |j| rr[base + j] = @intCast(i);
                    }
                }
                var args256 = pargs;
                args256.rows = @intCast(rows256);
                if (p77) |pipe7| {
                    _ = try timeRowsT(queue, pipe7, &args256, x256, experts[0], sums256, routes256, descs, table, 64, 128, rows256, 256, 1);
                    const msH = try timeRowsT(queue, pipe7, &args256, x256, experts[0], sums256, routes256, descs, table, 64, 128, rows256, 256, 5);
                    const hashH = hashLiveRows(sums256, rows256 * exl3_n, &plan_counts, &off256, exl3_n);
                    std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (H: nax77 -- the slab decoded once for 256 rows, {d} tiles (31 % less decode); against the 64-row plan-s {d:.3} ms: {d:.1} %; the live rows {x:0>16} {s})\n", .{ "nax77", msH, macs_live / (msH * 1e-3) / 1e12, tiles256, msA, 100.0 * (msH - msA) / msA, hashH, if (hashH == hash64) "IDENTICAL" else "DIFFER" });
                }
            }
        }
    }
    // The ENGINE's body and the stream-read arms.
    if (p59) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms59 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (THE ENGINE'S BODY -- nax59, the funnel path as its only path)\n", .{ "nax59", ms59, macs / (ms59 * 1e-3) / 1e12 });
        const hash59 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax5 {x:0>16} nax59 {x:0>16} {s}\n", .{ hash5, hash59, if (hash5 == hash59) "IDENTICAL" else "DIFFER" });
    }
    if (p57) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms57 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the 32-bit stream word -- the comparator's read)\n", .{ "nax57", ms57, macs / (ms57 * 1e-3) / 1e12 });
    }
    if (p53) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms53 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the cooperative read, one lane a word + a shuffle)\n", .{ "nax53", ms53, macs / (ms53 * 1e-3) / 1e12 });
    }
    if (p69) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms69 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (nax59 + the eight stores CONTIGUOUS -- the comparator's store form)\n", .{ "nax69", ms69, macs / (ms69 * 1e-3) / 1e12 });
    }
    if (p70) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms70 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (nax59 + the eight scattered stores GONE)\n", .{ "nax70", ms70, macs / (ms70 * 1e-3) / 1e12 });
    }
    if (p71) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms71 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (nax59 + the accumulator -- the calibration)\n", .{ "nax71", ms71, macs / (ms71 * 1e-3) / 1e12 });
    }
    if (p72) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms72 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (nax59 + the codebook arithmetic dropped)\n", .{ "nax72", ms72, macs / (ms72 * 1e-3) / 1e12 });
    }
    if (p73) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms73 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (nax59 + the extraction dropped)\n", .{ "nax73", ms73, macs / (ms73 * 1e-3) / 1e12 });
    }
    if (p64) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms64 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the eight stores CONTIGUOUS, one lane's run -- the comparator's store form)\n", .{ "nax64", ms64, macs / (ms64 * 1e-3) / 1e12 });
        const hash64 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax48 {x:0>16} nax64 {x:0>16} {s}\n", .{ hash48, hash64, if (hash48 == hash64) "IDENTICAL" else "DIFFER" });
    }
    if (p65) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms65 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the DECODE KEPT, the eight scattered stores GONE)\n", .{ "nax65", ms65, macs / (ms65 * 1e-3) / 1e12 });
    }
    if (p66) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms66 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the SHIPPED body + the accumulator -- the calibration)\n", .{ "nax66", ms66, macs / (ms66 * 1e-3) / 1e12 });
    }
    if (p67) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms67 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the codebook arithmetic dropped, the extraction and the stores kept)\n", .{ "nax67", ms67, macs / (ms67 * 1e-3) / 1e12 });
    }
    if (p68) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms68 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (the extraction dropped, the eight stores kept)\n", .{ "nax68", ms68, macs / (ms68 * 1e-3) / 1e12 });
    }
    if (p61) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
        const ms61 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=2, the activation staged and shared)\n", .{ "nax61", ms61, macs / (ms61 * 1e-3) / 1e12 });
        const hash61 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax5 {x:0>16} nax61 {x:0>16} {s}\n", .{ hash5, hash61, if (hash5 == hash61) "IDENTICAL" else "DIFFER" });
    }
    if (p63) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
        const ms63 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=2, the shipped per-warp activation read)\n", .{ "nax63", ms63, macs / (ms63 * 1e-3) / 1e12 });
        const hash63 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax5 {x:0>16} nax63 {x:0>16} {s}\n", .{ hash5, hash63, if (hash5 == hash63) "IDENTICAL" else "DIFFER" });
    }
    if (p62) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
        const ms62 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (ABLATION, the activation read hoisted out of the k loop; the output differs by construction)\n", .{ "nax62", ms62, macs / (ms62 * 1e-3) / 1e12 });
    }
    if (p37) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 128, 1);
        const ms37 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 128, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (4 warps, 2 col tiles a warp)\n", .{ "nax37", ms37, macs / (ms37 * 1e-3) / 1e12 });
        const hash37 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax3 {x:0>16} nax37 {x:0>16} {s}\n", .{ hash3, hash37, if (hash3 == hash37) "IDENTICAL" else "DIFFER" });
        std.debug.print("  sums: nax5 {x:0>16} nax37 {x:0>16} {s}\n", .{ hash5, hash37, if (hash5 == hash37) "IDENTICAL" else "DIFFER" });
    }
    if (p7) |pipe| {
        _ = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
        const ms7 = try timeExl3(queue, pipe, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
        std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}\n", .{ "nax7", ms7, macs / (ms7 * 1e-3) / 1e12 });
        const hash7 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
        std.debug.print("  sums: nax3 {x:0>16} nax7 {x:0>16} {s}\n", .{ hash3, hash7, if (hash3 == hash7) "IDENTICAL" else "DIFFER" });
        if (p8) |pipe8| {
            _ = try timeExl3(queue, pipe8, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
            const ms8 = try timeExl3(queue, pipe8, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (decode removed)\n", .{ "nax8", ms8, macs / (ms8 * 1e-3) / 1e12 });
        }
        if (p9) |pipe9| {
            _ = try timeExl3(queue, pipe9, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
            const ms9 = try timeExl3(queue, pipe9, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (one of four multiplies)\n", .{ "nax9", ms9, macs / (ms9 * 1e-3) / 1e12 });
        }
        if (p10) |pipe10| {
            _ = try timeExl3(queue, pipe10, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
            const ms10 = try timeExl3(queue, pipe10, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (codebook table)\n", .{ "nax10", ms10, macs / (ms10 * 1e-3) / 1e12 });
            const hash10 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax10 {x:0>16} {s}\n", .{ hash3, hash10, if (hash3 == hash10) "IDENTICAL" else "DIFFER" });
        }
        if (p11) |pipe11| {
            _ = try timeExl3(queue, pipe11, &args, x, experts[0], sums, routes, descs, table, 128, 512, 1);
            const ms11 = try timeExl3(queue, pipe11, &args, x, experts[0], sums, routes, descs, table, 128, 512, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (warp-specialised, ALU)\n", .{ "nax11", ms11, macs / (ms11 * 1e-3) / 1e12 });
            const hash11 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax11 {x:0>16} {s}\n", .{ hash3, hash11, if (hash3 == hash11) "IDENTICAL" else "DIFFER" });
        }
        if (p12) |pipe12| {
            _ = try timeExl3(queue, pipe12, &args, x, experts[0], sums, routes, descs, table, 128, 512, 1);
            const ms12 = try timeExl3(queue, pipe12, &args, x, experts[0], sums, routes, descs, table, 128, 512, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (warp-specialised, table)\n", .{ "nax12", ms12, macs / (ms12 * 1e-3) / 1e12 });
            const hash12 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax12 {x:0>16} {s}\n", .{ hash3, hash12, if (hash3 == hash12) "IDENTICAL" else "DIFFER" });
        }
        if (p13) |pipe13| {
            _ = try timeExl3(queue, pipe13, &args, x, experts[0], sums, routes, descs, table, 256, 512, 1);
            const ms13 = try timeExl3(queue, pipe13, &args, x, experts[0], sums, routes, descs, table, 256, 512, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=4)\n", .{ "nax13", ms13, macs / (ms13 * 1e-3) / 1e12 });
            const hash13 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax13 {x:0>16} {s}\n", .{ hash3, hash13, if (hash3 == hash13) "IDENTICAL" else "DIFFER" });
        }
        if (p14) |pipe14| {
            _ = try timeExl3(queue, pipe14, &args, x, experts[0], sums, routes, descs, table, 512, 1024, 1);
            const ms14 = try timeExl3(queue, pipe14, &args, x, experts[0], sums, routes, descs, table, 512, 1024, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=8)\n", .{ "nax14", ms14, macs / (ms14 * 1e-3) / 1e12 });
            const hash14 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax14 {x:0>16} {s}\n", .{ hash3, hash14, if (hash3 == hash14) "IDENTICAL" else "DIFFER" });
        }
        if (p15) |pipe15| {
            _ = try timeExl3(queue, pipe15, &args, x, experts[0], sums, routes, descs, table, 256, 1024, 1);
            const ms15 = try timeExl3(queue, pipe15, &args, x, experts[0], sums, routes, descs, table, 256, 1024, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (WS CG=4)\n", .{ "nax15", ms15, macs / (ms15 * 1e-3) / 1e12 });
            const hash15 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax15 {x:0>16} {s}\n", .{ hash3, hash15, if (hash3 == hash15) "IDENTICAL" else "DIFFER" });
        }
        if (p16) |pipe16| {
            _ = try timeExl3(queue, pipe16, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
            const ms16 = try timeExl3(queue, pipe16, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (decode only, table)\n", .{ "nax16", ms16, macs / (ms16 * 1e-3) / 1e12 });
        }
        if (p17) |pipe17| {
            _ = try timeExl3(queue, pipe17, &args, x, experts[0], sums, routes, descs, table, 128, 256, 1);
            const ms17 = try timeExl3(queue, pipe17, &args, x, experts[0], sums, routes, descs, table, 128, 256, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (decode only, ALU)\n", .{ "nax17", ms17, macs / (ms17 * 1e-3) / 1e12 });
        }
        if (p18) |pipe18| {
            _ = try timeExl3(queue, pipe18, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms18 = try timeExl3(queue, pipe18, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, table)\n", .{ "nax18", ms18, macs / (ms18 * 1e-3) / 1e12 });
            const hash18 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax18 {x:0>16} {s}\n", .{ hash3, hash18, if (hash3 == hash18) "IDENTICAL" else "DIFFER" });
        }
        if (p19) |pipe19| {
            _ = try timeExl3(queue, pipe19, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms19 = try timeExl3(queue, pipe19, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, ALU)\n", .{ "nax19", ms19, macs / (ms19 * 1e-3) / 1e12 });
            const hash19 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax19 {x:0>16} {s}\n", .{ hash3, hash19, if (hash3 == hash19) "IDENTICAL" else "DIFFER" });
        }
        if (p35) |pipe35| {
            _ = try timeExl3(queue, pipe35, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms35 = try timeExl3(queue, pipe35, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, two blocks ahead)\n", .{ "nax35", ms35, macs / (ms35 * 1e-3) / 1e12 });
            const hash35 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax35 {x:0>16} {s}\n", .{ hash3, hash35, if (hash3 == hash35) "IDENTICAL" else "DIFFER" });
        }
        if (p20) |pipe20| {
            _ = try timeExl3(queue, pipe20, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms20 = try timeExl3(queue, pipe20, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, MMA only)\n", .{ "nax20", ms20, macs / (ms20 * 1e-3) / 1e12 });
        }
        if (p21) |pipe21| {
            _ = try timeExl3(queue, pipe21, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms21 = try timeExl3(queue, pipe21, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, decode only)\n", .{ "nax21", ms21, macs / (ms21 * 1e-3) / 1e12 });
        }
        if (p40) |pipe40| {
            _ = try timeExl3(queue, pipe40, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms40 = try timeExl3(queue, pipe40, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, funnel read)\n", .{ "nax40", ms40, macs / (ms40 * 1e-3) / 1e12 });
            const hash40 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax5 {x:0>16} nax40 {x:0>16} {s}\n", .{ hash5, hash40, if (hash5 == hash40) "IDENTICAL" else "DIFFER" });
        }
        if (p52) |pipe52| {
            _ = try timeExl3(queue, pipe52, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms52 = try timeExl3(queue, pipe52, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, funnel read, cooperative load + shuffle)\n", .{ "nax52", ms52, macs / (ms52 * 1e-3) / 1e12 });
            const hash52 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax5 {x:0>16} nax52 {x:0>16} {s}\n", .{ hash5, hash52, if (hash5 == hash52) "IDENTICAL" else "DIFFER" });
        }
        if (p50) |pipe50| {
            _ = try timeExl3(queue, pipe50, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms50 = try timeExl3(queue, pipe50, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, funnel read, 32-bit extraction)\n", .{ "nax50", ms50, macs / (ms50 * 1e-3) / 1e12 });
            const hash50 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax5 {x:0>16} nax50 {x:0>16} {s}\n", .{ hash5, hash50, if (hash5 == hash50) "IDENTICAL" else "DIFFER" });
        }
        if (p41) |pipe41| {
            _ = try timeExl3(queue, pipe41, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms41 = try timeExl3(queue, pipe41, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, no value ALU)\n", .{ "nax41", ms41, macs / (ms41 * 1e-3) / 1e12 });
        }
        if (p43) |pipe43| {
            _ = try timeExl3(queue, pipe43, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms43 = try timeExl3(queue, pipe43, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, funnel + resident table)\n", .{ "nax43", ms43, macs / (ms43 * 1e-3) / 1e12 });
            const hash43 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax5 {x:0>16} nax43 {x:0>16} {s}\n", .{ hash5, hash43, if (hash5 == hash43) "IDENTICAL" else "DIFFER" });
        }
        if (p44) |pipe44| {
            _ = try timeExl3(queue, pipe44, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms44 = try timeExl3(queue, pipe44, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, funnel + threadgroup LUT)\n", .{ "nax44", ms44, macs / (ms44 * 1e-3) / 1e12 });
            const hash44 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax5 {x:0>16} nax44 {x:0>16} {s}\n", .{ hash5, hash44, if (hash5 == hash44) "IDENTICAL" else "DIFFER" });
        }
        if (p46) |pipe46| {
            _ = try timeExl3(queue, pipe46, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms46 = try timeExl3(queue, pipe46, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, double-buffered + funnel)\n", .{ "nax46", ms46, macs / (ms46 * 1e-3) / 1e12 });
            const hash46 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax5 {x:0>16} nax46 {x:0>16} {s}\n", .{ hash5, hash46, if (hash5 == hash46) "IDENTICAL" else "DIFFER" });
        }
        if (p47) |pipe47| {
            _ = try timeExl3(queue, pipe47, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms47 = try timeExl3(queue, pipe47, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, funnel, cheap value -- a diagnostic)\n", .{ "nax47", ms47, macs / (ms47 * 1e-3) / 1e12 });
        }
        if (p22) |pipe22| {
            _ = try timeExl3(queue, pipe22, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms22 = try timeExl3(queue, pipe22, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, table, single buffer)\n", .{ "nax22", ms22, macs / (ms22 * 1e-3) / 1e12 });
            const hash22 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax22 {x:0>16} {s}\n", .{ hash3, hash22, if (hash3 == hash22) "IDENTICAL" else "DIFFER" });
        }
        if (p23) |pipe23| {
            _ = try timeExl3(queue, pipe23, &args, x, experts[0], sums, routes, descs, table, 128, 128, 1);
            const ms23 = try timeExl3(queue, pipe23, &args, x, experts[0], sums, routes, descs, table, 128, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (av reuse, ALU)\n", .{ "nax23", ms23, macs / (ms23 * 1e-3) / 1e12 });
            const hash23 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax23 {x:0>16} {s}\n", .{ hash3, hash23, if (hash3 == hash23) "IDENTICAL" else "DIFFER" });
        }
        if (p24) |pipe24| {
            _ = try timeExl3(queue, pipe24, &args, x, experts[0], sums, routes, descs, table, 128, 128, 1);
            const ms24 = try timeExl3(queue, pipe24, &args, x, experts[0], sums, routes, descs, table, 128, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (av reuse, table)\n", .{ "nax24", ms24, macs / (ms24 * 1e-3) / 1e12 });
            const hash24 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax24 {x:0>16} {s}\n", .{ hash3, hash24, if (hash3 == hash24) "IDENTICAL" else "DIFFER" });
        }
        if (p25) |pipe25| {
            _ = try timeExl3(queue, pipe25, &args, x, experts[0], sums, routes, descs, table, 64, 128, 1);
            const ms25 = try timeExl3(queue, pipe25, &args, x, experts[0], sums, routes, descs, table, 64, 128, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (CG=1, threadgroup LUT)\n", .{ "nax25", ms25, macs / (ms25 * 1e-3) / 1e12 });
            const hash25 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums.slice(f32, @as(usize, exl3_rows) * exl3_n)));
            std.debug.print("  sums: nax3 {x:0>16} nax25 {x:0>16} {s}\n", .{ hash3, hash25, if (hash3 == hash25) "IDENTICAL" else "DIFFER" });
        }
        const dense_trellis_bytes = (dense_k / 16) * (dense_n / 16) * exl3_tile_bytes;
        const dtr = try device.buffer(dense_trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer dtr.deinit();
        var drng: u64 = 0xD1CEB00C;
        for (dtr.slice(u16, dense_trellis_bytes / 2)) |*v| {
            drng ^= drng << 13;
            drng ^= drng >> 7;
            drng ^= drng << 17;
            v.* = @truncate(drng);
        }
        const dx = try device.buffer(@as(usize, dense_rows) * dense_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer dx.deinit();
        @memset(dx.slice(u16, @as(usize, dense_rows) * dense_k), 0x3c00);
        const dsums = try device.buffer(@as(usize, dense_rows) * dense_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer dsums.deinit();
        @memset(dsums.slice(u32, @as(usize, dense_rows) * dense_n), 0);
        const dargs = Exl3Args{
            .k = dense_k,
            .n = dense_n,
            .rows = @intCast(dense_rows),
            .halves = exl3_halves,
            .codebook = 2,
            .su_signs = 0,
            .sv_signs = 0,
            .has_bias = 0,
            .output_f32 = 1,
            .input_ids = 0,
            .output_ids = 0,
            .routed = 0,
            .expert_id = 0,
            .repeat = 1,
            .indexed = 0,
            .experts = 0,
            .splits = 1,
            .x_stride = dense_k,
            .y_stride = dense_n,
            .tile_k_stride = exl3_k_stride,
            .tile_n_stride = exl3_n_stride,
        };
        const dmacs = @as(f64, @floatFromInt(dense_rows)) * @as(f64, dense_n) * @as(f64, dense_k);
        std.debug.print("exl3 dense bench: {d} rows x {d} x {d}, 128x64 tile\n", .{ dense_rows, dense_n, dense_k });
        if (p2) |pipe2| {
            _ = try timeDense(queue, pipe2, &dargs, dx, dtr, dsums, 64, 256, 1);
            const ms2 = try timeDense(queue, pipe2, &dargs, dx, dtr, dsums, 64, 256, 5);
            std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (dense shipped nax2)\n", .{ "nax2", ms2, dmacs / (ms2 * 1e-3) / 1e12 });
            const dh2 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
            if (p26) |pipe26| {
                _ = try timeDense(queue, pipe26, &dargs, dx, dtr, dsums, 64, 256, 1);
                const ms26 = try timeDense(queue, pipe26, &dargs, dx, dtr, dsums, 64, 256, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (dense DB+table)\n", .{ "nax26", ms26, dmacs / (ms26 * 1e-3) / 1e12 });
                const dh26 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
                std.debug.print("  sums: nax2 {x:0>16} nax26 {x:0>16} {s}\n", .{ dh2, dh26, if (dh2 == dh26) "IDENTICAL" else "DIFFER" });
            }
            if (p27) |pipe27| {
                _ = try timeDense(queue, pipe27, &dargs, dx, dtr, dsums, 64, 256, 1);
                const ms27 = try timeDense(queue, pipe27, &dargs, dx, dtr, dsums, 64, 256, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (dense LUT)\n", .{ "nax27", ms27, dmacs / (ms27 * 1e-3) / 1e12 });
                const dh27 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
                std.debug.print("  sums: nax2 {x:0>16} nax27 {x:0>16} {s}\n", .{ dh2, dh27, if (dh2 == dh27) "IDENTICAL" else "DIFFER" });
            }
            if (p28) |pipe28| {
                _ = try timeDense(queue, pipe28, &dargs, dx, dtr, dsums, 128, 512, 1);
                const ms28 = try timeDense(queue, pipe28, &dargs, dx, dtr, dsums, 128, 512, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (dense 128x128 LUT)\n", .{ "nax28", ms28, dmacs / (ms28 * 1e-3) / 1e12 });
                const dh28 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
                std.debug.print("  sums: nax2 {x:0>16} nax28 {x:0>16} {s}\n", .{ dh2, dh28, if (dh2 == dh28) "IDENTICAL" else "DIFFER" });
            }
            if (p29) |pipe29| {
                _ = try timeDense(queue, pipe29, &dargs, dx, dtr, dsums, 128, 512, 1);
                const ms29 = try timeDense(queue, pipe29, &dargs, dx, dtr, dsums, 128, 512, 5);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (dense 128x128 ALU)\n", .{ "nax29", ms29, dmacs / (ms29 * 1e-3) / 1e12 });
                const dh29 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
                std.debug.print("  sums: nax2 {x:0>16} nax29 {x:0>16} {s}\n", .{ dh2, dh29, if (dh2 == dh29) "IDENTICAL" else "DIFFER" });
            }
            if (p2) |pipe2b| {
                std.debug.print("exl3 attention shapes, shipped nax2 128x64 tile:\n", .{});
                std.debug.print("  {s:>18} {s:>9} {s:>9}\n", .{ "M x N x K", "ms", "TMAC/s" });
                try benchDenseShape(device, queue, pipe2b, dense_rows, 32768, 1280); // wq_b, short k
                try benchDenseShape(device, queue, pipe2b, dense_rows, 1024, 4096); // wo_a, narrow n
                try benchDenseShape(device, queue, pipe2b, dense_rows, 5120, 8192); // wo_b, long k
                std.debug.print("  cold-vs-hot (8 distinct 12-16 MB trellis buffers against one repeated):\n", .{});
                try benchDenseCold(device, queue, pipe2b, dense_rows, 32768, 1280, 8);
                try benchDenseCold(device, queue, pipe2b, dense_rows, 1024, 4096, 8);
                try benchDenseCold(device, queue, pipe2b, dense_rows, 5120, 8192, 8);
                std.debug.print("dense 128x64 tile diagnostics (RG=8, 256 thr): nax31 decode+MMA, nax30 MMA only, nax32 one of four multiplies\n", .{});
                const diag = [_]Shape{
                    .{ .m = @intCast(dense_rows), .k = 5120, .n = 8192 },
                    .{ .m = @intCast(dense_rows), .k = 1280, .n = 32768 },
                    .{ .m = @intCast(dense_rows), .k = 4096, .n = 1024 },
                };
                for (diag) |d| {
                    std.debug.print("  shape {d}x{d}x{d}\n", .{ d.m, d.n, d.k });
                    if (p31) |q31| try benchDenseShape(device, queue, q31, d.m, d.n, d.k);
                    var dh31: u64 = 0;
                    if (p31) |q31| {
                        _ = try timeDense(queue, q31, &dargs, dx, dtr, dsums, 128, 256, 1);
                        dh31 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
                    }
                    if (p56) |q56| {
                        std.debug.print("    nax56 RG=8 Rpg=2 (a simdgroup owns 32 rows, 128 thr)\n", .{});
                        _ = try benchDenseShapeAt(device, queue, q56, d.m, d.n, d.k, 128, 128);
                        const dh56 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dsums.slice(f32, @as(usize, dense_rows) * dense_n)));
                        std.debug.print("    sums: nax31 {x:0>16} nax56 {x:0>16} {s}\n", .{ dh31, dh56, if (dh31 == dh56) "IDENTICAL" else "DIFFER" });
                    }
                    if (p30) |q30| try benchDenseShape(device, queue, q30, d.m, d.n, d.k);
                    if (p32) |q32| try benchDenseShape(device, queue, q32, d.m, d.n, d.k);
                    if (p33) |q33| {
                        std.debug.print("    nax33 RG=16 decode+MMA\n", .{});
                        try benchDenseShapeAt(device, queue, q33, d.m, d.n, d.k, 256, 512);
                    }
                    if (p34) |q34| {
                        std.debug.print("    nax34 RG=32 decode+MMA\n", .{});
                        try benchDenseShapeAt(device, queue, q34, d.m, d.n, d.k, 512, 1024);
                    }
                    if (p35) |q35| {
                        std.debug.print("    nax35 RG=4 two-ahead\n", .{});
                        try benchDenseShapeAt(device, queue, q35, d.m, d.n, d.k, 64, 128);
                    }
                    if (p36) |q36| {
                        std.debug.print("    nax36 RG=8 two-ahead\n", .{});
                        try benchDenseShapeAt(device, queue, q36, d.m, d.n, d.k, 128, 256);
                    }
                }
            }
            // The 128-column double-buffered body, priced before its fused epilogue is built.
            if (p31) |q31| {
                if (p79 != null or p80 != null) {
                    const dtable = try device.buffer(3 * 65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                    defer dtable.deinit();
                    {
                        const tw: [*]u16 = @ptrCast(@alignCast(dtable.contents()));
                        for (0..3) |cb| for (0..65536) |st| {
                            const e: tf.exl3_format.Codebook = @fromBackingInt(@intCast(cb));
                            tw[cb * 65536 + st] = @bitCast(tf.exl3_format.codebookValue(e, @intCast(st)));
                        };
                    }
                    std.debug.print("128-column double-buffered dense body vs nax31, {d} rows:\n", .{dense_rows});
                    std.debug.print("  {s:>5} {s:>16} {s:>10} {s:>8} {s:>8}\n", .{ "shape", "M x N x K", "body", "ms", "TMAC/s" });
                    try benchDenseCompare128(device, queue, q31, p79, p80, p82, p83, dtable, dense_rows, 5120, 8192); // wo_b
                    try benchDenseCompare128(device, queue, q31, p79, p80, p82, p83, dtable, dense_rows, 32768, 1280); // wq_b
                    try benchDenseCompare128(device, queue, q31, p79, p80, p82, p83, dtable, dense_rows, 1024, 4096); // wo_a
                }
            }
            // The 128-row grouped tile: one expert a 128-row block, 384 tiles, the comparator's shape. The decode is
            // once a tile; the MMA rows double (mostly padding), which is the trade this prices.
            if (p2) |pipe128| {
                const rows128: usize = 384 * 128;
                const x128 = try device.buffer(rows128 * exl3_k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer x128.deinit();
                @memset(x128.slice(u16, rows128 * exl3_k), 0x3c00);
                const sums128 = try device.buffer(rows128 * exl3_n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer sums128.deinit();
                @memset(sums128.slice(u32, rows128 * exl3_n), 0);
                const routes128 = try device.buffer(rows128 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                defer routes128.deinit();
                for (routes128.slice(u32, rows128), 0..) |*r, i| r.* = @intCast(i / 128);
                const args128 = Exl3Args{
                    .k = exl3_k,
                    .n = exl3_n,
                    .rows = @intCast(rows128),
                    .halves = exl3_halves,
                    .codebook = 2,
                    .su_signs = 0,
                    .sv_signs = 0,
                    .has_bias = 0,
                    .output_f32 = 1,
                    .input_ids = 0,
                    .output_ids = 0,
                    .routed = 0,
                    .expert_id = 0,
                    .repeat = 1,
                    .indexed = 2,
                    .experts = exl3_tiles,
                    .splits = 1,
                    .x_stride = exl3_k,
                    .y_stride = exl3_n,
                    .tile_k_stride = exl3_k_stride,
                    .tile_n_stride = exl3_n_stride,
                };
                _ = try timeGrouped128(queue, pipe128, &args128, x128, experts[0], sums128, routes128, descs, 1);
                const ms128 = try timeGrouped128(queue, pipe128, &args128, x128, experts[0], sums128, routes128, descs, 5);
                const macs128 = @as(f64, rows128) * @as(f64, exl3_n) * @as(f64, exl3_k);
                std.debug.print("  {s:>8} {d:>9.3} {d:>9.1}  (384 x 128-row grouped, padded MACs)\n", .{ "nax2-128", ms128, macs128 / (ms128 * 1e-3) / 1e12 });
                const hash128 = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sums128.slice(f32, rows128 * exl3_n)));
                std.debug.print("  sums: 128-row {x:0>16}\n", .{hash128});
            }
        }
    } else {
        std.debug.print("  sums: nax3 {x:0>16} (nax7 absent)\n", .{hash3});
    }
}

// THE FP8/E4M3 DENSE ARM. nax31 reads the 5-6 bit EXL3 trellis and decodes it; this arm reads a one-byte-a-weight
// E4M3 tensor already in the kernel's 16-k x 64-n block order and expands it to half with the comparator's exact
// bit trick. Both are timed in one run at the attention's three shapes, and the fp8 path is validated against a
// host fp32 sum on a fixed pattern (row 0, columns 0..7).
fn fp8Value(b: u8) f32 {
    const bits: u16 = (@as(u16, b & 0x7F) << 7) | (@as(u16, b & 0x80) << 8);
    const h: f16 = @bitCast(bits);
    return @as(f32, @floatCast(h)) * 256.0;
}

fn fp8Code(n: usize, k: usize) u8 {
    return @intCast((n * 13 + k * 7) % 0x7E);
}

// The engine's own release-fp8 prompt MMA (e4m3 weights, one e8m0 scale a 32-block) at the attention's two big
// prompt shapes. The first gate: the library rate is 95.3/75.5 TMAC/s; this measures OUR kernel
// (`tf_dsv41_fp8_mma_bf16`) on synthetic weights, no pack and no serve.
const MxShape = struct { name: []const u8, m: u32, k: u32, n: u32 };

// The same e4m3 x e8m0/32 body with bfloat16 MMA operands and fp32 accumulation (the fp8_mma arm's fp32 simdgroup
// matrices are the rate question). Synthetic weights; no pack and no serve.
const Bf16xArgs = extern struct { K: u32, N: u32, R: u32 };

fn mxDenseBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    const fp8 = tf.dsv41_metal.fp8_mma;
    const shapes = [_]MxShape{
        .{ .name = "wq_b", .m = 3619, .k = 1280, .n = 32768 },
        .{ .name = "wo_a", .m = 3619, .k = 32768, .n = 1024 },
        .{ .name = "wq_b@1", .m = 1, .k = 1280, .n = 32768 },
        .{ .name = "wo_a@1", .m = 1, .k = 32768, .n = 1024 },
    };
    std.debug.print("engine fp8 MMA (e4m3 x e8m0/32, tf_dsv41_fp8_mma_bf16), synthetic weights\n", .{});
    std.debug.print("  {s:>10} {s:>5} {s:>15} {s:>9} {s:>9} {s:>9}\n", .{ "shape", "RS", "M x N x K", "ms", "TMAC/s", "GB/s" });
    for ([_]u32{ 4, 8, 16 }) |rs| {
        var pipes = if (rs == 4) try fp8.Pipelines.load(device) else blk: {
            const text = try std.fmt.allocPrint(gpa, "#define TF_RS {d}\n{s}", .{ rs, ks.dsv41.fp8_mma });
            defer gpa.free(text);
            break :blk try fp8.Pipelines.loadSource(device, text);
        };
        defer pipes.deinit();
        for (shapes) |s| {
            const m: usize = s.m;
            const n: usize = s.n;
            const k: usize = s.k;
            const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer x.deinit();
            const w = try device.buffer(n * k, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer w.deinit();
            const sc = try device.buffer((n / 32) * (k / 32), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer sc.deinit();
            const y = try device.buffer(m * n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer y.deinit();
            for (x.slice(u16, m * k), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64.0)) >> 16);
            @memset(w.slice(u8, n * k), 0x38); // e4m3 1.0, finite
            @memset(sc.slice(u8, (n / 32) * (k / 32)), 127); // e8m0 1.0
            @memset(y.slice(u8, m * n * 2), 0);
            const reps: usize = if (m == 1) 20 else 1;
            var samples: [16]f64 = undefined;
            const trials: usize = 9;
            for (0..trials) |t| {
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                for (0..reps) |_| {
                    try fp8.encode(&pipes, e, .{ .buf = x }, .{ .buf = w }, .{ .buf = sc }, .{ .buf = y }, .{ .k = s.k, .n = s.n, .rows = s.m, .out = .bf16 });
                }
                e.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("  {s} rs{d}: failed: {s}\n", .{ s.name, rs, msg });
                    return error.DispatchFailed;
                }
                samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
            }
            std.mem.sort(f64, samples[0..trials], {}, std.sort.asc(f64));
            const ms = samples[trials / 2];
            const macs = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
            const gbs = @as(f64, @floatFromInt(n * k)) / (ms * 1e-3) / 1e9;
            std.debug.print("  {s:>10} {d:>5} {d:>5}x{d:>6}x{d:>5} {d:>9.3} {d:>9.1} {d:>9.1}\n", .{ s.name, rs, m, n, k, ms, macs / (ms * 1e-3) / 1e12, gbs });
        }
    }
}

fn fp8DenseBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator) !void {
    const src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(src);
    var owned: ?[]u8 = null;
    var text: []const u8 = src;
    if (!device.tensorUnits()) {
        owned = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", src });
        text = owned.?;
    }
    defer if (owned) |o| gpa.free(o);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p31 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax31", false);
    defer p31.deinit();
    const pf8 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_fp8", false);
    defer pf8.deinit();
    const pf8w = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_fp8w", false);
    defer pf8w.deinit();
    const p30 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax30", false);
    defer p30.deinit();
    const rows: usize = dense_rows;
    const i8arm = std.c.getenv("TF_FP8_I8") != null;
    const wide = std.c.getenv("TF_FP8_WIDE") != null;
    const sc1 = try device.buffer(32768 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sc1.deinit();
    for (sc1.slice(f32, 32768)) |*v| v.* = 1.0;

    // Validity first: a 128-row, 1024-column, 1280-k case with a fixed pattern, against the host's fp32 sum.
    {
        const m: usize = 128;
        const n: usize = 1024;
        const k: usize = 1280;
        const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        @memset(x.slice(u16, m * k), 0x3c00);
        const sums = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sums.deinit();
        @memset(sums.slice(u32, m * n), 0);
        const w8 = try device.buffer(n * k, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer w8.deinit();
        const nb64 = n / 64;
        for (0..n) |nn| for (0..k) |kk| {
            const at = ((kk / 16) * nb64 + nn / 64) * 1024 + (kk % 16) * 64 + (nn % 64);
            w8.slice(u8, n * k)[at] = fp8Code(nn, kk);
        };
        const args = Exl3Args{
            .k = @intCast(k),
            .n = @intCast(n),
            .rows = @intCast(m),
            .halves = 1,
            .codebook = 0,
            .su_signs = 0,
            .sv_signs = 0,
            .has_bias = 0,
            .output_f32 = 1,
            .input_ids = 0,
            .output_ids = 0,
            .routed = 0,
            .expert_id = 0,
            .repeat = 1,
            .indexed = 0,
            .experts = 0,
            .splits = 1,
            .x_stride = @intCast(k),
            .y_stride = @intCast(n),
            .tile_k_stride = 0,
            .tile_n_stride = 0,
        };
        _ = try timeShapeSc(queue, pf8, &args, x, w8, sums, sc1, m, n, 64, 256, 128, 1);
        var worst: f32 = 0;
        for (0..8) |nn| {
            var want: f32 = 0;
            for (0..k) |kk| want += fp8Value(fp8Code(nn, kk));
            const got = sums.slice(f32, m * n)[nn];
            if (nn < 3) std.debug.print("  col {d}: got {e:.4} want {e:.4}\n", .{ nn, got, want });
            const err = @abs(got - want) / @max(1.0, @abs(want));
            worst = @max(worst, err);
        }
        std.debug.print("fp8 validity 128x1024x1280: worst relative |host sum - body| over 8 columns = {e:.3}\n", .{worst});
    }

    // Priced one wo_a group as if it were the layer (it is 8) and had wo_b's K/N swapped; the width is the
    // projection's own halves (5-bit for the fused o/wq_b groups, 6-bit for wq_a/wkv), not the bench's old 6 (a
    // 3-bit synthetic trellis).
    const shapes = [_]Shape{
        .{ .m = @intCast(rows), .k = 1280, .n = 32768 }, // wq_b  K 1280 (q_lora)      N 32768 (heads*head_dim)
        .{ .m = @intCast(rows), .k = 4096, .n = 1024 }, // wo_a  K 4096 (q_width/8)   N 1024 (o_lora), one of 8 groups
        .{ .m = @intCast(rows), .k = 8192, .n = 5120 }, // wo_b  K 8192 (8*o_lora)    N 5120 (hidden)
        .{ .m = @intCast(rows), .k = 5120, .n = 1280 }, // wq_a  K 5120 (hidden)      N 1280 (q_lora)
        .{ .m = @intCast(rows), .k = 5120, .n = 512 }, // wkv   K 5120 (hidden)      N 512 (one kv head x 512)
        .{ .m = @intCast(rows), .k = 5120, .n = 2304 }, // shared w1/w3  K 5120 (hidden) N 2304 (moe_intermediate), x2
        .{ .m = @intCast(rows), .k = 2304, .n = 5120 }, // shared w2     K 2304          N 5120 (hidden)
    };
    const hals = [_]u32{ 10, 10, 10, 12, 12, 10, 10 }; // the pack's halves: 80,80,80,96,96,80,80 16-bit words a tile
    const names = [_][]const u8{ "wq_b", "wo_a", "wo_b", "wq_a", "wkv", "sh_w13", "sh_w2" };
    std.debug.print("fp8 dense arm at the PACK'S own shapes and widths: nax31 (EXL3) against the 1-byte int8/E4M3 body, {d} rows\n", .{rows});
    std.debug.print("  {s:>6} {s:>17} {s:>4} {s:>9} {s:>9} {s:>9} {s:>9} {s:>8} {s:>8}\n", .{ "shape", "M x N x K", "half", "nax31 ms", "nax30 ms", "byte ms", "byte TMAC", "byte GB/s", "byte vs 31" });
    for (shapes, 0..) |s, si| {
        const m = s.m;
        const n = s.n;
        const k = s.k;
        const hv = hals[si];
        const x = try device.buffer(@as(usize, m) * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        @memset(x.slice(u16, @as(usize, m) * k), 0x3c00);
        const sums = try device.buffer(@as(usize, m) * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer sums.deinit();
        @memset(sums.slice(u32, @as(usize, m) * n), 0);
        var args = Exl3Args{
            .k = @intCast(k),
            .n = @intCast(n),
            .rows = @intCast(m),
            .halves = hv,
            .codebook = 2,
            .su_signs = 0,
            .sv_signs = 0,
            .has_bias = 0,
            .output_f32 = 1,
            .input_ids = 0,
            .output_ids = 0,
            .routed = 0,
            .expert_id = 0,
            .repeat = 1,
            .indexed = 0,
            .experts = 0,
            .splits = 1,
            .x_stride = @intCast(k),
            .y_stride = @intCast(n),
            .tile_k_stride = (n / 16) * exl3_n_stride,
            .tile_n_stride = exl3_n_stride,
        };
        const trellis_bytes = (k / 16) * (n / 16) * hv * 16;
        const tr = try device.buffer(trellis_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer tr.deinit();
        var rng: u64 = 0xD1CEB00C ^ @as(u64, n);
        for (tr.slice(u16, trellis_bytes / 2)) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            v.* = @truncate(rng);
        }
        _ = try timeShape(queue, p31, &args, x, tr, sums, null, m, n, 64, 256, 128, 1);
        const ms31 = try timeShape(queue, p31, &args, x, tr, sums, null, m, n, 64, 256, 128, 5);
        _ = try timeShape(queue, p30, &args, x, tr, sums, null, m, n, 64, 256, 128, 1);
        const ms30 = try timeShape(queue, p30, &args, x, tr, sums, null, m, n, 64, 256, 128, 5);
        const w8 = try device.buffer(n * k, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer w8.deinit();
        const nb64 = n / 64;
        for (0..n) |nn| for (0..k) |kk| {
            const at = ((kk / 16) * nb64 + nn / 64) * 1024 + (kk % 16) * 64 + (nn % 64);
            w8.slice(u8, n * k)[at] = fp8Code(nn, kk);
        };
        args.halves = if (i8arm) 0 else 1;
        const pipe8 = if (wide) pf8w else pf8;
        _ = try timeShapeSc(queue, pipe8, &args, x, w8, sums, sc1, m, n, 64, 256, 128, 1);
        const ms8 = try timeShapeSc(queue, pipe8, &args, x, w8, sums, sc1, m, n, 64, 256, 128, 5);
        const macs = @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
        // The fp8 tensor is re-read once for each 128-row block of the prompt.
        const passes = @as(f64, @floatFromInt((m + 127) / 128));
        const gbs = @as(f64, @floatFromInt(n * k)) * passes / (ms8 * 1e-3) / 1e9;
        std.debug.print("  {s:>6} {d:>5}x{d:>6}x{d:>5} {d:>4} {d:>9.3} {d:>9.3} {d:>9.3} {d:>9.1} {d:>9.1} {d:>7.1}%\n", .{ names[si], m, n, k, hv, ms31, ms30, ms8, macs / (ms8 * 1e-3) / 1e12, gbs, 100.0 * (ms8 - ms31) / ms31 });
    }
}

// The per-row-scaled fp8 timing helper (buffer 4 is the scale row).
fn timeShapeSc(queue: mtl.Queue, pipe: mtl.Pipeline, args: *const Exl3Args, x: mtl.Buffer, trellis: mtl.Buffer, sums: mtl.Buffer, scales: mtl.Buffer, m: usize, n: usize, cols: usize, threads: usize, tg_rows: usize, trials: usize) !f64 {
    const tn = (n + cols - 1) / cols;
    const tm = (m + tg_rows - 1) / tg_rows;
    var samples: [16]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(trellis, 0, 1);
        e.setBuffer(sums, 0, 2);
        e.setBytes(std.mem.asBytes(args), 3);
        e.setBuffer(scales, 0, 4);
        e.dispatchGroups(.{ .width = tn, .height = tm, .depth = 1 }, .{ .width = threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("  shape fp8: failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    return median(samples[0..nt]);
}

// The pre-rotated int8 pack's GPU body against the exact CPU linear map of the stored codes/scales.
fn i8bBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator, k: usize, n: usize, fp8b: []const u8, scb: []const u8, block: usize) !void {
    if (k % block != 0) return error.BadMeta;
    const nb = k / block;
    if (scb.len != n * nb * 4) return error.BadFiles;
    const src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(src);
    var owned: ?[]u8 = null;
    var text: []const u8 = src;
    if (!device.tensorUnits()) {
        owned = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", src });
        text = owned.?;
    }
    defer if (owned) |o| gpa.free(o);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const acc = std.c.getenv("TF_FP8_ACC") != null;
    const wide = std.c.getenv("TF_FP8_WIDE") != null and !acc;
    const pipe = try mtl.Pipeline.init(device, lib, if (acc) "tf_exl3_matmul_prefill_i8a" else if (wide) "tf_exl3_matmul_prefill_i8bw" else "tf_exl3_matmul_prefill_i8b", false);
    defer pipe.deinit();
    const pipe1 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_fp8", false);
    defer pipe1.deinit();
    const m: usize = if (std.c.getenv("TF_FP8_ROWS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 64) else 64;
    const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    for (x.slice(u16, m * k), 0..) |*v, i| {
        const sv: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 2654435761) % 2001)) - 1000)) / 1000.0;
        v.* = @truncate(@as(u32, @bitCast(sv)) >> 16);
    }
    const sums = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums.deinit();
    const sums1 = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums1.deinit();
    const w8 = try device.buffer(k * n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer w8.deinit();
    @memcpy(w8.contents()[0 .. k * n], fp8b);
    const scale1 = std.c.getenv("TF_FP8_SCALE1") != null;
    const sc = try device.buffer(n * nb * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sc.deinit();
    if (scale1) {
        for (sc.slice(f32, n * nb)) |*v| v.* = 1.0;
    } else {
        @memcpy(sc.contents()[0 .. n * nb * 4], scb);
    }
    const sc1 = try device.buffer(n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sc1.deinit();
    @memcpy(sc1.contents()[0 .. n * 4], scb[0 .. n * 4]);
    var args = Exl3Args{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(m),
        .halves = 0,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 0,
        .experts = 0,
        .splits = 1,
        .x_stride = @intCast(k),
        .y_stride = @intCast(n),
        .tile_k_stride = (n / 16) * 16,
        .tile_n_stride = 16,
    };
    @memset(sums.slice(u32, m * n), 0);
    const ms = try timeShapeSc(queue, pipe, &args, x, w8, sums, sc, m, n, 64, 256, 128, 5);
    const ms1 = try timeShapeSc(queue, pipe1, &args, x, w8, sums1, sc1, m, n, 64, 256, 128, 5);

    // The exact stored weight, W_hat[k,n] = code[k,n] * scale[k/block, n], and the same activation the kernel saw.
    const what = try gpa.alloc(f64, k * n);
    defer gpa.free(what);
    const scf = std.mem.bytesAsSlice(f32, scb);
    for (0..nb) |b| {
        for (0..n) |nn| {
            for (b * block..(b + 1) * block) |kk| {
                const at = ((kk / 16) * (n / 64) + nn / 64) * 1024 + (kk % 16) * 64 + (nn % 64);
                const code = fp8b[at];
                if (acc) {
                    const q: f64 = @floatFromInt(@as(i8, @bitCast(code)));
                    const s: f64 = @floatCast(if (scale1) 1.0 else scf[b * n + nn]);
                    what[kk * n + nn] = q * s;
                } else {
                    const qh: f16 = @floatFromInt(@as(i8, @bitCast(code)));
                    const sh: f16 = @floatCast(if (scale1) 1.0 else scf[b * n + nn]);
                    const ph: f16 = @floatCast(@as(f32, @floatCast(qh)) * @as(f32, @floatCast(sh)));
                    what[kk * n + nn] = @floatCast(ph);
                }
            }
        }
    }
    const xh = x.slice(f16, m * k);
    const gpu = sums.slice(f32, m * n);
    var se: f64 = 0;
    var sb: f64 = 0;
    var worst: f64 = 0;
    for (0..m) |r| {
        for (0..n) |nn| {
            var cpu: f64 = 0;
            for (0..k) |kk| cpu += @as(f64, @floatCast(xh[r * k + kk])) * what[kk * n + nn];
            const d = @as(f64, gpu[r * n + nn]) - cpu;
            se += d * d;
            sb += cpu * cpu;
            worst = @max(worst, @abs(d) / @max(1.0, @abs(cpu)));
        }
    }
    const tmac = 2.0 * @as(f64, @floatFromInt(m * n * k)) / (ms * 1e9);
    std.debug.print("int8b kernel: K {d} N {d} block {d} {s}, {d} rows: i8b {d:.3} ms ({d:.1} TMAC/s), i8 single {d:.3} ms, overhead {d:.2}%\n", .{
        k, n, block, if (wide) "wide" else "narrow", m, ms, tmac, ms1, 100.0 * (ms - ms1) / ms1,
    });
    if (m <= 64) std.debug.print("  GPU-vs-CPU exact-map NRMS = {e:.4}; worst relative pointwise = {e:.4}\n", .{ std.math.sqrt(se / sb), worst });
}

// The real-weight agreement harness. Reads one layer's repacked fp8 tensor + per-row scales and the same
// projection's EXL3 trellis, runs both bodies on one fixed activation, and reports NRMS and per-row argmax
// agreement. Layer-level, no serve; the model-level 20-prompt gate stays for the flag-gated port.
fn fp8RealBench(device: mtl.Device, queue: mtl.Queue, gpa: std.mem.Allocator, io: std.Io) !void {
    const base_z = std.c.getenv("TF_FP8_BASE") orelse return error.NoFp8Base;
    const base = std.mem.span(base_z);
    const meta_path = try std.fmt.allocPrint(gpa, "{s}.meta", .{base});
    defer gpa.free(meta_path);
    const meta = try std.Io.Dir.cwd().readFileAlloc(io, meta_path, gpa, .limited(256));
    defer gpa.free(meta);
    var it = std.mem.tokenizeAny(u8, meta, " \n\r\t");
    const k = try std.fmt.parseInt(usize, it.next() orelse return error.BadMeta, 10);
    const n = try std.fmt.parseInt(usize, it.next() orelse return error.BadMeta, 10);
    const halves = try std.fmt.parseInt(u32, it.next() orelse return error.BadMeta, 10);
    const fmt = it.next() orelse "fp8";
    const is_i8 = std.mem.eql(u8, fmt, "int8");
    var block: usize = 0;
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "block=")) block = std.fmt.parseInt(usize, tok[6..], 10) catch 0;
    }
    const nblocks: usize = if (block != 0) k / block else 1;
    const sc_len = n * nblocks * 4;
    const fp8_path = try std.fmt.allocPrint(gpa, "{s}.fp8", .{base});
    defer gpa.free(fp8_path);
    const sc_path = try std.fmt.allocPrint(gpa, "{s}.scale", .{base});
    defer gpa.free(sc_path);
    const fp8b = try std.Io.Dir.cwd().readFileAlloc(io, fp8_path, gpa, .limited(k * n + 16));
    defer gpa.free(fp8b);
    const scb = try std.Io.Dir.cwd().readFileAlloc(io, sc_path, gpa, .limited(sc_len + 16));
    defer gpa.free(scb);
    if (fp8b.len != k * n or scb.len != sc_len) return error.BadFiles;
    if (block != 0) {
        try i8bBench(device, queue, gpa, k, n, fp8b, scb, block);
        return;
    }
    const tr_path = try std.fmt.allocPrint(gpa, "{s}.trellis", .{base});
    defer gpa.free(tr_path);
    const tbytes = k / 16 * n / 16 * halves * 16;
    const trb = try std.Io.Dir.cwd().readFileAlloc(io, tr_path, gpa, .limited(tbytes + 16));
    defer gpa.free(trb);
    if (trb.len != tbytes) return error.BadFiles;

    const src = try std.mem.replaceOwned(u8, gpa, ks.core_exl3_projection_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(src);
    var owned: ?[]u8 = null;
    var text: []const u8 = src;
    if (!device.tensorUnits()) {
        owned = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", src });
        text = owned.?;
    }
    defer if (owned) |o| gpa.free(o);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p31 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_nax31", false);
    defer p31.deinit();
    const pf8 = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_fp8", false);
    defer pf8.deinit();

    const m: usize = 64;
    const x = try device.buffer(m * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer x.deinit();
    for (x.slice(u16, m * k), 0..) |*v, i| {
        const s: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 2654435761) % 2001)) - 1000)) / 1000.0;
        v.* = @truncate(@as(u32, @bitCast(s)) >> 16);
    }
    const sums31 = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums31.deinit();
    const sums8 = try device.buffer(m * n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sums8.deinit();
    const tr = try device.buffer(tbytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer tr.deinit();
    @memcpy(tr.contents()[0..tbytes], trb);
    const w8 = try device.buffer(k * n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer w8.deinit();
    @memcpy(w8.contents()[0 .. k * n], fp8b);
    const sc = try device.buffer(n * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer sc.deinit();
    @memcpy(sc.contents()[0 .. n * 4], scb);
    var args = Exl3Args{
        .k = @intCast(k),
        .n = @intCast(n),
        .rows = @intCast(m),
        .halves = halves,
        .codebook = 2,
        .su_signs = 0,
        .sv_signs = 0,
        .has_bias = 0,
        .output_f32 = 1,
        .input_ids = 0,
        .output_ids = 0,
        .routed = 0,
        .expert_id = 0,
        .repeat = 1,
        .indexed = 0,
        .experts = 0,
        .splits = 1,
        .x_stride = @intCast(k),
        .y_stride = @intCast(n),
        .tile_k_stride = (n / 16) * (halves * 16),
        .tile_n_stride = halves * 16,
    };
    @memset(sums31.slice(u32, m * n), 0);
    _ = try timeShape(queue, p31, &args, x, tr, sums31, null, m, n, 64, 256, 128, 2);
    args.halves = if (is_i8) 0 else 1;
    args.splits = if (std.c.getenv("TF_FP8_WIDE") != null) 8 else 1;
    @memset(sums8.slice(u32, m * n), 0);
    _ = try timeShapeSc(queue, pf8, &args, x, w8, sums8, sc, m, n, 64, 256, 128, 2);
    const a31 = sums31.slice(f32, m * n);
    const a8 = sums8.slice(f32, m * n);
    var se: f64 = 0;
    var sb: f64 = 0;
    var worst: f64 = 0;
    for (0..m * n) |i| {
        const d = @as(f64, a8[i]) - @as(f64, a31[i]);
        se += d * d;
        sb += @as(f64, a31[i]) * @as(f64, a31[i]);
        worst = @max(worst, @abs(d) / @max(1.0, @abs(@as(f64, a31[i]))));
    }
    var agree: usize = 0;
    for (0..m) |r| {
        var b31: usize = 0;
        var b8: usize = 0;
        for (1..n) |c| {
            if (a31[r * n + c] > a31[r * n + b31]) b31 = c;
            if (a8[r * n + c] > a8[r * n + b8]) b8 = c;
        }
        if (b31 == b8) agree += 1;
    }
    std.debug.print("real-weight layer agreement: {s} K {d} N {d} halves {d}, {d} rows\n", .{ base, k, n, halves, m });
    std.debug.print("  NRMS fp8-vs-EXL3 = {e:.4}; worst relative pointwise = {e:.4}; per-row argmax {d}/{d} = {d:.2}%\n", .{ std.math.sqrt(se / sb), worst, agree, m, 100.0 * @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(m)) });
    // The bf16-destination twin must be the same body with one cast at the end. Run it on the same bytes and
    // activation and compare the bf16 output against the f32 body's own sums.
    if (std.c.getenv("TF_BENCH_FP8B") != null) {
        const pb = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul_prefill_fp8b", false);
        defer pb.deinit();
        const yb = try device.buffer(m * n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer yb.deinit();
        @memset(yb.contents()[0 .. m * n * 2], 0);
        var ab = args;
        ab.output_f32 = 0;
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pb);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w8, 0, 1);
        e.setBuffer(yb, 0, 2);
        e.setBytes(std.mem.asBytes(&ab), 3);
        e.setBuffer(sc, 0, 4);
        e.dispatchGroups(.{ .width = (n + 63) / 64, .height = (m + 127) / 128, .depth = 1 }, .{ .width = 256 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("  fp8b dispatch failed: {s}\n", .{msg});
        const yv = yb.slice(u16, m * n);
        var bse: f64 = 0;
        var bsb: f64 = 0;
        var bworst: f64 = 0;
        var bad: usize = 0;
        for (0..m * n) |i| {
            const got: f32 = @bitCast(@as(u32, yv[i]) << 16);
            const d = @as(f64, got) - @as(f64, a8[i]);
            bse += d * d;
            bsb += @as(f64, a8[i]) * @as(f64, a8[i]);
            bworst = @max(bworst, @abs(d) / @max(1.0, @abs(@as(f64, a8[i]))));
            if (d != 0) bad += 1;
        }
        std.debug.print("  fp8b vs fp8 f32: NRMS {e:.3}, worst {e:.3}, exact {d}/{d}\n", .{
            std.math.sqrt(bse / bsb), bworst, m * n - bad, m * n,
        });
    }
}
