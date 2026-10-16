//! Device test: GPTQ 4/128 GEMMs whose activation group means need their low fp16 half, prompt and decode against FP64.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const loader = @import("load.zig");
const kb = @import("xpu_qwen27_kernels");
const ze = q.ze;

const N = 256;
const K = 1024;
const G = 128;
const GC = K / G;
const M = 64;
/// Worst error allowed, relative to the sum of |s·(q − 8)·x| over a column: a lost low half of a group mean exceeds it.
const BOUND = 2e-6;

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    var d: dev.Dev = .{ .g = ze.Gpu.open(0xe223) catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    }, .list = undefined, .k = undefined };
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0);
    defer d.deinit();
    const mg = try d.g.module(&kb.glue);
    const mm = try d.g.module(&kb.gemm128);
    d.k.repack_gptq = try d.kernel(mg, "repack_gptq", 16);
    d.k.dec_wide_g128 = try d.kernel(mm, "w4k1_decode", 64);
    d.k.ngen_wide_g128 = try d.kernel(try d.g.module(&kb.ngen_wide128), "w4g_dpb", 256);
    const rowsum = try d.kernel(mm, "rowsum_hl", 16);

    // weights balanced per group (pairs 8 ± d), so a group's bias −8·s·G cancels its products up to the means' rounding
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    const qv = try gpa.alloc(u8, K * N);
    defer gpa.free(qv);
    for (0..N) |n| for (0..K / 2) |i| {
        const dv = rnd.uintLessThan(u8, 8);
        qv[(2 * i) * N + n] = 8 + dv;
        qv[(2 * i + 1) * N + n] = 8 - dv;
    };
    const qw = try gpa.alloc(u32, K / 8 * N);
    defer gpa.free(qw);
    for (0..K / 8) |w| for (0..N) |n| {
        var o: u32 = 0;
        for (0..8) |i| o |= @as(u32, qv[(8 * w + i) * N + n]) << @intCast(4 * i);
        qw[w * N + n] = o;
    };
    const sc = try gpa.alloc(f16, GC * N);
    defer gpa.free(sc);
    for (sc) |*s| s.* = @floatCast((0.5 + 1.5 * rnd.float(f64)) / 64.0);
    // activations constant over a group but for its last entry, a few ulps off: the group mean needs bits below fp16
    const a = try gpa.alloc(f16, M * K);
    defer gpa.free(a);
    for (0..M) |m| for (0..GC) |g| {
        const mag: f16 = @floatCast(std.math.pow(f64, 10, -3 + 2 * rnd.float(f64)));
        const v: f16 = if (rnd.boolean()) mag else -mag;
        for (0..G) |k| a[m * K + g * G + k] = v;
        const bits: u16 = @bitCast(v);
        a[m * K + g * G + G - 1] = @bitCast(bits + 1 + rnd.uintLessThan(u16, 3));
    };

    const dqw = try d.upload(std.mem.sliceAsBytes(qw));
    const dsc = try d.upload(std.mem.sliceAsBytes(sc));
    const da = try d.upload(std.mem.sliceAsBytes(a));
    const proj: loader.Proj = .{ .w = try d.alloc_(K / 8 * N * 4), .s = try d.alloc_(GC * N * 2), .bg = try d.alloc_(GC * N * 2), .n = N, .k = K, .wide = true, .gs = G };
    try d.launch(d.k.repack_gptq, .{ .x = (N + 15) / 16, .y = K / 8 }, .{ dqw, dsc, proj.w, proj.s, proj.bg, @as(i32, N), @as(i32, K), @as(i32, N), @as(i32, 0), @as(i32, G), @as(i32, 0) });
    const pscr = try d.alloc_(M * N * 4);

    var worst: [2]f64 = .{ 0, 0 };
    for ([_]u32{ M, 1 }, 0..) |rows, path| {
        const mp: u32 = (rows + 7) / 8 * 8;
        const ahi = try d.alloc_(mp * GC * 2);
        const alo = try d.alloc_(mp * GC * 2);
        const out = try d.alloc_(rows * N * 2);
        try d.launch(rowsum, .{ .x = (mp + 15) / 16, .y = GC }, .{ da, ahi, alo, @as(i32, @intCast(rows)), @as(i32, @intCast(mp)), @as(i32, K), @as(i32, GC) });
        try fwd.gemm(&d, pscr, da, ahi, alo, proj, out, rows);
        const host = try gpa.alloc(f16, rows * N);
        defer gpa.free(host);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, host.ptr, out, rows * N * 2, null, 0, null));
        try d.sync();
        for (0..rows) |m| for (0..N) |n| {
            var ref: f64 = 0;
            var mag: f64 = 0;
            for (0..K) |k| {
                const t = @as(f64, sc[(k / G) * N + n]) * (@as(f64, @floatFromInt(qv[k * N + n])) - 8) * @as(f64, a[m * K + k]);
                ref += t;
                mag += @abs(t);
            }
            worst[path] = @max(worst[path], @abs(@as(f64, host[m * N + n]) - ref) / mag);
        };
    }
    std.debug.print("group means: worst error / Σ|terms|: prompt {e:.2}, decode {e:.2} (bound {e:.0})\n", .{ worst[0], worst[1], BOUND });
    if (worst[0] > BOUND or worst[1] > BOUND) return error.GroupMeansLost;
    std.debug.print("ok\n", .{});
}
