//! GLM's KDA vs a float64 recurrence rounding where the kernel does; fused = three-kernel bytes; replay rebuilds state.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const bf16 = qr.helpers.bf16Of;

const kda = glm.kda;
const heads = 32;
const n = heads * kda.dk;
const dk = kda.dk;
const p_stride = 3 * n + 64;
const lower = -5.0;
const eps = 1e-6;

fn floats(r: *Rig, count: usize, lo: f32, hi: f32) ![]f32 {
    const out = try r.a.alloc(f32, count);
    for (out) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
    return out;
}

/// The value rounded to bf16, as a float64.
fn rb(x: f64) f64 {
    return bf(bf16(@floatCast(x)));
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

const Inputs = struct { p: []u16, a: []u16, g: []u16, cs: []u16, cw: []f32, state: []f32, a_log: []f32, dt_bias: []f32, norm: []u16 };

/// float64 chain a head: conv+silu, L2 q/k, decay, bf16 beta, delta rule, y = bf16(S q), RMS norm, gate. Out [rows][n].
fn reference(r: *Rig, in: Inputs, rows: usize) ![]f64 {
    const out = try r.a.alloc(f64, rows * n);
    const s = try r.a.alloc(f64, dk * dk);
    var q: [dk]f64 = undefined;
    var k: [dk]f64 = undefined;
    var v: [dk]f64 = undefined;
    var g: [dk]f64 = undefined;
    var y: [dk]f64 = undefined;
    const cc = 3 * n;
    for (0..heads) |h| {
        for (s, in.state[h * dk * dk ..][0 .. dk * dk]) |*d, x| d.* = x;
        const rate = @exp(@as(f64, in.a_log[h]));
        for (0..rows) |row| {
            for (0..3) |part| for (0..dk) |i| {
                const c = part * n + h * dk + i;
                var acc: f64 = 0;
                for (0..4) |tap| {
                    const at = row + tap;
                    const x = if (at < 3) bf(in.cs[at * cc + c]) else bf(in.p[(at - 3) * p_stride + c]);
                    acc += @as(f64, in.cw[c * 4 + tap]) * x;
                }
                const act = rb(acc * sigmoid(acc));
                switch (part) {
                    0 => q[i] = act,
                    1 => k[i] = act,
                    else => v[i] = act,
                }
            };
            for ([_]*[dk]f64{ &q, &k }, [_]f64{ 1 / @sqrt(@as(f64, dk)), 1 }) |x, extra| {
                var ss: f64 = 0;
                for (x) |e| ss += e * e;
                const inv = 1 / @sqrt(ss + 1e-6) * extra;
                for (x) |*e| e.* *= inv;
            }
            for (&g, 0..) |*e, i| e.* = @exp(lower * sigmoid(rate * (bf(in.a[row * n + h * dk + i]) + in.dt_bias[h * dk + i])));
            const beta = rb(sigmoid(bf(in.p[row * p_stride + 3 * n + h])));
            for (0..dk) |j| {
                const srow = s[j * dk ..][0..dk];
                var kv: f64 = 0;
                for (srow, g, k) |*e, gi, ki| {
                    e.* *= gi;
                    kv += e.* * ki;
                }
                const delta = (v[j] - kv) * beta;
                for (srow, k) |*e, ki| e.* += ki * delta;
            }
            var ss: f64 = 0;
            for (&y, 0..) |*e, j| {
                var o: f64 = 0;
                for (s[j * dk ..][0..dk], q) |e2, qi| o += e2 * qi;
                e.* = rb(o);
                ss += e.* * e.*;
            }
            const rinv = 1 / @sqrt(ss / dk + eps);
            for (y, 0..) |e, t| out[row * n + h * dk + t] = bf(in.norm[t]) * e * rinv * sigmoid(bf(in.g[row * n + h * dk + t]));
        }
    }
    return out;
}

/// Fused and three-kernel chains, each replayed; fused out vs float64 within a bf16 ulp (boundaries) + 2^-8 of row max.
fn window(r: *Rig, ks: *const kda.Kernels, rows: usize) !void {
    const in: Inputs = .{
        .p = try r.bfs(rows * p_stride, 1.0), // q | k | v, then beta's logits at 3 n
        .a = try r.bfs(rows * n, 2.0),
        .g = try r.bfs(rows * n, 2.0),
        .cs = try r.bfs(3 * 3 * n, 1.0),
        .cw = try floats(r, 3 * n * 4, -0.5, 0.5),
        .state = try floats(r, heads * kda.dv * dk, -0.05, 0.05),
        .a_log = try floats(r, heads, -1, 1),
        .dt_bias = try floats(r, n, -0.5, 0.5),
        .norm = try r.bfs(dk, 1.0),
    };
    const d = [_]u64{
        try r.dev(std.mem.sliceAsBytes(in.p)),     try r.dev(std.mem.sliceAsBytes(in.a)),       try r.dev(std.mem.sliceAsBytes(in.g)),
        try r.dev(std.mem.sliceAsBytes(in.cs)),    try r.dev(std.mem.sliceAsBytes(in.cw)),      try r.dev(std.mem.sliceAsBytes(in.state)),
        try r.dev(std.mem.sliceAsBytes(in.a_log)), try r.dev(std.mem.sliceAsBytes(in.dt_bias)), try r.dev(std.mem.sliceAsBytes(in.norm)),
    };
    const ch: kda.Chain = .{ .p = d[0], .p_stride = p_stride, .b_off = 3 * n, .a = d[1], .a_stride = n, .g = d[2], .g_stride = n, .conv_state = d[3], .conv_w = d[4], .state_in = d[5], .a_log = d[6], .dt_bias = d[7], .norm_w = d[8], .eps = eps, .lower = lower, .heads = heads };
    const sizes = [_]usize{ rows * n * 2, heads * kda.dv * dk * 4, rows * n * 4, rows * n * 2, rows * n * 4, rows * heads * 4 };
    const what = [_][]const u8{ "out", "state", "saved k", "saved v", "saved decay", "saved beta" };
    var runs: [2][6]u64 = undefined;
    for (&runs, [_]bool{ false, true }) |*o, wide| {
        for (o, sizes) |*b, size| b.* = try r.zeros(size);
        const w: ?kda.Wide = if (wide) .{ .q = try r.zeros(rows * n * 4), .y = try r.zeros(rows * n * 2) } else null;
        try kda.chain(ks, r.s, ch, rows, .{ .out = o[0], .state_out = o[1], .k = o[2], .v = o[3], .g = o[4], .b = o[5] }, w);
        const replayed = try r.zeros(sizes[1]);
        try kda.replay(ks, r.s, d[5], o[2], o[3], o[4], o[5], rows, heads, replayed);
        try ref.sameRows(r, if (wide) "KDA three-kernel chain: the replay rebuilds its state" else "KDA fused chain: the replay rebuilds its state", o[1], replayed, sizes[1]);
    }
    var name: [96]u8 = undefined;
    for (runs[0], runs[1], sizes, what) |fused, three, size, label|
        try ref.sameRows(r, try std.fmt.bufPrint(&name, "KDA {d} rows: the three-kernel chain's {s} equal the fused chain's", .{ rows, label }), fused, three, size);
    const want = try reference(r, in, rows);
    const tol = try r.a.alloc(f64, want.len);
    for (0..rows) |row| {
        var most: f64 = 0;
        for (want[row * n ..][0..n]) |e| most = @max(most, @abs(e));
        for (tol[row * n ..][0..n], want[row * n ..][0..n]) |*t, e| t.* = @abs(e) * std.math.pow(f64, 2, -8) + most * std.math.pow(f64, 2, -8) + 1e-30;
    }
    try ref.closeCols(r, try std.fmt.bufPrint(&name, "KDA {d} rows x {d} heads, fused chain against float64 (column = head x 128 + t)", .{ rows, heads }), runs[0][0], false, want, tol, n);
}

/// `glm-ref`'s KDA: 70 rows (a prompt window) and 3 rows (a decode window) on 32 heads.
pub fn chains(r: *Rig) !void {
    var mod = try cuda.Module.load(r.gpu.d, cuda.kernels.glm_kda);
    defer mod.unload();
    const ks = try kda.Kernels.load(mod);
    for ([_]usize{ 70, 3 }) |rows| try window(r, &ks, rows);
    check.pass("KDA {d} heads: within float64 tolerance, fused and three-kernel chains equal at 70 and 3 rows, replays rebuild the state", .{heads});
}
