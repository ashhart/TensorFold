//! gdn-back: the Gated DeltaNet backward (cuda.gdn_train) against float64 at Flash Next's shapes, synthetic tensors.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ref = @import("gdn_back_ref.zig");
const Gpu = check.Gpu;
const gt = cuda.gdn_train;

const NK = 16;
const NV = 48;
var HIDDEN: usize = 2560;
var GROUP: usize = 32;
const EPS: f32 = 1e-6;
const DK = gt.dk;
const DV = gt.dv;
const SPLIT = gt.split;

/// Tolerances: fp32 chains against float64 (worst column), and the projections, their left side narrowed to bf16.
const tol_f32 = 1e-3;
const tol_gemm = 3e-2;

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn alloc(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(bytes, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle); // NaN everywhere: an element no kernel writes cannot pass
        return b.ptr;
    }

    fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        const out = try r.a.alloc(T, n);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    fn uniform(r: *Rig, lo: f32, hi: f32) f32 {
        return lo + r.rng.float(f32) * (hi - lo);
    }

    fn bfs(r: *Rig, n: usize, lo: f32, hi: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = ref.bf16Bits(r.uniform(lo, hi));
        return out;
    }

    fn f32s(r: *Rig, n: usize, lo: f32, hi: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = r.uniform(lo, hi);
        return out;
    }

    /// The worst column of got against want, both [rows, cols] (got at stride `gs` from `g0`, want likewise).
    fn compare(r: *Rig, label: []const u8, rows: usize, cols: usize, got: anytype, want: anytype, tol: f64) void {
        var worst: f64 = 0;
        var worst_col: usize = 0;
        var most: f64 = 0;
        var scale: f64 = 0;
        var nonfinite: usize = 0;
        for (0..cols) |c| {
            var e: f64 = 0;
            var m: f64 = 0;
            for (0..rows) |i| {
                const g = got.at(i, c);
                const w = want.at(i, c);
                if (!std.math.isFinite(g)) nonfinite += 1;
                e = @max(e, @abs(g - w));
                m = @max(m, @abs(w));
            }
            most = @max(most, e);
            scale = @max(scale, m);
            const rel = if (m > 0) e / m else if (e > 0) std.math.inf(f64) else 0;
            if (rel > worst) {
                worst = rel;
                worst_col = c;
            }
        }
        const ok = nonfinite == 0 and worst <= tol;
        if (!ok) r.bad += 1;
        std.debug.print("RESULT gdn-back {s:<22} worst column rel {e:9.2} (col {d:>5}), max abs {e:9.2}, ref scale {e:9.2}{s}\n", .{ label, worst, worst_col, most, scale, if (nonfinite > 0) " NONFINITE" else if (ok) "" else "  OVER TOLERANCE" });
    }
};

/// A [rows, cols] window of a flat slice (f64, f32, or bf16 bits): element (i, c) at i * stride + c0 + c.
fn View(comptime T: type) type {
    return struct {
        x: []const T,
        stride: usize,
        c0: usize = 0,
        rows: ?[]const usize = null, // the rows compared, when a subset

        fn at(v: @This(), i: usize, c: usize) f64 {
            const row = if (v.rows) |rs| rs[i] else i;
            const e = v.x[row * v.stride + v.c0 + c];
            return switch (T) {
                f64 => e,
                f32 => e,
                u16 => ref.bf16Value(e),
                else => @compileError("view of " ++ @typeName(T)),
            };
        }
    };
}

fn view(comptime T: type, x: []const T, stride: usize, c0: usize) View(T) {
    return .{ .x = x, .stride = stride, .c0 = c0 };
}

fn widen(a: std.mem.Allocator, comptime T: type, x: []const T) ![]f64 {
    const out = try a.alloc(f64, x.len);
    for (out, x) |*o, v| o.* = switch (T) {
        u16 => ref.bf16Value(v),
        f32 => v,
        else => @compileError("widen"),
    };
    return out;
}

/// bf16 results against the reference's rounded values: elements that differ and the largest difference in ulps.
fn ulps(label: []const u8, got: []const u16, want: []const f64) void {
    var differ: usize = 0;
    var most: u32 = 0;
    for (got, want) |g, w| {
        const wb = ref.bf16Bits(@floatCast(w));
        if (g != wb) {
            differ += 1;
            const gi: i32 = if (g & 0x8000 != 0) -@as(i32, g & 0x7fff) else g;
            const wi: i32 = if (wb & 0x8000 != 0) -@as(i32, wb & 0x7fff) else wb;
            most = @max(most, @abs(gi - wi));
        }
    }
    std.debug.print("RESULT gdn-back forward {s:<14} {d} of {d} bf16 values differ from float64's rounding, at most {d} ulp\n", .{ label, differ, got.len, most });
}

/// A random MLX affine 4-bit [n, k] matrix with `GROUP`-wide groups, values about +-0.04.
const Q4Host = struct {
    w: []u32,
    s: []u16,
    b: []u16,
    n: usize,
    k: usize,

    fn random(r: *Rig, n: usize, k: usize) !Q4Host {
        const m: Q4Host = .{ .w = try r.a.alloc(u32, n * k / 8), .s = try r.a.alloc(u16, n * k / GROUP), .b = try r.a.alloc(u16, n * k / GROUP), .n = n, .k = k };
        for (m.w) |*v| v.* = r.rng.int(u32);
        for (m.s, m.b) |*s, *b| {
            const sc = r.uniform(0.002, 0.008);
            s.* = ref.bf16Bits(sc);
            b.* = ref.bf16Bits(-7.5 * sc * r.uniform(0.8, 1.2));
        }
        return m;
    }

    /// Row i in float64: scale code + bias, undequantized to bf16.
    fn row(m: Q4Host, i: usize, out: []f64) void {
        for (0..m.k) |j| {
            const at = i * m.k + j;
            const code: f64 = @floatFromInt((m.w[at / 8] >> @intCast(4 * (at % 8))) & 0xf);
            out[j] = @as(f64, ref.bf16Value(m.s[at / GROUP])) * code + ref.bf16Value(m.b[at / GROUP]);
        }
    }

    fn upload(m: Q4Host, r: *Rig) !gt.Q4 {
        return .{ .w = try r.dev(std.mem.sliceAsBytes(m.w)), .s = try r.dev(std.mem.sliceAsBytes(m.s)), .b = try r.dev(std.mem.sliceAsBytes(m.b)), .n = m.n, .k = m.k, .group = GROUP };
    }
};

/// out[i] = x[rows[i]] times the matrix [m.n, m.k] in float64; `bf16_operands` rounds both sides first, as the GPU.
fn timesMatrix(a: std.mem.Allocator, x: []const f64, rows: []const usize, m: Q4Host, bf16_operands: bool) ![]f64 {
    const out = try a.alloc(f64, rows.len * m.k);
    @memset(out, 0);
    const w = try a.alloc(f64, m.k);
    for (0..m.n) |n| {
        m.row(n, w);
        if (bf16_operands) for (w) |*v| {
            v.* = ref.bfRound(v.*);
        };
        for (rows, 0..) |r, i| {
            const xv = if (bf16_operands) ref.bfRound(x[r * m.n + n]) else x[r * m.n + n];
            if (xv == 0) continue;
            const o = out[i * m.k ..][0..m.k];
            for (o, w) |*y, wv| y.* += xv * wv;
        }
    }
    return out;
}

const Case = struct { rows: usize, prefix: bool, every: usize }; // every: rows the projections' references cover

fn runCase(r: *Rig, case: Case, kern: *const gt.Kernels) !void {
    const rows = case.rows;
    const s = gt.Shape.init(rows, NK, NV);
    const C: usize = @intCast(s.conv);
    const P: usize = @intCast(s.proj);
    const d: ref.Dims = .{ .rows = rows, .nk = NK, .nv = NV };
    std.debug.print("--- gdn-back: {d} rows, {s}\n", .{ rows, if (case.prefix) "after a prefix (conv history and state)" else "from the sequence start" });

    // inputs: projection rows by column kind, the prefix, weights, the layer output's gradient
    const p16 = try r.a.alloc(u16, rows * P);
    for (0..rows) |i| for (0..P) |c| {
        const v = if (c < C) r.uniform(-1.5, 1.5) else r.uniform(-3, 3);
        p16[i * P + c] = ref.bf16Bits(v);
    };
    const hist16: ?[]u16 = if (case.prefix) try r.bfs(3 * C, -1.5, 1.5) else null;
    const cw16 = try r.bfs(C * gt.taps, -0.6, 0.6);
    const a_log = try r.f32s(NV, -2, 0.5);
    const dt_bias = try r.f32s(NV, -1, 1);
    const nw16 = try r.bfs(DV, 0.5, 1.5);
    const init: ?[]f32 = if (case.prefix) try r.f32s(NV * DV * DK, -0.3, 0.3) else null;
    const gy = try r.f32s(rows * HIDDEN, -0.01, 0.01);
    const in_q = try Q4Host.random(r, P, HIDDEN);
    const out_q = try Q4Host.random(r, HIDDEN, NV * DV);

    const layer: gt.Layer = .{
        .nk = NK,
        .nv = NV,
        .hidden = HIDDEN,
        .eps = EPS,
        .conv_w = try r.dev(std.mem.sliceAsBytes(cw16)),
        .a_log = try r.dev(std.mem.sliceAsBytes(a_log)),
        .dt_bias = try r.dev(std.mem.sliceAsBytes(dt_bias)),
        .norm_w = try r.dev(std.mem.sliceAsBytes(nw16)),
        .in_proj = try in_q.upload(r),
        .out_proj = try out_q.upload(r),
    };
    const p_d = try r.dev(std.mem.sliceAsBytes(p16));
    const hist_d: ?u64 = if (hist16) |h| try r.dev(std.mem.sliceAsBytes(h)) else null;
    const init_d: ?u64 = if (init) |x| try r.dev(std.mem.sliceAsBytes(x)) else null;
    const gy_d = try r.dev(std.mem.sliceAsBytes(gy));
    const bytes = gt.Scratch.bytes(rows, NK, NV, HIDDEN);
    const sc = gt.Scratch.carve(try r.alloc(bytes), rows, NK, NV, HIDDEN);
    const dx_d = try r.alloc(rows * HIDDEN * 4);

    const o: gt.Ops = .{ .k = kern, .s = r.s };
    try o.backward(layer, sc, p_d, hist_d, init_d, gy_d, dx_d, rows, false);
    try r.s.synchronize();

    // time: the whole layer, and the layer without its two projections
    var start = try cuda.Event.init(r.gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(r.gpu.d, true);
    defer end.deinit();
    var whole: [5]f64 = undefined;
    var core: [5]f64 = undefined;
    for (&whole, &core) |*w, *cr| {
        try start.record(r.s);
        try o.backward(layer, sc, p_d, hist_d, init_d, gy_d, dx_d, rows, false);
        try end.record(r.s);
        try end.synchronize();
        w.* = try start.elapsedMs(end);
        try start.record(r.s);
        try o.forward(layer, sc, p_d, hist_d, init_d, rows);
        try o.normBack(p_d, layer.norm_w, sc, s, EPS);
        try o.scanBack(sc, s);
        try o.qkBack(sc, s);
        try o.gatesBack(p_d, layer.a_log, layer.dt_bias, sc, s);
        try o.convBack(p_d, hist_d, layer.conv_w, sc, s);
        try end.record(r.s);
        try end.synchronize();
        cr.* = try start.elapsedMs(end);
    }
    const ck_bytes = s.chunks() * NV * DV * DK * 4;
    std.debug.print("RESULT gdn-back time {d} rows: layer backward {d:.3} ms (forward for training, chain and gates {d:.3} ms, the rest the two projections' dequant and GEMM); scratch {d:.1} MiB (checkpoints {d:.1}, replay {d:.1}, dequantized projection {d:.1})\n", .{
        rows,                                                check.median(&whole),                          check.median(&core),
        @as(f64, @floatFromInt(bytes)) / (1 << 20),          @as(f64, @floatFromInt(ck_bytes)) / (1 << 20), @as(f64, @floatFromInt(gt.ck * NV * DV * DK * 4)) / (1 << 20),
        @as(f64, @floatFromInt(P * HIDDEN * 2)) / (1 << 20),
    });
    // each step alone, then the core again: every buffer holds this pass's values
    {
        var ev: [8]cuda.Event = undefined;
        for (&ev) |*e| e.* = try cuda.Event.init(r.gpu.d, true);
        defer for (&ev) |*e| e.deinit();
        try ev[0].record(r.s);
        try o.front(p_d, hist_d, layer.conv_w, layer.a_log, layer.dt_bias, sc, s);
        try ev[1].record(r.s);
        try o.scanFwd(init_d, sc, s);
        try ev[2].record(r.s);
        try o.normBack(p_d, layer.norm_w, sc, s, EPS);
        try ev[3].record(r.s);
        try o.scanBack(sc, s);
        try ev[4].record(r.s);
        try o.qkBack(sc, s);
        try ev[5].record(r.s);
        try o.gatesBack(p_d, layer.a_log, layer.dt_bias, sc, s);
        try ev[6].record(r.s);
        try o.convBack(p_d, hist_d, layer.conv_w, sc, s);
        try ev[7].record(r.s);
        try ev[7].synchronize();
        var ms: [7]f32 = undefined;
        for (&ms, 0..) |*m, i| m.* = try ev[i].elapsedMs(ev[i + 1]);
        std.debug.print("RESULT gdn-back steps {d} rows (ms): front {d:.3}, scan_fwd {d:.3}, norm_back {d:.3}, scan_back {d:.3}, qk_back {d:.3}, gates_back {d:.3}, conv_back {d:.3}\n", .{ rows, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6] });
    }

    const g_act = try r.back(u16, sc.act, rows * C);
    const g_q = try r.back(f32, sc.q, rows * NK * DK);
    const g_k = try r.back(f32, sc.k, rows * NK * DK);
    const g_g = try r.back(f32, sc.g, rows * NV);
    const g_beta = try r.back(f32, sc.beta, rows * NV);
    const g_y = try r.back(u16, sc.y, rows * NV * DV);
    const g_dout = try r.back(f32, sc.dout, rows * NV * DV);
    const g_dy = try r.back(f32, sc.dy, rows * NV * DV);
    const g_dqp = try r.back(f32, sc.dqp, rows * NV * SPLIT * DK);
    const g_dkp = try r.back(f32, sc.dkp, rows * NV * SPLIT * DK);
    const g_dact = try r.back(f32, sc.dact, rows * C);
    const g_dgp = try r.back(f32, sc.dg, rows * NV * SPLIT);
    const g_dbp = try r.back(f32, sc.dbeta, rows * NV * SPLIT);
    // the chain blocks' shares summed in block order, as the gates' backward sums them
    const g_dg = try r.a.alloc(f32, rows * NV);
    const g_dbeta = try r.a.alloc(f32, rows * NV);
    for (g_dg, g_dbeta, 0..) |*x, *y, i| {
        x.* = 0;
        y.* = 0;
        for (0..SPLIT) |p| {
            x.* += g_dgp[i * SPLIT + p];
            y.* += g_dbp[i * SPLIT + p];
        }
    }
    const g_dproj = try r.back(f32, sc.dproj, rows * P);
    const g_dx = try r.back(f32, dx_d, rows * HIDDEN);

    // the reference
    const in: ref.Inputs = .{
        .p = try widen(r.a, u16, p16),
        .hist = if (hist16) |h| try widen(r.a, u16, h) else null,
        .conv_w = try widen(r.a, u16, cw16),
        .a_log = try widen(r.a, f32, a_log),
        .dt_bias = try widen(r.a, f32, dt_bias),
        .norm_w = try widen(r.a, u16, nw16),
        .init = if (init) |x| try widen(r.a, f32, x) else null,
        .eps = EPS,
    };
    const f = try ref.forward(r.a, d, in, true);
    ulps("act", g_act, f.act);
    ulps("beta", blk: {
        const b = try r.a.alloc(u16, rows * NV);
        for (b, g_beta) |*x, v| x.* = ref.bf16Bits(v);
        break :blk b;
    }, f.beta);
    // the rest of the forward from the GPU's conv outputs and betas
    const act64 = try widen(r.a, u16, g_act);
    const beta64 = try widen(r.a, f32, g_beta);
    const fg = try ref.forwardFrom(r.a, d, in, act64, beta64, true);
    ulps("y", g_y, fg.y);
    r.compare("forward q", rows, NK * DK, view(f32, g_q, NK * DK, 0), view(f64, fg.q, NK * DK, 0), tol_f32);
    r.compare("forward k", rows, NK * DK, view(f32, g_k, NK * DK, 0), view(f64, fg.k, NK * DK, 0), tol_f32);
    r.compare("forward g", rows, NV, view(f32, g_g, NV, 0), view(f64, fg.g, NV, 0), tol_f32);
    r.compare("forward y", rows, NV * DV, view(u16, g_y, NV * DV, 0), view(f64, fg.y, NV * DV, 0), 2e-2);

    // the projections on every `every`-th row
    var picked: std.ArrayList(usize) = .empty;
    var i: usize = 0;
    while (i < rows) : (i += case.every) try picked.append(r.a, i);
    const gy64 = try widen(r.a, f32, gy);
    const dout_ref = try timesMatrix(r.a, gy64, picked.items, out_q, false);
    const sub = picked.items;
    r.compare("dout (o_proj back)", sub.len, NV * DV, View(f32){ .x = g_dout, .stride = NV * DV, .rows = sub }, view(f64, dout_ref, NV * DV, 0), tol_gemm);
    const dout_b16 = try timesMatrix(r.a, gy64, sub, out_q, true);
    r.compare("dout, bf16 operands", sub.len, NV * DV, View(f32){ .x = g_dout, .stride = NV * DV, .rows = sub }, view(f64, dout_b16, NV * DV, 0), tol_f32);

    // the backward at the GPU's forward point
    const y64 = try widen(r.a, u16, g_y);
    const dout64 = try widen(r.a, f32, g_dout);
    const gr = try ref.backward(r.a, d, in, act64, beta64, y64, dout64, true);
    const dq_sum = try r.a.alloc(f64, rows * NK * DK);
    const dk_sum = try r.a.alloc(f64, rows * NK * DK);
    for (0..rows) |row| for (0..NK) |hk| for (0..DK) |e| {
        var x: f64 = 0;
        var y: f64 = 0;
        for (hk * (NV / NK)..(hk + 1) * (NV / NK)) |h| {
            for (0..SPLIT) |p| {
                x += g_dqp[((row * NV + h) * SPLIT + p) * DK + e];
                y += g_dkp[((row * NV + h) * SPLIT + p) * DK + e];
            }
        }
        dq_sum[(row * NK + hk) * DK + e] = x;
        dk_sum[(row * NK + hk) * DK + e] = y;
    };
    r.compare("dy (norm back)", rows, NV * DV, view(f32, g_dy, NV * DV, 0), view(f64, gr.dy, NV * DV, 0), tol_f32);
    r.compare("dz", rows, NV * DV, view(f32, g_dproj, P, C), view(f64, gr.dproj, P, C), tol_f32);
    r.compare("dq (chain)", rows, NK * DK, view(f64, dq_sum, NK * DK, 0), view(f64, gr.dq, NK * DK, 0), tol_f32);
    r.compare("dk (chain)", rows, NK * DK, view(f64, dk_sum, NK * DK, 0), view(f64, gr.dk, NK * DK, 0), tol_f32);
    r.compare("dv (chain)", rows, NV * DV, view(f32, g_dact, C, 2 * NK * DK), view(f64, gr.dact, C, 2 * NK * DK), tol_f32);
    r.compare("dg (chain)", rows, NV, view(f32, g_dg, NV, 0), view(f64, gr.dg, NV, 0), tol_f32);
    r.compare("dbeta (chain)", rows, NV, view(f32, g_dbeta, NV, 0), view(f64, gr.dbeta, NV, 0), tol_f32);
    r.compare("dact q (L2 back)", rows, NK * DK, view(f32, g_dact, C, 0), view(f64, gr.dact, C, 0), tol_f32);
    r.compare("dact k (L2 back)", rows, NK * DK, view(f32, g_dact, C, NK * DK), view(f64, gr.dact, C, NK * DK), tol_f32);
    r.compare("dproj q (conv back)", rows, NK * DK, view(f32, g_dproj, P, 0), view(f64, gr.dproj, P, 0), tol_f32);
    r.compare("dproj k (conv back)", rows, NK * DK, view(f32, g_dproj, P, NK * DK), view(f64, gr.dproj, P, NK * DK), tol_f32);
    r.compare("dproj v (conv back)", rows, NV * DV, view(f32, g_dproj, P, 2 * NK * DK), view(f64, gr.dproj, P, 2 * NK * DK), tol_f32);
    r.compare("dproj b (beta back)", rows, NV, view(f32, g_dproj, P, d.bAt()), view(f64, gr.dproj, P, d.bAt()), tol_f32);
    r.compare("dproj a (g back)", rows, NV, view(f32, g_dproj, P, d.aAt()), view(f64, gr.dproj, P, d.aAt()), tol_f32);
    const dx_ref = try timesMatrix(r.a, gr.dproj, sub, in_q, false);
    r.compare("dx (in_proj back)", sub.len, HIDDEN, View(f32){ .x = g_dx, .stride = HIDDEN, .rows = sub }, view(f64, dx_ref, HIDDEN, 0), tol_gemm);
    const dx_b16 = try timesMatrix(r.a, try widen(r.a, f32, g_dproj), sub, in_q, true);
    r.compare("dx, bf16 operands", sub.len, HIDDEN, View(f32){ .x = g_dx, .stride = HIDDEN, .rows = sub }, view(f64, dx_b16, HIDDEN, 0), tol_f32);
}

pub fn run(gpu: Gpu) !void {
    return runShape(gpu, 2560, 32);
}
pub fn runShape(gpu: Gpu, hidden: usize, group: usize) !void {
    const old_h = HIDDEN;
    const old_g = GROUP;
    defer {
        HIDDEN = old_h;
        GROUP = old_g;
    }
    HIDDEN = hidden;
    GROUP = group;
    return runCurrent(gpu);
}
fn runCurrent(gpu: Gpu) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP gdn-back: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x6d17_bac4);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, true), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
        r.s.deinit();
    }
    var kern = try gt.Kernels.load(gpu.d);
    defer kern.deinit();

    // the reference against its own central differences: small heads, a prefix, every column kind of P
    {
        var fa: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer fa.deinit();
        const a = fa.allocator();
        for ([_]ref.Dims{ .{ .rows = 6, .nk = 2, .nv = 6 }, .{ .rows = 9, .nk = 1, .nv = 3 } }) |d| {
            const p = try a.alloc(f64, d.rows * d.proj());
            for (p, 0..) |*v, i| v.* = if (i % d.proj() < d.conv()) r.uniform(-1.5, 1.5) else r.uniform(-3, 3);
            const hist = try a.alloc(f64, 3 * d.conv());
            for (hist) |*v| v.* = r.uniform(-1.5, 1.5);
            const cw = try a.alloc(f64, d.conv() * ref.taps);
            for (cw) |*v| v.* = r.uniform(-0.6, 0.6);
            const al = try a.alloc(f64, d.nv);
            const db = try a.alloc(f64, d.nv);
            for (al, db) |*x, *y| {
                x.* = r.uniform(-2, 0.5);
                y.* = r.uniform(-1, 1);
            }
            const nw = try a.alloc(f64, d.dv);
            for (nw) |*v| v.* = r.uniform(0.5, 1.5);
            const s0 = try a.alloc(f64, d.nv * d.dv * d.dk);
            for (s0) |*v| v.* = r.uniform(-0.3, 0.3);
            const dout = try a.alloc(f64, d.rows * d.nv * d.dv);
            for (dout) |*v| v.* = r.uniform(-1, 1);
            const in: ref.Inputs = .{ .p = p, .hist = hist, .conv_w = cw, .a_log = al, .dt_bias = db, .norm_w = nw, .init = s0, .eps = EPS };
            const fd = try ref.finiteCheck(a, d, in, dout, r.rng, 120);
            const ok = fd.worst < 1e-5;
            if (!ok) r.bad += 1;
            std.debug.print("RESULT gdn-back finite differences ({d} rows, {d} key / {d} value heads): 120 coordinates of P, worst |fd - analytic| / max(|analytic|, 1e-3 max) {e:.2}{s}\n", .{ d.rows, d.nk, d.nv, fd.worst, if (ok) "" else "  OVER TOLERANCE" });
        }
    }

    for ([_]Case{ .{ .rows = 64, .prefix = true, .every = 1 }, .{ .rows = 256, .prefix = false, .every = 8 }, .{ .rows = 37, .prefix = true, .every = 4 } }) |case| {
        try runCase(&r, case, &kern);
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
    try check.expect(r.bad == 0, "gdn-back: {d} checks over tolerance", .{r.bad});
    check.pass("gdn-back: forward for training and every input gradient within tolerance of float64", .{});
}
