//! The 27B's input-gradient kernels against float64 (q27-mlp-back, q27-attn-back, q27-head-back), 64 and 256 rows.
const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen35");
const check = @import("check.zig");
const rig = @import("q27_back_rig.zig");
const attn = @import("q27_back_attn.zig");
const to = qwen27.train_ops;
const Ops = to.Ops;
const Gpu = check.Gpu;
const Rig = rig.Rig;
const HQ4 = rig.HQ4;
const D = rig.D;
const I = rig.I;
const NP = rig.NP;
const eps = rig.eps;
const threads = rig.threads;
const row_counts = rig.row_counts;
const tol_col = rig.tol_col;
const tol_l2 = rig.tol_l2;
const bf16 = rig.bf16;
const bf = rig.bf;
const rb = rig.rb;
const sig = rig.sig;
const silu = rig.silu;
const siluGrad = rig.siluGrad;
const dot = rig.dot;
const axpy = rig.axpy;
const rms = rig.rms;
const toF32 = rig.toF32;
const toBf16 = rig.toBf16;
const parallel = rig.parallel;
const normRow = rig.normRow;
const normBack = rig.normBack;

// ---- the dense MLP and its pocket -----------------------------------------------------------------------------------

const Mlp = struct {
    gate: HQ4, // [I, D]
    up: HQ4,
    down: HQ4, // [D, I]
    norm: []f64, // [D] bf16 values
    pg: []f64, // pocket gate rows [NP, D] (fp32 values)
    pu: []f64,
    pd: []f64,
};

/// A case's forward: y the normed rows, gate and up the projections, act, out = down act + the pocket's.
const MlpCase = struct {
    m: *const Mlp,
    rows: usize,
    round: bool,
    h: []const f64,
    y: []f64,
    gate: []f64,
    up: []f64,
    g: []const f64,
    dy: []f64,
    dh: []f64,

    fn init(al: std.mem.Allocator, m: *const Mlp, rows: usize, round: bool, h: []const f64, g: []const f64) !MlpCase {
        var c: MlpCase = .{ .m = m, .rows = rows, .round = round, .h = h, .g = g, .y = try al.alloc(f64, rows * D), .gate = try al.alloc(f64, rows * I), .up = try al.alloc(f64, rows * I), .dy = try al.alloc(f64, rows * D), .dh = try al.alloc(f64, rows * D) };
        parallel(rows, @as(*const MlpCase, &c), forwardRow);
        parallel(rows, @as(*const MlpCase, &c), backRow);
        return c;
    }

    fn forwardRow(c: *const MlpCase, r: usize) void {
        const y = c.y[r * D ..][0..D];
        normRow(c.h[r * D ..][0..D], c.m.norm, c.round, y);
        for (0..I) |i| {
            const a = dot(c.m.gate.row(i), y);
            const b = dot(c.m.up.row(i), y);
            c.gate[r * I + i] = if (c.round) rb(a) else a;
            c.up[r * I + i] = if (c.round) rb(b) else b;
        }
    }

    /// dy = gate^T dgate + up^T dup + the pocket's, then dh through the norm, from g at this point of the forward.
    fn backRow(c: *const MlpCase, r: usize) void {
        const g = c.g[r * D ..][0..D];
        const y = c.y[r * D ..][0..D];
        const dy = c.dy[r * D ..][0..D];
        @memset(dy, 0);
        var dact: [I]f64 = undefined;
        @memset(&dact, 0);
        for (0..D) |o| axpy(&dact, g[o], c.m.down.row(o));
        for (0..I) |i| {
            const gv = c.gate[r * I + i];
            const uv = c.up[r * I + i];
            axpy(dy, dact[i] * uv * siluGrad(gv), c.m.gate.row(i));
            axpy(dy, dact[i] * silu(gv), c.m.up.row(i));
        }
        for (0..NP) |j| {
            const zg = dot(c.m.pg[j * D ..][0..D], y);
            const zu = dot(c.m.pu[j * D ..][0..D], y);
            const e = dot(c.m.pd[j * D ..][0..D], g);
            axpy(dy, e * siluGrad(zg) * zu, c.m.pg[j * D ..][0..D]);
            axpy(dy, e * silu(zg), c.m.pu[j * D ..][0..D]);
        }
        normBack(c.h[r * D ..][0..D], c.m.norm, dy, c.dh[r * D ..][0..D]);
    }
};

/// g . out(row r of h) in float64, the pocket included: the part of the loss row r moves.
fn mlpLoss(m: *const Mlp, h: []const f64, g: []const f64) f64 {
    var y: [D]f64 = undefined;
    normRow(h, m.norm, false, &y);
    var gd: [I]f64 = undefined;
    @memset(&gd, 0);
    for (0..D) |o| axpy(&gd, g[o], m.down.row(o));
    var l: f64 = 0;
    for (0..I) |i| l += gd[i] * silu(dot(m.gate.row(i), &y)) * dot(m.up.row(i), &y);
    for (0..NP) |j| l += dot(m.pd[j * D ..][0..D], g) * silu(dot(m.pg[j * D ..][0..D], &y)) * dot(m.pu[j * D ..][0..D], &y);
    return l;
}

fn mlpTest(r: *Rig, o: Ops) !void {
    const amp = @sqrt(3.0 / @as(f64, D));
    const m: Mlp = blk: {
        const norm = try r.a.alloc(f64, D);
        for (norm) |*v| v.* = rb(1 + r.uni(-0.3, 0.3));
        const pg = try r.f64s(NP * D, -2 * amp, 2 * amp, false);
        const pu = try r.f64s(NP * D, -2 * amp, 2 * amp, false);
        const pd = try r.f64s(NP * D, -0.05, 0.05, false);
        for (pg, pu, pd) |*a, *b, *c| {
            a.* = @as(f32, @floatCast(a.*));
            b.* = @as(f32, @floatCast(b.*));
            c.* = @as(f32, @floatCast(c.*));
        }
        break :blk .{ .gate = try r.q4(I, D, 2 * amp), .up = try r.q4(I, D, 2 * amp), .down = try r.q4(D, I, @sqrt(3.0 / @as(f64, I))), .norm = norm, .pg = pg, .pu = pu, .pd = pd };
    };
    const R = row_counts[row_counts.len - 1];
    const h = try r.f64s(R * D, -2, 2, true);
    const g = try r.f64s(R * D, -1, 1, false);
    for (g) |*v| v.* = @as(f32, @floatCast(v.*));
    {
        // the float64 reference against central differences of the loss in h (rows 2 and 5)
        const c = try MlpCase.init(r.a, &m, 8, false, h[0 .. 8 * D], g[0 .. 8 * D]);
        var num: std.ArrayList(f64) = .empty;
        var want: std.ArrayList(f64) = .empty;
        const step = 1e-5;
        for ([_]usize{ 2, 5 }) |row| for ([_]usize{ 7, 2049, 5119 }) |i| {
            const hr = try r.a.dupe(f64, h[row * D ..][0..D]);
            const v = hr[i];
            hr[i] = v + step;
            const lp = mlpLoss(&m, hr, g[row * D ..][0..D]);
            hr[i] = v - step;
            const lm = mlpLoss(&m, hr, g[row * D ..][0..D]);
            try num.append(r.a, (lp - lm) / (2 * step));
            try want.append(r.a, c.dh[row * D + i]);
        };
        r.fd("mlp dh", num.items, want.items, rms(c.dh));
    }
    const dm: to.Mlp = .{ .gate = try m.gate.dev(r), .up = try m.up.dev(r), .down = try m.down.dev(r) };
    const pk: to.Pocket = .{ .gate = try r.dev(f32, try toF32(r.a, m.pg)), .up = try r.dev(f32, try toF32(r.a, m.pu)), .down = try r.dev(f32, try toF32(r.a, m.pd)), .n = NP };
    const norm_d = try r.dev(u16, try toBf16(r.a, m.norm));
    for (row_counts) |rows| {
        const c = try MlpCase.init(r.a, &m, rows, true, h[0 .. rows * D], g[0 .. rows * D]);
        const kept: to.MlpKept = .{ .y = try r.dev(u16, try toBf16(r.a, c.y)), .gate = try r.dev(u16, try toBf16(r.a, c.gate)), .up = try r.dev(u16, try toBf16(r.a, c.up)) };
        const hd = try r.dev(u16, try toBf16(r.a, h[0 .. rows * D]));
        const gd = try r.dev(f32, try toF32(r.a, g[0 .. rows * D]));
        const t: to.MlpScratch = .{ .dact = try r.zeros(rows * I * 4), .dgate = try r.zeros(rows * I * 4), .dup = try r.zeros(rows * I * 4) };
        const parts = try r.zeros(@max(to.splitsFor(rows, D, I), to.splitsFor(rows, I, D)) * rows * I * 4);
        const dy = try r.zeros(rows * D * 4);
        const dh = try r.zeros(rows * D * 4);
        try o.mlpBack(dm, kept, pk, gd, dy, t, rows, parts);
        try o.rmsBack(hd, norm_d, dy, dh, rows, D, @floatCast(eps));
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "mlp {d} rows dy (MLP and pocket)", .{rows}), try r.back(f32, dy, rows * D), c.dy, D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "mlp {d} rows dh (through the post norm)", .{rows}), try r.back(f32, dh, rows * D), c.dh, D, tol_col, tol_l2);
        // the pocket's own rows forward, and its output vectors' gradient at a row each
        const zero_out = try r.zeros(rows * D * 2);
        try o.pocketRows(pk, kept.y, zero_out, rows, D);
        const want_out = try r.a.alloc(f64, rows * D);
        for (0..rows) |i| {
            const y = c.y[i * D ..][0..D];
            const row = want_out[i * D ..][0..D];
            @memset(row, 0);
            for (0..NP) |j| axpy(row, silu(dot(m.pg[j * D ..][0..D], y)) * dot(m.pu[j * D ..][0..D], y), m.pd[j * D ..][0..D]);
        }
        const got_b = try r.back(u16, zero_out, rows * D);
        const got = try r.a.alloc(f32, rows * D);
        for (got, got_b) |*x, b| x.* = @floatCast(bf(b));
        r.compare(try std.fmt.bufPrint(&buf, "mlp {d} rows pocket output (bf16)", .{rows}), got, want_out, D, tol_col, tol_l2);
        const row_of = try r.a.alloc(i32, NP);
        for (row_of, 0..) |*x, j| x.* = if (j == 3) -1 else @intCast((j * 7) % rows);
        const grad = try r.zeros(NP * D * 4);
        try o.pocketGrad(pk, kept.y, gd, try r.dev(i32, row_of), 0.5, grad, D);
        const want_g = try r.a.alloc(f64, NP * D);
        @memset(want_g, 0);
        for (0..NP) |j| if (row_of[j] >= 0) {
            const rr: usize = @intCast(row_of[j]);
            const y = c.y[rr * D ..][0..D];
            axpy(want_g[j * D ..][0..D], 0.5 * silu(dot(m.pg[j * D ..][0..D], y)) * dot(m.pu[j * D ..][0..D], y), g[rr * D ..][0..D]);
        };
        r.compare(try std.fmt.bufPrint(&buf, "mlp {d} rows pocket output gradient", .{rows}), try r.back(f32, grad, NP * D), want_g, D, 1e-4, 1e-5);
    }
    r.release();
}

// ---- entry ----------------------------------------------------------------------------------------------------------

pub const Case = enum { mlp, attn, head, all };

pub fn run(gpu: Gpu, case: Case) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_27b0);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var k = try to.Kernels.load(gpu.d);
    defer k.deinit();
    const o: Ops = .{ .k = &k, .s = s };
    const t0 = check.now(gpu.io);
    if (case == .mlp or case == .all) try mlpTest(&r, o);
    if (case == .attn or case == .all) try attn.attnTest(&r, o);
    if (case == .head or case == .all) try attn.headTest(&r, o);
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    try check.expect(r.bad == 0, "q27-back {t}: {d} of {d} checks outside tolerance", .{ case, r.bad, r.checks });
    check.pass("q27-back {t}: {d} checks within tolerance ({d:.1} s with the float64 references)", .{ case, r.checks, secs });
}
