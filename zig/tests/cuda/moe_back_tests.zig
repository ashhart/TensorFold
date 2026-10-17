//! moe-back: routed and shared SwiGLU experts and their router (cuda.train_q4), MLX stacks and packed, against float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const q4 = cuda.train_q4;
const rig = @import("train_back_rig.zig");
const Rig = rig.Rig;
const HQ4 = rig.HQ4;
const Timer = rig.Timer;
const D = rig.D;
const eps = rig.eps;
const row_counts = rig.row_counts;
const tol_col = rig.tol_col;
const tol_l2 = rig.tol_l2;
const bf16 = rig.bf16;
const bf = rig.bf;
const rb = rig.rb;
const rf = rig.rf;
const sig = rig.sig;
const silu = rig.silu;
const siluGrad = rig.siluGrad;
const dot = rig.dot;
const axpy = rig.axpy;
const dotF = rig.dotF;
const rms = rig.rms;
const toF32 = rig.toF32;
const toBf16 = rig.toBf16;
const parallel = rig.parallel;
const mib = rig.mib;

const E = 512;
const W = 640;
const TOPK = 10;
const SLOTS = TOPK + 1;
/// Pocket neurons a family's case adds (fn-moe-back), and the stride of their activation rows.
const pocket_n = 16;
const pocket_stride = 128;

const Moe = struct {
    router: []u16, // [E + 1, D] bf16: the routed experts' rows, then the shared expert's gate
    gate: HQ4, // [E W, D]
    up: HQ4, // [E W, D]
    down: HQ4, // [E D, W]
    sgate: HQ4, // [W, D]
    sup: HQ4,
    sdown: HQ4, // [D, W]
    pg: []f32, // the pocket [p, D]
    pu: []f32,
    pd: []f32,
    p: usize,

    const Mats = struct { g: *const HQ4, u: *const HQ4, d: *const HQ4, go: usize, do: usize };

    fn expert(m: *const Moe, e: usize) Mats {
        if (e == E) return .{ .g = &m.sgate, .u = &m.sup, .d = &m.sdown, .go = 0, .do = 0 };
        return .{ .g = &m.gate, .u = &m.up, .d = &m.down, .go = e * W, .do = e * D };
    }

    fn logit(m: *const Moe, e: usize, x: []const f64) f64 {
        var s: f64 = 0;
        for (0..D) |i| s += bf(m.router[e * D + i]) * x[i];
        return s;
    }
};

/// One row's slots: top_k by logit (lower id first), their softmax, the shared gate sigmoid(logit E); bf16 if `round`.
fn route(m: *const Moe, x: []const f64, round: bool, ids: []i32, wts: []f64) void {
    var l: [E]f64 = undefined;
    for (0..E) |e| l[e] = m.logit(e, x);
    var taken: [E]bool = @splat(false);
    for (0..TOPK) |k| {
        var best: usize = E;
        for (0..E) |e| if (!taken[e] and (best == E or l[e] > l[best])) {
            best = e;
        };
        taken[best] = true;
        ids[k] = @intCast(best);
    }
    var total: f64 = 0;
    for (0..TOPK) |k| total += @exp(l[@intCast(ids[k])] - l[@intCast(ids[0])]);
    for (0..TOPK) |k| {
        const w = @exp(l[@intCast(ids[k])] - l[@intCast(ids[0])]) / total;
        wts[k] = if (round) rb(w) else w;
    }
    ids[TOPK] = E;
    const ls = m.logit(E, x);
    wts[TOPK] = if (round) rb(sig(rb(ls))) else sig(ls);
}

/// The reference on one row from g = dL/d the MoE output: dx, each slot's a = dL/dw, dlog, the pocket's output.
fn moeBack(m: *const Moe, x: []const f64, g: []const f64, ids: []const i32, wts: []const f64, dx: []f64, a: []f64, dlog: []f64, pout: []f64) void {
    @memset(dx, 0);
    var gv: [W]f64 = undefined;
    var uv: [W]f64 = undefined;
    var dact: [W]f64 = undefined;
    for (0..SLOTS) |k| {
        const mt = m.expert(@intCast(ids[k]));
        for (0..W) |i| {
            gv[i] = mt.g.rowDot(mt.go + i, x);
            uv[i] = mt.u.rowDot(mt.go + i, x);
        }
        @memset(&dact, 0);
        for (0..D) |d| mt.d.rowAxpy(mt.do + d, g[d], &dact);
        var s: f64 = 0;
        for (0..W) |i| {
            s += dact[i] * silu(gv[i]) * uv[i];
            mt.g.rowAxpy(mt.go + i, wts[k] * dact[i] * uv[i] * siluGrad(gv[i]), dx);
            mt.u.rowAxpy(mt.go + i, wts[k] * dact[i] * silu(gv[i]), dx);
        }
        a[k] = s;
    }
    const sg = wts[TOPK];
    @memset(pout, 0);
    for (0..m.p) |j| {
        const zg = dotF(m.pg[j * D ..][0..D], x);
        const zu = dotF(m.pu[j * D ..][0..D], x);
        const e = dotF(m.pd[j * D ..][0..D], g);
        const cg = sg * e * siluGrad(zg) * zu;
        const cu = sg * e * silu(zg);
        for (0..D) |d| {
            dx[d] += cg * m.pg[j * D + d] + cu * m.pu[j * D + d];
            pout[d] += sg * silu(zg) * zu * m.pd[j * D + d];
        }
        a[TOPK] += silu(zg) * zu * e;
    }
    var l: [SLOTS]f64 = undefined;
    for (0..SLOTS) |k| l[k] = m.logit(@intCast(ids[k]), x);
    var top: f64 = -std.math.inf(f64);
    for (0..TOPK) |k| top = @max(top, l[k]);
    var total: f64 = 0;
    for (0..TOPK) |k| total += @exp(l[k] - top);
    var mix: f64 = 0;
    for (0..TOPK) |k| mix += @exp(l[k] - top) / total * a[k];
    for (0..TOPK) |k| dlog[k] = @exp(l[k] - top) / total * (a[k] - mix);
    const ss = sig(l[TOPK]);
    dlog[TOPK] = a[TOPK] * ss * (1 - ss);
    for (0..SLOTS) |k| {
        const e: usize = @intCast(ids[k]);
        for (0..D) |i| dx[i] += dlog[k] * bf(m.router[e * D + i]);
    }
}

/// L = g . the MoE output on one row (its experts fixed by ids), a float64 forward without rounding.
fn moeLoss(m: *const Moe, x: []const f64, g: []const f64, ids: []const i32) f64 {
    var l: [SLOTS]f64 = undefined;
    for (0..SLOTS) |k| l[k] = m.logit(@intCast(ids[k]), x);
    var top: f64 = -std.math.inf(f64);
    for (0..TOPK) |k| top = @max(top, l[k]);
    var total: f64 = 0;
    for (0..TOPK) |k| total += @exp(l[k] - top);
    const sg = sig(l[TOPK]);
    var loss: f64 = 0;
    var act: [W]f64 = undefined;
    for (0..SLOTS) |k| {
        const mt = m.expert(@intCast(ids[k]));
        for (0..W) |i| act[i] = silu(mt.g.rowDot(mt.go + i, x)) * mt.u.rowDot(mt.go + i, x);
        var y: f64 = 0;
        for (0..D) |d| y += g[d] * mt.d.rowDot(mt.do + d, &act);
        loss += y * if (k < TOPK) @exp(l[k] - top) / total else sg;
    }
    for (0..m.p) |j| loss += sg * silu(dotF(m.pg[j * D ..][0..D], x)) * dotF(m.pu[j * D ..][0..D], x) * dotF(m.pd[j * D ..][0..D], g);
    return loss;
}

const MoeCase = struct {
    m: *const Moe,
    x: []const f64,
    g: []const f64,
    ids: []const i32,
    wts: []const f64,
    dx: []f64,
    a: []f64,
    dlog: []f64,
    pout: []f64,

    fn init(al: std.mem.Allocator, m: *const Moe, rows: usize, x: []const f64, g: []const f64, ids: []const i32, wts: []const f64) !MoeCase {
        var c: MoeCase = .{ .m = m, .x = x, .g = g, .ids = ids, .wts = wts, .dx = try al.alloc(f64, rows * D), .a = try al.alloc(f64, rows * SLOTS), .dlog = try al.alloc(f64, rows * SLOTS), .pout = try al.alloc(f64, rows * D) };
        parallel(rows, @as(*const MoeCase, &c), row);
        return c;
    }

    fn row(c: *const MoeCase, r: usize, _: usize) void {
        moeBack(c.m, c.x[r * D ..][0..D], c.g[r * D ..][0..D], c.ids[r * SLOTS ..][0..SLOTS], c.wts[r * SLOTS ..][0..SLOTS], c.dx[r * D ..][0..D], c.a[r * SLOTS ..][0..SLOTS], c.dlog[r * SLOTS ..][0..SLOTS], c.pout[r * D ..][0..D]);
    }
};

const RouteCtx = struct { m: *const Moe, x: []const f64, round: bool, ids: []i32, wts: []f64 };
fn routeRow(c: *const RouteCtx, r: usize, _: usize) void {
    route(c.m, c.x[r * D ..][0..D], c.round, c.ids[r * SLOTS ..][0..SLOTS], c.wts[r * SLOTS ..][0..SLOTS]);
}

/// The layer's case: `family` void for the core's layer alone, else a family's Ops with its pocket beside the experts.
fn moeTest(r: *Rig, core: q4.Ops, family: anytype) !void {
    const F = @TypeOf(family);
    const with_pocket = comptime F != void;
    const P: usize = if (with_pocket) pocket_n else 0;
    const R = row_counts[row_counts.len - 1];
    const amp_in = @sqrt(3.0 / @as(f64, D));
    var m: Moe = .{
        .router = try r.a.alloc(u16, (E + 1) * D),
        .gate = try r.q4(E * W, D, 2 * amp_in),
        .up = try r.q4(E * W, D, 2 * amp_in),
        .down = try r.q4(E * D, W, @sqrt(3.0 / @as(f64, W))),
        .sgate = try r.q4(W, D, 2 * amp_in),
        .sup = try r.q4(W, D, 2 * amp_in),
        .sdown = try r.q4(D, W, @sqrt(3.0 / @as(f64, W))),
        .pg = try r.f32s(P * D, -3 * amp_in, 3 * amp_in),
        .pu = try r.f32s(P * D, -3 * amp_in, 3 * amp_in),
        .pd = try r.f32s(P * D, -0.05, 0.05),
        .p = P,
    };
    for (m.router) |*v| v.* = bf16(r.uni(-6, 6) / @sqrt(@as(f64, D)));
    const x = try r.f64s(R * D, -1, 1, .bf);
    const g = try r.f64s(R * D, -1, 1, .f);
    const ids = try r.a.alloc(i32, R * SLOTS);
    const wts = try r.a.alloc(f64, R * SLOTS);
    parallel(R, &RouteCtx{ .m = &m, .x = x, .round = true, .ids = ids, .wts = wts }, routeRow);
    // the float64 reference on two rows of smooth weights, against central differences in x and the pocket's outputs
    {
        const sw = try r.a.alloc(f64, R * SLOTS);
        parallel(64, &RouteCtx{ .m = &m, .x = x, .round = false, .ids = try r.a.alloc(i32, R * SLOTS), .wts = sw }, routeRow);
        const c = try MoeCase.init(r.a, &m, 64, x, g, ids, sw);
        var num: std.ArrayList(f64) = .empty;
        var ref: std.ArrayList(f64) = .empty;
        var nump: std.ArrayList(f64) = .empty;
        var refp: std.ArrayList(f64) = .empty;
        const step = 1e-4;
        for ([_]usize{ 1, 50 }) |row| {
            const xr = try r.a.dupe(f64, x[row * D ..][0..D]);
            const gr = g[row * D ..][0..D];
            const ir = ids[row * SLOTS ..][0..SLOTS];
            for ([_]usize{ 0, 777, 1600, 2559 }) |i| {
                const v = xr[i];
                xr[i] = v + step;
                const lp = moeLoss(&m, xr, gr, ir);
                xr[i] = v - step;
                const lm = moeLoss(&m, xr, gr, ir);
                xr[i] = v;
                try num.append(r.a, (lp - lm) / (2 * step));
                try ref.append(r.a, c.dx[row * D + i]);
            }
            // a pocket neuron's output vector: dL/dpd_j[i] = gate act_j g[i]
            const j = row % @as(usize, @max(P, 1));
            if (P > 0) for ([_]usize{ 9, 2000 }) |i| {
                const v = m.pd[j * D + i];
                const hi: f32 = v + 1e-2;
                const lo: f32 = v - 1e-2;
                m.pd[j * D + i] = hi;
                const lp = moeLoss(&m, xr, gr, ir);
                m.pd[j * D + i] = lo;
                const lm = moeLoss(&m, xr, gr, ir);
                m.pd[j * D + i] = v;
                try nump.append(r.a, (lp - lm) / (@as(f64, hi) - @as(f64, lo)));
                try refp.append(r.a, sw[row * SLOTS + TOPK] * silu(dotF(m.pg[j * D ..][0..D], xr)) * dotF(m.pu[j * D ..][0..D], xr) * gr[i]);
            };
        }
        r.fd("moe dx", num.items, ref.items, rms(c.dx));
        if (P > 0) r.fd("moe pocket output vectors", nump.items, refp.items, 0);
    }
    const c = try MoeCase.init(r.a, &m, R, x, g, ids, wts);
    // the device side
    const e: q4.Experts = blk: {
        const gq = try m.gate.dev(r);
        const uq = try m.up.dev(r);
        const dq = try m.down.dev(r);
        const sg = try m.sgate.dev(r);
        const su = try m.sup.dev(r);
        const sd = try m.sdown.dev(r);
        break :blk .{ .gate = .{ gq.w, gq.s, gq.b }, .up = .{ uq.w, uq.s, uq.b }, .down = .{ dq.w, dq.s, dq.b }, .shared_gate = .{ sg.w, sg.s, sg.b }, .shared_up = .{ su.w, su.s, su.b }, .shared_down = .{ sd.w, sd.s, sd.b }, .routed = E, .width = W, .dim = D };
    };
    const router = try r.dev(u16, m.router);
    // the forward's activation rows for the most rows a case runs
    const pocket = if (with_pocket) F.Pocket{ .gate = try r.dev(f32, m.pg), .up = try r.dev(f32, m.pu), .down = try r.dev(f32, m.pd), .n = P, .act = try r.zeros(row_counts[row_counts.len - 1] * pocket_stride * 4) } else {};
    const pk = try packExperts(r, &m);
    std.debug.print("RESULT moe weights: {d:.1} MiB (512 routed experts as MLX stacks and packed, the shared expert, the router, a pocket of {d})\n", .{ mib(r.bytes), P });
    for (row_counts) |rows| {
        const pairs = rows * SLOTS;
        const wts32 = try toF32(r.a, wts[0 .. rows * SLOTS]);
        const kept: q4.MoeKept = .{ .x = try r.dev(u16, try toBf16(r.a, x[0 .. rows * D])), .ids = try r.dev(i32, ids[0 .. rows * SLOTS]), .wts = try r.dev(f32, wts32) };
        const gd = try r.dev(f32, try toF32(r.a, g[0 .. rows * D]));
        const before = r.bytes;
        const dx = try r.zeros(rows * D * 4);
        const t = try moeScratch(r, rows);
        const scratch = r.bytes - before;
        const row_of = try r.a.alloc(i32, @as(usize, @max(P, 1)));
        for (row_of, 0..) |*v, j| v.* = if (j == 3) -1 else @intCast((j * 37 + 5) % rows);
        const rows_dev = try r.dev(i32, row_of);
        const grad = try r.zeros(@as(usize, @max(P, 1)) * D * 4);
        const pout = try r.zeros(rows * D * 4);
        var tm = try Timer.init(r);
        defer tm.deinit();
        try tm.start(r);
        if (with_pocket) try family.moeBack(e, router, kept, pocket, gd, dx, t, rows, TOPK) else try core.moeBack(e, router, kept, gd, dx, t, rows, TOPK);
        if (with_pocket) try family.pocketGrad(pocket, kept, gd, rows_dev, 1.0, grad, D, TOPK);
        const ms = try tm.stop(r);
        if (with_pocket) try family.pocketRows(pocket, kept, pout, true, rows, D, TOPK);
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows dx", .{rows}), try r.back(f32, dx, rows * D), c.dx[0 .. rows * D], D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows slot scores a = dL/dw", .{rows}), try r.back(f32, t.a, pairs), c.a[0..pairs], SLOTS, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows router logits' gradients", .{rows}), try r.back(f32, t.dlog, pairs), c.dlog[0..pairs], SLOTS, tol_col, tol_l2);
        if (with_pocket) r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows pocket forward", .{rows}), try r.back(f32, pout, rows * D), c.pout[0 .. rows * D], D, tol_col, tol_l2);
        // the pocket's output vectors at their rows
        const want = try r.a.alloc(f64, P * D);
        @memset(want, 0);
        for (row_of[0..P], 0..) |row, j| if (row >= 0) {
            const ri: usize = @intCast(row);
            const xr = x[ri * D ..][0..D];
            const k = @as(f64, wts32[ri * SLOTS + TOPK]) * silu(dotF(m.pg[j * D ..][0..D], xr)) * dotF(m.pu[j * D ..][0..D], xr);
            for (0..D) |i| want[j * D + i] = k * g[ri * D + i];
        };
        if (with_pocket) r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows pocket output vectors' gradients", .{rows}), try r.back(f32, grad, P * D), want, D, tol_col, tol_l2);
        std.debug.print("RESULT moe {d} rows: {d:.3} ms on the GPU, {d:.1} MiB of gradients and scratch\n", .{ rows, ms, mib(scratch) });
        // the same layer over the packed experts the engine keeps: the reference's tolerance, and the stacks' bits
        const dx2 = try r.zeros(rows * D * 4);
        const t2 = try moeScratch(r, rows);
        try tm.start(r);
        if (with_pocket) try family.moeBackPacked(pk, router, kept, pocket, gd, dx2, t2, rows, TOPK) else try core.moeBackPacked(pk, router, kept, gd, dx2, t2, rows, TOPK);
        const ms2 = try tm.stop(r);
        const got2 = try r.back(f32, dx2, rows * D);
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows packed dx", .{rows}), got2, c.dx[0 .. rows * D], D, tol_col, tol_l2);
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(got2), std.mem.sliceAsBytes(try r.back(f32, dx, rows * D)));
        r.checks += 1;
        if (!same) r.bad += 1;
        std.debug.print("RESULT moe {d} rows packed: {d:.3} ms, dx {s} the MLX stacks' bits{s}\n", .{ rows, ms2, if (same) "equals" else "differs from", if (same) "" else "  <- FAIL" });
    }
}

fn moeScratch(r: *Rig, rows: usize) !q4.MoeScratch {
    const pairs = rows * SLOTS;
    return .{
        .logits = try r.zeros(rows * (E + 1) * 4),
        .order = try r.zeros(pairs * 4),
        .inv = try r.zeros(pairs * 4),
        .offsets = try r.zeros((E + 1) * 4),
        .tiles = try r.zeros(2 * q4.tileCap(pairs, E + 1) * 4),
        .gu = try r.zeros(pairs * 2 * W * 4),
        .dact = try r.zeros(pairs * W * 4),
        .dgu = try r.zeros(pairs * 2 * W * 4),
        .dxp = try r.zeros(pairs * D * 4),
        .a = try r.zeros(pairs * 4),
        .dlog = try r.zeros(rows * SLOTS * 4),
    };
}

/// experts_pack.cu's kernel at group 32, as Flash Next's engine packs its experts.
const pack_symbol: [:0]const u8 = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiii";

/// Bytes of one expert's packed [n, k] blocks (experts_pack.cu at group 32: 160 words a 32 x 32 block).
fn packedBytes(n: usize, k: usize) usize {
    return (n / 32) * (k / 32) * 160 * 4;
}

/// The routed stack's experts, then the shared one, packed as the engine packs them at load.
fn packOne(r: *Rig, f: cuda.Function, routed: *const HQ4, shared: *const HQ4, n: usize, k: usize) !u64 {
    const per = packedBytes(n, k);
    const out = try r.zeros((E + 1) * per);
    inline for (.{ .{ routed, E, 0 }, .{ shared, 1, E * per } }) |c| {
        const m = c[0];
        var a: cuda.Args = .{};
        for ([_]u64{ try r.dev(u32, m.words), try r.dev(u16, m.sc), try r.dev(u16, m.bi), out + c[2] }) |v| a.add(v);
        for ([_]usize{ n, k / 8, k / 32, n / 32 }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(k / 32), .y = @intCast(n / 32), .z = c[1] }, .block = .{ .x = 160 } }, r.s, &a);
    }
    return out;
}

/// The engine's packed experts: gate | up block by block, down alone, the shared expert at index E.
fn packExperts(r: *Rig, m: *const Moe) !q4.PackedExperts {
    var mod = try cuda.Module.load(r.gpu.d, cuda.kernels.experts_pack);
    defer mod.unload();
    const f = try mod.function(pack_symbol);
    const gate = try packOne(r, f, &m.gate, &m.sgate, W, D);
    const up = try packOne(r, f, &m.up, &m.sup, W, D);
    const down = try packOne(r, f, &m.down, &m.sdown, D, W);
    const block = 160 * 4;
    const blocks = (E + 1) * packedBytes(W, D) / block;
    const g = try r.back(u8, gate, blocks * block);
    const u = try r.back(u8, up, blocks * block);
    const both = try r.a.alloc(u8, 2 * blocks * block);
    for (0..blocks) |b| {
        @memcpy(both[2 * b * block ..][0..block], g[b * block ..][0..block]);
        @memcpy(both[(2 * b + 1) * block ..][0..block], u[b * block ..][0..block]);
    }
    return .{ .up = try r.dev(u8, both), .down = down, .routed = E, .width = W, .dim = D };
}

pub fn run(gpu: Gpu) !void {
    try runCase(gpu, "moe-back", void, {});
}

/// The case on the GPU: `Ops` void for the core's layer, else a family's Ops over `kernels`, with its pocket.
pub fn runCase(gpu: Gpu, name: []const u8, comptime Ops: type, kernels: anytype) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP {s}: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{name});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_e0e0);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var kq = try q4.Kernels.load(gpu.d);
    defer kq.deinit();
    const t0 = check.now(gpu.io);
    try moeTest(&r, .{ .k = &kq, .s = s }, if (Ops == void) {} else Ops{ .k = kernels, .s = s });
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    try check.expect(r.bad == 0, "{s}: {d} of {d} checks outside tolerance", .{ name, r.bad, r.checks });
    check.pass("{s}: {d} checks within tolerance ({d:.1} s with the float64 references)", .{ name, r.checks, secs });
}
