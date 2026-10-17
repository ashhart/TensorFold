//! fn-hc-back: Flash Next's hyper-connection backward (fn_back) against float64, a block's and the final mixer's.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");
const check = @import("check.zig");
const Gpu = check.Gpu;
const fb = flashnext.fn_back;
const Ops = fb.Ops;
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

const S = 4;
const WIDE = S * D;
const LOW = 320;
const NV = LOW + S;

const Hc = struct {
    down: HQ4, // [LOW, WIDE]
    inj: ?HQ4, // [S, WIDE]
    up: HQ4, // [WIDE, LOW]
    scale: []f32, // [WIDE] 1 + w

    fn nv(h: *const Hc) usize {
        return LOW + if (h.inj != null) @as(usize, S) else 0;
    }
};

fn makeHc(r: *Rig, inject: bool) !Hc {
    const scale = try r.a.alloc(f32, WIDE);
    for (scale) |*v| v.* = @floatCast(1 + r.uni(-0.3, 0.3));
    return .{
        .down = try r.q4(LOW, WIDE, 4 * @sqrt(3.0 / @as(f64, WIDE))),
        .inj = if (inject) try r.q4(S, WIDE, 4 * @sqrt(3.0 / @as(f64, WIDE))) else null,
        .up = try r.q4(WIDE, LOW, 3 * @sqrt(3.0 / @as(f64, LOW))),
        .scale = scale,
    };
}

/// One row's read-out as glue.py computes it (rounded where it stores bf16 or fp32 when `round`).
fn hcForward(hc: *const Hc, h: []const f64, round: bool, normed: []f64, v: []f64, u: []f64, inj: []f64, mixed: []f64) void {
    for (0..S) |s| {
        const hs = h[s * D ..][0..D];
        const rinv = 1 / @sqrt(dot(hs, hs) / D + eps);
        for (0..D) |i| {
            const x = hs[i] * rinv * hc.scale[s * D + i];
            normed[s * D + i] = if (round) rb(x) else x;
        }
    }
    var act: [LOW]f64 = undefined;
    for (0..LOW) |i| {
        const x = hc.down.rowDot(i, normed);
        v[i] = if (round) rf(x) else x;
        act[i] = if (round) rb(silu(rb(v[i] / S))) else silu(v[i] / S);
    }
    if (hc.inj) |m| for (0..S) |s| {
        const x = m.rowDot(s, normed);
        v[LOW + s] = if (round) rf(x) else x;
        inj[s] = if (round) rb(2 * rb(sig(rb(v[LOW + s] / S)))) else 2 * sig(v[LOW + s] / S);
    };
    for (0..WIDE) |j| {
        const x = hc.up.rowDot(j, &act);
        u[j] = if (round) rb(x) else x;
    }
    for (0..D) |i| {
        var t: f64 = 0;
        for (0..S) |s| t += if (round) rb(rb(sig(u[s * D + i])) * normed[s * D + i]) else sig(u[s * D + i]) * normed[s * D + i];
        mixed[i] = if (round) rb(t / S) else t / S;
    }
}

/// The reference: dL/dh from G = dL/dh' (null on the final mixer), the branch b and gm = dL/dmixed; db and dinj too.
fn hcBack(hc: *const Hc, h: []const f64, normed: []const f64, v: []const f64, u: []const f64, inj: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64, db: []f64, dinj: []f64, dh: []f64) void {
    var dij: [S]f64 = @splat(0);
    if (G) |g| {
        for (0..D) |d| {
            var t: f64 = 0;
            for (0..S) |s| t += g[s * D + d] * inj[s];
            db[d] = t;
        }
        for (0..S) |s| {
            dij[s] = dot(g[s * D ..][0..D], b);
            dinj[s] = dij[s];
        }
    }
    var dn: [WIDE]f64 = undefined;
    var du: [WIDE]f64 = undefined;
    for (0..S) |s| for (0..D) |d| {
        const sg = sig(u[s * D + d]);
        dn[s * D + d] = gm[d] * sg / S;
        du[s * D + d] = gm[d] * normed[s * D + d] * sg * (1 - sg) / S;
    };
    var dact: [LOW]f64 = @splat(0);
    for (0..WIDE) |j| hc.up.rowAxpy(j, du[j], &dact);
    for (0..LOW) |i| hc.down.rowAxpy(i, dact[i] * siluGrad(v[i] / S) / S, &dn);
    if (hc.inj) |m| for (0..S) |s| {
        const sg = sig(v[LOW + s] / S);
        m.rowAxpy(s, dij[s] * 2 * sg * (1 - sg) / S, &dn);
    };
    for (0..S) |s| {
        const hs = h[s * D ..][0..D];
        const rinv = 1 / @sqrt(dot(hs, hs) / D + eps);
        var t: f64 = 0;
        for (0..D) |i| t += hc.scale[s * D + i] * dn[s * D + i] * hs[i];
        for (0..D) |i| {
            const base = if (G) |g| g[s * D + i] else 0;
            dh[s * D + i] = base + rinv * hc.scale[s * D + i] * dn[s * D + i] - rinv * rinv * rinv * hs[i] * t / D;
        }
    }
}

/// L = G . (h + b inject) + gm . mixed for one row, a float64 forward without rounding.
fn hcLoss(hc: *const Hc, h: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64) f64 {
    const normed = std.heap.page_allocator.alloc(f64, 2 * WIDE + NV + S + D) catch unreachable;
    defer std.heap.page_allocator.free(normed);
    const u = normed[WIDE..][0..WIDE];
    const v = normed[2 * WIDE ..][0..NV];
    const inj = normed[2 * WIDE + NV ..][0..S];
    const mixed = normed[2 * WIDE + NV + S ..][0..D];
    hcForward(hc, h, false, normed[0..WIDE], v, u, inj, mixed);
    var l = dot(gm, mixed);
    if (G) |g| for (0..S) |s| for (0..D) |d| {
        l += g[s * D + d] * (h[s * D + d] + b[d] * inj[s]);
    };
    return l;
}

/// Kept tensors and reference gradients of `rows` rows (inputs h, b, G, gm).
const HcCase = struct {
    hc: *const Hc,
    rows: usize,
    round: bool,
    h: []const f64,
    b: []const f64,
    G: ?[]const f64,
    gm: []const f64,
    normed: []f64,
    v: []f64,
    u: []f64,
    inj: []f64,
    mixed: []f64,
    db: []f64,
    dinj: []f64,
    dh: []f64,

    fn init(a: std.mem.Allocator, hc: *const Hc, rows: usize, round: bool, h: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64) !HcCase {
        var c: HcCase = .{ .hc = hc, .rows = rows, .round = round, .h = h, .b = b, .G = G, .gm = gm, .normed = try a.alloc(f64, rows * WIDE), .v = try a.alloc(f64, rows * NV), .u = try a.alloc(f64, rows * WIDE), .inj = try a.alloc(f64, rows * S), .mixed = try a.alloc(f64, rows * D), .db = try a.alloc(f64, rows * D), .dinj = try a.alloc(f64, rows * S), .dh = try a.alloc(f64, rows * WIDE) };
        @memset(c.inj, 0);
        @memset(c.db, 0);
        @memset(c.dinj, 0);
        parallel(rows, @as(*const HcCase, &c), row);
        return c;
    }

    fn row(c: *const HcCase, r: usize, _: usize) void {
        const nv = c.hc.nv();
        const h = c.h[r * WIDE ..][0..WIDE];
        hcForward(c.hc, h, c.round, c.normed[r * WIDE ..][0..WIDE], c.v[r * NV ..][0..nv], c.u[r * WIDE ..][0..WIDE], c.inj[r * S ..][0..S], c.mixed[r * D ..][0..D]);
        hcBack(c.hc, h, c.normed[r * WIDE ..][0..WIDE], c.v[r * NV ..][0..nv], c.u[r * WIDE ..][0..WIDE], c.inj[r * S ..][0..S], c.b[r * D ..][0..D], if (c.G) |g| g[r * WIDE ..][0..WIDE] else null, c.gm[r * D ..][0..D], c.db[r * D ..][0..D], c.dinj[r * S ..][0..S], c.dh[r * WIDE ..][0..WIDE]);
    }
};

/// The reference's gradients in h and b at a few coordinates of two rows, against central differences.
fn hcFd(r: *Rig, name: []const u8, hc: *const Hc, h: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64) !void {
    const rows = 64;
    const c = try HcCase.init(r.a, hc, rows, false, h[0 .. rows * WIDE], b[0 .. rows * D], if (G) |g| g[0 .. rows * WIDE] else null, gm[0 .. rows * D]);
    var num: std.ArrayList(f64) = .empty;
    var ref: std.ArrayList(f64) = .empty;
    var numb: std.ArrayList(f64) = .empty;
    var refb: std.ArrayList(f64) = .empty;
    const step = 1e-4;
    for ([_]usize{ 0, 37 }) |row| {
        const hr = try r.a.dupe(f64, h[row * WIDE ..][0..WIDE]);
        const br = try r.a.dupe(f64, b[row * D ..][0..D]);
        const gr: ?[]const f64 = if (G) |g| g[row * WIDE ..][0..WIDE] else null;
        const gmr = gm[row * D ..][0..D];
        for ([_]usize{ 5, D + 1000, 2 * D + 77, 3 * D + 2559, 2 * D + 1500 }) |i| {
            const x = hr[i];
            hr[i] = x + step;
            const lp = hcLoss(hc, hr, br, gr, gmr);
            hr[i] = x - step;
            const lm = hcLoss(hc, hr, br, gr, gmr);
            hr[i] = x;
            try num.append(r.a, (lp - lm) / (2 * step));
            try ref.append(r.a, c.dh[row * WIDE + i]);
        }
        if (G != null) for ([_]usize{ 3, 1290, 2500 }) |i| {
            const x = br[i];
            br[i] = x + step;
            const lp = hcLoss(hc, hr, br, gr, gmr);
            br[i] = x - step;
            const lm = hcLoss(hc, hr, br, gr, gmr);
            br[i] = x;
            try numb.append(r.a, (lp - lm) / (2 * step));
            try refb.append(r.a, c.db[row * D + i]);
        };
    }
    var buf: [96]u8 = undefined;
    r.fd(try std.fmt.bufPrint(&buf, "{s} dh", .{name}), num.items, ref.items, rms(c.dh));
    if (G != null) r.fd(try std.fmt.bufPrint(&buf, "{s} dbranch", .{name}), numb.items, refb.items, rms(c.db));
}

fn hcKept(r: *Rig, c: *const HcCase, rows: usize) !Ops.HcKept {
    const nv = c.hc.nv();
    const v = try r.a.alloc(f32, rows * nv);
    for (0..rows) |i| for (0..nv) |j| {
        v[i * nv + j] = @floatCast(c.v[i * NV + j]);
    };
    return .{
        .h = try r.dev(u16, try toBf16(r.a, c.h[0 .. rows * WIDE])),
        .normed = try r.dev(u16, try toBf16(r.a, c.normed[0 .. rows * WIDE])),
        .v = try r.dev(f32, v),
        .up = try r.dev(u16, try toBf16(r.a, c.u[0 .. rows * WIDE])),
        .inject = try r.dev(u16, try toBf16(r.a, c.inj[0 .. rows * S])),
    };
}

fn hcDev(r: *Rig, hc: *const Hc) !Ops.Hc {
    return .{ .down = try hc.down.dev(r), .inject = if (hc.inj) |*m| try m.dev(r) else null, .up = try hc.up.dev(r), .scale = try r.dev(f32, hc.scale), .streams = S, .d = D, .low = LOW, .eps = @floatCast(eps) };
}

fn hcScratch(r: *Rig, rows: usize) !Ops.HcScratch {
    return .{ .dn = try r.zeros(rows * WIDE * 4), .du = try r.zeros(rows * WIDE * 4), .dact = try r.zeros(rows * LOW * 4), .dv = try r.zeros(rows * NV * 4) };
}

/// The GPU read-out backward (and write-back's, when the block injects) on `rows` rows against the reference case.
fn hcGpu(r: *Rig, o: Ops, name: []const u8, dhc: Ops.Hc, c: *const HcCase, rows: usize) !void {
    const kept = try hcKept(r, c, rows);
    const before = r.bytes;
    const g = if (c.G) |gg| try r.dev(f32, try toF32(r.a, gg[0 .. rows * WIDE])) else try r.zeros(rows * WIDE * 4);
    const gm = try r.dev(f32, try toF32(r.a, c.gm[0 .. rows * D]));
    const db = try r.zeros(rows * D * 4);
    const dinj = try r.zeros(rows * S * 4);
    const t = try hcScratch(r, rows);
    const scratch = r.bytes - before;
    const branch = try r.dev(u16, try toBf16(r.a, c.b[0 .. rows * D]));
    var tm = try Timer.init(r);
    defer tm.deinit();
    try tm.start(r);
    if (c.G != null) try o.hcBranchBack(dhc, g, kept.inject, branch, db, dinj, rows);
    try o.hcReadoutBack(dhc, kept, gm, if (c.G != null) dinj else 0, g, t, rows);
    const ms = try tm.stop(r);
    var buf: [128]u8 = undefined;
    if (c.G != null) {
        r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dbranch", .{ name, rows }), try r.back(f32, db, rows * D), c.db[0 .. rows * D], D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dinject", .{ name, rows }), try r.back(f32, dinj, rows * S), c.dinj[0 .. rows * S], S, tol_col, tol_l2);
    }
    const got = try r.back(f32, g, rows * WIDE);
    r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dh", .{ name, rows }), got, c.dh[0 .. rows * WIDE], WIDE, tol_col, tol_l2);
    if (c.G) |gg| {
        // the read-out's own part, the residual's gradient (exact) taken out
        const want = try r.a.alloc(f64, rows * WIDE);
        for (got, want, 0..) |*x, *w, i| {
            x.* = @floatCast(@as(f64, x.*) - rf(gg[i]));
            w.* = c.dh[i] - rf(gg[i]);
        }
        r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dh less the residual", .{ name, rows }), got, want, WIDE, tol_col, tol_l2);
    }
    std.debug.print("RESULT {s} {d} rows: {d:.3} ms on the GPU, {d:.1} MiB of gradients and scratch\n", .{ name, rows, ms, mib(scratch) });
}

fn hcTest(r: *Rig, o: Ops) !void {
    const block = try makeHc(r, true);
    const mixer = try makeHc(r, false);
    const R = row_counts[row_counts.len - 1];
    const h = try r.a.alloc(f64, R * WIDE);
    for (0..R) |i| for (0..S) |s| for (0..D) |d| {
        h[(i * S + s) * D + d] = rb(r.uni(-1, 1) * @as(f64, @floatFromInt(@as(usize, 1) << @intCast(s))) * 0.5);
    };
    const b = try r.f64s(R * D, -1, 1, .bf);
    const G = try r.f64s(R * WIDE, -1, 1, .f);
    const gm = try r.f64s(R * D, -1, 1, .f);
    try hcFd(r, "hc block", &block, h, b, G, gm);
    try hcFd(r, "hc final mixer", &mixer, h, b, null, gm);
    const cb = try HcCase.init(r.a, &block, R, true, h, b, G, gm);
    const cm = try HcCase.init(r.a, &mixer, R, true, h, b, null, gm);
    const db = try hcDev(r, &block);
    const dm = try hcDev(r, &mixer);
    std.debug.print("RESULT hc weights: {d:.1} MiB a read-out\n", .{mib(block.down.bytes() + block.inj.?.bytes() + block.up.bytes() + WIDE * 4)});
    for (row_counts) |rows| {
        try hcGpu(r, o, "hc block", db, &cb, rows);
        try hcGpu(r, o, "hc final mixer", dm, &cm, rows);
    }
}

pub fn run(gpu: Gpu) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP fn-hc-back: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_f1a5);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var k = try fb.Kernels.load(gpu.d);
    defer k.deinit();
    const t0 = check.now(gpu.io);
    try hcTest(&r, .{ .k = &k, .s = s });
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    try check.expect(r.bad == 0, "fn-hc-back: {d} of {d} checks outside tolerance", .{ r.bad, r.checks });
    check.pass("fn-hc-back: {d} checks within tolerance ({d:.1} s with the float64 references)", .{ r.checks, secs });
}
