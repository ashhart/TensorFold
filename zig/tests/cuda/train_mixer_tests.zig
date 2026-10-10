//! Sliding Weights' mixer gradients (train_mixers.cu, train.cu's routing) at Nemotron's shapes against f64 references.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const ops = nemotron.train_ops;
const gm = nemotron.glue_math;

const D = 2688;
const E = 128;
const top_k = 6;
const NS = 8;
const H = 32; // attention heads
const KV = 2;
const HD = 128;
const QD = H * HD;
const NQKV = QD + 2 * KV * HD;
const MH = 64; // Mamba heads
const DH = 64;
const G = 8;
const N = 128;
const XD = MH * DH;
const CD = XD + 2 * G * N;
const PW = XD + CD + MH;
const eps: f32 = 1e-5;

pub const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    t: ops.Train,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    worst: f64 = 0,

    pub fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    pub fn zeros(r: *Rig, comptime T: type, n: usize) !u64 {
        const host = try r.a.alloc(T, n);
        @memset(std.mem.sliceAsBytes(host), 0);
        return r.dev(T, host);
    }

    pub fn bfs(r: *Rig, n: usize, scale: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    pub fn f32s(r: *Rig, n: usize, lo: f32, hi: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
        return out;
    }

    pub fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        try r.s.synchronize();
        const out = try r.a.alloc(T, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n * @sizeOf(T)), "cuMemcpyDtoH");
        return out;
    }

    /// The f32 device values at `ptr` against `want`: the largest difference over the largest reference value.
    pub fn close(r: *Rig, what: []const u8, ptr: u64, want: []const f64, tol: f64) !void {
        const got = try r.back(f32, ptr, want.len);
        var diff: f64 = 0;
        var top: f64 = 0;
        for (got, want) |x, y| {
            diff = @max(diff, @abs(x - y));
            top = @max(top, @abs(y));
        }
        const rel = diff / @max(top, 1e-30);
        r.worst = @max(r.worst, rel);
        std.debug.print("RESULT {s}: largest difference {e:.3} of the largest value {e:.3} ({e:.2})\n", .{ what, diff, top, rel });
        try check.expect(rel < tol, "{s} within {e:.0} of its f64 reference ({e:.2})", .{ what, tol, rel });
    }
};

pub fn bf(v: u16) f64 {
    return gm.bf16ToF32(v);
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var k = try nemotron.kernels.Kernels.load(gpu.gpa, gpu.io, gpu.ctx, null);
    try k.loadTrain();
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .t = .{ .f = &k.train, .s = s, .d = gpu.d }, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    try attention(&r, 37);
    try route(&r, 37);
    try mamba(&r, 21);
    check.pass("train-mixers: every gradient within {e:.2} of its f64 reference", .{r.worst});
}

/// Causal attention's backward with shared kv heads, q, k and v read from one qkv row.
fn attention(r: *Rig, rows: usize) !void {
    const qkv = try r.bfs(rows * NQKV, 1.0);
    const dout = try r.f32s(rows * QD, -1, 1);
    const scale: f64 = 1 / @sqrt(@as(f64, HD));
    const want = try r.a.alloc(f64, rows * NQKV);
    @memset(want, 0);
    const p = try r.a.alloc(f64, rows);
    const dp = try r.a.alloc(f64, rows);
    for (0..H) |h| {
        const kv = h / (H / KV);
        for (0..rows) |i| {
            var top: f64 = -std.math.inf(f64);
            for (0..i + 1) |j| {
                var sc: f64 = 0;
                var dv: f64 = 0;
                for (0..HD) |d| {
                    sc += bf(qkv[i * NQKV + h * HD + d]) * bf(qkv[j * NQKV + QD + kv * HD + d]);
                    dv += dout[i * QD + h * HD + d] * bf(qkv[j * NQKV + QD + KV * HD + kv * HD + d]);
                }
                p[j] = sc * scale;
                dp[j] = dv;
                top = @max(top, p[j]);
            }
            var total: f64 = 0;
            for (p[0 .. i + 1]) |*x| {
                x.* = @exp(x.* - top);
                total += x.*;
            }
            var mix: f64 = 0;
            for (p[0 .. i + 1], dp[0 .. i + 1]) |*x, y| {
                x.* /= total;
                mix += x.* * y;
            }
            for (0..i + 1) |j| {
                const ds = p[j] * (dp[j] - mix);
                for (0..HD) |d| {
                    want[i * NQKV + h * HD + d] += scale * ds * bf(qkv[j * NQKV + QD + kv * HD + d]);
                    want[j * NQKV + QD + kv * HD + d] += scale * ds * bf(qkv[i * NQKV + h * HD + d]);
                    want[j * NQKV + QD + KV * HD + kv * HD + d] += p[j] * dout[i * QD + h * HD + d];
                }
            }
        }
    }
    const hs: ops.Heads = .{ .rows = @intCast(rows), .heads = H, .kv_heads = KV, .dim = HD, .nqkv = NQKV, .scale = @floatCast(scale) };
    const qd = try r.dev(u16, qkv);
    const od = try r.dev(f32, dout);
    const probs = try r.zeros(f32, H * rows * rows);
    const dsc = try r.zeros(f32, H * rows * rows);
    const out = try r.zeros(f32, rows * NQKV);
    try r.t.attnQ(qd, od, probs, dsc, out, hs);
    try r.t.attnKv(qd, od, probs, dsc, out, hs);
    try r.close("attention dq, dk, dv", out, want, 1e-4);
}

/// The routing weights' gradient through normalized sigmoid scores into the router's input, added to dx.
fn route(r: *Rig, rows: usize) !void {
    const sk = 6;
    const part = try r.f32s(sk * rows * E, -0.5, 0.5);
    const ids = try r.a.alloc(i32, rows * NS);
    for (0..rows) |t| {
        var used: [E]bool = @splat(false);
        for (0..top_k) |j| {
            var e = r.rng.uintLessThan(usize, E);
            while (used[e]) e = r.rng.uintLessThan(usize, E);
            used[e] = true;
            ids[t * NS + j] = @intCast(e);
        }
        ids[t * NS + top_k] = E;
        ids[t * NS + top_k + 1] = E + 1;
    }
    const ys = try r.bfs(rows * NS * D, 1.0);
    const g = try r.f32s(rows * D, -1, 1);
    const gate = try r.bfs(E * D, 0.05);
    const dx0 = try r.f32s(rows * D, -1, 1);
    const want = try r.a.alloc(f64, rows * D);
    for (0..rows) |t| {
        var a: [top_k]f64 = undefined;
        var p: [top_k]f64 = undefined;
        var total: f64 = 0;
        for (0..top_k) |j| {
            a[j] = 0;
            for (0..D) |i| a[j] += g[t * D + i] * bf(ys[(t * NS + j) * D + i]);
            var z: f64 = 0;
            for (0..sk) |sl| z += part[(sl * rows + t) * E + @as(usize, @intCast(ids[t * NS + j]))];
            p[j] = 1 / (1 + @exp(-z));
            total += p[j];
        }
        var mix: f64 = 0;
        for (0..top_k) |j| mix += a[j] * p[j] / total;
        for (0..D) |i| {
            var sum: f64 = dx0[t * D + i];
            for (0..top_k) |j| sum += 2.5 / total * (a[j] - mix) * p[j] * (1 - p[j]) * bf(gate[@as(usize, @intCast(ids[t * NS + j])) * D + i]);
            want[t * D + i] = sum;
        }
    }
    const dx = try r.dev(f32, dx0);
    try r.t.routeBack(try r.dev(f32, part), sk, try r.dev(i32, ids), try r.dev(u16, ys), try r.dev(f32, g), try r.dev(u16, gate), dx, rows, E, top_k, NS, D, 2.5);
    try r.close("route back", dx, want, 1e-4);
}

/// The Mamba-2 mixer from its in-projection's rows to its gated norm, in f64: the objective sum(dn * out).
const Mixer = struct {
    rows: usize,
    cw: []const f32,
    cb: []const f32,
    dtb: []const f32,
    a: []const f32,
    d: []const f32,
    w: []const u16,
    dn: []const f32,

    fn loss(m: Mixer, al: std.mem.Allocator, proj: []const f64, act_out: ?[]u16) !f64 {
        const rows = m.rows;
        const act = try al.alloc(f64, rows * CD);
        defer al.free(act);
        for (0..rows) |t| for (0..CD) |c| {
            var pre: f64 = m.cb[c];
            for (0..4) |j| if (t + j >= 3) {
                pre += m.cw[j * CD + c] * proj[(t + j - 3) * PW + XD + c];
            };
            act[t * CD + c] = pre / (1 + @exp(-pre));
            if (act_out) |o| o[t * CD + c] = gm.f32ToBf16(@floatCast(act[t * CD + c]));
        };
        const y = try al.alloc(f64, rows * XD);
        defer al.free(y);
        const st = try al.alloc(f64, DH * N);
        defer al.free(st);
        for (0..MH) |h| {
            const g = h / (MH / G);
            @memset(st, 0);
            for (0..rows) |t| {
                const v = proj[t * PW + XD + CD + h] + m.dtb[h];
                const dt = @max(v, 0) + @log(1 + @exp(-@abs(v)));
                const decay = @exp(dt * m.a[h]);
                for (0..DH) |p| {
                    const x = act[t * CD + h * DH + p];
                    var out: f64 = 0;
                    for (0..N) |n| {
                        const at = p * N + n;
                        st[at] = decay * st[at] + dt * x * act[t * CD + XD + g * N + n];
                        out += st[at] * act[t * CD + XD + G * N + g * N + n];
                    }
                    y[t * XD + h * DH + p] = out + m.d[h] * x;
                }
            }
        }
        var total: f64 = 0;
        const width = XD / G;
        for (0..rows) |t| for (0..G) |g| {
            var sq: f64 = 0;
            for (0..width) |j| {
                const i = g * width + j;
                const z = proj[t * PW + i];
                const gated = y[t * XD + i] * z / (1 + @exp(-z));
                sq += gated * gated;
            }
            const inv = 1 / @sqrt(sq / @as(f64, @floatFromInt(width)) + eps);
            for (0..width) |j| {
                const i = g * width + j;
                const z = proj[t * PW + i];
                total += m.dn[t * XD + i] * bf(m.w[i]) * inv * y[t * XD + i] * z / (1 + @exp(-z));
            }
        };
        return total;
    }
};

/// The Mamba-2 backward (dt, scan, gate, conv, raw dt) against the f64 mixer's central differences at sampled columns.
fn mamba(r: *Rig, rows: usize) !void {
    const proj16 = try r.bfs(rows * PW, 1.0);
    const proj = try r.a.alloc(f64, rows * PW);
    for (proj, proj16) |*x, v| x.* = bf(v);
    for (0..rows) |t| for (0..MH) |h| {
        proj[t * PW + XD + CD + h] = -2 + 2 * r.rng.float(f64); // raw dt near the softplus knee
        proj16[t * PW + XD + CD + h] = gm.f32ToBf16(@floatCast(proj[t * PW + XD + CD + h]));
        proj[t * PW + XD + CD + h] = bf(proj16[t * PW + XD + CD + h]);
    };
    const m: Mixer = .{ .rows = rows, .cw = try r.f32s(4 * CD, -0.5, 0.5), .cb = try r.f32s(CD, -0.1, 0.1), .dtb = try r.f32s(MH, -1, 0), .a = try r.f32s(MH, -2, -0.2), .d = try r.f32s(MH, 0.5, 1.5), .w = try r.bfs(XD, 1.0), .dn = try r.f32s(rows * XD, -1, 1) };
    const act16 = try r.a.alloc(u16, rows * CD);
    _ = try m.loss(r.a, proj, act16);
    const shape: ops.Shape = .{ .rows = @intCast(rows), .heads = MH, .dh = DH, .groups = G, .n = N, .inner = XD, .conv = CD, .proj = PW, .lo = 0, .hi = std.math.inf(f32) };
    const pd = try r.dev(u16, proj16);
    const actd = try r.dev(u16, act16);
    const dtb = try r.dev(f32, m.dtb);
    const ad = try r.dev(f32, m.a);
    const dd = try r.dev(f32, m.d);
    const dt = try r.zeros(f32, rows * MH);
    const ytot = try r.zeros(f32, rows * XD);
    const ckpt = try r.zeros(f32, MH * (rows / 16 + 1) * DH * N);
    const states = try r.zeros(f32, MH * 16 * DH * N);
    const dytot = try r.zeros(f32, rows * XD);
    const dproj = try r.zeros(f32, rows * PW);
    const dact = try r.zeros(f32, rows * CD);
    const dbp = try r.zeros(f32, rows * MH * N);
    const dcp = try r.zeros(f32, rows * MH * N);
    const ddt = try r.zeros(f32, rows * MH);
    try r.t.dt(pd, dtb, dt, shape);
    try r.t.ssmFwd(actd, dt, ad, dd, ytot, ckpt, shape);
    try r.t.gateBack(ytot, pd, try r.dev(u16, m.w), try r.dev(f32, m.dn), dytot, dproj, shape, eps);
    try r.t.ssmBack(actd, dt, ad, dd, dytot, ckpt, states, dact, dbp, dcp, ddt, shape);
    try r.t.ssmBc(dbp, dcp, dact, shape);
    try r.t.convBack(dact, pd, try r.dev(f32, m.cw), try r.dev(f32, m.cb), dproj, shape);
    try r.t.dtBack(pd, dtb, ddt, dproj, shape);
    const got = try r.back(f32, dproj, rows * PW);
    // central differences at columns of z, x, B, C and dt, early and late rows
    const cols = [_]usize{ 5, 1500, XD + 7, XD + 2000, XD + XD + 3, XD + XD + G * N + 9, XD + CD + 2, XD + CD + 40 };
    var diff: f64 = 0;
    var top: f64 = 0;
    for (cols) |col| for ([_]usize{ 0, rows / 2, rows - 1 }) |t| {
        const at = t * PW + col;
        const h = 1e-4 * @max(1, @abs(proj[at]));
        const keep = proj[at];
        proj[at] = keep + h;
        const up = try m.loss(r.a, proj, null);
        proj[at] = keep - h;
        const down = try m.loss(r.a, proj, null);
        proj[at] = keep;
        const want = (up - down) / (2 * h);
        diff = @max(diff, @abs(got[at] - want));
        top = @max(top, @abs(want));
        std.debug.print("RESULT mamba dproj row {d} col {d}: {e:.5} against central {e:.5}\n", .{ t, col, got[at], want });
    };
    const rel = diff / @max(top, 1e-30);
    r.worst = @max(r.worst, rel);
    std.debug.print("RESULT mamba backward: largest difference {e:.3} of the largest value {e:.3} ({e:.2})\n", .{ diff, top, rel });
    try check.expect(rel < 2e-2, "the Mamba backward within 2% of the f64 mixer's central differences ({e:.2})", .{rel});
}
