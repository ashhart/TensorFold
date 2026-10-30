//! LFM3-Omni's Metal norms and rotary (om_rmsnorm, om_rmsnorm_1k, om_head_norm, om_layernorm, om_qk_prep, om_rope).
const std = @import("std");
const mtl = @import("metal");
const k = @import("metal_omni_common.zig");

const c = k.c;
const R = k.R;
const Probe = k.Probe;
const eps: f64 = @as(f32, 1e-5);
const sq_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

fn eps32() f32 {
    return 1e-5;
}

/// x as given, then the nonfinite fill's NaN, +Inf, -Inf and both infinities in rows 0 to 3.
fn plant(x: []u16, rows: usize, d: usize) void {
    const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d / 2 }, .{ 2, d - 1 }, .{ 3, 0 }, .{ 3, d - 1 } };
    const put = [_]u16{ k.nan16, k.inf16, k.ninf16, k.inf16, k.ninf16 };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * d + p[1]] = v;
    };
}

/// Row values: uniform in [-1, 1] with a larger one every 64, or (stress) integers in [-32, 32].
fn values(rng: std.Random, x: []u16, fill: k.g.Fill) void {
    if (fill == .stress) return k.intsBf16(rng, x, 32);
    for (x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1));
}

/// rsqrt(ss / n + eps) as the kernels step it.
fn invNorm(ss: R, n: usize) R {
    return ss.div(R.of(@floatFromInt(n))).add(R.of(eps)).rsqrt();
}

// --- om_rmsnorm and om_rmsnorm_1k: variant 0 plain, 1 the residual and its post norm, 2 a weight row a row (mode 1)

const NormArgs = extern struct { D: i32, ld: i32, ldy: i32, eps: f32, mode: i32, ldw: i32, has_r: i32, has_pw: i32 };

fn wide(p: *const Probe) bool {
    return p.variant & 0x10 != 0;
}

fn normPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n != s.k or s.k == 0 or (wide(p) and s.k < 1024)) return false;
    const v = p.variant & 0xf;
    const d = s.k;
    const x = try p.alloc(u16, 0, s.rows * d);
    const w = try p.alloc(u16, 1, if (v == 2) s.rows * d else d);
    const r = try p.alloc(u16, 2, if (v == 1) s.rows * d else 1);
    const pw = try p.alloc(u16, 3, if (v == 1) d else 1);
    _ = try p.alloc(u16, 4, s.rows * d);
    values(rng, x, p.fill);
    k.randomBf16(rng, w, 0.5, 1.5);
    if (v == 1) {
        values(rng, r, p.fill);
        if (p.fill == .stress) @memset(pw, 0) else k.randomBf16(rng, pw, 0.5, 1.5);
    }
    if (p.fill == .nonfinite) plant(x, s.rows, d);
    if (v == 1) try p.out(0, true) else try p.in(0);
    for (1..4) |i| try p.in(i);
    try p.out(4, false);
    return true;
}

fn normLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const v = p.variant & 0xf;
    const d = p.shape.k;
    p.bind(e, &.{ 0, if (v == 1) 2 else null, 1, 4, if (v == 1) 3 else null }, NormArgs{ .D = k.i32_(d), .ld = k.i32_(d), .ldy = k.i32_(d), .eps = eps32(), .mode = if (v == 2) 1 else 0, .ldw = if (v == 2) k.i32_(d) else 0, .has_r = @intFromBool(v == 1), .has_pw = @intFromBool(v == 1) });
    Probe.go(e, .{ p.shape.rows, 1, 1 }, if (wide(p)) 1024 else 256);
}

fn normCount(p: *Probe) usize {
    return p.shape.rows * p.shape.k * @as(usize, if (p.variant & 0xf == 1) 2 else 1);
}

fn normEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const v = p.variant & 0xf;
    const d = p.shape.k;
    const nt: usize = if (wide(p)) 1024 else 256;
    const depth = (d + nt - 1) / nt + 5 + nt / 32 + 1;
    const xs = try p.a.alloc(R, 2 * d);
    defer p.a.free(xs);
    const t = xs[d..];
    const x = p.get(u16, 0);
    const w = p.get(u16, 1);
    const r = p.get(u16, 2);
    const pw = p.get(u16, 3);
    const ys = if (v == 1) out[p.shape.rows * d ..] else out;
    for (0..p.shape.rows) |row| {
        var inv_r = R.of(0);
        if (v == 1) {
            for (t, 0..) |*q, i| q.* = R.bf16(r[row * d + i]).mulExact(R.bf16(r[row * d + i]));
            inv_r = invNorm(k.sum(mode, p.scratch, t, depth), d);
        }
        for (xs[0..d], 0..) |*q, i| {
            q.* = R.bf16(x[row * d + i]);
            if (v == 1) {
                const add = R.bf16(pw[i]).mul(R.bf16(r[row * d + i]).mul(inv_r).bf()).bf();
                q.* = q.add(add).bf();
                out[row * d + i] = q.*;
            }
            t[i] = q.mulExact(q.*);
        }
        const inv = invNorm(k.sum(mode, p.scratch, t, depth), d);
        for (xs[0..d], 0..) |q, i| {
            const wi = R.bf16(if (v == 2) w[row * d + i] else w[i]);
            ys[row * d + i] = if (v == 2) q.mul(inv).mul(wi).bf() else wi.mul(q.mul(inv).bf()).bf();
        }
    }
}

const norm_def: k.Def = .{ .prep = normPrep, .launch = normLaunch, .count = normCount, .eval = normEval };

fn depth256(s: c.Shape) usize {
    return (s.k + 255) / 256 + 14;
}
fn depth1k(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 38;
}

fn normSpec(name: []const u8, is_wide: bool, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = if (is_wide) depth1k else depth256, .ops = 8, .mids = 1, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, if (is_wide) .{ .size = 1024, .min = 1024 } else .{ .size = 256 } }, .served = served, .sums = .{ .bf16 = 0, .why = sq_why } };
}

// --- om_head_norm: rows of T x H heads of hd, in place; shape rows T, n the row's width (H hd), k hd

const HeadArgs = extern struct { T: i32, H: i32, hd: i32, ld: i32, eps: f32 };

fn headPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.k == 0 or s.n % s.k != 0 or s.k > 128) return false;
    const x = try p.alloc(u16, 0, s.rows * s.n);
    const w = try p.alloc(u16, 1, s.k);
    values(rng, x, p.fill);
    k.randomBf16(rng, w, 0.5, 1.5);
    if (p.fill == .nonfinite) plant(x, s.rows, s.n);
    try p.out(0, true);
    try p.in(1);
    return true;
}

fn headLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const h = s.n / s.k;
    p.bind(e, &.{ 0, 1 }, HeadArgs{ .T = k.i32_(s.rows), .H = k.i32_(h), .hd = k.i32_(s.k), .ld = k.i32_(s.n), .eps = eps32() });
    Probe.go(e, .{ k.blocks(s.rows * h * 32, 256), 1, 1 }, 256);
}

fn headCount(p: *Probe) usize {
    return p.shape.rows * p.shape.n;
}

/// One head's RMS norm as om_head_norm and om_qk_prep round it: bf16(w bf16(x inv)).
fn headRow(mode: k.Mode, scratch: []f64, x: []const u16, w: []const u16, t: []R, out: []R) void {
    const hd = x.len;
    for (t, x) |*q, xv| q.* = R.bf16(xv).mulExact(R.bf16(xv));
    const inv = invNorm(k.sum(mode, scratch, t, (hd + 31) / 32 + 6), hd);
    for (out, x, w) |*o, xv, wv| o.* = R.bf16(wv).mul(R.bf16(xv).mul(inv).bf()).bf();
}

fn headEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const t = try p.a.alloc(R, s.k);
    defer p.a.free(t);
    const x = p.get(u16, 0);
    for (0..s.rows * (s.n / s.k)) |hh| headRow(mode, p.scratch, x[hh * s.k ..][0..s.k], p.get(u16, 1), t, out[hh * s.k ..][0..s.k]);
}

fn headDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
}

const head_def: k.Def = .{ .prep = headPrep, .launch = headLaunch, .count = headCount, .eval = headEval };

// --- om_layernorm: the projector's LayerNorm with bias, one rounding

const LnArgs = extern struct { D: i32, eps: f32 };

fn lnPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n != s.k or s.k == 0) return false;
    const x = try p.alloc(u16, 0, s.rows * s.k);
    const w = try p.alloc(u16, 1, s.k);
    const bias = try p.alloc(u16, 2, s.k);
    _ = try p.alloc(u16, 3, s.rows * s.k);
    if (p.fill == .stress) {
        // integers whose row sums are zero: the mean and every deviation exact
        const h = s.k / 2;
        for (0..s.rows) |row| {
            k.intsBf16(rng, x[row * s.k ..][0..h], 64);
            for (0..h) |i| x[row * s.k + h + i] = x[row * s.k + i] ^ 0x8000;
            if (s.k % 2 == 1) x[row * s.k + s.k - 1] = 0;
        }
    } else values(rng, x, p.fill);
    k.randomBf16(rng, w, 0.5, 1.5);
    k.randomBf16(rng, bias, -0.5, 0.5);
    if (p.fill == .nonfinite) plant(x, s.rows, s.k);
    for (0..3) |i| try p.in(i);
    try p.out(3, false);
    return true;
}

fn lnLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1, 2, 3 }, LnArgs{ .D = k.i32_(p.shape.k), .eps = eps32() });
    Probe.go(e, .{ p.shape.rows, 1, 1 }, 256);
}

fn lnCount(p: *Probe) usize {
    return p.shape.rows * p.shape.k;
}

fn lnEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const d = p.shape.k;
    const t = try p.a.alloc(R, 2 * d);
    defer p.a.free(t);
    const dev = t[d..];
    const x = p.get(u16, 0);
    const w = p.get(u16, 1);
    const bias = p.get(u16, 2);
    const depth = depth256(p.shape) - 1;
    for (0..p.shape.rows) |row| {
        for (t[0..d], 0..) |*q, i| q.* = R.bf16(x[row * d + i]);
        const mean = k.sum(mode, p.scratch, t[0..d], depth).div(R.of(@floatFromInt(d)));
        for (dev, 0..) |*q, i| {
            q.* = R.bf16(x[row * d + i]).sub(mean);
            t[i] = q.mulExact(q.*);
        }
        const rstd = invNorm(k.sum(mode, p.scratch, t[0..d], depth + 1), d);
        for (dev, 0..) |q, i| out[row * d + i] = q.mul(rstd).mul(R.bf16(w[i])).add(R.bf16(bias[i])).bf();
    }
}

const ln_def: k.Def = .{ .prep = lnPrep, .launch = lnLaunch, .count = lnCount, .eval = lnEval };

// --- rotary: the stacks' heads, their rotary modes and frequencies

/// The codec (fp32 cos/sin), the backbone's full layers (bf16 cos/sin, half rotated), the encoder, the depth decoder.
const Rot = struct { h: usize, hkv: usize, hd: usize, theta: f32, fraction: f32, pos0: usize, mode: u32 };
const rots = [4]Rot{
    .{ .h = 8, .hkv = 2, .hd = 64, .theta = 1e6, .fraction = 1, .pos0 = 90, .mode = 0 },
    .{ .h = 16, .hkv = 4, .hd = 128, .theta = 1e6, .fraction = 0.5, .pos0 = 300, .mode = 1 },
    .{ .h = 8, .hkv = 4, .hd = 96, .theta = 1e6, .fraction = 1, .pos0 = 0, .mode = 2 },
    .{ .h = 32, .hkv = 8, .hd = 32, .theta = 1e6, .fraction = 1, .pos0 = 0, .mode = 1 },
};

/// The engine's inverse frequencies (spec.zig's invFreq), in fp32.
fn invFreq(out: []f32, r: Rot) void {
    const rot: usize = @as(usize, @intFromFloat(r.fraction * @as(f32, @floatFromInt(r.hd)))) / 2;
    for (out, 0..) |*x, i| x.* = if (i < rot) 1.0 / std.math.pow(f32, r.theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(r.hd))) else 0;
}

fn cos(x: f64) f64 {
    return @cos(x);
}
fn sin(x: f64) f64 {
    return @sin(x);
}

/// A head's pair (a, z) at position `pos` and frequency `inv` in rotary mode `m`, as om_rope rounds it.
fn rotate(a: R, z: R, pos: usize, inv: f32, m: u32) [2]R {
    const fr = R.of(@floatFromInt(pos)).mul(R.of(inv));
    var cs = fr.lip(cos, 1, k.fn_ulps);
    var sn = fr.lip(sin, 1, k.fn_ulps);
    if (m != 0) {
        cs = cs.bf();
        sn = sn.bf();
    }
    if (m == 2) return .{ a.mul(cs).bf().add(z.neg().mul(sn).bf()).bf(), z.mul(cs).bf().add(a.mul(sn).bf()).bf() };
    return .{ a.mul(cs).sub(z.mul(sn)).bf(), z.mul(cs).add(a.mul(sn)).bf() };
}

const RopeArgs = extern struct { T: i32, H: i32, hd: i32, ld: i32, pos0: i32, mode: i32 };

/// om_rope: shape rows T, n heads, k hd; the variant is the mode (and its stack's frequencies).
fn ropePrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    const r = rots[p.variant];
    if (s.k != r.hd or s.n == 0) return false;
    const x = try p.alloc(u16, 0, s.rows * s.n * s.k);
    const inv = try p.alloc(f32, 1, s.k / 2);
    values(rng, x, if (p.fill == .stress) .random else p.fill);
    if (p.fill == .nonfinite) plant(x, s.rows, s.n * s.k);
    invFreq(inv, r);
    try p.out(0, true);
    try p.in(1);
    return true;
}

fn ropeLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const r = rots[p.variant];
    p.bind(e, &.{ 0, 1 }, RopeArgs{ .T = k.i32_(s.rows), .H = k.i32_(s.n), .hd = k.i32_(s.k), .ld = k.i32_(s.n * s.k), .pos0 = k.i32_(r.pos0), .mode = @intCast(r.mode) });
    Probe.go(e, .{ k.blocks(s.rows * s.n * s.k / 2, 256), 1, 1 }, 256);
}

fn ropeCount(p: *Probe) usize {
    return p.shape.rows * p.shape.n * p.shape.k;
}

fn ropeEval(p: *Probe, _: k.Mode, out: []R) !void {
    const s = p.shape;
    const x = p.get(u16, 0);
    const inv = p.get(f32, 1);
    const half = s.k / 2;
    for (0..s.rows) |t| for (0..s.n) |h| {
        const at = (t * s.n + h) * s.k;
        for (0..half) |i| {
            const o = rotate(R.bf16(x[at + i]), R.bf16(x[at + i + half]), rots[p.variant].pos0 + t, inv[i], rots[p.variant].mode);
            out[at + i] = o[0];
            out[at + i + half] = o[1];
        }
    };
}

const rope_def: k.Def = .{ .prep = ropePrep, .launch = ropeLaunch, .count = ropeCount, .eval = ropeEval };

// --- om_qk_prep: the few-row path's heads, normed and rotated, k and v into the caches; variant as `rots`

const QkArgs = extern struct { T: i32, H: i32, Hkv: i32, hd: i32, ld: i32, pos0: i32, mode: i32, nk: i32, eps: f32 };

fn qkLd(r: Rot) usize {
    return r.h * r.hd + 2 * r.hkv * r.hd + r.h;
}

fn qkPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    const r = rots[p.variant];
    if (s.k != r.hd or s.n != r.h or s.rows == 0) return false;
    const ld = qkLd(r);
    const nk = r.hkv * r.hd;
    const x = try p.alloc(u16, 0, s.rows * ld);
    const qn = try p.alloc(u16, 1, r.hd);
    const kn = try p.alloc(u16, 2, r.hd);
    const inv = try p.alloc(f32, 3, r.hd / 2);
    const kc = try p.alloc(u16, 4, (r.pos0 + s.rows) * nk);
    const vc = try p.alloc(u16, 5, (r.pos0 + s.rows) * nk);
    values(rng, x, p.fill);
    k.randomBf16(rng, qn, 0.5, 1.5);
    k.randomBf16(rng, kn, 0.5, 1.5);
    k.randomBf16(rng, kc, -1, 1);
    k.randomBf16(rng, vc, -1, 1);
    if (p.fill == .nonfinite) plant(x, s.rows, ld);
    invFreq(inv, r);
    try p.out(0, true);
    for (1..4) |i| try p.in(i);
    try p.out(4, true);
    try p.out(5, true);
    return true;
}

fn qkLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const r = rots[p.variant];
    p.bind(e, &.{ 0, 1, 2, 3, 4, 5 }, QkArgs{ .T = k.i32_(s.rows), .H = k.i32_(r.h), .Hkv = k.i32_(r.hkv), .hd = k.i32_(r.hd), .ld = k.i32_(qkLd(r)), .pos0 = k.i32_(r.pos0), .mode = @intCast(r.mode), .nk = k.i32_(r.hkv * r.hd), .eps = eps32() });
    Probe.go(e, .{ s.rows * (r.h + r.hkv), 1, 1 }, 32);
}

fn qkCount(p: *Probe) usize {
    const r = rots[p.variant];
    return p.shape.rows * qkLd(r) + 2 * (r.pos0 + p.shape.rows) * r.hkv * r.hd;
}

fn qkEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const r = rots[p.variant];
    const ld = qkLd(r);
    const nk = r.hkv * r.hd;
    const x = p.get(u16, 0);
    const kc_out = out[s.rows * ld ..][0 .. (r.pos0 + s.rows) * nk];
    const vc_out = out[s.rows * ld + kc_out.len ..];
    for (out[0 .. s.rows * ld], x) |*o, v| o.* = R.bf16(v);
    for (kc_out, p.get(u16, 4)) |*o, v| o.* = R.bf16(v);
    for (vc_out, p.get(u16, 5)) |*o, v| o.* = R.bf16(v);
    var t: [128]R = undefined;
    var sv: [128]R = undefined;
    const half = r.hd / 2;
    for (0..s.rows) |row| for (0..r.h + r.hkv) |h| {
        const q = h < r.h;
        const at = row * ld + (if (q) h * r.hd else r.h * r.hd + (h - r.h) * r.hd);
        headRow(mode, p.scratch, x[at..][0..r.hd], p.get(u16, if (q) 1 else 2), t[0..r.hd], sv[0..r.hd]);
        const dst = if (q) out[at..][0..r.hd] else kc_out[(r.pos0 + row) * nk + (h - r.h) * r.hd ..][0..r.hd];
        for (0..half) |i| {
            const o = rotate(sv[i], sv[i + half], r.pos0 + row, p.get(f32, 3)[i], r.mode);
            dst[i] = o[0];
            dst[i + half] = o[1];
        }
        if (!q) for (0..r.hd) |i| {
            vc_out[(r.pos0 + row) * nk + (h - r.h) * r.hd + i] = R.bf16(x[row * ld + r.h * r.hd + nk + (h - r.h) * r.hd + i]);
        };
    };
}

const qk_def: k.Def = .{ .prep = qkPrep, .launch = qkLaunch, .count = qkCount, .eval = qkEval };

// --- the entries, each with the control that drops (or moves) one rounding point

const sq = "for (int i = lane; i < a.hd; i += 32) ";
const rope_shapes = [3][]const c.Shape{
    &.{ .{ .rows = 6, .n = 10, .k = 64 }, .{ .rows = 48, .n = 10, .k = 64 } },
    &.{ .{ .rows = 6, .n = 24, .k = 128 }, .{ .rows = 40, .n = 24, .k = 128 } },
    &.{ .{ .rows = 6, .n = 12, .k = 96 }, .{ .rows = 94, .n = 12, .k = 96 } },
};

pub const entries = [_]k.Entry{
    .{ .def = &norm_def, .spec = normSpec("metal lfm3 omni om_rmsnorm", false, &.{ .{ .rows = 94, .n = 768, .k = 768 }, .{ .rows = 48, .n = 512, .k = 512 }, .{ .rows = 64, .n = 2048, .k = 2048 } }), .file = .omni, .pipe = "om_rmsnorm", .variant = 0, .control = .{ "y[i] = a.mode == 0 ? b(f(W[i]) * rb(v))", "y[i] = a.mode == 0 ? b(f(W[i]) * (v))" }, .what = "drops bf16(x rsqrt)" },
    .{ .def = &norm_def, .spec = normSpec("metal lfm3 omni om_rmsnorm residual, post norm", false, &.{ .{ .rows = 48, .n = 512, .k = 512 }, .{ .rows = 64, .n = 2048, .k = 2048 } }), .file = .omni, .pipe = "om_rmsnorm", .variant = 1, .control = .{ "const float add = a.has_pw ? rb(f(PW[i]) * rb(f(r[i]) * inv_r)) : f(r[i]);", "const float add = a.has_pw ? rb(f(PW[i]) * (f(r[i]) * inv_r)) : f(r[i]);" }, .what = "drops bf16(r rsqrt)" },
    .{ .def = &norm_def, .spec = normSpec("metal lfm3 omni om_rmsnorm_1k", true, &.{.{ .rows = 1, .n = 2048, .k = 2048 }}), .file = .omni, .pipe = "om_rmsnorm_1k", .variant = 0x10, .control = .{ "y[i] = a.mode == 0 ? b(f(W[i]) * rb(v))", "y[i] = a.mode == 0 ? b(f(W[i]) * (v))" }, .what = "drops bf16(x rsqrt)" },
    .{ .def = &norm_def, .spec = normSpec("metal lfm3 omni om_rmsnorm_1k residual, post norm", true, &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 12, .n = 1024, .k = 1024 } }), .file = .omni, .pipe = "om_rmsnorm_1k", .variant = 0x11, .control = .{ "v = rb(v + add);", "v = (v + add);" }, .what = "drops bf16(x + r)" },
    .{ .def = &norm_def, .spec = normSpec("metal lfm3 omni om_rmsnorm_1k row weights", true, &.{.{ .rows = 12, .n = 1024, .k = 1024 }}), .file = .omni, .pipe = "om_rmsnorm_1k", .variant = 0x12, .control = .{ ": b(v * f(W[i]));", ": b(rb(v) * f(W[i]));" }, .what = "adds bf16(x rsqrt) to mode 1" },
    .{ .def = &head_def, .spec = .{ .name = "metal lfm3 omni om_head_norm", .accum = .f32, .depth = headDepth, .ops = 8, .mids = 1, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, .{ .size = 32, .min = 32, .step = 32 } }, .served = &.{ .{ .rows = 40, .n = 2048, .k = 128 }, .{ .rows = 94, .n = 768, .k = 96 }, .{ .rows = 48, .n = 512, .k = 64 }, .{ .rows = 12, .n = 1024, .k = 32 } }, .sums = .{ .bf16 = 0, .why = sq_why } }, .file = .omni, .pipe = "om_head_norm", .control = .{ sq ++ "x[i] = b(f(W[i]) * rb(f(x[i]) * inv));", sq ++ "x[i] = b(f(W[i]) * (f(x[i]) * inv));" }, .what = "drops bf16(x rsqrt)" },
    .{ .def = &ln_def, .spec = .{ .name = "metal lfm3 omni om_layernorm", .accum = .f32, .depth = depth256, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{.{ .rows = 94, .n = 768, .k = 768 }}, .sums = .{ .bf16 = 0, .why = "the deviations' squares cannot cancel and the mean's error moves every output alike, under its ulps" } }, .file = .omni, .pipe = "om_layernorm", .control = .{ "b((f(x[i]) - mean) * rstd * f(W[i]) + f(Bb[i]))", "b(rb((f(x[i]) - mean) * rstd) * f(W[i]) + f(Bb[i]))" }, .what = "adds bf16((x - mean) rstd)" },
    .{ .def = &rope_def, .spec = ropeSpec("metal lfm3 omni om_rope fp32 cos sin (codec)", rope_shapes[0]), .file = .omni, .pipe = "om_rope", .variant = 0, .control = .{ "x[p] = b(xa * c - z * s);", "x[p] = b(xa * c - z * rb(s));" }, .what = "adds bf16(sin) to mode 0" },
    .{ .def = &rope_def, .spec = ropeSpec("metal lfm3 omni om_rope bf16 cos sin (backbone)", rope_shapes[1]), .file = .omni, .pipe = "om_rope", .variant = 1, .control = .{ "    c = rb(c);\n    s = rb(s);", "    c = rb(c);\n    s = (s);" }, .what = "drops bf16(sin)" },
    .{ .def = &rope_def, .spec = ropeSpec("metal lfm3 omni om_rope bf16 steps (encoder)", rope_shapes[2]), .file = .omni, .pipe = "om_rope", .variant = 2, .control = .{ "x[p] = b(rb(xa * c) + rb(-z * s));", "x[p] = b((xa * c) + rb(-z * s));" }, .what = "drops bf16(a cos)" },
    .{ .def = &qk_def, .spec = qkSpec("metal lfm3 omni om_qk_prep fp32 cos sin (codec)", &.{ .{ .rows = 1, .n = 8, .k = 64 }, .{ .rows = 6, .n = 8, .k = 64 } }), .file = .rows, .pipe = "om_qk_prep", .variant = 0, .control = .{ "out[p] = b(xa * c - z * s);", "out[p] = b(xa * c - rb(z * s));" }, .what = "adds bf16(z sin) to mode 0" },
    .{ .def = &qk_def, .spec = qkSpec("metal lfm3 omni om_qk_prep bf16 cos sin (backbone)", &.{ .{ .rows = 1, .n = 16, .k = 128 }, .{ .rows = 16, .n = 16, .k = 128 } }), .file = .rows, .pipe = "om_qk_prep", .variant = 1, .control = .{ "sv[i] = rb(f(w[i]) * rb(f(x[i]) * inv));", "sv[i] = rb(f(w[i]) * (f(x[i]) * inv));" }, .what = "drops bf16(x rsqrt)" },
    .{ .def = &qk_def, .spec = qkSpec("metal lfm3 omni om_qk_prep bf16 cos sin (depth decoder)", &.{.{ .rows = 12, .n = 32, .k = 32 }}), .file = .rows, .pipe = "om_qk_prep", .variant = 3, .control = .{ "      c = rb(c);\n      s = rb(s);", "      c = (c);\n      s = rb(s);" }, .what = "drops bf16(cos)" },
};

fn ropeSpec(name: []const u8, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .none, .depth = k.noSums, .ops = 8, .out = .bf16, .in_place = true, .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "rotations: no sums" } };
}

fn qkDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
}

fn qkSpec(name: []const u8, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = qkDepth, .ops = 16, .mids = 2, .out = .bf16, .in_place = true, .tiles = .{ .{ .size = 16, .min = 1 }, .{}, .{} }, .served = served, .sums = .{ .bf16 = 0, .why = sq_why } };
}
