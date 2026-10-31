//! id_repo and id_qk under the contract: RePo's angles in fp32, and the heads rotated, normed and rounded to fp16.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");

const c = id.c;
const g = id.g;
const H = id.H;
const D = id.D;
const Range = id.Range;
const Ends = id.Ends;
const hf = id.hf;
const kref = id.kref;

const streams: []const c.Shape = &.{ .{ .rows = 12294, .n = 512, .k = 128 }, .{ .rows = 8192, .n = 512, .k = 128 }, .{ .rows = 4101, .n = 512, .k = 128 } };

// --- id_repo: u = silu(g) c, d = u final^T (48), angles x f pi as cos and sin [rows, 16, 32] --------------------

const pi32: f64 = @as(f32, std.math.pi);

/// d's FMA chain over 128.
fn repoDepth(_: c.Shape) usize {
    return 128 + 1;
}

/// A pair's axis: 10 x, 10 y, then 12 z.
fn axis(j: usize) usize {
    return if (j < 10) 0 else if (j < 20) 1 else 2;
}

pub const Repo = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    rows: usize = 0,
    stress: bool = false,
    /// gate sums then content sums, 128 each a row
    gc: []f32 = &.{},
    /// the final map [48, 128]
    wf: []f32 = &.{},
    freqs: []f32 = &.{},
    inv: f32 = 0.5,
    ins: [3]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: 16 heads x 32 pairs; k: RePo's hidden width.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = repoDepth, .ops = id.exp_ulps + id.div_ulps + 2, .out = .f32, .served = streams };
    }

    pub fn deinit(t: *Repo) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Repo) void {
        t.a.free(t.gc);
        t.a.free(t.wf);
        t.a.free(t.freqs);
        t.gc = &.{};
        t.wf = &.{};
        t.freqs = &.{};
    }
    pub fn kernel(t: *Repo, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Repo);

    pub fn prepare(t: *Repo, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 512 or shape.k != 128) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.stress = fill == .stress;
        t.gc = try t.a.alloc(f32, shape.rows * 256);
        t.wf = try t.a.alloc(f32, 48 * 128);
        t.freqs = try t.a.alloc(f32, 32);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // gates of 20 to 40 (sigmoid 1 in fp32), integer contents: u's halves differ by a gate twice, d small
            for (0..shape.rows) |r| for (0..64) |i| {
                const gr = t.gc[r * 256 ..][0..256];
                gr[i] = id.int(rng, 20, 40) / t.inv;
                gr[i + 64] = gr[i];
                gr[128 + i] = id.sign(rng) * id.int(rng, 1, 8) / t.inv;
                gr[128 + i + 64] = gr[128 + i] + @as(f32, if (i < 2) 1 / t.inv else 0);
            };
            for (0..48) |o| for (0..64) |i| {
                t.wf[o * 128 + i] = id.sign(rng) * id.int(rng, 1, 32);
                t.wf[o * 128 + 64 + i] = -t.wf[o * 128 + i];
            };
            id.randomF32(rng, t.freqs, 0.001, 0.01);
        } else {
            id.randomF32(rng, t.gc, -4, 4);
            id.randomF32(rng, t.wf, -0.2, 0.2);
            id.randomF32(rng, t.freqs, 0.05, 1.5);
        }
        if (fill == .nonfinite) {
            const at = [_]usize{ 0 * 256 + 5, 1 * 256 + 7, 2 * 256 + 128 + 3 };
            for (at, [_]f32{ id.nan32, id.inf32, -id.inf32 }) |p, v| if (p < t.gc.len) {
                t.gc[p] = v;
            };
            t.freqs[4] = id.nan32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.gc));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wf));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.freqs));
        for (&t.outs) |*o| o.* = try t.rig.output(shape.rows * 512 * 4);
        return true;
    }

    pub fn run(t: *Repo) !void {
        try t.lib.need("id_repo");
        const p = t.rig.begin();
        t.lib.k.repo(p.e, kref(t.ins[0]), t.inv, kref(t.ins[1]), kref(t.ins[2]), kref(t.outs[0]), kref(t.outs[1]), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Repo) usize {
        return 2 * t.rows * 512;
    }

    /// Row r's u values in float64 and their ranges through the kernel's exp, add, division and product.
    fn us(t: *const Repo, r: usize, u: *[128]f64, ur: *[128]Range) void {
        const inv: f64 = t.inv;
        for (u, ur, 0..) |*v, *vr, i| {
            const a = @as(f64, t.gc[r * 256 + i]) * inv;
            const cc = @as(f64, t.gc[r * 256 + 128 + i]) * inv;
            v.* = a / (1 + @exp(-a)) * cc;
            // a >= 20: 1 + exp(-a) is 1 in fp32 and a / 1 is a, so u = a c exactly
            if (t.stress) {
                v.* = a * cc;
                vr.* = Range.of(v.*);
                continue;
            }
            const den = Range.of(@exp(-a)).fp32(id.exp_ulps).add(Range.of(1)).fp32(0.5);
            vr.* = Range.of(a).mul(den.recip()).fp32(id.div_ulps).mul(Range.of(cc)).fp32(0.5);
        }
    }

    /// Row r's outputs from its 48 offsets d (ranges dr).
    fn row(t: *Repo, ends: *Ends, r: usize, d: *const [48]f64, dr: *const [48]Range, want: []f64, mag: []f64) void {
        for (0..512) |i| {
            const h = i / 32;
            const j = i % 32;
            const x = d[h * 3 + axis(j)];
            const xr = dr[h * 3 + axis(j)];
            const f: f64 = t.freqs[j];
            const th = std.math.tanh(f);
            const ang = (x * th + x * (f - th)) * pi32;
            // tanh's error cancels between the two products: fp32 rounds each product, the sum and the scale
            const xm = @max(@abs(xr.lo), @abs(xr.hi));
            const e = 2 * 0x1p-24 * pi32 * xm * (@abs(th) + 2 * @abs(f - th) + 2 * @abs(f)) + 0x1p-140;
            const lo = @min(xr.lo * f, xr.hi * f) * pi32 - e;
            const hi = @max(xr.lo * f, xr.hi * f) * pi32 + e;
            ends.put(want, mag, r * 512 + i, @cos(ang), id.cosOver(lo, hi).fp32(id.trig_ulps));
            ends.put(want, mag, (t.rows + r) * 512 + i, @sin(ang), id.sinOver(lo, hi).fp32(id.trig_ulps));
        }
    }

    pub fn reference(t: *Repo, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const gd = c.gamma(@floatFromInt(repoDepth(.{ .rows = 1, .n = 1, .k = 1 })), c.unit(.f32));
        var u: [128]f64 = undefined;
        var ur: [128]Range = undefined;
        var d: [48]f64 = undefined;
        var dr: [48]Range = undefined;
        for (0..t.rows) |r| {
            t.us(r, &u, &ur);
            for (&d, &dr, 0..) |*dv, *dv_r, o| {
                var s: f64 = 0;
                var rr = Range.of(0);
                var abs: f64 = 0;
                for (u, ur, t.wf[o * 128 ..][0..128]) |uv, uvr, wv| {
                    s += uv * wv;
                    rr = rr.add(uvr.mul(Range.of(wv)));
                    abs += @max(@abs(uvr.lo), @abs(uvr.hi)) * @abs(@as(f64, wv));
                }
                dv.* = s;
                dv_r.* = if (t.stress) rr else rr.widen(gd * abs);
            }
            t.row(&t.ends, r, &d, &dr, want, mag);
        }
    }
    pub fn bounds(t: *Repo, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Repo, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Repo, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var u: [128]f64 = undefined;
        var ur: [128]Range = undefined;
        var d: [48]f64 = undefined;
        var dr: [48]Range = undefined;
        var terms: [128]f64 = undefined;
        for (0..t.rows) |r| {
            t.us(r, &u, &ur);
            for (&d, &dr, 0..) |*dv, *dv_r, o| {
                for (&terms, u, t.wf[o * 128 ..][0..128]) |*v, uv, wv| v.* = uv * wv;
                dv.* = contract.rows.sumIn(acc, order, &terms);
                dv_r.* = Range.of(dv.*);
            }
            t.row(&scratch, r, &d, &dr, out, mag);
        }
    }
};

// --- id_qk: (q, k) pairs turned by cos and sin and rounded, F.normalize, times gamma and 8, rounded; v copied -----

const qk_why = "an fp16 output follows the head's sum of squares as its square root: a bf16 sum is off a few ulps";

/// A lane's two squares, then the simdgroup tree.
fn qkDepth(_: c.Shape) usize {
    return 1 + 5;
}

pub const Qk = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    rows: usize = 0,
    stress: bool = false,
    qkv: []f16 = &.{},
    /// gq then gk [1024] each
    gam: []f32 = &.{},
    /// cos then sin [rows, 512] each
    rot: []f32 = &.{},
    ins: [5]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: heads; k: the head width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = qkDepth,
            .ops = id.sqrt_ulps + id.div_ulps + 2,
            .out = .f16,
            .served = &.{ .{ .rows = 12294, .n = 16, .k = D }, .{ .rows = 8192, .n = 16, .k = D }, .{ .rows = 4101, .n = 16, .k = D } },
            .sums = .{ .bf16 = 0, .why = qk_why },
        };
    }

    pub fn deinit(t: *Qk) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Qk) void {
        t.a.free(t.qkv);
        t.a.free(t.gam);
        t.a.free(t.rot);
        t.qkv = &.{};
        t.gam = &.{};
        t.rot = &.{};
    }
    pub fn kernel(t: *Qk, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Qk);

    pub fn prepare(t: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 16 or shape.k != D) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.stress = fill == .stress;
        const n = shape.rows;
        t.qkv = try t.a.alloc(f16, n * 3 * H);
        t.gam = try t.a.alloc(f32, 2 * H);
        t.rot = try t.a.alloc(f32, 2 * n * 512);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        id.randomF16(rng, t.qkv, -4, 4);
        id.randomF32(rng, t.gam, 0.25, 2);
        for (0..n * 512) |i| {
            if (t.stress) {
                // quarter turns: exact rotations of integer heads, whose squares sum to ~2^20 exactly
                const turns = [4][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ -1, 0 }, .{ 0, -1 } };
                const turn = turns[rng.uintLessThan(usize, 4)];
                t.rot[i] = turn[0];
                t.rot[n * 512 + i] = turn[1];
                continue;
            }
            const angle = rng.float(f32) * 2 * std.math.pi;
            t.rot[i] = @cos(angle);
            t.rot[n * 512 + i] = @sin(angle);
        }
        if (t.stress) for (0..n) |r| for (0..2 * H) |i| {
            t.qkv[r * 3 * H + i] = @floatCast(id.int(rng, -128, 128));
        };
        if (fill == .nonfinite) {
            // q NaN, k +inf, v -inf, in turn; then a NaN cos
            const at = [_]usize{ 0 * 3 * H + 0 * D + 5, 1 * 3 * H + H + 2 * D + 9, 2 * 3 * H + 2 * H + 3 * D + 1 };
            for (at, [_]f16{ id.nan16, id.inf16, -id.inf16 }) |p, v| if (p < t.qkv.len) {
                t.qkv[p] = v;
            };
            if (n > 3) t.rot[3 * 512 + 4 * 32 + 6] = id.nan32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gam[0..H]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.gam[H..]));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.rot[0 .. n * 512]));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.rot[n * 512 ..]));
        for (&t.outs) |*o| o.* = try t.rig.output(n * H * 2);
        return true;
    }

    pub fn run(t: *Qk) !void {
        try t.lib.need("id_qk");
        const p = t.rig.begin();
        const in = t.ins;
        t.lib.k.heads(p.e, kref(in[0]), kref(in[1]), kref(in[2]), kref(in[3]), kref(in[4]), kref(t.outs[0]), kref(t.outs[1]), kref(t.outs[2]), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Qk) usize {
        return 3 * t.rows * H;
    }

    /// Head h of row r of q (0) or k (1): its pairs turned and rounded, as float64 values and ranges.
    fn turned(t: *const Qk, which: usize, r: usize, h: usize, v: *[D]f64, vr: *[D]Range) void {
        const x = t.qkv[r * 3 * H + which * H + h * D ..][0..D];
        for (0..D / 2) |p| {
            const cs: f64 = t.rot[r * 512 + h * 32 + p];
            const sn: f64 = t.rot[(t.rows + r) * 512 + h * 32 + p];
            const a: f64 = x[2 * p];
            const b: f64 = x[2 * p + 1];
            v[2 * p] = hf(a * cs - b * sn);
            v[2 * p + 1] = hf(a * sn + b * cs);
            const ac = Range.of(a * cs).fp32(0.5);
            const bs = Range.of(b * sn).fp32(0.5);
            const as_ = Range.of(a * sn).fp32(0.5);
            const bc = Range.of(b * cs).fp32(0.5);
            vr[2 * p] = ac.sub(bs).fp32(0.5).fp16();
            vr[2 * p + 1] = as_.add(bc).fp32(0.5).fp16();
        }
    }

    /// The head's outputs from its sum of squares `ss` (range `ssr`).
    fn head(t: *Qk, ends: *Ends, which: usize, r: usize, h: usize, v: *const [D]f64, vr: *const [D]Range, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const e12: f64 = @as(f32, 1e-12);
        // torch's clamp_min keeps a NaN norm (@max would drop it)
        const norm = if (std.math.isNan(ss)) ss else @max(@sqrt(ss), e12);
        const nr0 = ssr.up(id.sqrt).fp32(id.sqrt_ulps);
        const nr = Range{ .lo = @max(nr0.lo, e12), .hi = @max(nr0.hi, e12) };
        const gam = t.gam[which * H + h * D ..][0..D];
        const base = which * t.rows * H + (h * t.rows + r) * D;
        for (0..D) |i| {
            const gv: f64 = gam[i];
            const w = hf(v[i] / norm * gv * 8);
            const rr = vr[i].mul(nr.recip()).fp32(id.div_ulps).mul(Range.of(gv)).fp32(0.5).mul(Range.of(8)).fp32(0.5).fp16();
            ends.put(want, mag, base + i, w, rr);
        }
    }

    /// v's heads, copied bit for bit.
    fn values(t: *Qk, ends: *Ends, want: []f64, mag: []f64) void {
        for (0..t.rows) |r| for (0..16) |h| for (0..D) |i| {
            const f: f64 = t.qkv[r * 3 * H + 2 * H + h * D + i];
            ends.put(want, mag, 2 * t.rows * H + (h * t.rows + r) * D + i, f, Range.of(f));
        };
    }

    pub fn reference(t: *Qk, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const gd = c.gamma(@floatFromInt(qkDepth(.{ .rows = 1, .n = 1, .k = 1 })), c.unit(.f32));
        var v: [D]f64 = undefined;
        var vr: [D]Range = undefined;
        for (0..2) |which| for (0..t.rows) |r| for (0..16) |h| {
            t.turned(which, r, h, &v, &vr);
            var ss: f64 = 0;
            var lo: f64 = 0;
            var hi: f64 = 0;
            for (v, vr) |x, xr| {
                ss += x * x;
                const q = xr.square();
                lo += q.lo;
                hi += q.hi;
            }
            const ssr = if (t.stress) Range.of(ss) else Range{ .lo = lo * (1 - gd), .hi = hi * (1 + gd) };
            t.head(&t.ends, which, r, h, &v, &vr, ss, ssr, want, mag);
        };
        t.values(&t.ends, want, mag);
    }
    pub fn bounds(t: *Qk, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Qk, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var v: [D]f64 = undefined;
        var vr: [D]Range = undefined;
        var terms: [D]f64 = undefined;
        for (0..2) |which| for (0..t.rows) |r| for (0..16) |h| {
            t.turned(which, r, h, &v, &vr);
            for (&terms, v) |*s, x| s.* = x * x;
            const ss = contract.rows.sumIn(acc, order, &terms);
            t.head(&scratch, which, r, h, &v, &vr, ss, Range.of(ss), out, mag);
        };
        t.values(&scratch, out, mag);
    }
};
