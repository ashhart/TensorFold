//! Idol's summing glue under the contract: RePo's angles, q and k turned and normed, the latent's input embed.
const std = @import("std");
const cuda = @import("cuda");
const idol = @import("idol_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const Fill = r.g.Fill;

const why = "gate 4's integers keep fp32 exact; fp16 and bf16 sums miss by a margin it reports but cannot bound ahead";

fn summed(name: []const u8, out: c.Elem, served: []const c.Shape, depth: *const fn (c.Shape) usize) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = depth, .ops = 8, .mids = 2, .out = out, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
}

fn rowsOf(n: usize, k: usize) [3]c.Shape {
    return .{ .{ .rows = r.joint, .n = n, .k = k }, .{ .rows = r.latent, .n = n, .k = k }, .{ .rows = r.cond, .n = n, .k = k } };
}

/// tf_idol_qk: q and k of [rows, 3072] turned by cos, sin [rows, 512], F.normalize times gamma times 8; v copied.
pub const Qk = struct {
    pub const kinds = [_]r.Kind{ .f16, .f32, .f32, .f32, .f32 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    const served = rowsOf(3 * r.hidden, r.D);

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == 3 * r.hidden and s.k == r.D;
    }

    /// A lane's two squares and their sum, then the warp's 5 shuffles.
    fn depth(_: c.Shape) usize {
        return 7;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return summed(name, .f16, &served, depth);
    }

    pub fn sizes(s: c.Shape) [5]usize {
        return .{ s.rows * 3 * r.hidden, r.hidden, r.hidden, s.rows * 512, s.rows * 512 };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * 3 * r.hidden;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[5][]f64) void {
        if (mode == .stress) {
            r.integers(rng, v[0], 64);
            @memset(v[3], 1);
            @memset(v[4], 0);
        } else {
            r.uniform(rng, v[0], -4, 4);
            for (v[3], v[4]) |*co, *si| {
                const t = rng.float(f64) * 2 * std.math.pi;
                co.* = @cos(t);
                si.* = @sin(t);
            }
        }
        r.uniform(rng, v[1], 0.5, 1.5);
        r.uniform(rng, v[2], 0.5, 1.5);
        if (mode == .nonfinite) {
            r.plant(v[0], &.{ 5, 3072 + 1024 + 77, 2 * 3072 + 2048 + 9 });
            r.plant(v[1], &.{300});
        }
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [5]u64, o: u64, shape: c.Shape, _: u32) !void {
        try idol.qk(k, s, dev[0], dev[1], dev[2], dev[3], dev[4], o, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [5][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        var rot: [64]f64 = undefined;
        var rr: [64]Range = undefined;
        var sq: [32]f64 = undefined;
        var lo: [32]f64 = undefined;
        var hi: [32]f64 = undefined;
        for (0..shape.rows) |row| for (0..r.heads) |hd| {
            const base = row * 3 * r.hidden + hd * r.D;
            for (0..r.D) |j| e.put(want, mag, base + 2048 + j, v[0][base + 2048 + j], Range.of(v[0][base + 2048 + j]));
            for ([_]usize{ 0, 1024 }, [_][]const f64{ v[1], v[2] }) |off, gamma| {
                for (0..32) |l| {
                    const x0 = v[0][base + off + 2 * l];
                    const x1 = v[0][base + off + 2 * l + 1];
                    const co = v[3][row * 512 + hd * 32 + l];
                    const si = v[4][row * 512 + hd * 32 + l];
                    rot[2 * l] = r.h(x0 * co - x1 * si);
                    rot[2 * l + 1] = r.h(x0 * si + x1 * co);
                    const a = Range.of(x0).mul(Range.of(co)).fp32(0.5);
                    const b = Range.of(x1).mul(Range.of(si)).fp32(0.5);
                    const p = Range.of(x0).mul(Range.of(si)).fp32(0.5);
                    const q = Range.of(x1).mul(Range.of(co)).fp32(0.5);
                    rr[2 * l] = a.sub(b).fp32(0.5).half();
                    rr[2 * l + 1] = p.add(q).fp32(0.5).half();
                    sq[l] = rot[2 * l] * rot[2 * l] + rot[2 * l + 1] * rot[2 * l + 1];
                    const t = rr[2 * l].square().fp32(0.5).add(rr[2 * l + 1].square().fp32(0.5)).fp32(0.5);
                    lo[l] = t.lo;
                    hi[l] = t.hi;
                }
                const s = sum.total(&sq, &lo, &hi, 5);
                const n = clampMin(@sqrt(s.v));
                const nr = s.r.up(sqrt).fp32(r.sqrt_ulps).up(clampMin);
                for (0..r.D) |j| {
                    const g = gamma[hd * r.D + j];
                    const w = r.h(rot[j] / n * g * 8);
                    const wr = rr[j].div(nr).fp32(r.div_ulps).mul(Range.of(g)).fp32(0.5).mul(Range.of(8)).fp32(0.5).half();
                    e.put(want, mag, base + off + j, w, wr);
                }
            }
        };
    }
};

fn sqrt(x: f64) f64 {
    return @sqrt(@max(x, 0));
}

/// F.normalize's clamp_min(1e-12) in fp32, NaN kept.
fn clampMin(x: f64) f64 {
    return if (x < r.f(1e-12)) r.f(1e-12) else x;
}

/// tf_idol_repo: from gate | content [rows, 256], d = final (silu(gate) content), angles; out cos then sin [rows, 512].
pub const Repo = struct {
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;
    const served = rowsOf(r.hidden, 128);
    const pi = r.f(std.math.pi);

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == r.hidden and s.k == 128;
    }

    /// The final map's 128 fused steps in turn.
    fn depth(_: c.Shape) usize {
        return 128;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return summed(name, .f32, &served, depth);
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * 256, 48 * 128, 32 };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * 1024;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
        if (mode == .stress) {
            // gate 64..80: silu is the gate itself in fp32, so every term is an integer and every fp32 sum exact
            for (0..s.rows) |row| {
                for (v[0][row * 256 ..][0..128]) |*x| x.* = @floatFromInt(64 + rng.uintLessThan(u32, 17));
                r.integers(rng, v[0][row * 256 + 128 ..][0..128], 8);
            }
            r.integers(rng, v[1], 8);
            @memset(v[2], 0x1p-19);
            return;
        }
        r.uniform(rng, v[0], -4, 4);
        r.uniform(rng, v[1], -0.2, 0.2);
        r.uniform(rng, v[2], 0.5, 3);
        if (mode == .nonfinite) {
            r.plant(v[0], &.{ 7, 256 + 130, 2 * 256 + 50 });
            r.plant(v[1], &.{ 128 * 5 + 3, 128 * 20 });
            r.plant(v[2], &.{31});
        }
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try idol.repo(k, s, dev[0], dev[1], dev[2], o, o + shape.rows * 512 * 4, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        var u: [128]f64 = undefined;
        var ur: [128]Range = undefined;
        var t: [128]f64 = undefined;
        var lo: [128]f64 = undefined;
        var hi: [128]f64 = undefined;
        var d: [48]f64 = undefined;
        var dr: [48]Range = undefined;
        const half = shape.rows * 512;
        for (0..shape.rows) |row| {
            const g = v[0][row * 256 ..][0..256];
            for (0..128) |k| {
                const a = g[k];
                const cv = g[128 + k];
                u[k] = a / (1 + @exp(-a)) * cv;
                const den = Range.of(-a).up(exp).fp32(r.fn_ulps).add(Range.of(1)).fp32(0.5);
                ur[k] = Range.of(a).div(den).fp32(r.div_ulps).mul(Range.of(cv)).fp32(0.5);
            }
            for (0..48) |o| {
                for (0..128) |k| {
                    const w = v[1][o * 128 + k];
                    t[k] = u[k] * w;
                    const p = ur[k].mul(Range.of(w));
                    lo[k] = p.lo;
                    hi[k] = p.hi;
                }
                const s = sum.total(&t, &lo, &hi, 128);
                d[o] = s.v;
                dr[o] = s.r;
            }
            for (0..512) |i| {
                const hd = i / 32;
                const j = i % 32;
                const axis: usize = if (j < 10) 0 else if (j < 20) 1 else 2;
                const x = d[hd * 3 + axis];
                const xr = dr[hd * 3 + axis];
                const fr = v[2][j];
                const ft = std.math.tanh(fr);
                const ang = (x * ft + x * (fr - ft)) * pi;
                const ftr = Range.of(fr).up(tanh).fp32(r.fn_ulps);
                const a1 = xr.mul(ftr).fp32(0.5);
                const a2 = xr.mul(Range.of(fr).sub(ftr).fp32(0.5)).fp32(0.5);
                const ar = a1.add(a2).fp32(0.5).mul(Range.of(pi)).fp32(0.5);
                const at = row * 512 + i;
                e.put(want, mag, at, @cos(ang), ar.trig(false).fp32(r.fn_ulps));
                e.put(want, mag, half + at, @sin(ang), ar.trig(true).fp32(r.fn_ulps));
            }
        }
    }
};

fn exp(x: f64) f64 {
    return @exp(x);
}

fn tanh(x: f64) f64 {
    return std.math.tanh(x);
}

/// tf_idol_embed: x = fp16(fp16(z w^T + b) + pe), 16 fp32 inputs a latent row, its fp16 position embedding.
pub const Embed = struct {
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32, .f16 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    const served = [_]c.Shape{.{ .rows = r.latent, .n = r.hidden, .k = 16 }};

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == r.hidden and s.k == 16;
    }

    fn depth(_: c.Shape) usize {
        return 16;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return summed(name, .f16, &served, depth);
    }

    pub fn sizes(s: c.Shape) [4]usize {
        return .{ s.rows * 16, r.hidden * 16, r.hidden, s.rows * r.hidden };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * r.hidden;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[4][]f64) void {
        if (mode == .stress) {
            for (v) |col| r.integers(rng, col, 8);
            return;
        }
        r.uniform(rng, v[0], -3, 3);
        r.uniform(rng, v[1], -0.5, 0.5);
        r.uniform(rng, v[2], -0.5, 0.5);
        r.uniform(rng, v[3], -1, 1);
        if (mode == .nonfinite) for (v, 0..) |col, i| r.plant(col, &.{ 3 + 40 * i, 900 + i });
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, _: u32) !void {
        try idol.embed(k, s, dev[0], dev[1], dev[2], dev[3], o, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [4][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        var t: [16]f64 = undefined;
        for (0..shape.rows) |row| for (0..r.hidden) |col| {
            for (&t, 0..) |*x, k| x.* = v[0][row * 16 + k] * v[1][col * 16 + k];
            const s = sum.total(&t, &t, &t, 16);
            const b = v[2][col];
            const at = row * r.hidden + col;
            const pe = v[3][at];
            e.put(want, mag, at, r.h(r.h(s.v + b) + pe), s.r.add(Range.of(b)).fp32(0.5).half().add(Range.of(pe)).fp32(0.5).half());
        };
    }
};

test "q and k turned and normed by their warp's sum keep the float64 value inside its range" {
    const a = std.testing.allocator;
    const shape: c.Shape = .{ .rows = 1, .n = 3 * r.hidden, .k = r.D };
    var bufs: [5][]f64 = undefined;
    for (&bufs, Qk.sizes(shape)) |*b, n| b.* = try a.alloc(f64, n);
    defer for (bufs) |b| a.free(b);
    var prng = std.Random.DefaultPrng.init(3);
    Qk.fill(prng.random(), .random, shape, 0, &bufs);
    var e: r.Ends = .{ .a = a };
    defer e.deinit();
    try e.size(Qk.count(shape));
    const want = try a.alloc(f64, Qk.count(shape));
    defer a.free(want);
    const mag = try a.alloc(f64, Qk.count(shape));
    defer a.free(mag);
    try Qk.ref(a, shape, 0, .{ bufs[0], bufs[1], bufs[2], bufs[3], bufs[4] }, .{ .range = false }, &e, want, mag);
    for (want, e.lo, e.hi) |w, lo, hi| try std.testing.expect(lo <= w and w <= hi);
}
