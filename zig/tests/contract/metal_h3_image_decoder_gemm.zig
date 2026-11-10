//! h3d_gemm under the contract: y = bf16(x w^T + b), and its LayerScale form y += bf16(that s), against float64.
const std = @import("std");
const mtl = @import("metal");
const parts = @import("metal_h3_image_decoder_parts.zig");

const c = parts.c;
const Range = parts.Range;
const Ends = parts.Ends;
const Sum = parts.Sum;
const bf = parts.bf;
const value = parts.value;
const W = parts.W;
const F = parts.F;

/// Rows a pass multiplies: one tile of 16 x 16 cells and its 5 tokens after, the batch of 16, one small tile of 65.
const passes = [_]usize{ 261, 16 * 261, 65 };

fn served(comptime nk: []const [2]usize) [passes.len * nk.len]c.Shape {
    var out: [passes.len * nk.len]c.Shape = undefined;
    for (passes, 0..) |r, i| for (nk, 0..) |p, j| {
        out[i * nk.len + j] = .{ .rows = r, .n = p[0], .k = p[1] };
    };
    return out;
}

/// to_qkv (3,072 x 1,024), w1 (8,192 x 1,024) and proj_out (768 x 1,024).
const plain = served(&.{ .{ 3 * W, W }, .{ 2 * F, W }, .{ 768, W } });
/// to_out (1,024 x 1,024) and w2 (1,024 x 4,096), each added to the stream through its LayerScale.
const scaled = served(&.{ .{ W, W }, .{ W, F } });

/// The k-deep chain of the simdgroup products' fp32 sums, then the bias.
fn chain(s: c.Shape) usize {
    return s.k + 1;
}

/// float64 x . w and sum |x_i w_i| over fp32 copies of bf16 values.
fn dot(x: []const f32, w: []const f32) [2]f64 {
    var s: @Vector(8, f64) = @splat(0);
    var m: @Vector(8, f64) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const a: @Vector(8, f64) = @floatCast(@as(@Vector(8, f32), x[i..][0..8].*));
        const b: @Vector(8, f64) = @floatCast(@as(@Vector(8, f32), w[i..][0..8].*));
        s += a * b;
        m += @abs(a * b);
    }
    var sv = @reduce(.Add, s);
    var mv = @reduce(.Add, m);
    while (i < x.len) : (i += 1) {
        const p = @as(f64, x[i]) * w[i];
        sv += p;
        mv += @abs(p);
    }
    return .{ sv, mv };
}

fn widen(a: std.mem.Allocator, b: []const u16) ![]f32 {
    const out = try a.alloc(f32, b.len);
    for (out, b) |*o, v| o.* = c.fromBf16(v);
    return out;
}

pub const Gemm = struct {
    /// The LayerScale form (mode 1): y = bf16(y + bf16(bf16(x w^T + b) s)), in place.
    layer_scale: bool,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []u16 = &.{},
    s: []u16 = &.{},
    y0: []u16 = &.{},

    /// Rows: rows; n: the projection's outputs (64 a tile); k: its inputs (32 a step).
    pub fn spec(name: []const u8, k: Gemm) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = chain,
            .mids = if (k.layer_scale) 2 else 0,
            .out = .bf16,
            .in_place = k.layer_scale,
            .tiles = .{ .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 32, .step = 32, .min = 32 } },
            .served = if (k.layer_scale) &scaled else &plain,
        };
    }

    /// Gate 4's rows: integers, k's second half repeating the first, its weights negated plus 1 on 4: every sum is 4.
    fn mirror(k: *Gemm, rng: std.Random, xb: []u16, wb: []u16, s: c.Shape) void {
        const h = s.k / 2;
        for (0..s.rows) |r| for (0..h) |i| {
            const v = if (i < 4) 1 else rng.intRangeAtMost(i32, 1, 48);
            xb[r * s.k + i] = parts.integer(v);
            xb[r * s.k + h + i] = parts.integer(v);
        };
        for (0..s.n) |o| for (0..h) |i| {
            const v = rng.intRangeAtMost(i32, 0, 48);
            wb[o * s.k + i] = parts.integer(v);
            wb[o * s.k + h + i] = parts.integer(-v + @as(i32, if (i < 4) 1 else 0));
        };
        @memset(k.b, 0);
        @memset(k.s, parts.integer(2));
        parts.integers(rng, k.y0, -8, 8);
    }

    pub fn prepare(k: *Gemm, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.n == 0 or s.n % 64 != 0 or s.k == 0 or s.k % 32 != 0) return false;
        const al = t.host();
        const xb = try al.alloc(u16, s.rows * s.k);
        const wb = try al.alloc(u16, s.n * s.k);
        k.b = try al.alloc(u16, s.n);
        k.s = try al.alloc(u16, s.n);
        k.y0 = try al.alloc(u16, s.rows * s.n);
        if (t.fill == .stress) k.mirror(rng, xb, wb, s) else {
            parts.uniform(rng, xb, -2, 2);
            const a: f32 = @floatCast(@sqrt(3.0 / @as(f64, @floatFromInt(s.k))));
            parts.uniform(rng, wb, -a, a);
            parts.uniform(rng, k.b, -0.1, 0.1);
            parts.uniform(rng, k.s, 0.1, 0.5);
            parts.uniform(rng, k.y0, -2, 2);
        }
        if (t.fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, s.k / 3 }, .{ 1, s.k / 2 }, .{ 2, s.k - 1 }, .{ 3, 0 }, .{ 3, s.k - 1 } };
            for (at, parts.plants) |p, v| if (p[0] < s.rows) {
                xb[p[0] * s.k + p[1]] = v;
            };
            // the LayerScale form's old stream value and scale too
            if (k.layer_scale and s.rows > 4) k.y0[4 * s.n + 3] = parts.nan16;
            if (k.layer_scale) k.s[5] = parts.inf16;
        }
        k.x = try widen(al, xb);
        k.w = try widen(al, wb);
        try t.input(xb);
        try t.input(wb);
        try t.input(k.b);
        if (k.layer_scale) try t.input(k.s);
        try t.output(s.rows * s.n * 2);
        return true;
    }

    /// The launcher's gemm: 64 x 64 tiles a group of 128; mode 0 binds the bias again where the scale goes.
    pub fn launch(k: *Gemm, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        const out: usize = if (k.layer_scale) 4 else 3;
        // the stream the LayerScale form adds into, fresh before every run
        if (k.layer_scale) @memcpy(t.rig.bytes(t.bufs[out], k.y0.len * 2), std.mem.sliceAsBytes(k.y0));
        for (0..3) |i| t.bind(e, i, i);
        t.bind(e, if (k.layer_scale) 3 else 2, 3);
        t.bind(e, out, 4);
        e.setValue(parts.Shape{ .m = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k), .mode = @intFromBool(k.layer_scale) }, 5);
        e.dispatchGroups(mtl.Size.of(s.n / 64, (s.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn count(_: *Gemm, s: c.Shape) usize {
        return s.rows * s.n;
    }

    /// Output (r, o) from its sum: the bias and one rounding, then for LayerScale the scale and the old value.
    fn finish(k: *const Gemm, s: c.Shape, sum: Sum.Out, r: usize, o: usize, ends: *Ends, want: []f64, mag: []f64) void {
        const at = r * s.n + o;
        const b = value(k.b[o]);
        const v = bf(sum.v + b);
        const range = sum.r.add(Range.of(b)).fp32(0.5).bf16();
        if (!k.layer_scale) return ends.put(want, mag, at, v, range);
        const sc = value(k.s[o]);
        const old = value(k.y0[at]);
        const y = range.mul(Range.of(sc)).fp32(0.5).bf16().add(Range.of(old)).fp32(0.5).bf16();
        ends.put(want, mag, at, bf(old + bf(v * sc)), y);
    }

    const Job = struct {
        k: *const Gemm,
        s: c.Shape,
        sum: Sum,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        fn rows(j: Job, from: usize, to: usize) void {
            const kk = j.s.k;
            for (from..to) |r| for (0..j.s.n) |o| {
                const d = dot(j.k.x[r * kk ..][0..kk], j.k.w[o * kk ..][0..kk]);
                j.k.finish(j.s, .{ .v = d[0], .r = j.sum.spread(d[0], d[1], kk) }, r, o, j.ends, j.want, j.mag);
            };
        }
    };

    pub fn outputs(k: *Gemm, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        const s = t.shape;
        if (sum.acc == null) return parts.parallel(s.rows, Job{ .k = k, .s = s, .sum = sum, .ends = ends, .want = want, .mag = mag }, Job.rows);
        // gate 4's replays: each output's terms in the accumulator and order asked
        const terms = try t.a.alloc(f64, s.k);
        defer t.a.free(terms);
        for (0..s.rows) |r| for (0..s.n) |o| {
            for (terms, k.x[r * s.k ..][0..s.k], k.w[o * s.k ..][0..s.k]) |*p, a, b| p.* = @as(f64, a) * b;
            k.finish(s, sum.of(terms, s.k), r, o, ends, want, mag);
        };
    }
};
