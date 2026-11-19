//! id_linear (plain and GELU), id_split and id_gemm32 under the contract: products against float64, sums proven fp32.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");

const c = id.c;
const g = id.g;
const H = id.H;
const Range = id.Range;
const Ends = id.Ends;
const hf = id.hf;
const kref = id.kref;

/// Rows of x and of w (with the bias) repeat at these periods, prime to every tile: 97 x 131 dots a shape.
const row_period = 97;
const col_period = 131;

pub const Kind = enum { linear, gelu, split };

/// An output's float64 value and its range.
const Out = struct { w: f64, r: Range };

/// The M5 tensor unit's fp32 sums: 36 roundings of 2^-24 of |acc| + sum |a b| a 16-deep step (attention_range's).
fn naxDepth(s: c.Shape) usize {
    return s.k / 16 * contract.attention_range.mma_step + 8;
}

/// float64's dot of an fp16 row pair and the tensor unit's allowance on its fp32 sums, step by 16-deep step.
fn naxDot(x: []const f16, w: []const f16) [3]f64 {
    var s: f64 = 0;
    var spread: f64 = 0;
    var mag: f64 = 0;
    var i: usize = 0;
    while (i < x.len) : (i += 16) {
        var t: f64 = 0;
        var abs: f64 = 0;
        for (x[i..@min(i + 16, x.len)], w[i..@min(i + 16, x.len)]) |a, b| {
            const p = @as(f64, a) * @as(f64, b);
            t += p;
            abs += @abs(p);
        }
        spread += @abs(s) + abs;
        s += t;
        mag += abs;
    }
    return .{ s, spread * contract.attention_range.mma_step * 0x1p-24, mag };
}

const gelu_c: f64 = @as(f32, 0.7978845608028654);
const gelu_k: f64 = @as(f32, 0.044715);

fn inner(x: f64) f64 {
    return gelu_c * (x + gelu_k * x * x * x);
}

/// torch's tanh GELU in float64 at the kernel's fp32 constants.
fn gelu(x: f64) f64 {
    return 0.5 * x * (1 + std.math.tanh(inner(x)));
}

/// The GELU's fp32 steps over a range of fp16 inputs: its inner polynomial rises, tanh rises, the rest as products.
fn geluRange(v: Range) Range {
    const t = (Range{ .lo = inner(v.lo), .hi = inner(v.hi) }).fp32(3).up(id.tanh).fp32(id.tanh_ulps);
    return (Range{ .lo = 0.5 * v.lo, .hi = 0.5 * v.hi }).mul(t.add(Range.of(1)).fp32(0.5)).fp32(0.5);
}

const Served = struct { served: []const c.Shape, tiles: [3]c.Tile };

/// Forms by (M % 64 == 0)(N % 128 == 0); rows: 64-row tiles in whole-row forms, a last tile's rows first in the rest.
fn envelope(kind: Kind, form: u2) Served {
    const row_tail: c.Tile = .{ .size = 64 };
    const row_whole: c.Tile = .{ .size = 2, .step = 1, .min = 1 };
    const col_tail: c.Tile = .{ .size = 128 };
    const col_whole: c.Tile = .{ .size = 256, .step = 128, .min = 128 };
    const k_tile: c.Tile = .{ .size = 256, .step = 32, .min = 32 };
    const rt = if (form & 2 != 0) row_whole else row_tail;
    if (kind == .split) return .{
        .served = if (form & 2 != 0) &.{.{ .rows = 8192 / 64, .n = 256, .k = 3 * H }} else &.{ .{ .rows = 6, .n = 256, .k = 3 * H }, .{ .rows = 12294, .n = 256, .k = 3 * H }, .{ .rows = 4101, .n = 256, .k = 3 * H } },
        .tiles = .{ rt, .{}, .{} },
    };
    const ct = if (form & 1 != 0) col_whole else col_tail;
    const tiles: [3]c.Tile = .{ rt, ct, k_tile };
    if (kind == .gelu) return .{
        .tiles = tiles,
        .served = switch (form) {
            3 => &.{.{ .rows = 8192 / 64, .n = 4 * H, .k = H }},
            1 => &.{ .{ .rows = 6, .n = 4 * H, .k = H }, .{ .rows = 12294, .n = 4 * H, .k = H }, .{ .rows = 4101, .n = 4 * H, .k = H } },
            // no served call takes a width short of 128 columns: these forms are checked at one
            2 => &.{.{ .rows = 4096 / 64, .n = 1000, .k = H }},
            0 => &.{ .{ .rows = 5, .n = 1000, .k = H }, .{ .rows = 4101, .n = 1000, .k = H } },
        },
    };
    return .{ .tiles = tiles, .served = switch (form) {
        3 => &.{ .{ .rows = 8192 / 64, .n = 3 * H, .k = H }, .{ .rows = 8192 / 64, .n = H, .k = H }, .{ .rows = 8192 / 64, .n = H, .k = 4 * H } },
        1 => &.{ .{ .rows = 6, .n = 3 * H, .k = H }, .{ .rows = 12294, .n = 3 * H, .k = H }, .{ .rows = 12294, .n = H, .k = H }, .{ .rows = 12294, .n = H, .k = 4 * H }, .{ .rows = 4101, .n = 3 * H, .k = H }, .{ .rows = 4101, .n = H, .k = H }, .{ .rows = 4101, .n = H, .k = 4 * H } },
        2 => &.{.{ .rows = 4096 / 64, .n = 1000, .k = H }},
        0 => &.{ .{ .rows = 5, .n = 1000, .k = H }, .{ .rows = 4101, .n = 1000, .k = H } },
    } };
}

/// The pipeline a kind and form run.
pub fn pipeline(kind: Kind, form: u2) []const u8 {
    const names = [3][4][]const u8{
        .{ "id_linear_00", "id_linear_01", "id_linear_10", "id_linear_11" },
        .{ "id_linear_gelu_00", "id_linear_gelu_01", "id_linear_gelu_10", "id_linear_gelu_11" },
        .{ "id_split_0", "id_split_0", "id_split_1", "id_split_1" },
    };
    return names[@backingInt(kind)][form];
}

pub const Gemm16 = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    kind: Kind,
    /// split's forms are 1 and 3: its N, 256, is whole 128-column tiles.
    form: u2,
    name: []const u8,
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    stress: bool = false,
    x: []f16 = &.{},
    w: []f16 = &.{},
    b: []f16 = &.{},
    out: rig_.Ref = undefined,
    ins: [3]rig_.Ref = undefined,
    ends: Ends,

    pub fn spec(name: []const u8, kind: Kind, form: u2) c.Spec {
        const e = envelope(kind, form);
        return .{
            .name = name,
            .accum = .f32,
            .depth = naxDepth,
            .ops = 1,
            .out = if (kind == .split) .f32 else .f16,
            .tiles = e.tiles,
            .served = e.served,
        };
    }

    pub fn deinit(t: *Gemm16) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Gemm16) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.a.free(t.b);
        t.x = &.{};
        t.w = &.{};
        t.b = &.{};
    }
    pub fn kernel(t: *Gemm16, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.kind, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Gemm16);

    pub fn prepare(t: *Gemm16, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const m = if (t.form & 2 != 0) shape.rows * 64 else shape.rows;
        if (m == 0 or shape.n == 0 or shape.k == 0 or shape.k % 2 != 0) return false;
        if (t.kind == .split and (shape.n != 256 or shape.k != 3 * H)) return false;
        const form = @as(u2, @intFromBool(m % 64 == 0)) * 2 + @intFromBool(shape.n % 128 == 0);
        if (form != t.form) return false;
        t.rig.reset();
        t.free();
        t.m = m;
        t.n = shape.n;
        t.k = shape.k;
        t.stress = fill == .stress;
        const K = shape.k;
        const px = @min(m, row_period);
        const pw = @min(shape.n, col_period);
        t.x = try t.a.alloc(f16, px * K);
        t.w = try t.a.alloc(f16, pw * K);
        t.b = try t.a.alloc(f16, pw);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // integers in [hi / 2, hi), hi^2 K <= 2^24: fp32 sums exact in any order, a 32nd of a half past fp16's
            const hi: i32 = @intFromFloat(@sqrt(@as(f64, 0x1p24) / @as(f64, @floatFromInt(K))));
            for (0..px) |r| for (0..K / 2) |i| {
                const v = id.int(rng, @divTrunc(hi, 2), hi - 1);
                t.x[r * K + i] = @floatCast(v);
                // x's halves differ by 1 in their first 8: the sums cancel to a few hundred
                t.x[r * K + K / 2 + i] = @floatCast(v - @as(f32, if (i < 8) 1 else 0));
            };
            for (0..pw) |j| for (0..K / 2) |i| {
                const v = id.int(rng, @divTrunc(hi, 2), hi - 1);
                t.w[j * K + i] = @floatCast(v);
                t.w[j * K + K / 2 + i] = @floatCast(-v);
            };
            for (t.b) |*v| v.* = @floatCast(id.int(rng, -4, 4));
        } else {
            id.randomF16(rng, t.x, -2, 2);
            id.randomF16(rng, t.w, -0.25, 0.25);
            id.randomF16(rng, t.b, -1, 1);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, K / 3 }, .{ 1, K / 2 }, .{ 2, K - 1 }, .{ 3, 0 }, .{ 3, K - 1 } };
            for (at, [_]f16{ id.nan16, id.inf16, -id.inf16, id.inf16, -id.inf16 }) |p, v| if (p[0] < px) {
                t.x[p[0] * K + p[1]] = v;
            };
            if (pw > 2) t.b[2] = id.inf16;
        }
        const xs = try id.periodic(f16, t.a, t.x, K, m);
        defer t.a.free(xs);
        const ws = try id.periodic(f16, t.a, t.w, K, shape.n);
        defer t.a.free(ws);
        const bs = try id.periodic(f16, t.a, t.b, 1, shape.n);
        defer t.a.free(bs);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(xs));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(ws));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(bs));
        t.out = try t.rig.output(m * shape.n * @as(usize, if (t.kind == .split) 4 else 2));
        return true;
    }

    pub fn run(t: *Gemm16) !void {
        try t.lib.need(pipeline(t.kind, t.form));
        const p = t.rig.begin();
        const k = &t.lib.k;
        switch (t.kind) {
            .split => k.split(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.out), t.m),
            else => k.linear(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.m, t.n, t.k, t.kind == .gelu),
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Gemm16) usize {
        return t.m * t.n;
    }

    /// The output of a sum `s` (range `sr`) of column j: its bias, fp16 rounding and GELU; split's fp32 sum as is.
    fn finish(t: *const Gemm16, s: f64, sr: Range, j: usize) Out {
        if (t.kind == .split) return .{ .w = s, .r = sr.fp32(0.5) };
        const b: f64 = t.b[j];
        const v = hf(s + b);
        const vr = sr.add(Range.of(b)).fp32(0.5).fp16();
        if (t.kind == .linear) return .{ .w = v, .r = vr };
        return .{ .w = hf(gelu(v)), .r = geluRange(vr).fp16() };
    }

    /// The distinct outputs' values and ranges by `value(r, j)`, then every output from its row's and column's.
    fn table(t: *Gemm16, ends: *Ends, want: []f64, mag: []f64, ctx: anytype, comptime value: anytype) !void {
        const px = t.x.len / t.k;
        const pw = t.w.len / t.k;
        const tw = try t.a.alloc(f64, 3 * px * pw);
        defer t.a.free(tw);
        for (0..px) |r| for (0..pw) |j| {
            const o = value(ctx, t, r, j);
            tw[3 * (r * pw + j) ..][0..3].* = .{ o.w, o.r.lo, o.r.hi };
        };
        for (0..t.m) |r| for (0..t.n) |j| {
            const e = tw[3 * ((r % px) * pw + j % pw) ..][0..3];
            ends.put(want, mag, r * t.n + j, e[0], .{ .lo = e[1], .hi = e[2] });
        };
    }

    pub fn reference(t: *Gemm16, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const f = struct {
            fn value(_: void, s: *const Gemm16, r: usize, j: usize) Out {
                const d = naxDot(s.x[r * s.k ..][0..s.k], s.w[j * s.k ..][0..s.k]);
                return s.finish(d[0], if (s.stress) Range.of(d[0]) else Range.of(d[0]).widen(d[1]), j);
            }
        };
        try t.table(&t.ends, want, mag, {}, f.value);
    }
    pub fn bounds(t: *Gemm16, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Gemm16, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Gemm16, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        const terms = try t.a.alloc(f64, t.k);
        defer t.a.free(terms);
        const Ctx = struct { acc: c.Accum, order: g.Order, terms: []f64 };
        const f = struct {
            fn value(x: Ctx, s: *const Gemm16, r: usize, j: usize) Out {
                for (x.terms, s.x[r * s.k ..][0..s.k], s.w[j * s.k ..][0..s.k]) |*v, a, b| v.* = @as(f64, a) * @as(f64, b);
                const sum = id.rows.sumIn(x.acc, x.order, x.terms);
                return s.finish(sum, Range.of(sum), j);
            }
        };
        try t.table(&scratch, out, mag, Ctx{ .acc = acc, .order = order, .terms = terms }, f.value);
    }
};

// --- id_gemm32: fp32 x [M, K] times w [N, K]^T, sums that must be fp32's (the NAX fp32 path's are not) -----------

/// Each 8x8 step's eight FMAs in one chain over K.
fn gemm32Depth(s: c.Shape) usize {
    return s.k + 8;
}

pub const Gemm32 = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = gemm32Depth,
            .out = .f32,
            .tiles = .{ .{ .size = 64 }, .{ .size = 64, .step = 8, .min = 8 }, .{ .size = 32, .step = 32, .min = 32 } },
            .served = &.{ .{ .rows = 4101, .n = H, .k = 1280 }, .{ .rows = 4101, .n = H, .k = 128 }, .{ .rows = 8192, .n = 16, .k = H } },
        };
    }

    pub fn deinit(t: *Gemm32) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Gemm32) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Gemm32, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Gemm32);

    pub fn prepare(t: *Gemm32, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n_ok = shape.n % 64 == 0 or (shape.n < 64 and shape.n % 8 == 0);
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0 or !n_ok) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const K = shape.k;
        const px = @min(shape.rows, row_period);
        const pw = @min(shape.n, col_period);
        t.x = try t.a.alloc(f32, px * K);
        t.w = try t.a.alloc(f32, pw * K);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // 13-bit integers against +-1, in turn on either side: exact fp32 products and sums below 2^24
            for (0..px) |r| for (0..K) |i| {
                t.x[r * K + i] = id.sign(rng) * if (i % 2 == 0) id.int(rng, 4097, 8191) else 1;
            };
            for (0..pw) |j| for (0..K) |i| {
                t.w[j * K + i] = id.sign(rng) * if (i % 2 == 1) id.int(rng, 4097, 8191) else 1;
            };
        } else {
            id.randomF32(rng, t.x, -2, 2);
            id.randomF32(rng, t.w, -0.25, 0.25);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, K / 3 }, .{ 1, K / 2 }, .{ 2, K - 1 }, .{ 3, 0 }, .{ 3, K - 1 } };
            for (at, [_]f32{ id.nan32, id.inf32, -id.inf32, id.inf32, -id.inf32 }) |p, v| if (p[0] < px) {
                t.x[p[0] * K + p[1]] = v;
            };
        }
        const xs = try id.periodic(f32, t.a, t.x, K, shape.rows);
        defer t.a.free(xs);
        const ws = try id.periodic(f32, t.a, t.w, K, shape.n);
        defer t.a.free(ws);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(xs));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(ws));
        t.out = try t.rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(t: *Gemm32) !void {
        try t.lib.need("id_gemm32");
        const p = t.rig.begin();
        t.lib.k.linear32(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.out), t.shape.rows, t.shape.n, t.shape.k, t.shape.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *Gemm32) usize {
        return t.shape.rows * t.shape.n;
    }

    fn each(t: *Gemm32, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) !void {
        const K = t.shape.k;
        const px = t.x.len / K;
        const pw = t.w.len / K;
        const tw = try t.a.alloc(f64, 2 * px * pw);
        defer t.a.free(tw);
        const terms = try t.a.alloc(f64, K);
        defer t.a.free(terms);
        for (0..px) |r| for (0..pw) |j| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (terms, t.x[r * K ..][0..K], t.w[j * K ..][0..K]) |*v, a, b| {
                v.* = @as(f64, a) * @as(f64, b);
                s += v.*;
                m += @abs(v.*);
            }
            tw[2 * (r * pw + j) ..][0..2].* = .{ if (acc) |x| id.rows.sumIn(x, order, terms) else s, m };
        };
        for (0..t.shape.rows) |r| for (0..t.shape.n) |j| {
            const e = tw[2 * ((r % px) * pw + j % pw) ..][0..2];
            want[r * t.shape.n + j] = e[0];
            mag[r * t.shape.n + j] = e[1];
        };
    }

    pub fn reference(t: *Gemm32, want: []f64, mag: []f64) !void {
        try t.each(want, mag, null, .sequential);
    }

    pub fn simulate(t: *Gemm32, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        try t.each(out, mag, acc, order);
    }
};
