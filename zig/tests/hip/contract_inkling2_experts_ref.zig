//! The HIP experts' host side: loop_model.zig's expert arithmetic as the device must reproduce it, bit for bit.
const std = @import("std");
const lm = @import("inkling2_loop_model");
const bf = @import("bf16_round");
const core = @import("core");

const threads = core.contract.reference_threads;

/// Every NaN the kernels write (experts.hpp kNan).
pub const nan_bits: u16 = 0x7fc0;

fn softcap(x: f32, cap: f32) f32 {
    return cap * std.math.tanh(x / cap);
}

/// softcap4(g) * sigmoid(g): loop_model.expertActivation's first product, with its f32 std.math.tanh and @exp.
pub fn gateFactor(g: f32) f32 {
    const sigmoid: f32 = 1 / (1 + @exp(-g));
    return softcap(g, 4) * sigmoid;
}

/// softcap25(u): expertActivation's last factor.
pub fn upFactor(u: f32) f32 {
    return softcap(u, 25);
}

/// A float64 reference value's bf16 bits as the kernels store it: NaN as nan_bits, else ties to even.
pub fn bits(v: f64) u16 {
    return if (std.math.isNan(v)) nan_bits else bf.bits(@floatCast(v));
}

/// Two fp32 outputs agree: both NaN, or the same bits.
pub fn same32(x: f32, y: f32) bool {
    return (std.math.isNan(x) and std.math.isNan(y)) or @as(u32, @bitCast(x)) == @as(u32, @bitCast(y));
}

pub const Format = enum { nvfp4, mxfp8 };

/// Inputs a scale covers: NVFP4's E4M3 per 16, MXFP8's UE8M0 per 32.
pub fn group(f: Format) usize {
    return if (f == .nvfp4) 16 else 32;
}

/// An expert matrix as the checkpoint stores it: experts of `rows` rows over `k` inputs, the first with id `first`.
pub const Mat = struct {
    format: Format,
    rows: usize,
    k: usize,
    codes: []const u8,
    scales: []const u8,
    amax: []const f32 = &.{},
    den: f32 = 1,
    first: usize = 0,

    pub fn rowBytes(m: Mat) usize {
        return if (m.format == .nvfp4) m.k / 2 else m.k;
    }

    pub fn rowScales(m: Mat) usize {
        return m.k / group(m.format);
    }

    pub fn count(m: Mat) usize {
        return m.codes.len / (m.rows * m.rowBytes());
    }

    /// Row `row` of expert `id`, decoded by loop_model into `out` (k values).
    pub fn decode(m: Mat, id: usize, row: usize, out: []f64) !void {
        if (id < m.first or id - m.first >= m.count()) return error.ExpertNotHeld;
        const e = id - m.first;
        const at = e * m.rows + row;
        const codes = m.codes[at * m.rowBytes() ..][0..m.rowBytes()];
        const scales = m.scales[at * m.rowScales() ..][0..m.rowScales()];
        switch (m.format) {
            .nvfp4 => try lm.nvfp4ReferenceRow(codes, scales, m.amax[e], m.den, out),
            .mxfp8 => try lm.mxfp8ReferenceRow(codes, scales, out),
        }
    }
};

/// A down call's host inputs: each route's x row and its expert id.
pub const Call = struct {
    mat: Mat,
    x: []const u16,
    ids: []const i32,
    n: usize,
    experts: usize = 128,
};

/// Columns lo..hi of every route: rows decoded once a held expert, then each route of that expert's dots.
const Cols = struct {
    call: *const Call,
    xs: []const f64,
    out: []u16,
    lo: usize,
    hi: usize,
    err: ?anyerror = null,

    pub fn run(j: *Cols) void {
        j.work() catch |e| {
            j.err = e;
        };
    }

    fn work(j: *Cols) !void {
        const cl = j.call;
        const k = cl.mat.k;
        var w: [2304]f64 = undefined;
        var dot: [1]f64 = undefined;
        for (0..cl.mat.count()) |e| {
            const id = cl.mat.first + e;
            for (j.lo..j.hi) |col| {
                var decoded = false;
                for (cl.ids, 0..) |r_id, r| {
                    if (r_id != @as(i32, @intCast(id))) continue;
                    if (!decoded) try cl.mat.decode(id, col, w[0..k]);
                    decoded = true;
                    try lm.expertProjection(w[0..k], j.xs[r * k ..][0..k], 1, &dot);
                    j.out[r * cl.n + col] = bits(dot[0]);
                }
            }
        }
    }
};

/// The kernel's outputs [routes][n] under loop_model: an id at or past `experts` gives a NaN row.
pub fn project(a: std.mem.Allocator, cl: Call, out: []u16) !void {
    const k = cl.mat.k;
    if (k > 2304 or out.len != cl.ids.len * cl.n) return error.BadShape;
    const xs = try a.alloc(f64, cl.x.len);
    defer a.free(xs);
    for (xs, cl.x) |*v, h| v.* = bf.value(h);
    for (cl.ids, 0..) |id, r| {
        if (id < 0 or id >= cl.experts) {
            @memset(out[r * cl.n ..][0..cl.n], nan_bits);
        } else if (@as(usize, @intCast(id)) < cl.mat.first or @as(usize, @intCast(id)) - cl.mat.first >= cl.mat.count()) return error.ExpertNotHeld;
    }
    var jobs: [threads.maximum]Cols = undefined;
    const n_jobs = threads.count(threads.maximum, cl.n);
    for (jobs[0..n_jobs], 0..) |*j, i| j.* = .{ .call = &cl, .xs = xs, .out = out, .lo = cl.n * i / n_jobs, .hi = cl.n * (i + 1) / n_jobs };
    threads.run(Cols, jobs[0..n_jobs], .normal);
    for (jobs[0..n_jobs]) |j| if (j.err) |e| return e;
}

/// One output's dot as a simulated accumulator sums it: terms w[c] * x[c], summed in `order` (gate 4).
pub fn simDot(acc: core.contract.spec.Accum, order: core.contract.rows.Order, w: []const f64, x: []const f64, terms: []f64) f64 {
    for (terms, w, x) |*t, a, b| t.* = a * b;
    return core.contract.rows.sumIn(acc, order, terms);
}

/// Gate 4's columns: 128 of w * (1 + 2^-7), 129 of -w, one of +1, the rest 0, in a seeded order.
pub const Stress = enum(u8) { zero, plus, minus, one };

pub fn stressColumns(k: usize, out: []Stress) void {
    @memset(out[0..k], .zero);
    // the +1 sits alone in the last scale group (its own scale); the 257 large terms spread over the others
    out[k - 1] = .one;
    var prng = std.Random.DefaultPrng.init(0x57e55);
    var placed: usize = 0;
    while (placed < 257) {
        const c = prng.random().uintLessThan(usize, k - 32);
        if (out[c] != .zero) continue;
        out[c] = if (placed < 128) .plus else .minus;
        placed += 1;
    }
}

/// The stress terms' value: w = 1.5 * 2^16, so every large term overflows fp16 and is a multiple of 256.
pub const stress_w: f64 = 98304;

pub fn stressTerm(s: Stress) f64 {
    return switch (s) {
        .zero => 0,
        .plus => stress_w * (1 + 0x1p-7),
        .minus => -stress_w,
        .one => 1,
    };
}

test "loop_model's activation is one fp32 product of the two factors, rounded once: all 65,536 g by 64 u, and back" {
    var prng = std.Random.DefaultPrng.init(0x1a2b);
    var samples: [64]u16 = undefined;
    for (&samples, 0..) |*s, i| s.* = if (i < 8) ([_]u16{ 0, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x3f80, 0xc120, 0x0001 })[i] else prng.random().int(u16);
    for (0..65536) |i| for (samples) |j| for ([_][2]u16{ .{ @intCast(i), j }, .{ j, @intCast(i) } }) |p| {
        const g = bf.value(p[0]);
        const u = bf.value(p[1]);
        const want = bits(lm.expertActivation(g, u));
        const got = if (std.math.isNan(gateFactor(g) * upFactor(u))) nan_bits else bf.bits(gateFactor(g) * upFactor(u));
        if (want != got) {
            std.debug.print("g 0x{x:0>4} u 0x{x:0>4}: loop_model 0x{x:0>4}, factors 0x{x:0>4}\n", .{ p[0], p[1], want, got });
            return error.TestUnexpectedResult;
        }
    };
}

test "the reference decodes MXFP8 scale 255 as 2^128 (code 0x30 is 2^127), where an fp32 2^(s - 127) overflows" {
    try std.testing.expectEqual(@as(f64, 0x1p127), lm.mxfp8ReferenceWeight(0x30, 255));
    try std.testing.expectEqual(@as(f64, 0x1p-133), lm.mxfp8ReferenceWeight(0x08, 0)); // bf16's least subnormal
    try std.testing.expectEqual(@as(f64, 0), lm.mxfp8ReferenceWeight(0x01, 0)); // 2^-136: under half that step
    const cuda_scale: f32 = @bitCast(@as(u32, 255) << 23); // modelopt.cu e8m0(255)
    try std.testing.expect(std.math.isInf(cuda_scale));
}

test "stress rows: fp32 sums are exact in every order, fp16 sums are not finite, bf16 sums miss by 100x" {
    const c = core.contract.spec;
    for ([_]usize{ 2304, 1152 }) |k| {
        var cols: [2304]Stress = undefined;
        stressColumns(k, &cols);
        var w: [2304]f64 = undefined;
        var x: [2304]f64 = undefined;
        var terms: [2304]f64 = undefined;
        for (cols[0..k], w[0..k], x[0..k]) |s, *a, *b| {
            a.* = switch (s) {
                .zero => 0,
                .one => 1,
                else => stress_w,
            };
            b.* = switch (s) {
                .plus => 1 + 0x1p-7,
                .minus => -1,
                else => 1,
            };
            try std.testing.expectEqual(stressTerm(s), a.* * b.*);
        }
        const exact = simDot(.none, .sequential, w[0..k], x[0..k], terms[0..k]);
        try std.testing.expectEqual(@as(f64, 1), exact);
        for ([_]core.contract.rows.Order{ .sequential, .pairwise, .strided }) |o| {
            try std.testing.expectEqual(exact, simDot(.f32, o, w[0..k], x[0..k], terms[0..k]));
            try std.testing.expect(!std.math.isFinite(simDot(.f16, o, w[0..k], x[0..k], terms[0..k])));
            const b16 = simDot(.bf16, o, w[0..k], x[0..k], terms[0..k]);
            const allow_down = c.boundExact(.{ .name = "", .accum = .f32, .depth = undefined, .out = .bf16, .served = &.{} }, 1);
            try std.testing.expect(@abs(lm.bf16(b16) - 1) >= 100 * allow_down);
        }
    }
}
