//! tf_uocr_matvec and tf_uocr_gather under the contract: a step's projections and a prompt's expert rows.
const std = @import("std");
const ud = @import("contract_uocr_decoder.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = ud.c;
const g = ud.g;
const mi = ud.mi;
const H = ud.H;
const D = ud.D;

/// A lane's 8 values a step (k / 256 steps), each step's 8 products in two 4-chains, then the warp's 5 shuffles.
fn mvDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 4 + 5;
}

pub const Matvec = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: one step; n: output rows; k: the input width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = mvDepth,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 256, .step = 8, .min = 8 } },
            // q | k | v, router q | k, o, the shared expert's gate | up and down, the dense layer's, the head
            .served = &.{
                .{ .rows = 1, .n = 3 * H, .k = H },         .{ .rows = 1, .n = 2 * D, .k = H },
                .{ .rows = 1, .n = H, .k = H },             .{ .rows = 1, .n = 2 * ud.shared_ffn, .k = H },
                .{ .rows = 1, .n = H, .k = ud.shared_ffn }, .{ .rows = 1, .n = 2 * ud.dense_ffn, .k = H },
                .{ .rows = 1, .n = H, .k = ud.dense_ffn },  .{ .rows = 1, .n = ud.vocab, .k = H },
            },
        };
    }

    pub fn deinit(t: *Matvec) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Matvec) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Matvec) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Matvec);

    pub fn prepare(t: *Matvec, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 8 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try a.alloc(u16, shape.k);
        t.w = try a.alloc(u16, shape.n * shape.k);
        if (fill == .stress) {
            // inputs 1, each weight row mirrored over its columns: fp32 sums exact, fp16 and bf16 lose bits
            @memset(t.x, c.toBf16(1));
            const col = try a.alloc(f32, shape.k);
            defer a.free(col);
            for (0..shape.n) |r| {
                attn.mirrored(rng, col);
                for (col, t.w[r * shape.k ..][0..shape.k]) |v, *o| o.* = c.toBf16(v);
            }
        } else {
            for (t.x) |*v| v.* = attn.normal(rng);
            mi.randomBf16(rng, t.w, -0.25, 0.25);
        }
        if (fill == .nonfinite) {
            t.w[3] = mi.nan16;
            if (shape.n > 2) t.w[2 * shape.k + shape.k - 1] = mi.inf16;
            if (shape.n > 4) t.w[4 * shape.k + 5] = mi.ninf16;
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.w);
        t.out = try t.rig.output(shape.n * 2);
        return true;
    }

    pub fn run(t: *Matvec) !void {
        try t.k.run("matvec", t.rig.s, .{ (t.shape.n + 31) / 32, 1, 1 }, 256, .{ t.ins[0], t.ins[1], t.out, t.shape.n, t.shape.k });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Matvec) usize {
        return t.shape.n;
    }

    pub fn reference(t: *Matvec, want: []f64, mag: []f64) !void {
        const k = t.shape.k;
        for (want, mag, 0..) |*w, *m, o| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (t.w[o * k ..][0..k], t.x) |a, b| {
                const p = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                s += p;
                abs += @abs(p);
            }
            w.* = s;
            m.* = abs;
        }
    }

    pub fn simulate(t: *Matvec, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const k = t.shape.k;
        const terms = try a.alloc(f64, k);
        defer a.free(terms);
        for (out, 0..) |*y, o| {
            for (terms, t.w[o * k ..][0..k], t.x) |*v, w, x| v.* = @as(f64, c.fromBf16(w)) * c.fromBf16(x);
            y.* = mi.rows.sumIn(acc, order, terms);
        }
    }
};

pub const Gather = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    rows: []u32 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: rows gathered; n: the width (1,280); k: rows to gather from.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .out = .bf16,
            // a prompt pass's 6 expert slots a row
            .served = &.{ .{ .rows = 6 * 2048, .n = H, .k = 2048 }, .{ .rows = 6 * 37, .n = H, .k = 37 } },
        };
    }

    pub fn deinit(t: *Gather) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.rows);
        t.rig.deinit();
    }
    pub fn kernel(t: *Gather) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gather);

    pub fn prepare(t: *Gather, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.rows);
        t.x = &.{};
        t.rows = &.{};
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try a.alloc(u16, shape.k * H);
        mi.randomBf16(rng, t.x, -4, 4);
        t.rows = try a.alloc(u32, shape.rows);
        for (t.rows) |*r| r.* = rng.uintLessThan(u32, @intCast(shape.k));
        if (fill == .nonfinite) for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
            t.x[@as(usize, t.rows[i % shape.rows]) * H + 7 * i] = v;
        };
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.rows);
        t.out = try t.rig.output(shape.rows * H * 2);
        return true;
    }

    pub fn run(t: *Gather) !void {
        try t.k.run("gather", t.rig.s, .{ t.shape.rows, ud.kernels.blocks(H), 1 }, 256, .{ t.ins[0], t.ins[1], t.out });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gather) usize {
        return t.shape.rows * H;
    }

    pub fn reference(t: *Gather, want: []f64, mag: []f64) !void {
        for (0..t.shape.rows) |i| for (0..H) |e| {
            const v: f64 = c.fromBf16(t.x[@as(usize, t.rows[i]) * H + e]);
            want[i * H + e] = v;
            mag[i * H + e] = 0;
        };
    }
};
