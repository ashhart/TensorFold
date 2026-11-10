//! h3d_post_quant, h3d_embed and h3d_swiglu under the contract: each output against float64 at its bf16 points.
const std = @import("std");
const mtl = @import("metal");
const parts = @import("metal_h3_image_decoder_parts.zig");

const c = parts.c;
const Range = parts.Range;
const Ends = parts.Ends;
const Sum = parts.Sum;
const bf = parts.bf;
const value = parts.value;
const C = parts.channels;
const W = parts.W;
const F = parts.F;

/// Gate 4's weights: channels 12-23 negate 0-11 plus 1 on 4, against inputs repeating 0-11: exact fp32, small sums.
fn mirror(rng: std.Random, w: []u16, outs: usize) void {
    for (0..outs) |o| for (0..C / 2) |ch| {
        const v = rng.intRangeAtMost(i32, 128, 255);
        w[o * C + ch] = parts.integer(v);
        w[o * C + C / 2 + ch] = parts.integer(-v + @as(i32, if (ch < 4) 1 else 0));
    };
}

/// Gate 4's inputs for one cell: integers in [64, 127], channels 12-23 repeating 0-11.
fn mirrored(rng: std.Random) [C]f32 {
    var out: [C]f32 = undefined;
    for (0..C / 2) |ch| {
        out[ch] = @floatFromInt(rng.intRangeAtMost(i32, 64, 127));
        out[ch + C / 2] = out[ch];
    }
    return out;
}

/// Output `o` of a cell's 24 inputs: the 24-deep fp32 chain, the bias, one rounding to bf16.
fn product(sum: Sum, w: []const u16, bias: u16, in: *const [C]f64, o: usize) Sum.Out {
    var terms: [C]f64 = undefined;
    for (&terms, in, w[o * C ..][0..C]) |*t, x, wv| t.* = value(wv) * x;
    const s = sum.of(&terms, C);
    const b = value(bias);
    return .{ .v = bf(s.v + b), .r = s.r.add(Range.of(b)).fp32(0.5).bf16() };
}

/// The 24-deep chain, then the bias.
fn chain(s: c.Shape) usize {
    return s.k + 1;
}

/// Planted inputs for gate 1: (tile, cell, channel) of each of `parts.plants`; the last two in one cell.
const spots = [_][3]usize{ .{ 0, 0, 3 }, .{ 1, 1, 5 }, .{ 2, 2, 7 }, .{ 3, 3, 0 }, .{ 3, 3, 1 } };

// --- h3d_post_quant: a tile's cells of the fp32 latent rounded to bf16, then 24 x 24 with bias to bf16 -----------

pub const PostQuant = struct {
    lat: []f32 = &.{},
    org: [][2]u32 = &.{},
    pw: []u16 = &.{},
    pb: []u16 = &.{},
    hw: [2]usize = .{ 0, 0 },
    rows: usize = 0,
    cols: usize = 0,

    /// Rows: tiles; n: cells a tile (h x w); k: the channels each output sums. One 16 x 16 tile, 16, a 4 x 15 one.
    pub fn spec(name: []const u8, _: PostQuant) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = chain,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 8 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 256, .k = C }, .{ .rows = 16, .n = 256, .k = C }, .{ .rows = 1, .n = 60, .k = C } },
        };
    }

    /// Cell `i` of tile `t` in the latent's plane.
    fn place(k: *const PostQuant, t: usize, i: usize) usize {
        return (k.org[t][0] + i / k.hw[1]) * k.cols + k.org[t][1] + i % k.hw[1];
    }

    pub fn prepare(k: *PostQuant, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.k != C) return false;
        k.hw = parts.geometry(s.n) orelse return false;
        // a latent wider than the tile, so each tile reads at its own origin
        k.rows = k.hw[0] + 3;
        k.cols = k.hw[1] + 5;
        const plane = k.rows * k.cols;
        const al = t.host();
        k.lat = try al.alloc(f32, C * plane);
        k.org = try al.alloc([2]u32, s.rows);
        k.pw = try al.alloc(u16, C * C);
        k.pb = try al.alloc(u16, C);
        for (k.org) |*o| o.* = .{ rng.uintAtMost(u32, @intCast(k.rows - k.hw[0])), rng.uintAtMost(u32, @intCast(k.cols - k.hw[1])) };
        if (t.fill == .stress) {
            for (0..plane) |i| for (mirrored(rng), 0..) |v, ch| {
                k.lat[ch * plane + i] = v;
            };
            mirror(rng, k.pw, C);
            @memset(k.pb, 0);
        } else {
            for (k.lat) |*v| v.* = rng.float(f32) * 6 - 3;
            parts.uniform(rng, k.pw, -0.35, 0.35);
            parts.uniform(rng, k.pb, -0.1, 0.1);
            if (t.fill == .nonfinite) for (spots, parts.plants) |at, b| {
                k.lat[at[2] * plane + k.place(@min(at[0], s.rows - 1), at[1] % s.n)] = c.fromBf16(b);
            };
        }
        try t.input(k.lat);
        try t.input(k.org);
        try t.input(k.pw);
        try t.input(k.pb);
        try t.output(s.rows * s.n * C * 2);
        return true;
    }

    /// The launcher's embed stage, its first launch: threads [24, tiles x cells] in groups of 24 x 8.
    pub fn launch(k: *PostQuant, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        for (0..5) |i| t.bind(e, i, i);
        e.setValue(parts.Shape{ .tiles = @intCast(s.rows), .h = @intCast(k.hw[0]), .w = @intCast(k.hw[1]), .tokens = @intCast(s.n + parts.suffix), .rows = @intCast(k.rows), .cols = @intCast(k.cols), .m = @intCast(s.rows * (s.n + parts.suffix)) }, 5);
        e.dispatchThreads(mtl.Size.of(C, s.rows * s.n, 1), mtl.Size.of(C, 8, 1));
    }

    pub fn count(_: *PostQuant, s: c.Shape) usize {
        return s.rows * s.n * C;
    }

    pub fn outputs(k: *PostQuant, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        const s = t.shape;
        const plane = k.rows * k.cols;
        var in: [C]f64 = undefined;
        for (0..s.rows) |tl| for (0..s.n) |i| {
            const at = k.place(tl, i);
            for (&in, 0..) |*v, ch| v.* = bf(k.lat[ch * plane + at]);
            for (0..C) |o| {
                const y = product(sum, k.pw, k.pb[o], &in, o);
                ends.put(want, mag, (tl * s.n + i) * C + o, y.v, y.r);
            }
        };
    }
};

// --- h3d_embed: x_embedder (24 to 1,024 with bias) on a tile's cells to bf16, then the 4 registers and a zero token

pub const Embed = struct {
    tok: []u16 = &.{},
    ew: []u16 = &.{},
    eb: []u16 = &.{},
    reg: []u16 = &.{},
    hw: [2]usize = .{ 0, 0 },

    /// Rows: tiles; n: tokens a tile (its cells, the registers, the zero token); k: the channels a cell's output sums.
    pub fn spec(name: []const u8, _: Embed) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = chain,
            .out = .bf16,
            .served = &.{ .{ .rows = 1, .n = 261, .k = C }, .{ .rows = 16, .n = 261, .k = C }, .{ .rows = 1, .n = 65, .k = C } },
        };
    }

    pub fn prepare(k: *Embed, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.k != C or s.n <= parts.suffix) return false;
        const cells = s.n - parts.suffix;
        k.hw = parts.geometry(cells) orelse return false;
        const al = t.host();
        k.tok = try al.alloc(u16, s.rows * cells * C);
        k.ew = try al.alloc(u16, W * C);
        k.eb = try al.alloc(u16, W);
        k.reg = try al.alloc(u16, parts.registers * W);
        if (t.fill == .stress) {
            for (0..s.rows * cells) |i| for (mirrored(rng), 0..) |v, ch| {
                k.tok[i * C + ch] = c.toBf16(v);
            };
            mirror(rng, k.ew, W);
            @memset(k.eb, 0);
            parts.integers(rng, k.reg, -8, 8);
        } else {
            parts.uniform(rng, k.tok, -2, 2);
            parts.uniform(rng, k.ew, -0.35, 0.35);
            parts.uniform(rng, k.eb, -0.1, 0.1);
            parts.uniform(rng, k.reg, -1.5, 1.5);
            if (t.fill == .nonfinite) {
                for (spots, parts.plants) |at, b| k.tok[(@min(at[0], s.rows - 1) * cells + at[1] % cells) * C + at[2]] = b;
                k.reg[W + 7] = parts.nan16;
                k.reg[3 * W + 1000] = parts.inf16;
            }
        }
        try t.input(k.tok);
        try t.input(k.ew);
        try t.input(k.eb);
        try t.input(k.reg);
        try t.output(s.rows * s.n * W * 2);
        return true;
    }

    /// The launcher's embed stage, its second launch: threads [1,024, tiles x tokens] in groups of 256.
    pub fn launch(k: *Embed, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        for (0..5) |i| t.bind(e, i, i);
        e.setValue(parts.Shape{ .tiles = @intCast(s.rows), .h = @intCast(k.hw[0]), .w = @intCast(k.hw[1]), .tokens = @intCast(s.n), .m = @intCast(s.rows * s.n) }, 5);
        e.dispatchThreads(mtl.Size.of(W, s.rows * s.n, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn count(_: *Embed, s: c.Shape) usize {
        return s.rows * s.n * W;
    }

    pub fn outputs(k: *Embed, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        const s = t.shape;
        const cells = s.n - parts.suffix;
        var in: [C]f64 = undefined;
        for (0..s.rows) |tl| for (0..s.n) |tk| {
            const row = (tl * s.n + tk) * W;
            if (tk < cells) {
                for (&in, k.tok[(tl * cells + tk) * C ..][0..C]) |*v, b| v.* = value(b);
                for (0..W) |j| {
                    const y = product(sum, k.ew, k.eb[j], &in, j);
                    ends.put(want, mag, row + j, y.v, y.r);
                }
            } else for (0..W) |j| {
                const v = if (tk < cells + parts.registers) value(k.reg[(tk - cells) * W + j]) else 0;
                ends.put(want, mag, row + j, v, Range.of(v));
            }
        };
    }
};

// --- h3d_swiglu: act = bf16(bf16(SiLU(gate)) * value), each row [gate 4,096 | value 4,096] -------------------------

pub const Swiglu = struct {
    wide: []u16 = &.{},

    fn none(_: c.Shape) usize {
        return 0;
    }

    /// Rows: rows (one tile, the batch of 16, a small tile); n: the 4,096 outputs a row; k: 1, no sums.
    pub fn spec(name: []const u8, _: Swiglu) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = none,
            .ops = 2 * (parts.exp_ulps + parts.div_ulps) + 2,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 261, .n = F, .k = 1 }, .{ .rows = 4176, .n = F, .k = 1 }, .{ .rows = 65, .n = F, .k = 1 } },
        };
    }

    pub fn prepare(k: *Swiglu, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.n != F or s.k != 1) return false;
        k.wide = try t.host().alloc(u16, s.rows * 2 * F);
        for (0..s.rows) |r| {
            parts.uniform(rng, k.wide[r * 2 * F ..][0..F], -6, 6);
            parts.uniform(rng, k.wide[r * 2 * F + F ..][0..F], -3, 3);
        }
        if (t.fill == .nonfinite) {
            // NaN, +Inf and -Inf gates, then both infinities as values in one row
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 9 }, .{ 2, 100 }, .{ 3, F + 7 }, .{ 3, F + 8 } };
            for (at, parts.plants) |p, b| if (p[0] < s.rows) {
                k.wide[p[0] * 2 * F + p[1]] = b;
            };
        }
        try t.input(k.wide);
        try t.output(s.rows * F * 2);
        return true;
    }

    /// A block's FFN: threads [4,096, rows] in groups of 256.
    pub fn launch(_: *Swiglu, t: anytype, e: mtl.ComputeEncoder) void {
        const s = t.shape;
        for (0..2) |i| t.bind(e, i, i);
        e.setValue(parts.Shape{ .m = @intCast(s.rows) }, 2);
        e.dispatchThreads(mtl.Size.of(F, s.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn count(_: *Swiglu, s: c.Shape) usize {
        return s.rows * F;
    }

    const Job = struct {
        wide: []const u16,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        fn rows(j: Job, from: usize, to: usize) void {
            for (from..to) |r| for (0..F) |i| {
                const gv = value(j.wide[r * 2 * F + i]);
                const vv = value(j.wide[r * 2 * F + F + i]);
                const act = parts.silu(gv).bf16().mul(Range.of(vv)).fp32(0.5).bf16();
                j.ends.put(j.want, j.mag, r * F + i, bf(bf(gv / (1 + @exp(-gv))) * vv), act);
            };
        }
    };

    pub fn outputs(k: *Swiglu, t: anytype, _: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        parts.parallel(t.shape.rows, Job{ .wide = k.wide, .ends = ends, .want = want, .mag = mag }, Job.rows);
    }
};
