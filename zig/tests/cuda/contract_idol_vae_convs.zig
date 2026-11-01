//! tf_iv_conv_in and tf_iv_conv under the contract: Idol's image encoder's fp32 convolutions against float64.
const std = @import("std");
const cuda = @import("cuda");
const vae = @import("idol_vae_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const Fill = r.g.Fill;

/// NaN, +Inf and -Inf at `x`'s first values, and +Inf with -Inf side by side `apart` further on (one row holding both).
fn plant(x: []f64, apart: usize) void {
    r.plant(x, &.{ 0, 7, 14 });
    if (x.len > 3 * apart + 1) {
        x[3 * apart] = r.inf;
        x[3 * apart + 1] = -r.inf;
    }
}

// --- tf_iv_conv_in: conv3x3(2 x - 1) + b of the picture, 3 channels to 128 ------------------------------------------

pub const ConvIn = struct {
    pub const Kernels = vae.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;

    /// The 27 taps' products in one FMA chain (2 x - 1 rounded once before each), then the bias.
    fn depth(_: c.Shape) usize {
        return 27 + 2;
    }

    /// Rows: the picture's rows; n: the 128 output channels; k: its columns (the encoder's picture is 1024 x 1024).
    pub fn spec(name: []const u8, _: u32) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{ .size = 2 }, .{}, .{ .size = 2 } },
            .served = &.{ .{ .rows = 12, .n = 128, .k = 1024 }, .{ .rows = 1024, .n = 128, .k = 12 }, .{ .rows = 37, .n = 128, .k = 41 } },
        };
    }

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.k > 0 and s.n == 128;
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * s.k * 3, 27 * 128, 128 };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.k * 128;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
        if (mode == .stress) {
            // 2 x - 1 an odd integer (127 or 2041), channel 1 two more, channel 2 zero; weights 509, -509, 1 cancel
            for (0..s.rows * s.k) |p| {
                const x: f64 = if (p % 2 == 0) 2041 else 127;
                v[0][p * 3] = (x + 1) / 2;
                v[0][p * 3 + 1] = (x + 3) / 2;
                v[0][p * 3 + 2] = 0.5;
            }
            for (0..27) |row| @memset(v[1][row * 128 ..][0..128], switch (row % 3) {
                0 => 509,
                1 => -509,
                else => 1,
            });
            @memset(v[2], 0);
            return;
        }
        r.uniform(rng, v[0], 0, 1);
        r.uniform(rng, v[1], -0.5, 0.5);
        r.uniform(rng, v[2], -1, 1);
        if (mode == .nonfinite) plant(v[0], s.k * 3);
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try vae.convIn(k, s, dev[0], dev[1], dev[2], o, shape.rows, shape.k);
    }

    pub fn ref(a: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const buf = try a.alloc(f64, 3 * 27);
        defer a.free(buf);
        const terms = buf[0..27];
        const lo = buf[27..54];
        const hi = buf[54..];
        for (0..shape.rows * shape.k) |px| for (0..128) |o| {
            const py: isize = @intCast(px / shape.k);
            const pxx: isize = @intCast(px % shape.k);
            for (0..3) |ky| for (0..3) |kx| for (0..3) |ch| {
                const iy = py + @as(isize, @intCast(ky)) - 1;
                const ix = pxx + @as(isize, @intCast(kx)) - 1;
                const at = (ky * 3 + kx) * 3 + ch;
                if (iy < 0 or ix < 0 or iy >= shape.rows or ix >= shape.k) {
                    terms[at] = 0;
                    lo[at] = 0;
                    hi[at] = 0;
                    continue;
                }
                const x = v[0][(@as(usize, @intCast(iy)) * shape.k + @as(usize, @intCast(ix))) * 3 + ch];
                const w = v[1][at * 128 + o];
                const t = Range.of(2 * x - 1).fp32(0.5).mul(Range.of(w));
                terms[at] = (2 * x - 1) * w;
                lo[at] = t.lo;
                hi[at] = t.hi;
            };
            const s = sum.total(terms, lo, hi, 27);
            const b = v[2][o];
            e.put(want, mag, px * 128 + o, s.v + b, s.r.add(Range.of(b)).fp32(0.5));
        };
    }
};

// --- tf_iv_conv: same-size and stride-2 3x3s and 1x1s as one implicit GEMM, fp32 sums, + bias (+ a residual) ---------

/// A form the encoder launches: taps (9 or 1), stride, padding above and left, a residual, and its served shapes.
pub const Form = struct { taps: usize, stride: usize = 1, pad: usize = 0, add: bool = false, served: []const c.Shape };

/// A 3x3's rows are output rows of 16 columns (8: a 128-pixel tile), a 1x1's pixels; fewer than the engine's.
pub const forms = struct {
    pub const same: Form = .{ .taps = 9, .pad = 1, .served = &.{ .{ .rows = 9, .n = 128, .k = 128 }, .{ .rows = 9, .n = 256, .k = 128 }, .{ .rows = 9, .n = 256, .k = 256 }, .{ .rows = 9, .n = 512, .k = 256 }, .{ .rows = 9, .n = 512, .k = 512 }, .{ .rows = 9, .n = 64, .k = 512 } } };
    pub const residual: Form = .{ .taps = 9, .pad = 1, .add = true, .served = &.{ .{ .rows = 9, .n = 128, .k = 128 }, .{ .rows = 9, .n = 256, .k = 256 }, .{ .rows = 9, .n = 512, .k = 512 } } };
    pub const down: Form = .{ .taps = 9, .stride = 2, .served = &.{ .{ .rows = 9, .n = 128, .k = 128 }, .{ .rows = 9, .n = 256, .k = 256 }, .{ .rows = 9, .n = 512, .k = 512 } } };
    pub const one: Form = .{ .taps = 1, .served = &.{ .{ .rows = 1100, .n = 256, .k = 128 }, .{ .rows = 1100, .n = 512, .k = 256 }, .{ .rows = 1100, .n = 64, .k = 64 } } };
};

const out_cols = 16;

pub fn Conv(comptime f: Form) type {
    return struct {
        pub const Kernels = vae.Kernels;
        pub const kinds = [_]r.Kind{ .f32, .f32, .f32, .f32 };
        pub const out: r.Kind = .f32;
        pub const in_place: ?usize = null;

        /// Each output sums taps * cin products in one FMA chain, then the bias and the residual.
        fn depth(s: c.Shape) usize {
            return f.taps * s.k + 2;
        }

        /// Rows: output rows (3x3) or pixels (1x1); n: output channels; k: input channels.
        pub fn spec(name: []const u8, _: u32) c.Spec {
            return .{
                .name = name,
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .tiles = .{ if (f.taps == 9) .{ .size = vae.tile_m / out_cols } else .{ .size = vae.tile_m }, .{ .size = vae.tile_n, .step = 64, .min = 64 }, .{ .size = vae.tile_k, .step = vae.tile_k, .min = vae.tile_k } },
                .served = f.served,
            };
        }

        fn cols() usize {
            return if (f.taps == 9) out_cols else 1;
        }
        fn shapeOf(s: c.Shape) vae.ConvShape {
            return .{ .rows = s.rows, .cols = cols(), .in_rows = s.rows * f.stride, .in_cols = cols() * f.stride, .cin = s.k, .cout = s.n, .ldw = vae.ldwOf(s.n), .taps = f.taps, .stride = f.stride, .pad = f.pad };
        }

        pub fn accepts(s: c.Shape) bool {
            return s.rows > 0 and s.n > 0 and s.k > 0 and vae.convOk(shapeOf(s));
        }

        pub fn sizes(s: c.Shape) [4]usize {
            const g = shapeOf(s);
            return .{ g.in_rows * g.in_cols * s.k, f.taps * s.k * g.ldw, s.n, if (f.add) count(s) else 0 };
        }

        pub fn count(s: c.Shape) usize {
            return s.rows * cols() * s.n;
        }

        pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[4][]f64) void {
            const ldw = vae.ldwOf(s.n);
            if (mode == .stress) {
                // the channels' second half mirroring the first, weights 1 then -1: partial sums reach ~2^18 and cancel
                const half = s.k / 2;
                for (0..v[0].len / s.k) |p| for (0..s.k) |ci| {
                    const at = if (ci < half) ci else ci - half;
                    const x: f64 = if (at % 2 == 0) 127 else 2041;
                    v[0][p * s.k + ci] = if (ci < half) x else x + @as(f64, if (at < 4) 1 else 0);
                };
                for (0..f.taps * s.k) |k| for (0..ldw) |o| {
                    v[1][k * ldw + o] = if (o >= s.n) 0 else if (k % s.k < half) 1 else -1;
                };
                @memset(v[2], 0);
                r.integers(rng, v[3], 3);
                return;
            }
            r.uniform(rng, v[0], -2, 2);
            r.uniform(rng, v[1], -0.1, 0.1);
            r.uniform(rng, v[2], -1, 1);
            r.uniform(rng, v[3], -3, 3);
            if (mode == .nonfinite) plant(v[0], s.k);
        }

        pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, _: u32) !void {
            try vae.conv(k, s, dev[0], dev[1], dev[2], if (f.add) dev[3] else null, o, shapeOf(shape));
        }

        /// Output (p, o)'s products in the kernel's order, tap then input channel; a tap outside the input is 0.
        fn terms(shape: c.Shape, v: [4][]const f64, p: usize, o: usize, out_: []f64) void {
            const g = shapeOf(shape);
            const K = shape.k;
            const oy: isize = @intCast(p / g.cols);
            const ox: isize = @intCast(p % g.cols);
            const st: isize = @intCast(f.stride);
            const pd: isize = @intCast(f.pad);
            for (0..f.taps) |tap| {
                const iy = oy * st + @as(isize, @intCast(if (f.taps == 9) tap / 3 else 0)) - pd;
                const ix = ox * st + @as(isize, @intCast(if (f.taps == 9) tap % 3 else 0)) - pd;
                const inside = iy >= 0 and ix >= 0 and iy < g.in_rows and ix < g.in_cols;
                for (0..K) |ci| {
                    out_[tap * K + ci] = if (inside) blk: {
                        const at = (@as(usize, @intCast(iy)) * g.in_cols + @as(usize, @intCast(ix))) * K + ci;
                        break :blk v[0][at] * v[1][(tap * K + ci) * g.ldw + o];
                    } else 0;
                }
            }
        }

        pub fn ref(a: std.mem.Allocator, shape: c.Shape, _: u32, v: [4][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
            const K = f.taps * shape.k;
            const buf = try a.alloc(f64, K);
            defer a.free(buf);
            for (0..shape.rows * cols()) |p| for (0..shape.n) |o| {
                terms(shape, v, p, o, buf);
                const s = sum.total(buf, buf, buf, K);
                const i = p * shape.n + o;
                const b = v[2][o];
                var val = s.v + b;
                var vr = s.r.add(Range.of(b)).fp32(0.5);
                if (f.add) {
                    val += v[3][i];
                    vr = vr.add(Range.of(v[3][i])).fp32(0.5);
                }
                e.put(want, mag, i, val, vr);
            };
        }
    };
}

test "conv references: a delta kernel copies its input, the stride-2 3x3 pads right and below alone" {
    const a = std.testing.allocator;
    const C = Conv(forms.down);
    const shape: c.Shape = .{ .rows = 2, .n = 64, .k = 8 };
    var vals: [4][]f64 = undefined;
    for (&vals, C.sizes(shape)) |*v, n| {
        v.* = try a.alloc(f64, n);
        @memset(v.*, 0);
    }
    defer for (vals) |v| a.free(v);
    for (vals[0], 0..) |*x, i| x.* = @floatFromInt(i + 1);
    // tap (2, 2), channel 3 to output 5: output pixel (oy, ox) reads input (2 oy + 2, 2 ox + 2)
    const ldw = vae.ldwOf(64);
    vals[1][(8 * 8 + 3) * ldw + 5] = 1;
    var ends: r.Ends = .{ .a = a };
    defer ends.deinit();
    try ends.size(C.count(shape));
    const want = try a.alloc(f64, C.count(shape));
    defer a.free(want);
    const mag = try a.alloc(f64, C.count(shape));
    defer a.free(mag);
    try C.ref(a, shape, 0, .{ vals[0], vals[1], vals[2], vals[3] }, .{ .range = true }, &ends, want, mag);
    try std.testing.expectEqual(vals[0][(2 * 32 + 2) * 8 + 3], want[5]);
    try std.testing.expectEqual(vals[0][(2 * 32 + 30) * 8 + 3], want[14 * 64 + 5]);
    // past the right edge (input column 32) and below (input row 4): the padding's zeros
    try std.testing.expectEqual(@as(f64, 0), want[15 * 64 + 5]);
    for (want[16 * 64 ..]) |w| try std.testing.expectEqual(@as(f64, 0), w);
    try std.testing.expect(C.accepts(shape) and !C.accepts(.{ .rows = 1, .n = 64, .k = 12 }));
}
