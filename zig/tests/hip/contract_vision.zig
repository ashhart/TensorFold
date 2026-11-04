//! Shared vision primitives against independent scalar FP64 formulas at the full image budget and tile edges.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;

pub const Which = enum { layer_norm, gelu, rope, attention_rows };

fn depth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 8;
}

pub fn spec(which: Which) c.Spec {
    return .{
        .name = switch (which) {
            .layer_norm => "hip BF16 LayerNorm",
            .gelu => "hip BF16 exact GELU",
            .rope => "hip BF16 axial vision RoPE",
            .attention_rows => "hip FP32 attention to BF16 rows",
        },
        .accum = if (which == .layer_norm) .f32 else .none,
        .depth = depth,
        .ops = 10,
        .out = if (which == .rope) .f32 else .bf16,
        .tiles = if (which == .layer_norm) .{ .{}, .{ .size = 256 }, .{} } else .{ .{ .size = 256 }, .{}, .{} },
        .served = switch (which) {
            .layer_norm => &.{ .{ .rows = 57600, .n = 1536, .k = 1536 }, .{ .rows = 17, .n = 1536, .k = 1536 } },
            .gelu => &.{ .{ .rows = 14400, .n = 6144, .k = 1 }, .{ .rows = 17, .n = 6144, .k = 1 } },
            else => &.{ .{ .rows = 57600, .n = 1536, .k = 128 }, .{ .rows = 17, .n = 1536, .k = 128 } },
        },
        .sums = .{ .bf16 = 0, .why = "BF16 output and variance normalization can hide a BF16 sum's error" },
    };
}

pub const Vision = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    bias: []u16 = &.{},
    positions: [][2]u32 = &.{},
    dx: u64 = 0,
    dw: u64 = 0,
    db: u64 = 0,
    dp: u64 = 0,
    out: [3]u64 = @splat(0),
    kv: [2]u64 = @splat(0),
    lut: []f64 = &.{},

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, which: Which) Vision {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .which = which, .arena = .init(a) };
    }

    pub fn deinit(v: *Vision) void {
        v.arena.deinit();
        v.rig.deinit();
    }

    const vt = g.hooks(Vision);
    pub fn kernel(v: *Vision) g.Kernel {
        return .{ .spec = spec(v.which), .ctx = v, .vt = &vt };
    }

    pub fn prepare(v: *Vision, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.n > 65536) return false;
        if (v.which == .layer_norm and s.k != s.n) return false;
        if ((v.which == .rope or v.which == .attention_rows) and (s.k != 128 or s.n % 128 != 0)) return false;
        v.rig.reset();
        _ = v.arena.reset(.retain_capacity);
        const a = v.arena.allocator();
        v.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = s.rows * s.n;
        v.x = try a.alloc(u16, n * (if (v.which == .rope) @as(usize, 3) else 1));
        v.w = try a.alloc(u16, s.n);
        v.bias = try a.alloc(u16, s.n);
        for (v.x, 0..) |*x, i| x.* = c.toBf16(if (fill == .stress) (if (i % 2 == 0) @as(f32, 256) else -256) else rng.float(f32) * 16 - 8);
        for (v.w, v.bias) |*w, *b| {
            w.* = c.toBf16(if (fill == .stress) 1 else rng.float(f32) + 0.5);
            b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) - 0.5);
        }
        if (fill == .nonfinite) {
            const plants = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (plants, 0..) |x, i| if (i < s.rows) {
                v.x[i * s.n + @min(s.n - 1, i + 1)] = c.toBf16(x);
            };
        }
        if (v.which == .attention_rows) {
            const xf = try a.alloc(f32, n);
            for (xf, v.x) |*x, b| x.* = c.fromBf16(b);
            v.dx = try v.rig.input(std.mem.sliceAsBytes(xf));
        } else v.dx = try v.rig.input(std.mem.sliceAsBytes(v.x));
        v.dw = try v.rig.input(std.mem.sliceAsBytes(v.w));
        v.db = try v.rig.input(std.mem.sliceAsBytes(v.bias));
        v.out[0] = try v.rig.output(n * (if (v.which == .rope) @as(usize, 4) else 2));
        if (v.which == .rope) {
            v.positions = try a.alloc([2]u32, s.rows);
            for (v.positions, 0..) |*p, i| p.* = .{ @intCast(i / 2), @intCast(i % 2) };
            v.dp = try v.rig.input(std.mem.sliceAsBytes(v.positions));
            for (&v.kv, v.out[1..]) |*kv, *out| {
                kv.* = try v.rig.scratch(n * 2);
                out.* = try v.rig.output(n * 4);
            }
        }
        if (v.which == .gelu) {
            v.lut = try a.alloc(f64, 65536);
            for (v.lut, 0..) |*x, i| x.* = gelu(c.fromBf16(@intCast(i)));
        }
        return true;
    }

    pub fn run(v: *Vision) !void {
        var arena = hip.Arena.counting();
        const o: hip.ops.Ops = .{ .l = v.l, .bf16 = true, .stream = v.rig.s.handle, .arena = &arena };
        const s = v.shape;
        const x: hip.ops.Tensor = .{ .ptr = v.dx, .kind = .bf16 };
        const out: hip.ops.Tensor = .{ .ptr = v.out[0], .kind = .bf16 };
        switch (v.which) {
            .layer_norm => try hip.vision.layerNorm(o, x, v.dw, v.db, out, s.rows, s.n, 1e-6),
            .gelu => try hip.vision.gelu(o, x, out, s.rows, s.n),
            .attention_rows => try hip.vision.attentionRows(o, v.dx, out, s.rows, s.n / s.k, s.k),
            .rope => {
                try hip.vision.rope(o, x, v.dp, v.out[0], v.kv[0], v.kv[1], s.rows, s.n / s.k, s.k, 10000);
                for (v.kv, v.out[1..]) |kv, dst| try o.cast(.{ .ptr = kv, .kind = .bf16 }, .{ .ptr = dst, .kind = .f32 }, s.rows * s.n);
            },
        }
        try v.rig.s.synchronize();
    }

    pub fn count(v: *Vision) usize {
        return v.shape.rows * v.shape.n * (if (v.which == .rope) @as(usize, 3) else 1);
    }

    /// Simpson integration of the Gaussian is independent of the device erf implementation.
    fn gelu(x: f64) f64 {
        if (!std.math.isFinite(x)) return if (x > 0) x else std.math.nan(f64);
        if (@abs(x) > 12) return if (x > 0) x else 0;
        const z = x / @sqrt(@as(f64, 2));
        const step = z / 512;
        var sum: f64 = 1 + @exp(-z * z);
        for (1..512) |i| {
            const t = @as(f64, @floatFromInt(i)) * step;
            sum += @as(f64, if (i % 2 == 0) 2 else 4) * @exp(-t * t);
        }
        const erf = 2.0 / @sqrt(@as(f64, std.math.pi)) * (step / 3) * sum;
        return 0.5 * x * (1 + erf);
    }

    pub fn reference(v: *Vision, want: []f64, mag: []f64) !void {
        const s = v.shape;
        switch (v.which) {
            .layer_norm => try v.layer(want, mag, null, .sequential),
            .gelu => for (want, mag, v.x) |*y, *m, x| {
                y.* = v.lut[x];
                m.* = @abs(c.fromBf16(x));
            },
            .attention_rows => for (want, mag, 0..) |*y, *m, i| {
                const row = i / s.n;
                const head = (i % s.n) / s.k;
                const dim = i % s.k;
                y.* = c.fromBf16(v.x[(head * s.rows + row) * s.k + dim]);
                m.* = @abs(y.*);
            },
            .rope => {
                const n = s.rows * s.n;
                for (0..s.rows) |row| for (0..s.k) |dim| {
                    const j = dim % (s.k / 2);
                    const exponent = @as(f64, @floatFromInt(j % (s.k / 4))) / @as(f64, @floatFromInt(s.k / 4));
                    const angle = @as(f64, @floatFromInt(v.positions[row][j / (s.k / 4)])) / std.math.pow(f64, 10000, exponent);
                    const co = @cos(angle);
                    const si = @sin(angle);
                    const partner = if (dim < s.k / 2) dim + s.k / 2 else dim - s.k / 2;
                    const sign: f64 = if (dim < s.k / 2) -1 else 1;
                    for (0..s.n / s.k) |head| {
                        const src = row * s.n * 3 + head * s.k;
                        const dst = (head * s.rows + row) * s.k + dim;
                        for (0..2) |which| {
                            const a: f64 = c.fromBf16(v.x[src + which * s.n + dim]);
                            const b: f64 = c.fromBf16(v.x[src + which * s.n + partner]);
                            want[which * n + dst] = a * co + sign * b * si;
                            mag[which * n + dst] = @abs(a) + @abs(b);
                        }
                        want[2 * n + dst] = c.fromBf16(v.x[src + 2 * s.n + dim]);
                        mag[2 * n + dst] = 0;
                    }
                };
            },
        }
    }

    /// Propagate both mean and variance errors through LayerNorm; cancellation uses the input magnitude.
    fn layer(v: *Vision, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) !void {
        const s = v.shape;
        const a = v.arena.allocator();
        const terms = try a.alloc(f64, s.n);
        defer a.free(terms);
        for (0..s.rows) |row| {
            var mean: f64 = 0;
            var abs: f64 = 0;
            for (terms, 0..) |*t, j| {
                t.* = c.fromBf16(v.x[row * s.n + j]);
                mean += t.*;
                abs += @abs(t.*);
            }
            mean = (if (acc) |kind| rows_.sumIn(kind, order, terms) else mean) / @as(f64, @floatFromInt(s.n));
            var variance: f64 = 0;
            for (terms, 0..) |*t, j| {
                const d = @as(f64, c.fromBf16(v.x[row * s.n + j])) - mean;
                t.* = d * d;
                variance += t.*;
            }
            variance = (if (acc) |kind| rows_.sumIn(kind, order, terms) else variance) / @as(f64, @floatFromInt(s.n));
            const inv = 1 / @sqrt(variance + 1e-6);
            for (0..s.n) |j| {
                const i = row * s.n + j;
                const x: f64 = c.fromBf16(v.x[i]);
                const w: f64 = c.fromBf16(v.w[j]);
                want[i] = (x - mean) * inv * w + c.fromBf16(v.bias[j]);
                mag[i] = (3 * @abs(x - mean) + 2 * abs / @as(f64, @floatFromInt(s.n))) * inv * @abs(w);
            }
        }
    }

    pub fn simulate(v: *Vision, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try v.rig.a.alloc(f64, out.len);
        defer v.rig.a.free(mag);
        try v.layer(out, mag, acc, order);
    }

    pub fn bounds(v: *Vision, want: []const f64, mag: []const f64, allow: []f64) !void {
        var position: u32 = 0;
        if (v.which == .rope) for (v.positions) |p| {
            position = @max(position, @max(p[0], p[1]));
        };
        for (want, mag, allow) |w, m, *out| out.* = switch (v.which) {
            .layer_norm, .attention_rows => c.bound(spec(v.which), v.shape, w, m),
            .gelu => (0x1p-22 * m + 0x1p-8 * @abs(w)) * (1 + 0x1p-8) + 0x1p-133,
            // FP32 inv_freq, angle, sin/cos and products, then one BF16 rounding, including widened Q.
            .rope => (0x1p-21 * @max(@as(f64, @floatFromInt(position)), 1) * m + 0x1p-8 * @abs(w)) * (1 + 0x1p-8) + 0x1p-133,
        };
    }
};
