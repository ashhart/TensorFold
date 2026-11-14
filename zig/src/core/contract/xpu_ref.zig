//! The XPU glue's float64 references: the normed and residual rows, the gated row, the SwiGLU, embed and argmax.
const std = @import("std");
pub const attention = @import("xpu_attn_ref.zig");
const c = @import("contract.zig");
const rows = @import("rows.zig");

test {
    _ = attention;
}

/// glue.cl's GS, the group the glue's row sums cover, and ASC, the scale it stores them at.
pub const GS: usize = 64;
pub const ASC: f64 = 256;

/// A value rounded to bf16, round to nearest even, NaN and infinities kept: glue.cl's rbf.
pub fn bf16Round(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value rounded to fp16, round to nearest even, out of range to an infinity: glue.cl's convert_half_rte.
pub fn f16Round(v: f64) f64 {
    return @as(f16, @floatCast(v));
}

/// glue.cl's silu: x / (1 + e^-x), NaN for -Inf as x * sigmoid(x) gives.
pub fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// The normed row's sum of squares: a lane every 512th block, the sub-group's tree, then the group sums in order.
pub fn normDepth(dec: bool, s: c.Shape) usize {
    return s.n / 512 + @as(usize, if (dec) 36 else 12);
}

/// The gated row's sum of squares: eight a lane, the sub-group's tree.
pub fn gatedDepth(_: c.Shape) usize {
    return 12;
}

/// The SwiGLU row sums: a lane's share of a group, the sub-group's tree.
pub fn sumsDepth(_: c.Shape) usize {
    return GS / 16 + 4;
}

/// One call's norm: the kernel's eps and whether it is the 512-work-item decode form.
pub const Norm = struct { dec: bool, eps: f32 };

/// xr: the residual the norm reads, bf16(x + r) when there is one, else x as it lies.
fn residual(x: []const f64, r: []const f64, i: usize, has_r: bool) f64 {
    return if (has_r) bf16Round(x[i] + r[i]) else x[i];
}

/// The normed row, fp16(bf16(rms(x) · w)), beside the residual it reads; mag is the sums' share of each output.
pub fn normedRow(a: std.mem.Allocator, x: []const f64, r: []const f64, w: []const f64, n: Norm, shape: c.Shape, has_r: bool, want: []f64, mag: []f64) !void {
    const d = shape.n;
    const t = try a.alloc(f64, d);
    defer a.free(t);
    for (0..shape.rows) |row| {
        for (t, 0..) |*v, j| {
            const xr = residual(x, r, row * d + j, has_r);
            v.* = xr * xr;
        }
        var ss: f64 = 0;
        for (t) |v| ss += v;
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + n.eps);
        for (0..d) |j| {
            const xr = residual(x, r, row * d + j, has_r);
            const y = f16Round(bf16Round(xr * inv * w[j]));
            want[row * d + j] = y;
            mag[row * d + j] = @abs(y) / 2; // y ~ ss^-1/2: an error in the sum moves it by half
        }
    }
}

/// The residual the kernel writes in place, bf16(x + r) widened: one rounding, no sums.
pub fn residualRow(x: []const f64, r: []const f64, shape: c.Shape, want: []f64, mag: []f64) void {
    for (0..shape.rows * shape.n) |i| {
        want[i] = residual(x, r, i, true);
        mag[i] = 0;
    }
}

/// The normed row if its sum of squares ran in `acc` in `order`, the rest of the arithmetic exact.
pub fn simulateNormed(a: std.mem.Allocator, x: []const f64, r: []const f64, w: []const f64, n: Norm, shape: c.Shape, has_r: bool, acc: c.Accum, order: rows.Order, out: []f64) !void {
    const d = shape.n;
    const t = try a.alloc(f64, d);
    defer a.free(t);
    for (0..shape.rows) |row| {
        for (t, 0..) |*v, j| {
            const xr = residual(x, r, row * d + j, has_r);
            v.* = xr * xr;
        }
        const ss = rows.sumIn(acc, order, t);
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + n.eps);
        for (0..d) |j| out[row * d + j] = residual(x, r, row * d + j, has_r) * inv * w[j];
    }
}

/// The gated row, fp16(bf16(silu(z) · rms(y) · w)), a head of 128 a sub-group; mag is the sums' share of each output.
pub fn gatedRow(a: std.mem.Allocator, y: []const f64, z: []const f64, w: []const f64, eps: f32, shape: c.Shape, want: []f64, mag: []f64) !void {
    const vd = shape.n;
    const t = try a.alloc(f64, 128);
    defer a.free(t);
    for (0..shape.rows) |row| {
        var head: usize = 0;
        while (head < vd) : (head += 128) {
            for (t, 0..) |*v, j| {
                const yv = y[row * vd + head + j];
                v.* = yv * yv;
            }
            var ss: f64 = 0;
            for (t) |v| ss += v;
            const inv = 1 / @sqrt(ss / 128 + eps);
            for (0..128) |j| {
                const i = row * vd + head + j;
                const yv = f16Round(bf16Round(silu(z[i]) * (y[i] * inv * w[j])));
                want[i] = yv;
                mag[i] = @abs(yv) / 2;
            }
        }
    }
}

/// The gated row if each head's sum of squares ran in `acc` in `order`, the rest of the arithmetic exact.
pub fn simulateGated(a: std.mem.Allocator, y: []const f64, z: []const f64, w: []const f64, eps: f32, shape: c.Shape, acc: c.Accum, order: rows.Order, out: []f64) !void {
    const vd = shape.n;
    const t = try a.alloc(f64, 128);
    defer a.free(t);
    for (0..shape.rows) |row| {
        var head: usize = 0;
        while (head < vd) : (head += 128) {
            for (t, 0..) |*v, j| {
                const yv = y[row * vd + head + j];
                v.* = yv * yv;
            }
            const ss = rows.sumIn(acc, order, t);
            const inv = 1 / @sqrt(ss / 128 + eps);
            for (0..128) |j| {
                const i = row * vd + head + j;
                out[i] = silu(z[i]) * (y[i] * inv * w[j]);
            }
        }
    }
}

/// The SwiGLU activation, fp16(bf16(silu(g) · u)), gate at column c and up at Q27_FF + c.
pub fn swigluAct(gu: []const f64, ff: usize, want: []f64, mag: []f64) void {
    for (0..want.len) |i| {
        const row = i / ff;
        const col = i % ff;
        const g = gu[row * 2 * ff + col];
        const u = gu[row * 2 * ff + ff + col];
        want[i] = f16Round(bf16Round(silu(g) * u));
        mag[i] = 0;
    }
}

/// The SwiGLU row sums, mean(act) · ASC a group of GS before the fp16 store the bound allows, Inf past fp16's range.
pub fn swigluSums(act: []const f64, ff: usize, want: []f64, mag: []f64) void {
    const groups = ff / GS;
    for (0..act.len / ff) |row| for (0..groups) |g| {
        var sum: f64 = 0;
        var abs: f64 = 0;
        for (act[row * ff + g * GS ..][0..GS]) |v| {
            sum += v;
            abs += @abs(v);
        }
        const s = sum / @as(f64, @floatFromInt(GS)) * ASC;
        want[row * groups + g] = if (std.math.isInf(f16Round(s))) f16Round(s) else s;
        mag[row * groups + g] = abs / @as(f64, @floatFromInt(GS)) * ASC;
    };
}

/// The SwiGLU row sums if each group's sum ran in `acc` in `order`, the rest of the arithmetic exact.
pub fn simulateSums(a: std.mem.Allocator, act: []const f64, ff: usize, acc: c.Accum, order: rows.Order, out: []f64) !void {
    const groups = ff / GS;
    const t = try a.alloc(f64, GS);
    defer a.free(t);
    for (0..act.len / ff) |row| for (0..groups) |g| {
        @memcpy(t, act[row * ff + g * GS ..][0..GS]);
        out[row * groups + g] = rows.sumIn(acc, order, t) / @as(f64, @floatFromInt(GS)) * ASC;
    };
}

/// The embedding value glue.cl's embed stores: bf16(q · s + b), the product and the sum in fp32, one bf16 rounding.
pub fn embedValue(q: f64, s: f64, b: f64) f64 {
    const p: f32 = @floatCast(q * s); // q <= 15 and s bf16: exact in fp32
    const sum: f32 = p + @as(f32, @floatCast(b));
    return c.fromBf16(c.toBf16(sum));
}

/// The pick glue_quant.cl's argmax gives: a NaN skipped, the largest above -Inf, the lowest index on a tie.
pub fn argmaxPick(row: []const f64) i32 {
    var best: f64 = -std.math.inf(f64);
    var at: i32 = std.math.maxInt(i32);
    for (row, 0..) |v, i| {
        if (std.math.isNan(v)) continue;
        if (v > best) {
            best = v;
            at = @intCast(i);
        }
    }
    return at;
}

/// Every element of `row` that is a NaN, for the rule's check to see a planted row.
pub fn hasNan(row: []const f64) bool {
    for (row) |v| if (std.math.isNan(v)) return true;
    return false;
}

test "bf16 and fp16 rounding keep NaN, keep infinities and round to nearest even" {
    try std.testing.expectEqual(@as(f64, 1), bf16Round(1.00390625)); // halfway rounds to even
    try std.testing.expectEqual(@as(f64, 1.0078125), bf16Round(1.005859375));
    try std.testing.expect(std.math.isNan(bf16Round(std.math.nan(f64))));
    try std.testing.expectEqual(std.math.inf(f64), bf16Round(std.math.inf(f64)));
    try std.testing.expectEqual(@as(f64, 1), f16Round(1));
    try std.testing.expectEqual(std.math.inf(f64), f16Round(1e6));
    try std.testing.expect(std.math.isNan(f16Round(std.math.nan(f64))));
}

test "the normed row is the residual's norm, and its fp16 and bf16 replays lose the sum" {
    const a = std.testing.allocator;
    const shape = c.Shape{ .rows = 2, .n = 512, .k = 512 };
    const x = try a.alloc(f64, 2 * 512);
    defer a.free(x);
    const r = try a.alloc(f64, 2 * 512);
    defer a.free(r);
    const w = try a.alloc(f64, 512);
    defer a.free(w);
    const want = try a.alloc(f64, 2 * 512);
    defer a.free(want);
    const mag = try a.alloc(f64, 2 * 512);
    defer a.free(mag);
    const got = try a.alloc(f64, 2 * 512);
    defer a.free(got);
    var rng = std.Random.DefaultPrng.init(7);
    const rand = rng.random();
    for (w) |*v| v.* = bf16Round(0.75 + rand.float(f64) * 0.5);
    for (x) |*v| v.* = @floatFromInt(rand.intRangeAtMost(i32, -64, 64));
    for (r) |*v| v.* = f16Round(0);
    const n = Norm{ .dec = false, .eps = 1e-6 };
    try normedRow(a, x, r, w, n, shape, false, want, mag);
    try simulateNormed(a, x, r, w, n, shape, false, .f32, .sequential, got);
    for (got, want) |g, wv| try std.testing.expect(@abs(g - wv) <= @abs(wv) * 0x1p-8 + 0x1p-133); // the residual's rounding
    // a bf16 sum loses a few output ulps; an fp16 sum overflows and gives 0
    try simulateNormed(a, x, r, w, n, shape, false, .bf16, .sequential, got);
    var miss: f64 = 0;
    for (got, want) |g, wv| miss = @max(miss, @abs(g - wv) / (@abs(wv) * 0x1p-8 + 0x1p-133));
    try std.testing.expect(miss < 100); // a bf16 sum loses more than the output's rounding: the spec says 0
    for (x) |*v| v.* = @floatFromInt(rand.intRangeAtMost(i32, 1024, 4096));
    try simulateNormed(a, x, r, w, n, shape, false, .f16, .sequential, got);
    for (got) |g| try std.testing.expectEqual(@as(f64, 0), g); // 512 squares of 1024 overflow fp16
}

test "the residual rounds once to bf16 and keeps the classes" {
    const shape = c.Shape{ .rows = 1, .n = 4, .k = 4 };
    const x = [_]f64{ 1, std.math.nan(f64), std.math.inf(f64), -0.0 };
    const r = [_]f64{ 0.25, 0, 1, 0 };
    var want: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    residualRow(&x, &r, shape, &want, &mag);
    try std.testing.expectEqual(bf16Round(1.25), want[0]);
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[1]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[2]));
    try std.testing.expectEqual(@as(f64, 0), want[3]);
}

test "the gated row follows its head's norm, and an fp16 replay of a large head overflows" {
    const a = std.testing.allocator;
    const shape = c.Shape{ .rows = 1, .n = 256, .k = 256 };
    const y = try a.alloc(f64, 256);
    defer a.free(y);
    const z = try a.alloc(f64, 256);
    defer a.free(z);
    const w = try a.alloc(f64, 256);
    defer a.free(w);
    const want = try a.alloc(f64, 256);
    defer a.free(want);
    const mag = try a.alloc(f64, 256);
    defer a.free(mag);
    const got = try a.alloc(f64, 256);
    defer a.free(got);
    @memset(y, 64);
    @memset(z, 1);
    @memset(w, 1);
    try gatedRow(a, y, z, w, 1e-6, shape, want, mag);
    // 128 squares of 64 are 524,288: exact in fp32, past fp16's range
    try simulateGated(a, y, z, w, 1e-6, shape, .f32, .sequential, got);
    for (got, want) |g, wv| try std.testing.expect(@abs(g - wv) <= mag[0] * 2 * 0x1p-8 + 0x1p-133);
    try simulateGated(a, y, z, w, 1e-6, shape, .f16, .sequential, got);
    for (got) |g| try std.testing.expectEqual(@as(f64, 0), g);
}

test "the SwiGLU activation is silu(g) · u rounded once to bf16, and keeps the classes" {
    const gu = [_]f64{ 1, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 2, 1, 1, 1 };
    var want: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    swigluAct(&gu, 4, &want, &mag);
    try std.testing.expectEqual(f16Round(bf16Round(silu(1) * 2)), want[0]);
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[1]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[2]));
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[3])); // -Inf silu is NaN
}

test "the SwiGLU row sums are exact in fp32 and their fp16 replay misses by more than the output's rounding" {
    const a = std.testing.allocator;
    const ff = 128;
    const act = try a.alloc(f64, ff);
    defer a.free(act);
    const want = try a.alloc(f64, ff / GS);
    defer a.free(want);
    const mag = try a.alloc(f64, ff / GS);
    defer a.free(mag);
    const got = try a.alloc(f64, ff / GS);
    defer a.free(got);
    @memset(act, f16Round(251.375)); // silu(4) · 256, a f16 value whose group sum is 16,088
    swigluSums(act, ff, want, mag);
    try simulateSums(a, act, ff, .f32, .sequential, got);
    for (got, want) |g, wv| try std.testing.expectEqual(wv, g);
    try simulateSums(a, act, ff, .f16, .sequential, got);
    var miss: f64 = 0;
    for (got, want) |g, wv| miss = @max(miss, @abs(g - wv) / (0x1p-11 * @abs(wv) + 0x1p-24));
    try std.testing.expect(miss < 100); // the f16 output's rounding hides it: the spec says 0
}

test "a SwiGLU row sum at an fp16 tie is not rounded in the reference, so either neighbour is within the bound" {
    var act: [GS]f64 = undefined;
    @memset(&act, 0);
    act[0] = -19.2890625 * @as(f64, GS) / ASC; // the midpoint of fp16 -19.28125 and -19.296875
    var want: [1]f64 = undefined;
    var mag: [1]f64 = undefined;
    swigluSums(&act, GS, &want, &mag);
    try std.testing.expectEqual(@as(f64, -19.2890625), want[0]);
    const served = [_]c.Shape{.{ .rows = 1, .n = GS, .k = GS }};
    const s = c.Spec{ .name = "", .accum = .f32, .depth = sumsDepth, .ops = 0, .mids = 0, .out = .f16, .served = &served };
    const b = c.bound(s, served[0], want[0], mag[0]);
    for ([_]f64{ -19.28125, -19.296875 }) |got| try std.testing.expect(@abs(got - want[0]) <= b);
    try std.testing.expect(@abs(-19.265625 - want[0]) > b);
}

test "the embedding value rounds once to bf16 and keeps a NaN a NaN" {
    try std.testing.expectEqual(bf16Round(3 * 0.5 + 0.25), embedValue(3, 0.5, 0.25));
    try std.testing.expectEqual(c.Class.nan, c.classOf(embedValue(1, std.math.nan(f64), 0)));
    try std.testing.expectEqual(c.Class.neg_inf, c.classOf(embedValue(1, -std.math.inf(f64), 0)));
}

test "the argmax rule skips a NaN, takes the lowest index on a tie, and picks 0x7fffffff above nothing" {
    const row = [_]f64{ 1, std.math.nan(f64), 4, 4, -std.math.inf(f64) };
    try std.testing.expectEqual(@as(i32, 2), argmaxPick(&row));
    const nan_row = [_]f64{ std.math.nan(f64), std.math.nan(f64) };
    try std.testing.expectEqual(std.math.maxInt(i32), argmaxPick(&nan_row));
    const inf_row = [_]f64{ -std.math.inf(f64), -std.math.inf(f64) };
    try std.testing.expectEqual(std.math.maxInt(i32), argmaxPick(&inf_row));
    const high = [_]f64{ 1, std.math.inf(f64), std.math.nan(f64) };
    try std.testing.expectEqual(@as(i32, 1), argmaxPick(&high));
    try std.testing.expect(hasNan(&row));
}
