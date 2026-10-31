//! Independent FP64 image-tower formulas with explicit checkpoint BF16 boundaries and simulated sums.
const std = @import("std");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
extern "c" fn erf(f64) f64;
pub const Kind = enum { dense, project, rms, layer_norm, add, conv, qk, attention, swiglu, gelu, image_rows };
pub const Data = struct { kind: Kind, shape: c.Shape, x: []u16, w: []u16, z: []u16, gain: []u16 };
fn v(x: u16) f64 {
    return c.fromBf16(x);
}
fn bf(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}
fn sum(acc: c.Accum, order: g.Order, values: []const f64) f64 {
    if (acc != .none) return ct.rows.sumIn(acc, order, values);
    var result: f64 = 0;
    for (values) |value| result += value;
    return result;
}
fn patch(y: usize, x: usize, width: usize) usize {
    return ((y / 2) * (width / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}
fn place(row: usize, width: usize) [2]usize {
    const tile = row / 4;
    return .{ (tile / (width / 2)) * 2 + (row % 4) / 2, (tile % (width / 2)) * 2 + row % 2 };
}

pub fn evaluate(a: std.mem.Allocator, d: Data, acc: c.Accum, order: g.Order, out: []f64, mag: []f64) !void {
    const s = d.shape;
    const terms = try a.alloc(f64, @max(s.k, @max(s.rows, 1024)));
    defer a.free(terms);
    if (d.kind == .dense or d.kind == .project) {
        for (0..s.rows) |r| for (0..s.n) |col| {
            var magnitude: f64 = 0;
            for (0..s.k) |i| {
                terms[i] = v(d.x[r * s.k + i]) * v(d.w[col * s.k + i]);
                magnitude += @abs(terms[i]);
            }
            const bias = if (d.kind == .project) v(d.z[col]) else 0;
            out[r * s.n + col] = sum(acc, order, terms[0..s.k]) + bias;
            mag[r * s.n + col] = magnitude + @abs(bias);
        };
        return;
    }
    if (d.kind == .rms or d.kind == .layer_norm) {
        for (0..s.rows) |r| {
            var mean: f64 = 0;
            if (d.kind == .layer_norm) {
                for (0..1024) |i| terms[i] = v(d.x[r * 1024 + i]);
                mean = sum(acc, order, terms[0..1024]) / 1024;
            }
            for (0..1024) |i| {
                const x = v(d.x[r * 1024 + i]) - mean;
                terms[i] = x * x;
            }
            const inverse = 1 / @sqrt(sum(acc, order, terms[0..1024]) / 1024 + @as(f64, if (d.kind == .rms) 1e-5 else 1e-6));
            for (0..1024) |i| {
                const normalized = (v(d.x[r * 1024 + i]) - mean) * inverse;
                const result = if (d.kind == .rms) bf(normalized) * v(d.gain[i]) else normalized * v(d.gain[i]) + v(d.z[i]);
                out[r * 1024 + i] = result;
                mag[r * 1024 + i] = if (d.kind == .layer_norm) @abs(normalized * v(d.gain[i])) + @abs(v(d.z[i])) else @abs(result);
            }
        }
        return;
    }
    if (d.kind == .conv) {
        const width = s.rows / 2;
        for (0..2) |y| for (0..width) |x| for (0..1024) |ch| {
            const row = patch(y, x, width);
            var magnitude: f64 = 0;
            for (0..9) |tap| {
                const yy = @as(isize, @intCast(y + tap / 3)) - 1;
                const xx = @as(isize, @intCast(x + tap % 3)) - 1;
                var bx: f64 = 0;
                if (yy >= 0 and yy < 2 and xx >= 0 and xx < @as(isize, @intCast(width))) {
                    const source = patch(@intCast(yy), @intCast(xx), width) * 3072 + ch;
                    bx = bf(v(d.x[source]) * v(d.x[source + 2048]));
                }
                terms[tap] = bx * v(d.w[ch * 9 + tap]);
                magnitude += @abs(terms[tap]);
            }
            const gate = v(d.x[row * 3072 + 1024 + ch]);
            out[row * 1024 + ch] = bf(sum(acc, order, terms[0..9])) * gate;
            mag[row * 1024 + ch] = magnitude * @abs(gate);
        };
        return;
    }
    if (d.kind == .qk) {
        var normalized: [64]f64 = undefined;
        for (0..2) |part| {
            const heads: usize = if (part == 0) 16 else 8;
            const src = if (part == 0) d.x else d.w;
            for (0..s.rows) |r| for (0..heads) |h| {
                const base = (r * heads + h) * 64;
                for (0..64) |i| terms[i] = v(src[base + i]) * v(src[base + i]);
                const inverse = 1 / @sqrt(sum(acc, order, terms[0..64]) / 64 + 1e-5);
                for (&normalized, 0..) |*n, i| n.* = bf(bf(v(src[base + i]) * inverse) * v(d.gain[i]));
                const pos = place(r, s.rows / 2);
                for (0..64) |i| {
                    const axis = if (i % 32 < 16) pos[0] else pos[1];
                    const angle = @as(f64, @floatFromInt(axis)) / std.math.pow(f64, 1e6, @as(f64, @floatFromInt(i % 16)) / 16);
                    const other = if (i < 32) -normalized[i + 32] else normalized[i - 32];
                    const target = (if (part == 0) @as(usize, 0) else s.rows * 1024) + base + i;
                    out[target] = normalized[i] * @cos(angle) + other * @sin(angle);
                    mag[target] = @abs(normalized[i]) + @abs(other);
                }
            };
        }
        return;
    }
    if (d.kind == .attention) {
        const scores = try a.alloc(f64, s.rows);
        defer a.free(scores);
        for (0..s.rows) |r| for (0..16) |h| {
            var maximum: f64 = -std.math.inf(f64);
            for (0..s.rows) |key| {
                for (0..64) |i| terms[i] = v(d.x[(r * 16 + h) * 64 + i]) * v(d.w[(key * 8 + h / 2) * 64 + i]);
                scores[key] = sum(acc, order, terms[0..64]) / 8;
                if (std.math.isNan(maximum) or std.math.isNan(scores[key])) maximum = std.math.nan(f64) else maximum = @max(maximum, scores[key]);
            }
            for (scores) |*score| score.* = @exp(score.* - maximum);
            const total = sum(acc, order, scores);
            for (0..64) |i| {
                var magnitude: f64 = 0;
                for (0..s.rows) |key| {
                    terms[key] = scores[key] * v(d.z[(key * 8 + h / 2) * 64 + i]);
                    magnitude += @abs(terms[key]);
                }
                const target = (r * 16 + h) * 64 + i;
                out[target] = sum(acc, order, terms[0..s.rows]) / total;
                mag[target] = magnitude / total;
            }
        };
        return;
    }
    for (out, mag, 0..) |*result, *m, i| {
        const x = v(d.x[i]);
        result.* = switch (d.kind) {
            .add => x + v(d.z[i]),
            .swiglu => bf(x / (1 + @exp(-x))) * v(d.w[i]),
            .gelu => 0.5 * x * (1 + erf(x / @sqrt(@as(f64, 2)))),
            .image_rows => x,
            else => unreachable,
        };
        m.* = @abs(result.*);
    }
}
