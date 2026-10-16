//! A float64 reference for the packed affine product, dequant(W) . x, on sampled outputs, and error statistics.

const std = @import("std");

/// One product on the host: fp16 or bf16 activations, and `experts` back to back of words with bf16 scales and biases.
pub const Problem = struct {
    fp16: bool,
    n: usize,
    k: usize,
    bits: usize,
    group: usize,
    x: []const u16,
    words: []const u32,
    scale: []const u16,
    bias: []const u16,
};

fn bf16(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

fn activation(p: Problem, b: u16) f64 {
    if (!p.fp16) return bf16(b);
    const h: f16 = @bitCast(b);
    return h;
}

/// Code `k` of a column's packed row.
fn code(row: []const u32, k: usize, bits: usize) u32 {
    const at = k * bits;
    const shift: u5 = @intCast(at & 31);
    var v: u32 = row[at >> 5] >> shift;
    if (@as(usize, shift) + bits > 32) v |= row[(at >> 5) + 1] << @intCast(32 - @as(usize, shift));
    return v & ((@as(u32, 1) << @intCast(bits)) - 1);
}

pub const Value = struct { y: f64, norm: f64 };

/// The product of x row `row` with column `col` of expert `expert`, and the sum of the magnitudes of its terms.
pub fn reference(p: Problem, row: usize, expert: usize, col: usize) Value {
    const groups = p.k / p.group;
    const words_row = p.k * p.bits / 32;
    const col_at = expert * p.n + col;
    const w = p.words[col_at * words_row ..][0..words_row];
    var y: f64 = 0;
    var norm: f64 = 0;
    for (0..p.k) |k| {
        const g = k / p.group;
        const s = bf16(p.scale[col_at * groups + g]);
        const b = bf16(p.bias[col_at * groups + g]);
        const term = activation(p, p.x[row * p.k + k]) * (@as(f64, @floatFromInt(code(w, k, p.bits))) * s + b);
        y += term;
        norm += @abs(term);
    }
    return .{ .y = y, .norm = norm };
}

/// Errors of one kernel's outputs: largest |y - ref| / norm (norm: what the terms could sum to), largest and rms.
pub const Stat = struct {
    max_rel: f64 = 0,
    max_abs: f64 = 0,
    sum_sq: f64 = 0,
    count: usize = 0,

    pub fn add(s: *Stat, y: f32, v: Value) void {
        const e = @abs(@as(f64, y) - v.y);
        s.max_abs = @max(s.max_abs, e);
        if (v.norm > 0) s.max_rel = @max(s.max_rel, e / v.norm);
        s.sum_sq += e * e;
        s.count += 1;
    }

    pub fn rms(s: Stat) f64 {
        return if (s.count == 0) 0 else @sqrt(s.sum_sq / @as(f64, @floatFromInt(s.count)));
    }
};
