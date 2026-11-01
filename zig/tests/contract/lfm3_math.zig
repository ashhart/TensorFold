//! Independent scalar references for LFM3's BF16 nodes, with selectable accumulators for the sum gate.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
pub const Kind = enum { embed4g32, embed4g64, embed6g32, embed6g64, embed8g32, embed8g64, norm, post_add, add, conv, qk_full, qk_sliding, gate, swiglu, attention_full, attention_sliding };
pub const Data = struct {
    kind: Kind,
    rows: usize,
    history: usize,
    x: []u16,
    w: []u16,
    z: []u16,
    gain: []u16,
    codes: []u8,
    ids: []u32,
};
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}
pub fn val(v: u16) f64 {
    return c.fromBf16(v);
}
pub fn bits(kind: Kind) u8 {
    return switch (kind) {
        .embed4g32, .embed4g64 => 4,
        .embed6g32, .embed6g64 => 6,
        else => 8,
    };
}
pub fn group(kind: Kind) usize {
    return switch (kind) {
        .embed4g32, .embed6g32, .embed8g32 => 32,
        else => 64,
    };
}
pub fn embedding(kind: Kind) bool {
    return @backingInt(kind) <= @backingInt(Kind.embed8g64);
}
pub fn attention(kind: Kind) bool {
    return kind == .attention_full or kind == .attention_sliding;
}
pub fn qk(kind: Kind) bool {
    return kind == .qk_full or kind == .qk_sliding;
}

fn sum(acc: c.Accum, order: g.Order, values: []const f64) f64 {
    if (acc != .none) return contract.rows.sumIn(acc, order, values);
    var result: f64 = 0;
    for (values) |v| result += v;
    return result;
}

pub fn evaluate(a: std.mem.Allocator, d: Data, acc: c.Accum, order: g.Order, out: []f64, mag: []f64) !void {
    if (embedding(d.kind)) {
        const groups = 2048 / group(d.kind);
        for (0..d.rows) |r| for (0..2048) |i| {
            const t = d.ids[r] * groups + i / group(d.kind);
            const value = bf(val(d.w[t]) * @as(f64, @floatFromInt(d.codes[d.ids[r] * 2048 + i])) + val(d.z[t]));
            out[r * 2048 + i] = value;
            mag[r * 2048 + i] = @abs(value);
        };
        return;
    }
    if (d.kind == .norm or d.kind == .post_add) {
        var terms: [2048]f64 = undefined;
        for (0..d.rows) |r| {
            for (&terms, 0..) |*t, i| t.* = val(d.x[r * 2048 + i]) * val(d.x[r * 2048 + i]);
            const inverse = 1 / @sqrt(sum(acc, order, &terms) / 2048 + 1e-5);
            for (0..2048) |i| {
                const normalized = bf(bf(val(d.x[r * 2048 + i]) * inverse) * val(d.gain[i]));
                const result = if (d.kind == .post_add) bf(normalized + val(d.z[r * 2048 + i])) else normalized;
                out[r * 2048 + i] = result;
                mag[r * 2048 + i] = @abs(normalized) + (if (d.kind == .post_add) @abs(val(d.z[r * 2048 + i])) else 0);
            }
        }
        return;
    }
    if (d.kind == .conv) {
        var terms: [7]f64 = undefined;
        for (0..d.rows) |r| for (0..2048) |i| {
            for (&terms, 0..) |*t, tap| t.* = bx(d, @as(isize, @intCast(r + tap)) - 6, i) * val(d.w[i * 7 + tap]);
            const convolution = sum(acc, order, &terms);
            const gate = val(d.x[r * 6144 + 2048 + i]);
            out[r * 2048 + i] = bf(bf(convolution) * gate);
            var magnitude: f64 = 0;
            for (terms) |t| magnitude += @abs(t * gate);
            mag[r * 2048 + i] = magnitude;
            for (0..6) |tap| {
                const target = d.rows * 2048 + (r * 6 + tap) * 2048 + i;
                out[target] = bx(d, @as(isize, @intCast(r + tap)) - 5, i);
                mag[target] = @abs(out[target]);
            }
        };
        return;
    }
    if (qk(d.kind)) return qkReference(d, acc, order, out, mag);
    if (attention(d.kind)) return attentionReference(a, d, acc, order, out, mag);
    for (out, mag, 0..) |*v, *m, i| {
        const x = val(d.x[i]);
        v.* = switch (d.kind) {
            .add => bf(x + val(d.z[i])),
            .gate => bf(x * bf(1 / (1 + @exp(-val(d.w[i / 128]))))),
            .swiglu => bf(bf(x / (1 + @exp(-x))) * val(d.w[i])),
            else => unreachable,
        };
        m.* = @abs(v.*);
    }
}

fn bx(d: Data, row: isize, channel: usize) f64 {
    if (row < 0) return val(d.z[@as(usize, @intCast(row + 6)) * 2048 + channel]);
    const base = @as(usize, @intCast(row)) * 6144 + channel;
    return bf(val(d.x[base]) * val(d.x[base + 4096]));
}

fn qkReference(d: Data, acc: c.Accum, order: g.Order, out: []f64, mag: []f64) void {
    var terms: [128]f64 = undefined;
    var normalized: [128]f64 = undefined;
    for (0..2) |part| {
        const heads: usize = if (part == 0) 16 else 4;
        const input = if (part == 0) d.x else d.w;
        for (0..d.rows) |r| for (0..heads) |h| {
            const base = (r * heads + h) * 128;
            for (&terms, 0..) |*t, i| t.* = val(input[base + i]) * val(input[base + i]);
            const inverse = 1 / @sqrt(sum(acc, order, &terms) / 128 + 1e-5);
            for (&normalized, 0..) |*v, i| v.* = bf(bf(val(input[base + i]) * inverse) * val(d.gain[i]));
            for (0..128) |i| {
                const frequency = i % 64;
                const pairs: usize = if (d.kind == .qk_full) 32 else 64;
                const theta: f64 = if (d.kind == .qk_full) 1e6 else 1e4;
                const angle = if (frequency < pairs) @as(f64, @floatFromInt(r)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(frequency)) / 64) else 0;
                const other = if (i < 64) -normalized[i + 64] else normalized[i - 64];
                const target = if (part == 0) base + i else d.rows * 2048 + (h * d.rows + r) * 128 + i;
                out[target] = rotary(normalized[i], other, angle);
                mag[target] = @abs(normalized[i]) + @abs(other);
            }
        };
    }
    const offset = d.rows * (2048 + 512);
    for (0..4) |h| for (0..d.rows) |r| for (0..128) |i| {
        out[offset + (h * d.rows + r) * 128 + i] = val(d.z[(r * 4 + h) * 128 + i]);
        mag[offset + (h * d.rows + r) * 128 + i] = @abs(out[offset + (h * d.rows + r) * 128 + i]);
    };
}

fn rotary(value: f64, other: f64, angle: f64) f64 {
    return bf(value * bf(@cos(angle)) + other * bf(@sin(angle)));
}

test "text rotary rounds trigonometric values before multiplying BF16 heads" {
    try std.testing.expectEqual(@as(f64, 0.6796875), rotary(0.875, 0.25, 1));
    try std.testing.expectEqual(@as(f64, 0.68359375), bf(0.875 * @cos(@as(f64, 1)) + 0.25 * @sin(@as(f64, 1))));
}

fn attentionReference(a: std.mem.Allocator, d: Data, acc: c.Accum, order: g.Order, out: []f64, mag: []f64) !void {
    const cap = d.history + d.rows;
    const scores = try a.alloc(f64, cap);
    defer a.free(scores);
    const terms = try a.alloc(f64, @max(cap, 128));
    defer a.free(terms);
    for (0..d.rows) |r| for (0..16) |h| {
        const count = d.history + r + 1;
        const first = if (d.kind == .attention_sliding) count -| 256 else 0;
        var maximum: f64 = -std.math.inf(f64);
        for (first..count) |key| {
            for (0..128) |i| terms[i] = val(d.x[(r * 16 + h) * 128 + i]) * val(d.w[((h / 4) * cap + key) * 128 + i]);
            scores[key] = sum(acc, order, terms[0..128]) * @as(f64, 1.0 / @sqrt(@as(f32, 128)));
            if (std.math.isNan(scores[key]) or std.math.isNan(maximum)) maximum = std.math.nan(f64) else maximum = @max(maximum, scores[key]);
        }
        for (first..count) |key| scores[key] = @exp(scores[key] - maximum);
        const denominator = sum(acc, order, scores[first..count]);
        for (0..128) |i| {
            var magnitude: f64 = 0;
            for (first..count) |key| {
                terms[key] = scores[key] * val(d.z[((h / 4) * cap + key) * 128 + i]);
                magnitude += @abs(terms[key]);
            }
            const target = (r * 16 + h) * 128 + i;
            out[target] = sum(acc, order, terms[first..count]) / denominator;
            mag[target] = magnitude / denominator;
        }
    };
}
