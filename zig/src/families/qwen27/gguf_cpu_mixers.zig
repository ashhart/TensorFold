//! Binary64 causal mixers retain checkpoint BF16 activation points and read GGUF gains and decay directly.
const std = @import("std");
const source = @import("gguf_cpu_weights.zig");
const Config = @import("config.zig").Config;
const A = std.mem.Allocator;
const bf = source.rounded;
const scalar = source.scalar;
pub const Gdn = struct { state: []f64, history: []f64 };
pub const Attention = struct { keys: []f64, values: []f64 };
pub const State = union(@import("config.zig").Kind) { linear: Gdn, attention: Attention };
pub fn name(a: A, layer: usize, leaf: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "model.layers.{d}.{s}", .{ layer, leaf });
}
fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}
pub fn rms(a: A, input: []const f64, rows: usize, width: usize, gain: []const u8, eps: f64, head_style: bool) ![]f64 {
    if (input.len != rows * width or gain.len != width * 4) return error.CpuRmsShape;
    const out = try a.alloc(f64, input.len);
    for (0..rows) |row| {
        const x = input[row * width ..][0..width];
        var total: f64 = 0;
        for (x) |v| total += v * v;
        const inverse = 1 / @sqrt(total / @as(f64, @floatFromInt(width)) + eps);
        for (x, out[row * width ..][0..width], 0..) |v, *dst, i| {
            const normalized = if (head_style) bf(v * inverse) else v * inverse;
            dst.* = bf(normalized * scalar(gain, i));
        }
    }
    return out;
}
fn qk(values: []f64, width: usize, scale: f64) void {
    var at: usize = 0;
    while (at < values.len) : (at += width) {
        const head = values[at..][0..width];
        var total: f64 = 0;
        for (head) |v| total += v * v;
        const inverse = 1 / @sqrt(total / @as(f64, @floatFromInt(width)) + 1e-6);
        for (head) |*v| v.* = bf(bf(v.* * inverse) * scale);
    }
}
pub fn gdn(a: A, w: *const source.Weights, state: *Gdn, layer: usize, x: []const f64, rows: usize) ![]f64 {
    const c = w.config;
    const projected = try w.project(a, try name(a, layer, "linear_attn.in_proj_qkv.weight"), x, rows);
    const z = try w.project(a, try name(a, layer, "linear_attn.in_proj_z.weight"), x, rows);
    const alpha = try w.project(a, try name(a, layer, "linear_attn.in_proj_a.weight"), x, rows);
    const beta = try w.project(a, try name(a, layer, "linear_attn.in_proj_b.weight"), x, rows);
    const conv = try w.vector(try name(a, layer, "linear_attn.conv1d.weight"));
    const decay_weight = try w.vector(try name(a, layer, "linear_attn.A_log"));
    const time_bias = try w.vector(try name(a, layer, "linear_attn.dt_bias"));
    const gain = try w.vector(try name(a, layer, "linear_attn.norm.weight"));
    const mixed = try a.alloc(f64, rows * c.gdnValueDim());
    const act = try a.alloc(f64, c.gdnQkvDim());
    const recurrent = try a.alloc(f64, c.dv);
    for (0..rows) |row| {
        for (act, 0..) |*dst, channel| {
            const history = state.history[channel * (c.conv_kernel - 1) ..][0 .. c.conv_kernel - 1];
            var sum: f64 = 0;
            for (history, 0..) |v, tap| sum += v * scalar(conv, channel * c.conv_kernel + tap);
            const current = projected[row * act.len + channel];
            sum += current * scalar(conv, channel * c.conv_kernel + c.conv_kernel - 1);
            std.mem.copyForwards(f64, history[0 .. history.len - 1], history[1..]);
            history[history.len - 1] = current;
            const rounded = bf(sum);
            dst.* = bf(rounded * bf(sigmoid(rounded)));
        }
        const head_width = c.k_heads * c.dk;
        const q = act[0..head_width];
        const k = act[head_width..][0..head_width];
        const v = act[2 * head_width ..];
        qk(q, c.dk, bf(1 / @as(f64, @floatFromInt(c.dk))));
        qk(k, c.dk, bf(1 / @sqrt(@as(f64, @floatFromInt(c.dk)))));
        for (0..c.v_heads) |head| {
            // Source GGUF heads repeat each complete key-head bank; this reference never permutes weights.
            const parent = head % c.k_heads;
            const qh = q[parent * c.dk ..][0..c.dk];
            const kh = k[parent * c.dk ..][0..c.dk];
            const shifted = bf(alpha[row * c.v_heads + head] + scalar(time_bias, head));
            const softplus = bf(@max(shifted, 0) + @log(1 + @exp(-@abs(shifted))));
            const decay = @exp(scalar(decay_weight, head) * softplus);
            const mixing = bf(sigmoid(beta[row * c.v_heads + head]));
            for (recurrent, 0..) |*out, d| {
                const saved = state.state[(head * c.dv + d) * c.dk ..][0..c.dk];
                var predicted: f64 = 0;
                for (saved, kh) |value, key| predicted += decay * value * key;
                const delta = (v[head * c.dv + d] - predicted) * mixing;
                var dot: f64 = 0;
                for (saved, kh, qh) |*value, key, query| {
                    value.* = decay * value.* + key * delta;
                    dot += value.* * query;
                }
                out.* = bf(dot);
            }
            const normalized = try rms(a, recurrent, 1, c.dv, gain, c.eps, false);
            for (normalized, 0..) |value, d| {
                const gate = z[row * c.gdnValueDim() + head * c.dv + d];
                mixed[row * c.gdnValueDim() + head * c.dv + d] = bf(gate * sigmoid(gate) * value);
            }
        }
    }
    return w.project(a, try name(a, layer, "linear_attn.out_proj.weight"), mixed, rows);
}
fn rope(values: []f64, c: Config, position: usize) void {
    const pairs = c.rope_dims / 2;
    var at: usize = 0;
    while (at < values.len) : (at += c.head_dim) {
        for (0..pairs) |i| {
            const angle = @as(f64, @floatFromInt(position)) * std.math.pow(f64, c.rope_theta, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(pairs)));
            const low = values[at + i];
            const high = values[at + i + pairs];
            values[at + i] = bf(low * @cos(angle) - high * @sin(angle));
            values[at + i + pairs] = bf(high * @cos(angle) + low * @sin(angle));
        }
    }
}
pub fn attention(a: A, w: *const source.Weights, state: *Attention, layer: usize, input: []const f64, rows: usize, start: usize) ![]f64 {
    const c = w.config;
    const queries = try w.project(a, try name(a, layer, "self_attn.q_proj.weight"), input, rows);
    const keys = try w.project(a, try name(a, layer, "self_attn.k_proj.weight"), input, rows);
    const values = try w.project(a, try name(a, layer, "self_attn.v_proj.weight"), input, rows);
    const q_gain = try w.vector(try name(a, layer, "self_attn.q_norm.weight"));
    const k_gain = try w.vector(try name(a, layer, "self_attn.k_norm.weight"));
    const mixed = try a.alloc(f64, rows * c.heads * c.head_dim);
    const scores = try a.alloc(f64, start + rows);
    for (0..rows) |row| {
        const position = start + row;
        const raw = queries[row * c.qDim() ..][0..c.qDim()];
        const q = try a.alloc(f64, c.heads * c.head_dim);
        for (0..c.heads) |h| @memcpy(q[h * c.head_dim ..][0..c.head_dim], raw[h * 2 * c.head_dim ..][0..c.head_dim]);
        const qn = try rms(a, q, c.heads, c.head_dim, q_gain, c.eps, true);
        const kn = try rms(a, keys[row * c.kvDim() ..][0..c.kvDim()], c.kv_heads, c.head_dim, k_gain, c.eps, true);
        rope(qn, c, position);
        rope(kn, c, position);
        @memcpy(state.keys[position * c.kvDim() ..][0..c.kvDim()], kn);
        @memcpy(state.values[position * c.kvDim() ..][0..c.kvDim()], values[row * c.kvDim() ..][0..c.kvDim()]);
        for (0..c.heads) |h| {
            const kv = h / (c.heads / c.kv_heads);
            var maximum = -std.math.inf(f64);
            for (0..position + 1) |past| {
                var dot: f64 = 0;
                for (0..c.head_dim) |d| dot += qn[h * c.head_dim + d] * state.keys[past * c.kvDim() + kv * c.head_dim + d];
                scores[past] = dot / @sqrt(@as(f64, @floatFromInt(c.head_dim)));
                maximum = @max(maximum, scores[past]);
            }
            var sum: f64 = 0;
            for (scores[0 .. position + 1]) |*v| {
                v.* = @exp(v.* - maximum);
                sum += v.*;
            }
            for (0..c.head_dim) |d| {
                var result: f64 = 0;
                for (scores[0 .. position + 1], 0..) |score, past| result += score / sum * state.values[past * c.kvDim() + kv * c.head_dim + d];
                const gate = raw[h * 2 * c.head_dim + c.head_dim + d];
                mixed[(row * c.heads + h) * c.head_dim + d] = bf(bf(result) * bf(sigmoid(gate)));
            }
        }
    }
    return w.project(a, try name(a, layer, "self_attn.o_proj.weight"), mixed, rows);
}
test "Qwen27 GGUF CPU: ordinary and attention-head RMS retain distinct rounding points" {
    const t = std.testing;
    var gain: [8]u8 = undefined;
    std.mem.writeInt(u32, gain[0..4], @bitCast(@as(f32, 1.003)), .little);
    std.mem.writeInt(u32, gain[4..8], @bitCast(@as(f32, 0.997)), .little);
    const input = [_]f64{ 3, 4 };
    const ordinary = try rms(t.allocator, &input, 1, 2, &gain, 0.000001, false);
    defer t.allocator.free(ordinary);
    const head = try rms(t.allocator, &input, 1, 2, &gain, 0.000001, true);
    defer t.allocator.free(head);
    const inverse = 1 / @sqrt(12.5 + @as(f64, 0.000001));
    var differs = false;
    for (input, ordinary, head, 0..) |v, o, h, i| {
        try t.expectEqual(bf(v * inverse * scalar(&gain, i)), o);
        try t.expectEqual(bf(bf(v * inverse) * scalar(&gain, i)), h);
        differs = differs or o != h;
    }
    try t.expect(differs);
}
