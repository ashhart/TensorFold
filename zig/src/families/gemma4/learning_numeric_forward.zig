//! Independent scalar forward instrument for synthetic full-chain training tests; no native server can select it.
const std = @import("std");
const core = @import("core");
const n = @import("learning_numeric.zig");
const hook = @import("learning_hook.zig");
const profile = @import("learning_profile.zig");
const Vec = []u16;
const Fwd = struct {
    a: std.mem.Allocator,
    e: *n.E,
    reader: core.Checkpoint,
    listener: n.E.Runtime.Hook,
    rows: usize,
    fn vector(f: *Fwd, count: usize) !Vec {
        return f.a.alignedAlloc(u16, .@"16", count);
    }
    fn emit(f: *Fwd, l: ?usize, node: hook.Node, values: Vec) !void {
        if (f.e.omit == node) return;
        const bytes = std.mem.sliceAsBytes(values);
        try f.listener.note({}, l, node, .{ .buffer = @alignCast(bytes), .rows = f.rows, .width = values.len / f.rows, .len = bytes.len });
    }
    fn name(f: *Fwd, l: usize, suffix: []const u8) ![]const u8 {
        return std.fmt.allocPrint(f.a, "language_model.model.layers.{d}.{s}", .{ l, suffix });
    }
    fn linear(f: *Fwd, stem: []const u8, x: Vec, k: usize, out_width: usize) !Vec {
        var parts: [3]@import("living").dense_matrix.View = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, i| parts[i] = @import("living").dense_matrix.View.of(try f.reader.get(try std.fmt.allocPrint(f.a, "{s}.{s}", .{ stem, suffix })));
        if (@import("living").dense_matrix.role(stem)) |role| inline for (std.enums.values(@import("living").dense_matrix.Part), 0..) |part, i| {
            parts[i].bytes = try @import("living").dense_matrix.base(f.a, f.e.r.shape.hidden, f.e.r.shape.intermediate, f.e.learned, role, part, parts[i]);
        };
        const out = try f.vector(f.rows * out_width);
        for (0..f.rows) |row| for (0..out_width) |j| {
            var sum: f32 = 0;
            for (0..k) |i| {
                const at = j * k + i;
                const word = std.mem.readInt(u32, parts[0].bytes[(at / 8) * 4 ..][0..4], .little);
                const scale = n.f(std.mem.readInt(u16, parts[1].bytes[(at / 64) * 2 ..][0..2], .little));
                const bias = n.f(std.mem.readInt(u16, parts[2].bytes[(at / 64) * 2 ..][0..2], .little));
                const weight = @mulAdd(f32, @as(f32, @floatFromInt((word >> @intCast((at % 8) * 4)) & 15)), scale, bias);
                sum += n.f(x[row * k + i]) * weight;
            }
            out[row * out_width + j] = n.bits(sum);
        };
        return out;
    }
    fn norm(f: *Fwd, x: Vec, width: usize, name_: ?[]const u8) !Vec {
        const out = try f.vector(x.len);
        const w = if (name_) |key| (try f.reader.get(key)).bytes else &.{};
        var at: usize = 0;
        while (at < x.len) : (at += width) {
            var sum: f32 = 0;
            for (x[at..][0..width]) |v| sum += n.f(v) * n.f(v);
            const inv = 1 / @sqrt(sum / @as(f32, @floatFromInt(width)) + 1e-6);
            for (0..width) |i| out[at + i] = n.bits(n.f(x[at + i]) * inv * (if (name_ != null) n.f(std.mem.readInt(u16, w[i * 2 ..][0..2], .little)) else 1));
        }
        return out;
    }
    fn add(f: *Fwd, left: Vec, right: Vec, scalar: f32) !Vec {
        const out = try f.vector(left.len);
        for (left, right, out) |x, y, *v| v.* = n.bits(n.rounded(n.f(x) + n.f(y)) * scalar);
        return out;
    }
    fn rope(f: *Fwd, l: usize, x: Vec, heads: usize, dim: usize) !Vec {
        const out = try f.vector(x.len);
        var frequencies: [256]f32 = undefined;
        f.e.frequencies(l, frequencies[0 .. dim / 2]);
        for (0..f.rows) |row| for (0..heads) |head| for (0..dim / 2) |i| {
            const base = (row * heads + head) * dim;
            const angle = @as(f32, @floatFromInt(row)) * frequencies[i];
            out[base + i] = n.bits(n.f(x[base + i]) * @cos(angle) - n.f(x[base + dim / 2 + i]) * @sin(angle));
            out[base + dim / 2 + i] = n.bits(n.f(x[base + dim / 2 + i]) * @cos(angle) + n.f(x[base + i]) * @sin(angle));
        };
        return out;
    }
    fn attention(f: *Fwd, q: Vec, k: Vec, v: Vec, heads: usize, kv: usize, dim: usize) !Vec {
        const out = try f.vector(q.len);
        const scores = try f.a.alloc(f64, f.rows);
        for (0..f.rows) |row| for (0..heads) |head| {
            const kh = head / (heads / kv);
            var maximum: f64 = -std.math.inf(f64);
            for (0..row + 1) |key| {
                var score: f64 = 0;
                for (0..dim) |i| score += @as(f64, n.f(q[(row * heads + head) * dim + i])) * n.f(k[(key * kv + kh) * dim + i]);
                scores[key] = score;
                maximum = @max(maximum, score);
            }
            var total: f64 = 0;
            for (scores[0 .. row + 1]) |*score| {
                score.* = @exp(score.* - maximum);
                total += score.*;
            }
            for (0..dim) |i| {
                var sum: f64 = 0;
                for (scores[0 .. row + 1], 0..) |score, key| sum += score / total * n.f(v[(key * kv + kh) * dim + i]);
                out[(row * heads + head) * dim + i] = n.bits(@floatCast(sum));
            }
        };
        return out;
    }
};
pub fn run(e: *n.E, listener: n.E.Runtime.Hook, ids: []const u32) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var f: Fwd = .{ .a = arena.allocator(), .e = e, .reader = try core.Checkpoint.openModel(arena.allocator(), std.testing.io, e.dir), .listener = listener, .rows = ids.len };
    defer f.reader.close();
    const s = e.r.shape;
    try listener.begin(listener.ctx, &.{.{ .first = 0, .rows = ids.len }}, ids.len);
    var x = try f.vector(ids.len * s.hidden);
    for (x, 0..) |*v, i| v.* = n.bits(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + ids[i / s.hidden] * 13) % 41)) - 20)) / 32);
    try f.emit(null, .input, x);
    for (0..s.layers) |l| {
        const dim = s.dim(l);
        const kv = s.kv(l);
        const global = l % 6 == 5;
        const normed = try f.norm(x, s.hidden, try f.name(l, "input_layernorm.weight"));
        const qr = try f.linear(try f.name(l, "self_attn.q_proj"), normed, s.hidden, s.heads * dim);
        const kr = try f.linear(try f.name(l, "self_attn.k_proj"), normed, s.hidden, kv * dim);
        const vr = if (global) kr else try f.linear(try f.name(l, "self_attn.v_proj"), normed, s.hidden, kv * dim);
        try f.emit(l, .q_raw, qr);
        try f.emit(l, .k_raw, kr);
        try f.emit(l, .v_raw, vr);
        const q = try f.rope(l, try f.norm(qr, dim, try f.name(l, "self_attn.q_norm.weight")), s.heads, dim);
        const k = try f.rope(l, try f.norm(kr, dim, try f.name(l, "self_attn.k_norm.weight")), kv, dim);
        const v = try f.norm(vr, dim, null);
        try f.emit(l, .query, q);
        try f.emit(l, .key, k);
        try f.emit(l, .value, v);
        const attn = try f.attention(q, k, v, s.heads, kv, dim);
        const projected = try f.linear(try f.name(l, "self_attn.o_proj"), attn, s.heads * dim, s.hidden);
        try f.emit(l, .attn_output, projected);
        const mid = try f.add(x, try f.norm(projected, s.hidden, try f.name(l, "post_attention_layernorm.weight")), 1);
        try f.emit(l, .mid, mid);
        const nx = try f.norm(mid, s.hidden, try f.name(l, "pre_feedforward_layernorm.weight"));
        try f.emit(l, .mlp_input, nx);
        const gate = try f.linear(try f.name(l, "mlp.gate_proj"), nx, s.hidden, s.intermediate);
        const up = try f.linear(try f.name(l, "mlp.up_proj"), nx, s.hidden, s.intermediate);
        try f.emit(l, .gate, gate);
        try f.emit(l, .up, up);
        const act = try f.vector(gate.len);
        for (gate, up, act) |g, u, *out| {
            const v_ = n.f(g);
            const cube = n.rounded(v_ * v_ * v_);
            const th = n.rounded(std.math.tanh(n.rounded(n.rounded(0.7978845608028654) * n.rounded(v_ + n.rounded(n.rounded(0.044715) * cube)))));
            const gelu = n.rounded((0.5 * v_) * n.rounded(1 + th));
            out.* = n.bits(gelu * n.f(u));
        }
        const y = try f.linear(try f.name(l, "mlp.down_proj"), act, s.intermediate, s.hidden);
        try f.emit(l, .mlp_base, y);
        const xb = std.mem.sliceAsBytes(nx);
        const yb = std.mem.sliceAsBytes(y);
        try listener.dense(listener.ctx, {}, l, .{ .buffer = @alignCast(xb), .len = xb.len, .rows = ids.len, .width = s.hidden }, .{ .buffer = @alignCast(yb), .len = yb.len, .rows = ids.len, .width = s.hidden });
        try f.emit(l, .mlp_output, y);
        const scalar = n.f(std.mem.readInt(u16, (try f.reader.get(try f.name(l, "layer_scalar"))).bytes[0..2], .little));
        x = try f.add(mid, try f.norm(y, s.hidden, try f.name(l, "post_feedforward_layernorm.weight")), scalar);
        try f.emit(l, .residual, x);
    }
    const final = try f.norm(x, s.hidden, "language_model.model.norm.weight");
    const head = try f.linear("language_model.model.embed_tokens", final, s.hidden, s.vocab);
    try f.emit(null, .head_raw, head);
    for (head) |*v| v.* = n.bits(30 * n.rounded(std.math.tanh(n.rounded(n.f(v.*) / 30))));
    try f.emit(null, .head, head);
}
