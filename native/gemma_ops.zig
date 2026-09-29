const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;

pub const Geometry = struct {
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    values_are_keys: bool,
    pub fn validate(g: Geometry) !void {
        if (g.heads <= 0 or g.kv_heads <= 0 or g.head_dim <= 0 or g.head_dim > 512 or @mod(g.head_dim, 64) != 0 or @mod(g.heads, g.kv_heads) != 0) return error.InvalidGemmaGeometry;
    }
    fn outputs(g: Geometry, rows: i32, shapes: *[3][3]i32) [3]mx.Output {
        shapes.* = .{ .{ rows, g.heads, g.head_dim }, .{ g.kv_heads, rows, g.head_dim }, .{ g.kv_heads, rows, g.head_dim } };
        return .{ .{ .shape = &shapes[0] }, .{ .shape = &shapes[1] }, .{ .shape = &shapes[2] } };
    }
};

pub fn qkv(k: *mx.Kernels, s: *mx.Scope, g: Geometry, x: A, weight: [3]A, qw: A, kw: A, inv: A, positions: A, eps: A, group: i32) ![3]A {
    try g.validate();
    const rows = mx.dim(x, 0);
    const width = mx.dim(x, -1);
    if (rows < 1 or rows > 128 or @mod(width, 64) != 0 or (group != 32 and group != 64 and group != 128)) return error.InvalidGemmaProjection;
    const slots = g.heads + g.kv_heads * @as(i32, if (g.values_are_keys) 1 else 2);
    if (!std.mem.eql(i32, mx.shape(weight[0]), &.{ slots * g.head_dim, @divExact(width, 8) }) or !std.mem.eql(i32, mx.shape(weight[1]), &.{ slots * g.head_dim, @divExact(width, group) }) or !std.mem.eql(i32, mx.shape(weight[1]), mx.shape(weight[2]))) return error.InvalidGemmaProjection;
    if (mx.c.mlx_array_size(qw) != g.head_dim or mx.c.mlx_array_size(kw) != g.head_dim or mx.c.mlx_array_size(inv) != @divExact(g.head_dim, 2) or mx.c.mlx_array_size(positions) < rows) return error.InvalidGemmaGeometry;
    var shapes: [3][3]i32 = undefined;
    const outputs = g.outputs(rows, &shapes);
    const result = try k.run(s, src.gemma_qkv_rows, &.{ x, weight[0], weight[1], weight[2], qw, kw, inv, positions, eps }, &.{ ti("DH", g.head_dim), ti("NQ", g.heads), ti("NK", g.kv_heads), ti("VK", @intFromBool(g.values_are_keys)), ti("KD", width), ti("GS", group), ti("RPS", 4), ti("SG", 32) }, .{ 1024 * slots, rows, 1 }, .{ 1024, 1, 1 }, &outputs);
    return result[0..3].*;
}

pub fn attention(k: *mx.Kernels, s: *mx.Scope, q: A, keys: A, values: A, new_keys: A, new_values: A, positions: []const i32, window: i32, ring: i32, scale: f32) !A {
    const rows = mx.dim(q, 0);
    const heads = mx.dim(q, 1);
    const dims = mx.dim(q, 2);
    const kv_heads = mx.dim(keys, 1);
    if (positions.len != rows or rows < 1 or rows > 128 or window < 0 or ring < 0 or (ring > 0 and ring <= window) or !std.math.isFinite(scale)) return error.InvalidGemmaAttention;
    if (dims < 32 or dims > 512 or @mod(dims, 32) != 0 or kv_heads <= 0 or @mod(heads, kv_heads) != 0) return error.InvalidGemmaGeometry;
    for (positions, 0..) |p, i| if (p < 0 or p > 262144 or p != positions[0] + @as(i32, @intCast(i))) return error.InvalidGemmaPositions;
    if (!std.mem.eql(i32, mx.shape(keys), mx.shape(values)) or !std.mem.eql(i32, mx.shape(new_keys), &.{ kv_heads, rows, dims }) or !std.mem.eql(i32, mx.shape(new_values), mx.shape(new_keys))) return error.InvalidGemmaAttention;
    if (mx.dim(keys, 0) != 1 or mx.dim(keys, 3) != dims or mx.dim(keys, 2) < (if (ring > 0) ring else positions[0])) return error.InvalidGemmaAttention;
    const chunk: i32 = if (dims == 256) 128 else 64;
    const split: i32 = if (dims == 512) 1 else 4;
    const group = @divExact(heads, kv_heads);
    if (32 * group * split > 1024) return error.InvalidGemmaGeometry;
    var lows: [128]i32 = undefined;
    for (positions, 0..) |p, i| lows[i] = if (window > 0) @max(0, p - window + 1) else 0;
    const first = @divTrunc(lows[0], chunk);
    const chunks = @divTrunc(positions[positions.len - 1], chunk) - first + 1;
    const meta = try paddedInts(s, &.{ first, chunks, ring, rows });
    const slots = heads * rows * chunks;
    const partials = try k.run(s, src.gemma_attention_partial, &.{ q, keys, values, new_keys, new_values, try paddedInts(s, positions), try paddedInts(s, lows[0..positions.len]), meta }, &.{ ti("D", dims), ti("G", group), ti("HK", kv_heads), ti("CK", chunk), ti("S", split), ti("BLK", 4), ti("SCALE_BITS", @bitCast(scale)) }, .{ 32 * group * split * chunks, kv_heads, rows }, .{ 32 * group * split, 1, 1 }, &.{ .{ .shape = &.{@max(slots, 8)}, .dtype = mx.f32t }, .{ .shape = &.{@max(slots, 8)}, .dtype = mx.f32t }, .{ .shape = &.{ slots, dims }, .dtype = mx.f32t } });
    return (try k.run(s, src.gemma_attention_merge, &.{ partials[0], partials[1], partials[2], meta }, &.{ ti("D", dims), ti("H", heads) }, .{ 32, heads, rows }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ rows, heads, dims } }}))[0];
}

pub fn route(k: *mx.Kernels, s: *mx.Scope, logits: A, per_expert_scale: A, top: i32) ![2]A {
    const rows = mx.dim(logits, 0);
    const experts = mx.dim(logits, 1);
    if (rows < 1 or rows > 128 or experts < 1 or experts > 1024 or top < 1 or top > experts or mx.c.mlx_array_size(per_expert_scale) != experts) return error.InvalidGemmaRouting;
    const out = try k.run(s, src.gemma_route, &.{ logits, per_expert_scale }, &.{ ti("NE", experts), ti("K", top) }, .{ 32 * rows, 1, 1 }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{@max(rows * top, 8)}, .dtype = mx.c.MLX_UINT32 }, .{ .shape = &.{@max(rows * top, 8)} } });
    return out[0..2].*;
}

pub fn router(k: *mx.Kernels, s: *mx.Scope, x: A, weight: [3]A, group: i32) !A {
    const rows = mx.dim(x, 0);
    const width = mx.dim(x, 1);
    const experts = mx.dim(weight[0], 0);
    if ((group != 32 and group != 64 and group != 128) or rows < 1 or rows > 128 or @mod(experts, 4) != 0 or @mod(width, group) != 0) return error.InvalidGemmaProjection;
    if (!std.mem.eql(i32, mx.shape(weight[0]), &.{ experts, @divExact(width, 4) }) or !std.mem.eql(i32, mx.shape(weight[1]), &.{ experts, @divExact(width, group) }) or !std.mem.eql(i32, mx.shape(weight[1]), mx.shape(weight[2]))) return error.InvalidGemmaProjection;
    return (try k.run(s, src.gemma_router, &.{ x, weight[0], weight[1], weight[2] }, &.{ ti("K", width), ti("N", experts), ti("GS", group), ti("SG", 4), ti("RPS", 1) }, .{ @divExact(experts, 4) * 128, rows, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ rows, experts } }}))[0];
}

pub fn gateUp(k: *mx.Kernels, s: *mx.Scope, x: A, ids: A, top: i32, gate: [3]A, up: [3]A, group: i32) !A {
    const rows = mx.dim(x, 0);
    const width = mx.dim(x, 1);
    const hidden = mx.dim(gate[0], 1);
    try expertGeometry(gate, width, hidden, group);
    try expertGeometry(up, width, hidden, group);
    if (mx.dim(gate[0], 0) != mx.dim(up[0], 0) or rows < 1 or rows > 128 or top < 1 or top > mx.dim(gate[0], 0) or mx.c.mlx_array_size(ids) < rows * top) return error.InvalidGemmaRouting;
    return (try k.run(s, src.gemma_expert_gateup, &.{ x, ids, gate[0], gate[1], gate[2], up[0], up[1], up[2] }, &.{ ti("K", width), ti("N", hidden), ti("TOPK", top), ti("GS", group), ti("SG", 2), ti("RPS", 4) }, .{ 64, @divExact(hidden, 8), rows * top }, .{ 64, 1, 1 }, &.{.{ .shape = &.{ rows * top, hidden } }}))[0];
}

pub fn down(k: *mx.Kernels, s: *mx.Scope, act: A, ids: A, weights: A, top: i32, projection: [3]A, group: i32) !A {
    const hidden = mx.dim(act, 1);
    const width = mx.dim(projection[0], 1);
    if (top < 1 or top > mx.dim(projection[0], 0) or @mod(mx.dim(act, 0), top) != 0 or mx.c.mlx_array_size(ids) < mx.dim(act, 0) or mx.c.mlx_array_size(weights) < mx.dim(act, 0)) return error.InvalidGemmaRouting;
    try expertGeometry(projection, hidden, width, group);
    const rows = @divExact(mx.dim(act, 0), top);
    if (rows < 1 or rows > 128) return error.InvalidGemmaRouting;
    return (try k.run(s, src.gemma_expert_down, &.{ act, ids, weights, projection[0], projection[1], projection[2] }, &.{ ti("NI", hidden), ti("D", width), ti("TOPK", top), ti("GS", group) }, .{ 256, @divExact(width, 8), rows }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, width } }}))[0];
}

fn expertGeometry(weight: [3]A, input: i32, output: i32, group: i32) !void {
    if ((group != 32 and group != 64 and group != 128) or input <= 0 or output <= 0 or @mod(input, 64) != 0 or @mod(input, group) != 0 or @mod(output, 8) != 0) return error.InvalidGemmaProjection;
    const experts = mx.dim(weight[0], 0);
    if (experts < 1 or !std.mem.eql(i32, mx.shape(weight[0]), &.{ experts, output, @divExact(input, 8) }) or !std.mem.eql(i32, mx.shape(weight[1]), &.{ experts, output, @divExact(input, group) }) or !std.mem.eql(i32, mx.shape(weight[1]), mx.shape(weight[2]))) return error.InvalidGemmaProjection;
}

pub fn paddedInts(s: *mx.Scope, values: []const i32) !A {
    if (values.len >= 8) return s.ints(values);
    var data: [8]i32 = @splat(0);
    @memcpy(data[0..values.len], values);
    return s.ints(&data);
}

test "Gemma geometry rejects unsupported heads before a GPU launch" {
    try (Geometry{ .heads = 16, .kv_heads = 8, .head_dim = 256, .values_are_keys = false }).validate();
    try (Geometry{ .heads = 16, .kv_heads = 2, .head_dim = 512, .values_are_keys = true }).validate();
    try std.testing.expectError(error.InvalidGemmaGeometry, (Geometry{ .heads = 16, .kv_heads = 3, .head_dim = 256, .values_are_keys = false }).validate());
    try std.testing.expectError(error.InvalidGemmaGeometry, (Geometry{ .heads = 16, .kv_heads = 0, .head_dim = 256, .values_are_keys = false }).validate());
}
