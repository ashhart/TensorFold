//! The Qwen2 text graph, with a stream-owned dense BF16 KV cache and ordinary one-dimensional RoPE.
const std = @import("std");
const hip = @import("hip");
const cfg = @import("config.zig");
const Graph = @import("hip_graph.zig").Graph;
const Tensor = hip.ops.Tensor;

pub const Cache = struct {
    data: hip.DeviceBuffer,
    c: cfg.Config,
    total: usize,

    pub fn bytes(c: cfg.Config, total: usize) !usize {
        if (total == 0 or total > c.context) return error.DotsContext;
        return std.math.mul(usize, total, @as(usize, c.layers) * 2 * c.kvWidth() * 2);
    }

    pub fn init(r: *const hip.Runtime, c: cfg.Config, total: usize) !Cache {
        return .{ .data = try hip.DeviceBuffer.alloc(r, try bytes(c, total)), .c = c, .total = total };
    }
    pub fn deinit(cache: *Cache) void {
        cache.data.free();
        cache.* = undefined;
    }
    pub fn layer(cache: *const Cache, i: usize) !hip.ops.Ops.Cache {
        if (i >= cache.c.layers) return error.BadShape;
        const step = cache.total * cache.c.kvWidth() * 2;
        return .{ .k = try cache.data.address(i * 2 * step), .v = try cache.data.address((i * 2 + 1) * step), .kind = .bf16, .kv_heads = cache.c.kv_heads, .total = cache.total, .d = cache.c.headDim() };
    }

    /// Copy only the live prefix, head by head; source and destination may have different reserved capacities.
    pub fn copyPrefix(source: *const Cache, dest: *Cache, rows: usize, stream: hip.Stream) !void {
        if (rows > source.total or rows > dest.total or source.c.layers != dest.c.layers or
            source.c.kv_heads != dest.c.kv_heads or source.c.headDim() != dest.c.headDim()) return error.BadShape;
        if (source.data.r != dest.data.r or source.data.r != stream.r) return error.BadShape;
        if (source.data.base() == dest.data.base() or rows == 0) return;
        const row_bytes = @as(usize, source.c.headDim()) * 2;
        const copied_bytes = rows * row_bytes;
        for (0..source.c.layers * 2) |layer_kind| for (0..source.c.kv_heads) |head| {
            const index = layer_kind * source.c.kv_heads + head;
            const from = try source.data.address(index * source.total * row_bytes);
            try dest.data.copyFromAsync(index * dest.total * row_bytes, from, copied_bytes, stream);
        };
    }
};

fn weight(g: Graph, layer: usize, suffix: []const u8) !u64 {
    var name: [160]u8 = undefined;
    return g.w.get(try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ layer, suffix }));
}

/// The positions must be the contiguous range pos0..pos0+rows, as admitted by the stream owner.
pub fn block(g: Graph, x: Tensor, out: Tensor, positions: u64, rows: usize, pos0: usize, layer: usize, cache: *const Cache) !void {
    if (rows == 0 or pos0 >= cache.total or rows > cache.total - pos0 or layer >= g.c.layers or
        x.kind != .bf16 or out.kind != .bf16 or x.ptr == out.ptr) return error.BadShape;
    const mark = g.ops.arena.mark();
    defer g.ops.arena.release(mark);
    const c = g.c;
    const n = rows * c.hidden;
    const norm = try g.norm(x, try weight(g, layer, "input_layernorm.weight"), rows, c.hidden, c.eps);
    const q = try g.linear(norm, try weight(g, layer, "self_attn.q_proj.weight"), try weight(g, layer, "self_attn.q_proj.bias"), rows, c.hidden, c.hidden);
    const k = try g.linear(norm, try weight(g, layer, "self_attn.k_proj.weight"), try weight(g, layer, "self_attn.k_proj.bias"), rows, c.kvWidth(), c.hidden);
    const v = try g.linear(norm, try weight(g, layer, "self_attn.v_proj.weight"), try weight(g, layer, "self_attn.v_proj.bias"), rows, c.kvWidth(), c.hidden);
    const layer_cache = try cache.layer(layer);
    const qr = try g.ops.arena.of(f32, n);
    try hip.dense_text.rope(g.ops, q, positions, .{ .ptr = qr, .kind = .f32 }, rows, c.heads, c.headDim(), @floatCast(c.theta), false, cache.total);
    try hip.dense_text.rope(g.ops, k, positions, .{ .ptr = layer_cache.k, .kind = .bf16 }, rows, c.kv_heads, c.headDim(), @floatCast(c.theta), true, cache.total);
    try g.ops.kvWrite(v, layer_cache.v, rows, c.kv_heads, c.headDim(), cache.total, pos0);
    const att = try g.ops.arena.of(f32, n);
    try g.ops.causalPrefill(qr, layer_cache, att, rows, pos0 + rows, c.heads, 1 / @sqrt(@as(f32, @floatFromInt(c.headDim()))), pos0);
    const ar = try g.tensor(n);
    try hip.vision.attentionRows(g.ops, att, ar, rows, c.heads, c.headDim());
    const projected = try g.linear(ar, try weight(g, layer, "self_attn.o_proj.weight"), null, rows, c.hidden, c.hidden);
    try g.ops.add(x, projected, out, n);
    g.ops.arena.release(mark);
    const norm2 = try g.norm(out, try weight(g, layer, "post_attention_layernorm.weight"), rows, c.hidden, c.eps);
    const gate = try g.linear(norm2, try weight(g, layer, "mlp.gate_proj.weight"), null, rows, c.intermediate, c.hidden);
    const up = try g.linear(norm2, try weight(g, layer, "mlp.up_proj.weight"), null, rows, c.intermediate, c.hidden);
    try g.ops.siluMul(gate, up, gate, rows * c.intermediate);
    const down = try g.linear(gate, try weight(g, layer, "mlp.down_proj.weight"), null, rows, c.hidden, c.intermediate);
    try g.ops.add(out, down, out, n);
}

pub const ImageSpan = struct { first: usize, rows: usize, features: Tensor };

/// Token IDs and the image runs are validated by the request owner before this device-only gather.
pub fn embed(g: Graph, ids: u64, rows: usize, images: []const ImageSpan) !Tensor {
    if (rows == 0 or rows > g.c.context) return error.DotsContext;
    var end: usize = 0;
    for (images) |image| {
        if (image.rows == 0 or image.first < end or image.first >= rows or image.rows > rows - image.first or
            image.features.kind != .bf16) return error.BadShape;
        end = image.first + image.rows;
    }
    const out = try g.tensor(rows * g.c.hidden);
    try hip.dense_text.embed(g.ops, try g.w.get("model.embed_tokens.weight"), ids, out, rows, g.c.hidden, g.c.vocab);
    for (images) |image| try g.ops.cast(image.features, .{ .ptr = out.ptr + image.first * g.c.hidden * 2, .kind = .bf16 }, image.rows * g.c.hidden);
    return out;
}

pub const Result = struct { normalized: Tensor, logits: Tensor };

pub fn forward(g: Graph, input: Tensor, positions: u64, rows: usize, pos0: usize, cache: *const Cache, all_logits: bool) !Result {
    if (rows == 0 or pos0 >= cache.total or rows > cache.total - pos0 or input.kind != .bf16) return error.DotsContext;
    const c = g.c;
    const n = rows * c.hidden;
    var x = try g.tensor(n);
    var y = try g.tensor(n);
    try g.ops.cast(input, x, n);
    for (0..c.layers) |layer| {
        try block(g, x, y, positions, rows, pos0, layer, cache);
        std.mem.swap(Tensor, &x, &y);
    }
    const norm = try g.norm(x, try g.w.get("model.norm.weight"), rows, c.hidden, c.eps);
    const head_input: Tensor = .{ .ptr = norm.ptr + (if (all_logits) 0 else (rows - 1) * c.hidden * 2), .kind = .bf16 };
    return .{ .normalized = norm, .logits = try g.linear(head_input, try g.w.get("lm_head.weight"), null, if (all_logits) rows else 1, c.vocab, c.hidden) };
}

pub fn scratchBytes(c: cfg.Config, rows: usize, all_logits: bool) !usize {
    if (rows == 0 or rows > c.context) return error.DotsContext;
    const hidden = rows * @as(usize, c.hidden);
    const ff = rows * @as(usize, c.intermediate);
    return 12 * hidden * 2 + 2 * hidden * 4 + 2 * ff * 2 +
        (if (all_logits) rows else @as(usize, 1)) * c.vocab * 2 + 32 * 256;
}

test "dots HIP graph: text cache and scratch reject out of context requests" {
    const cases = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, cases.sample, cases.generation);
    try std.testing.expectEqual(@as(usize, 3758096384), try Cache.bytes(c, c.context));
    try std.testing.expectError(error.DotsContext, Cache.bytes(c, 0));
    try std.testing.expectError(error.DotsContext, scratchBytes(c, c.context + 1, false));
    const short: Cache = .{ .data = undefined, .c = c, .total = 17 };
    var other: Cache = .{ .data = undefined, .c = c, .total = 31 };
    try std.testing.expectError(error.BadShape, short.copyPrefix(&other, 18, undefined));
    std.testing.refAllDecls(Cache);
}
