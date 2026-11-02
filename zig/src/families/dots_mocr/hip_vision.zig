//! The dense vision tower and its 2x2 merger, on the caller's stream and scratch arena.
const std = @import("std");
const hip = @import("hip");
const Graph = @import("hip_graph.zig").Graph;
const Tensor = hip.ops.Tensor;

fn weight(g: Graph, layer: usize, suffix: []const u8) !u64 {
    var name: [160]u8 = undefined;
    return g.w.get(try std.fmt.bufPrint(&name, "vision_tower.blocks.{d}.{s}", .{ layer, suffix }));
}

/// x and out are distinct persistent buffers. All temporaries are released on this same ordered stream.
pub fn block(g: Graph, x: Tensor, out: Tensor, positions: u64, rows: usize, layer: usize) !void {
    if (layer >= g.c.vision.layers or x.kind != .bf16 or out.kind != .bf16 or x.ptr == out.ptr) return error.BadShape;
    const mark = g.ops.arena.mark();
    defer g.ops.arena.release(mark);
    const c = g.c.vision;
    const n = rows * c.hidden;
    const normalized = try g.norm(x, try weight(g, layer, "norm1.weight"), rows, c.hidden, c.eps);
    const qkv = try g.linear(normalized, try weight(g, layer, "attn.qkv.weight"), null, rows, 3 * c.hidden, c.hidden);
    const q = try g.ops.arena.of(f32, n);
    const k = try g.tensor(n);
    const v = try g.tensor(n);
    try hip.vision.rope(g.ops, qkv, positions, q, k.ptr, v.ptr, rows, c.heads, c.hidden / c.heads, 10000);
    const att = try g.ops.arena.of(f32, n);
    try hip.vision.fullAttention(g.ops, q, k.ptr, v.ptr, att, rows, c.heads, c.hidden / c.heads);
    const ar = try g.tensor(n);
    try hip.vision.attentionRows(g.ops, att, ar, rows, c.heads, c.hidden / c.heads);
    const projected = try g.linear(ar, try weight(g, layer, "attn.proj.weight"), null, rows, c.hidden, c.hidden);
    try g.ops.add(x, projected, out, n);
    g.ops.arena.release(mark);
    const norm2 = try g.norm(out, try weight(g, layer, "norm2.weight"), rows, c.hidden, c.eps);
    const gate = try g.linear(norm2, try weight(g, layer, "mlp.fc1.weight"), null, rows, c.intermediate, c.hidden);
    const up = try g.linear(norm2, try weight(g, layer, "mlp.fc3.weight"), null, rows, c.intermediate, c.hidden);
    try g.ops.siluMul(gate, up, gate, rows * c.intermediate);
    const down = try g.linear(gate, try weight(g, layer, "mlp.fc2.weight"), null, rows, c.hidden, c.intermediate);
    try g.ops.add(out, down, out, n);
}

/// The patch rows are already in merge order. Each image is encoded separately to keep attention isolated.
pub fn forward(g: Graph, pixels: Tensor, positions: u64, rows: usize) !Tensor {
    return (try stages(g, pixels, positions, rows)).merged;
}

pub const Stages = struct { normalized: Tensor, merged: Tensor };

pub fn stages(g: Graph, pixels: Tensor, positions: u64, rows: usize) !Stages {
    if (rows % 4 != 0) return error.DotsGrid;
    _ = try @import("hip_graph.zig").patchBytes(g.c, rows);
    const c = g.c.vision;
    const n = rows * c.hidden;
    var x = try g.tensor(n);
    var y = try g.tensor(n);
    const mark = g.ops.arena.mark();
    const patch = try g.patch(pixels, rows);
    try g.ops.cast(patch.normalized, x, n);
    g.ops.arena.release(mark);
    for (0..c.layers) |layer| {
        try block(g, x, y, positions, rows, layer);
        std.mem.swap(Tensor, &x, &y);
    }
    const normalized = try g.norm(x, try g.w.get("vision_tower.post_trunk_norm.weight"), rows, c.hidden, c.eps);
    return .{ .normalized = normalized, .merged = try merger(g, normalized, rows) };
}

pub fn merger(g: Graph, normalized: Tensor, rows: usize) !Tensor {
    if (rows == 0 or rows % 4 != 0) return error.DotsGrid;
    const c = g.c.vision;
    const n = rows * c.hidden;
    const ln = try g.tensor(n);
    try hip.vision.layerNorm(g.ops, normalized, try g.w.get("vision_tower.merger.ln_q.weight"), try g.w.get("vision_tower.merger.ln_q.bias"), ln, rows, c.hidden, 1e-6);
    const merged_rows = rows / 4;
    const wide = c.hidden * 4;
    const hidden = try g.linear(ln, try g.w.get("vision_tower.merger.mlp.0.weight"), try g.w.get("vision_tower.merger.mlp.0.bias"), merged_rows, wide, wide);
    try hip.vision.gelu(g.ops, hidden, hidden, merged_rows, wide);
    return g.linear(hidden, try g.w.get("vision_tower.merger.mlp.2.weight"), try g.w.get("vision_tower.merger.mlp.2.bias"), merged_rows, c.hidden, wide);
}

/// Persistent double buffering plus the largest block workspace. Every allocation is 256-byte aligned.
pub fn scratchBytes(c: @import("config.zig").Config, rows: usize) !usize {
    _ = try @import("hip_graph.zig").patchBytes(c, rows);
    const n = rows * @as(usize, c.vision.hidden);
    const ff = rows * @as(usize, c.vision.intermediate);
    // Two persistent hidden buffers, eight BF16 hidden temporaries, Q/attention FP32, and gate/up.
    return (2 + 8) * n * 2 + 2 * n * 4 + 2 * ff * 2 + 32 * 256;
}
