const std = @import("std");
const gpu = @import("tensor_metal");
const implementation = @import("sam3_kernels").attention_split;

pub fn run(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize) !gpu.Tensor {
    const batch, const heads, const rows, const keys, const dim = shape;
    const planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), rows);
    const key_planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), keys);
    if (batch == 0 or heads == 0 or rows == 0 or keys == 0 or dim != 256 or q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32 or
        q.count != try std.math.mul(usize, planes, dim) or k.count != try std.math.mul(usize, key_planes, dim) or k.count != v.count) return error.AttentionSplitShape;
    const chunk_blocks = implementation.chunk_blocks;
    const blocks = try std.math.divCeil(usize, keys, 32);
    const splits = try std.math.divCeil(usize, blocks, chunk_blocks);
    const probabilities, const stats, const partial = try g.splitAttentionWorkspace(rows, keys, heads, batch, chunk_blocks);
    const out = try g.allocDevice(q.count, .f32);
    errdefer out.deinit();
    const nq = try std.math.divCeil(usize, rows, 64);
    const values = [_]usize{
        batch,              heads,        rows,        keys,               1,   nq,          blocks,             rows / 64, keys / 32,   rows % 64,          keys % 32, 0,
        rows * heads * dim, dim,          heads * dim, keys * heads * dim, dim, heads * dim, keys * heads * dim, dim,       heads * dim, rows * heads * dim, dim,       heads * dim,
        splits,             chunk_blocks,
    };
    var params: [27]i32 = undefined;
    for (values, 0..) |value, index| params[index] = std.math.cast(i32, value) orelse return error.AttentionTooLarge;
    params[26] = @bitCast(@as(f32, 0.0625));
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const pipes = [3]gpu.mtl.Pipeline{
        try g.pipeline("sam_attention_split_scores"),
        try g.pipeline("sam_attention_split_values"),
        try g.pipeline("sam_attention_split_merge"),
    };
    const pass = g.command();
    implementation.encode(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ q, k, v, probabilities, stats, partial, out }, params);
    try g.finish(pass.cb, pass.enc);
    return out;
}
