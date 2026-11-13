const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").attention_split32;

pub fn run(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize) !gpu.Tensor {
    const batch, const heads, const rows, const keys, const dim = shape;
    const bh = try std.math.mul(usize, batch, heads);
    const planes = try std.math.mul(usize, bh, rows);
    const key_planes = try std.math.mul(usize, bh, keys);
    if (batch == 0 or heads == 0 or rows == 0 or keys == 0 or (dim != 32 and dim != 64) or keys > (if (dim == 32) @as(usize, 65536) else 8192) or
        q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32 or
        q.count != try std.math.mul(usize, planes, dim) or k.count != try std.math.mul(usize, key_planes, dim) or k.count != v.count) return error.AttentionSplit32Shape;
    const out = try g.allocDevice(q.count, .f32);
    errdefer out.deinit();
    const nq = try std.math.divCeil(usize, rows, 64);
    const blocks = try std.math.divCeil(usize, keys, 32);
    const splits = try std.math.divCeil(usize, blocks, kernel.chunk_blocks);
    const workspace = try g.deviceSplitAttentionWorkspace(planes, splits, dim, 1 + try std.math.mul(usize, bh, nq));
    const partial, const stats, const flags = workspace;
    const values = [_]usize{
        batch,              heads,               rows,        keys,               1,   nq,          blocks,             rows / 64, keys / 32,   rows % 64,          keys % 32, 0,
        rows * heads * dim, dim,                 heads * dim, keys * heads * dim, dim, heads * dim, keys * heads * dim, dim,       heads * dim, rows * heads * dim, dim,       heads * dim,
        splits,             kernel.chunk_blocks,
    };
    var params: [27]i32 = undefined;
    for (values, 0..) |value, i| params[i] = std.math.cast(i32, value) orelse return error.AttentionTooLarge;
    params[26] = @bitCast(1 / @sqrt(@as(f32, @floatFromInt(dim))));
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [6]gpu.mtl.Pipeline = undefined;
    const names = if (dim == 32) kernel.names else kernel.names64;
    for ([_]usize{ 0, 1, 4, 5, 3, 2 }) |i| pipes[i] = try g.pipeline(names[i]);
    const refs = [_]gpu.Tensor{ q, k, v, partial, stats, out, flags };
    if (g.operator_observer != null) {
        inline for (0..6) |phase| {
            const pass = g.command();
            g.operator_name = names[phase];
            kernel.encodePhase(gpu.mtl, pass.enc, pipes, refs, params, phase);
            try g.finish(pass.cb, pass.enc);
        }
    } else {
        const pass = g.command();
        kernel.encode(gpu.mtl, pass.enc, pipes, refs, params);
        try g.finish(pass.cb, pass.enc);
    }
    return out;
}
