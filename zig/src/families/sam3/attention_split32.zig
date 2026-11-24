const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").attention_split32;

pub fn run(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize) !gpu.Tensor {
    return runWith(g, q, k, v, shape, null, 0);
}

pub fn runPrechecked(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, flags: gpu.Tensor, wide_idx: usize) !gpu.Tensor {
    return runWith(g, q, k, v, shape, flags, wide_idx);
}

fn runWith(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, prechecked: ?gpu.Tensor, wide_idx: usize) !gpu.Tensor {
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
    const plan = kernel.splitsFor(batch, heads, nq, keys);
    const splits = plan.splits;
    if (prechecked != null and splits != 1) return error.AttentionSplit32Shape;
    const flag_count = 1 + try std.math.mul(usize, bh, nq);
    var own_flags: ?gpu.Tensor = null;
    defer if (own_flags) |f| f.deinit();
    const partial, const stats, const flags = if (prechecked) |f| blk: {
        if (f.count < flag_count + @intFromBool(wide_idx != 0) or (wide_idx != 0 and wide_idx >= f.count)) return error.AttentionSplit32Shape;
        break :blk .{ out, out, f };
    } else if (splits > 1)
        try g.deviceSplitAttentionWorkspace(planes, splits, dim, flag_count)
    else blk: {
        own_flags = try g.allocDevice(flag_count, .f32);
        break :blk .{ out, out, own_flags.? };
    };
    const values = [_]usize{
        batch,              heads,             rows,        keys,               1,   nq,          blocks,             rows / 64, keys / 32,   rows % 64,          keys % 32, wide_idx,
        rows * heads * dim, dim,               heads * dim, keys * heads * dim, dim, heads * dim, keys * heads * dim, dim,       heads * dim, rows * heads * dim, dim,       heads * dim,
        splits,             plan.chunk_blocks,
    };
    var params: [27]i32 = undefined;
    for (values, 0..) |value, i| params[i] = std.math.cast(i32, value) orelse return error.AttentionTooLarge;
    params[26] = @bitCast(1 / @sqrt(@as(f32, @floatFromInt(dim))));
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    if (g.operator_observer != null) g.operator_shape = .{ rows, keys, dim };
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [6]gpu.mtl.Pipeline = undefined;
    const names = if (dim == 32) kernel.names else kernel.names64;
    for ([_]usize{ 0, 1, 4, 5, 3, 2 }) |i| pipes[i] = try g.pipeline(names[i]);
    const refs = [_]gpu.Tensor{ q, k, v, partial, stats, out, flags };
    if (g.operator_observer != null) {
        inline for (0..6) |phase| {
            if ((phase >= 2 or prechecked == null) and (phase != 3 or splits > 1)) {
                const pass = g.command();
                g.operator_name = names[phase];
                kernel.encodePhase(gpu.mtl, pass.enc, pipes, refs, params, phase);
                try g.finish(pass.cb, pass.enc);
            }
        }
    } else {
        const pass = g.command();
        if (prechecked != null)
            kernel.encodePrechecked(gpu.mtl, pass.enc, pipes, refs, params)
        else
            kernel.encode(gpu.mtl, pass.enc, pipes, refs, params);
        try g.finish(pass.cb, pass.enc);
    }
    return out;
}
