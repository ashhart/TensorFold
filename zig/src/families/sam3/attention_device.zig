const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").attention_device;

pub fn run(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize) !gpu.Tensor {
    return runWith(g, q, k, v, shape, null, 0);
}

pub fn runPrechecked(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, flags: gpu.Tensor, wide_idx: usize) !gpu.Tensor {
    return runWith(g, q, k, v, shape, flags, wide_idx);
}

fn runWith(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize, prechecked: ?gpu.Tensor, wide_idx: usize) !gpu.Tensor {
    const batch, const heads, const rows, const keys, const dim = shape;
    const planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), rows);
    const key_planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), keys);
    if (batch == 0 or heads == 0 or rows == 0 or keys == 0 or keys > (if (dim == 32) @as(usize, 65536) else 8192) or
        (dim != 32 and dim != 64) or q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32 or
        q.count != try std.math.mul(usize, planes, dim) or k.count != try std.math.mul(usize, key_planes, dim) or k.count != v.count) return error.AttentionDeviceShape;
    const out = try g.allocDevice(q.count, .f32);
    errdefer out.deinit();
    const nq = try std.math.divCeil(usize, rows, 64);
    const min_flags = 1 + try std.math.mul(usize, nq, try std.math.mul(usize, heads, batch));
    var own_flags: ?gpu.Tensor = null;
    defer if (own_flags) |f| f.deinit();
    const flags = if (prechecked) |f| blk: {
        if (f.count < min_flags + @intFromBool(wide_idx != 0) or (wide_idx != 0 and wide_idx >= f.count)) return error.AttentionDeviceShape;
        break :blk f;
    } else blk: {
        own_flags = try g.allocDevice(min_flags, .f32);
        break :blk own_flags.?;
    };
    const values = [_]usize{
        batch,              heads, rows,        keys,               1,   nq,          (keys + 31) / 32,   rows / 64, keys / 32,   rows % 64,          keys % 32, wide_idx,
        rows * heads * dim, dim,   heads * dim, keys * heads * dim, dim, heads * dim, keys * heads * dim, dim,       heads * dim, rows * heads * dim, dim,       heads * dim,
    };
    var params: [25]i32 = undefined;
    for (values, 0..) |value, index| params[index] = std.math.cast(i32, value) orelse return error.AttentionTooLarge;
    params[24] = @bitCast(1 / @sqrt(@as(f32, @floatFromInt(dim))));
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [5]gpu.mtl.Pipeline = undefined;
    const names = if (dim == 32) kernel.names32 else kernel.names;
    for ([_]usize{ 0, 1, 3, 4, 2 }) |i| pipes[i] = try g.pipeline(names[i]);
    const pass = g.command();
    if (prechecked != null)
        kernel.encodePrechecked(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ q, k, v, out, flags }, params, true)
    else
        kernel.encode(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ q, k, v, out, flags }, params, true);
    try g.finish(pass.cb, pass.enc);
    return out;
}
