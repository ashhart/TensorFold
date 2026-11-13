const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").attention_device;

pub fn run(g: *gpu.Gpu, q: gpu.Tensor, k: gpu.Tensor, v: gpu.Tensor, shape: [5]usize) !gpu.Tensor {
    const batch, const heads, const rows, const keys, const dim = shape;
    const planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), rows);
    const key_planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), keys);
    if (batch == 0 or heads == 0 or rows == 0 or keys == 0 or keys > (if (dim == 32) @as(usize, 65536) else 8192) or
        (dim != 32 and dim != 64) or q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32 or
        q.count != try std.math.mul(usize, planes, dim) or k.count != try std.math.mul(usize, key_planes, dim) or k.count != v.count) return error.AttentionDeviceShape;
    const out = try g.allocDevice(q.count, .f32);
    errdefer out.deinit();
    const nq = try std.math.divCeil(usize, rows, 64);
    const flags = try g.allocDevice(1 + try std.math.mul(usize, nq, try std.math.mul(usize, heads, batch)), .f32);
    defer flags.deinit();
    const window = dim == 64 and heads == 16 and rows == 576 and keys == 576;
    const centered: ?gpu.Tensor = if (window) try g.allocDevice(try std.math.add(usize, try std.math.mul(usize, k.count, 2), try std.math.mul(usize, batch * heads, 64)), .f32) else null;
    defer if (centered) |tensor| tensor.deinit();
    const values = [_]usize{
        batch,              heads, rows,        keys,               1,   nq,          (keys + 31) / 32,   rows / 64, keys / 32,   rows % 64,          keys % 32, 0,
        rows * heads * dim, dim,   heads * dim, keys * heads * dim, dim, heads * dim, keys * heads * dim, dim,       heads * dim, rows * heads * dim, dim,       heads * dim,
    };
    var params: [25]i32 = undefined;
    for (values, 0..) |value, index| params[index] = std.math.cast(i32, value) orelse return error.AttentionTooLarge;
    params[24] = @bitCast(1 / @sqrt(@as(f32, @floatFromInt(dim))));
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [6]gpu.mtl.Pipeline = undefined;
    const names = if (dim == 32) kernel.names32 else kernel.names;
    for ([_]usize{ 0, 1, 4, 5, 2, 3 }) |i| pipes[i] = try g.pipeline(names[i]);
    const pass = g.command();
    kernel.encode(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ q, k, v, out, flags, centered orelse out }, params, true);
    try g.finish(pass.cb, pass.enc);
    return out;
}
