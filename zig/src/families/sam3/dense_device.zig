const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").dense_device;
pub const Layout = enum { nk, nk_gelu, nk_gelu_linear, nk_compensated, kn, kn_compensated };

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, layout: Layout) !gpu.Tensor {
    const output = try g.allocDevice(try std.math.mul(usize, rows, cols), .f32);
    errdefer output.deinit();
    try into(g, x, weight, bias, output, rows, cols, depth, layout);
    return output;
}

pub fn into(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, output: gpu.Tensor, rows: usize, cols: usize, depth: usize, layout: Layout) !void {
    if (rows == 0 or cols == 0 or depth == 0 or depth > 4736 or cols > 4736 or
        x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        x.count != try std.math.mul(usize, rows, depth) or weight.count != try std.math.mul(usize, cols, depth) or bias.count != cols or
        output.dtype != .f32 or output.count != try std.math.mul(usize, rows, cols)) return error.SamDeviceProjectionShape;
    const domain = try g.allocDevice(7, .f32);
    defer domain.deinit();
    const shape = [3]u32{
        @intCast(std.math.cast(i32, rows) orelse return error.SamDeviceProjectionShape),
        @intCast(cols),
        @intCast(depth),
    };
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [5]gpu.mtl.Pipeline = undefined;
    const names = switch (layout) {
        .nk => kernel.names,
        .nk_gelu => kernel.gelu_names,
        .nk_gelu_linear => kernel.gelu_linear_names,
        .nk_compensated => kernel.compensated_names,
        .kn => kernel.nn_names,
        .kn_compensated => kernel.nn_compensated_names,
    };
    for ([_]usize{ 0, 1, 2, 4, 3 }) |i| pipes[i] = try g.pipeline(names[i]);
    const pass = g.command();
    kernel.encode(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ x, weight, bias, output, domain }, shape, true);
    try g.finish(pass.cb, pass.enc);
}
