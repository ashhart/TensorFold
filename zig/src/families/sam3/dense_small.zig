const std = @import("std");
const gpu = @import("tensor_metal");

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
    if (rows == 0 or rows > 32 or cols == 0 or cols > 4736 or depth == 0 or depth > 4736 or
        x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        x.count != rows * depth or weight.count != cols * depth or bias.count != cols) return error.SamSmallProjectionShape;
    const output = try g.allocDevice(try std.math.mul(usize, rows, cols), .f32);
    errdefer output.deinit();
    const shape = [3]u32{ @intCast(rows), @intCast(cols), @intCast(depth) };
    try g.dispatch("tf_sam3_dense_small", &.{ x, weight, bias, output }, shape, gpu.mtl.Size.of((cols + 3) / 4 * 128, rows, 1), gpu.mtl.Size.of(128, 1, 1));
    return output;
}
