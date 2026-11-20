const std = @import("std");
const gpu = @import("tensor_metal");

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32) !gpu.Tensor {
    if (width == 0 or width > 4096 or x.count == 0 or x.count % width != 0 or
        weight.count != width or bias.count != width or x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        !std.math.isFinite(eps) or eps <= 0) return error.SamNormalizationShape;
    const out = try g.allocDevice(x.count, .f32);
    errdefer out.deinit();
    const rows = std.math.cast(u32, x.count / width) orelse return error.SamNormalizationShape;
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const pass = g.command();
    pass.enc.setPipeline(try g.pipeline("sam_norm_compensated"));
    for ([_]gpu.Tensor{ x, weight, bias, out }, 0..) |value, i| pass.enc.setBuffer(value.buffer, value.offset, i);
    pass.enc.setValue([2]u32{ rows, @intCast(width) }, 4);
    pass.enc.setValue(eps, 5);
    pass.enc.dispatchGroups(gpu.mtl.Size.of((@as(usize, rows) + 3) / 4, 1, 1), gpu.mtl.Size.of(128, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return out;
}
