const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").dense_device;
/// `wide` layouts admit any finite operand up to 2^36: device products below 2^-126 flush to zero.
pub const Layout = enum { nk, nk_half, nk_gelu, nk_gelu_linear, nk_compensated, kn, kn_compensated, wide, wide_compensated, wide_gelu };

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, layout: Layout) !gpu.Tensor {
    const output = try g.allocDevice(try std.math.mul(usize, rows, cols), .f32);
    errdefer output.deinit();
    try into(g, x, weight, bias, output, rows, cols, depth, layout);
    return output;
}

pub fn into(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, output: gpu.Tensor, rows: usize, cols: usize, depth: usize, layout: Layout) !void {
    return project(g, x, weight, bias, null, output, rows, cols, depth, layout);
}

/// The weight and bias half of a wide layout's domain, for a constant weight: one word, kept by the caller.
pub fn operandFlags(g: *gpu.Gpu, weight: gpu.Tensor, bias: gpu.Tensor, cols: usize, depth: usize) !gpu.Tensor {
    if (weight.dtype != .f32 or bias.dtype != .f32 or weight.count != try std.math.mul(usize, cols, depth) or bias.count != cols)
        return error.SamDeviceProjectionShape;
    const limit = g.buffer_cache_limit;
    g.buffer_cache_limit = 0;
    defer g.buffer_cache_limit = limit;
    const flags = try g.alloc(1, .f32);
    errdefer flags.deinit();
    @memset(flags.slice(u32, 1), 0);
    const shape = [3]u32{ 0, @intCast(cols), @intCast(depth) };
    try g.dispatch(kernel.operands_name, &.{ weight, bias, flags }, shape, gpu.mtl.Size.of(@min(cols * depth + cols, 1 << 18), 1, 1), gpu.mtl.Size.of(128, 1, 1));
    return flags;
}

/// A wide layout whose weight and bias flags come from `operandFlags`: only the input is scanned per call.
pub fn runCached(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, flags: gpu.Tensor, rows: usize, cols: usize, depth: usize, layout: Layout) !gpu.Tensor {
    if (layout != .wide and layout != .wide_compensated and layout != .wide_gelu) return error.SamDeviceProjectionShape;
    const output = try g.allocDevice(try std.math.mul(usize, rows, cols), .f32);
    errdefer output.deinit();
    try project(g, x, weight, bias, flags, output, rows, cols, depth, layout);
    return output;
}

fn project(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, flags: ?gpu.Tensor, output: gpu.Tensor, rows: usize, cols: usize, depth: usize, layout: Layout) !void {
    if (rows == 0 or cols == 0 or depth == 0 or depth > 4736 or cols > 4736 or
        x.dtype != .f32 or weight.dtype != (if (layout == .nk_half) gpu.DType.f16 else gpu.DType.f32) or bias.dtype != .f32 or
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
    if (g.operator_observer != null) g.operator_shape = .{ rows, cols, depth };
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [5]gpu.mtl.Pipeline = undefined;
    const names = switch (layout) {
        .nk => kernel.names,
        .nk_half => kernel.half_names,
        .nk_gelu => kernel.gelu_names,
        .nk_gelu_linear => kernel.gelu_linear_names,
        .nk_compensated => kernel.compensated_names,
        .kn => kernel.nn_names,
        .kn_compensated => kernel.nn_compensated_names,
        .wide => kernel.wide_names,
        .wide_compensated => kernel.wide_compensated_names,
        .wide_gelu => kernel.wide_gelu_names,
    };
    const cached = if (flags != null and x.offset % 16 == 0) flags else null;
    if (cached != null) for (kernel.cached_names, 0..) |name, i| {
        pipes[i] = try g.pipeline(name);
    };
    for ([_]usize{ 0, 1, 2, 4, 3 }) |i| if (cached == null or i >= 2) {
        pipes[i] = try g.pipeline(names[i]);
    };
    const pass = g.command();
    if (cached) |word|
        kernel.encodeCached(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ x, weight, bias, output, domain, word }, shape)
    else
        kernel.encode(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ x, weight, bias, output, domain }, shape, true);
    try g.finish(pass.cb, pass.enc);
}
