const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").dense_planes;

pub const Accumulation = enum { plain, compensated };
pub const Epilogue = union(enum) { none, gelu_planes, residual: gpu.Tensor, tile_residual: gpu.Tensor };

pub fn planesCount(n: usize) usize {
    return kernel.planesCount(n);
}

// The number of values a planes buffer of count halves holds, or null when no value count fits.
pub fn values(count: usize) ?usize {
    if (count < 16) return null;
    const n = (count - 16) / 5;
    return if (planesCount(n) == count) n else null;
}

// Projects FP32 rows or a planes buffer through an exact FP16 weight at the FP16 tensor rate.
pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, accumulation: Accumulation, epilogue: Epilogue) !gpu.Tensor {
    const outputs = try std.math.mul(usize, rows, cols);
    const output = if (epilogue == .gelu_planes) try g.allocDevice(planesCount(outputs), .f16) else try g.allocDevice(outputs, .f32);
    errdefer output.deinit();
    try into(g, x, weight, bias, output, rows, cols, depth, accumulation, epilogue);
    return output;
}

pub fn into(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, output: gpu.Tensor, rows: usize, cols: usize, depth: usize, accumulation: Accumulation, epilogue: Epilogue) !void {
    const n = try std.math.mul(usize, rows, depth);
    const outputs = try std.math.mul(usize, rows, cols);
    const packing = x.dtype == .f32;
    const residual: ?gpu.Tensor = switch (epilogue) {
        .residual, .tile_residual => |t| t,
        else => null,
    };
    if (rows == 0 or cols == 0 or depth == 0 or depth > 4736 or cols > 4736 or
        x.count != (if (packing) n else planesCount(n)) or (!packing and x.dtype != .f16) or
        weight.dtype != .f16 or weight.count != try std.math.mul(usize, cols, depth) or bias.dtype != .f32 or bias.count != cols or
        output.dtype != (if (epilogue == .gelu_planes) gpu.DType.f16 else gpu.DType.f32) or
        output.count != (if (epilogue == .gelu_planes) planesCount(outputs) else outputs) or
        (epilogue == .tile_residual and (rows != 5184 or accumulation != .plain)) or
        (residual != null and (residual.?.dtype != .f32 or residual.?.count != outputs))) return error.SamPlanesProjectionShape;
    const planes = if (packing) try g.allocDevice(planesCount(n), .f16) else x;
    defer if (packing) planes.deinit();
    const plan = try g.allocDevice(16, .f32);
    defer plan.deinit();
    const scratch = if (epilogue != .none) try g.allocDevice(outputs, .f32) else null;
    defer if (scratch) |t| t.deinit();
    const shape = [3]u32{ @intCast(std.math.cast(i32, rows) orelse return error.SamPlanesProjectionShape), @intCast(cols), @intCast(depth) };
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    if (g.operator_observer != null) g.operator_shape = .{ rows, cols, depth };
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [9]gpu.mtl.Pipeline = undefined;
    for (kernel.names, 0..) |name, i| pipes[i] = try g.pipeline(name);
    const code: kernel.Epilogue = switch (epilogue) {
        .none => .none,
        .gelu_planes => .gelu_planes,
        .residual => .residual,
        .tile_residual => .tile_residual,
    };
    const residual_name = code == .residual or code == .tile_residual;
    pipes[8] = try g.pipeline(switch (accumulation) {
        .plain => if (code == .tile_residual) "sam_dense_planes_tiled" else if (residual_name) "sam_dense_planes_residual" else "sam_dense_planes",
        .compensated => if (residual_name) "sam_dense_planes_compensated_residual" else "sam_dense_planes_compensated",
    });
    const pass = g.command();
    kernel.encode(gpu.mtl, pass.enc, pipes, [_]gpu.Tensor{ x, weight, bias, output, planes, plan, residual orelse output, scratch orelse output }, shape, packing, code);
    try g.finish(pass.cb, pass.enc);
}

// Splits FP32 rows into a planes buffer that run and into accept in place of the FP32 input.
pub fn pack(g: *gpu.Gpu, x: gpu.Tensor, rows: usize, depth: usize) !gpu.Tensor {
    const n = try std.math.mul(usize, rows, depth);
    if (x.dtype != .f32 or x.count != n or n == 0) return error.SamPlanesProjectionShape;
    const planes = try g.allocDevice(planesCount(n), .f16);
    errdefer planes.deinit();
    const plan = try g.allocDevice(16, .f32);
    defer plan.deinit();
    const shape = [3]u32{ @intCast(std.math.cast(i32, rows) orelse return error.SamPlanesProjectionShape), 0, @intCast(depth) };
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const pass = g.command();
    pass.enc.setPipeline(try g.pipeline("sam_dense_planes_clear"));
    for ([_]gpu.Tensor{ planes, plan, planes }, 0..) |t, i| pass.enc.setBuffer(t.buffer, t.offset, i);
    pass.enc.setValue(shape, 3);
    pass.enc.setValue([2]u32{ 1, 0 }, 4);
    pass.enc.dispatchThreads(gpu.mtl.Size.of(16, 1, 1), gpu.mtl.Size.of(16, 1, 1));
    pass.enc.barrier();
    pass.enc.setPipeline(try g.pipeline("sam_dense_planes_pack"));
    for ([_]gpu.Tensor{ x, planes, plan, planes, plan }, 0..) |t, i| pass.enc.setBuffer(t.buffer, t.offset, i);
    pass.enc.setValue(shape, 5);
    pass.enc.dispatchThreads(gpu.mtl.Size.of(@min(n, 1 << 20), 1, 1), gpu.mtl.Size.of(128, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return planes;
}
