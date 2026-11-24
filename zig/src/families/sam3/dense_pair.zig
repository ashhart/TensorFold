const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").dense_pair;

pub const Epilogue = union(enum) { none, gelu_pair, residual: gpu.Tensor, tile_residual: gpu.Tensor };

pub fn pairCount(n: usize) usize {
    return kernel.pairCount(n);
}

pub fn prepareWeight(g: *gpu.Gpu, weight: gpu.Tensor, bias: gpu.Tensor, cols: usize, depth: usize) !gpu.Tensor {
    const nw = try std.math.mul(usize, cols, depth);
    if (cols == 0 or depth == 0 or weight.dtype != .f32 or bias.dtype != .f32 or
        weight.count != nw or bias.count != cols) return error.SamPairProjectionShape;
    const w_pair = try g.allocDedicated(kernel.weightPairCount(nw), .f16);
    errdefer w_pair.deinit();
    const shape = [3]u32{ 0, @intCast(cols), @intCast(depth) };
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const clear = try g.pipeline("sam_norm_wide_clear");
    const prep = try g.pipeline("sam_dense_pair_prepare_weights");
    const pass = g.command();
    pass.enc.setPipeline(clear);
    pass.enc.setBuffer(w_pair.buffer, w_pair.offset + 4 * nw, 0);
    pass.enc.dispatchGroups(gpu.mtl.Size.of(1, 1, 1), gpu.mtl.Size.of(32, 1, 1));
    pass.enc.barrier();
    pass.enc.setPipeline(prep);
    pass.enc.setBuffer(weight.buffer, weight.offset, 0);
    pass.enc.setBuffer(bias.buffer, bias.offset, 1);
    pass.enc.setBuffer(w_pair.buffer, w_pair.offset, 2);
    pass.enc.setValue(shape, 3);
    pass.enc.dispatchThreads(gpu.mtl.Size.of(@min(nw + cols, 1 << 18), 1, 1), gpu.mtl.Size.of(128, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return w_pair;
}

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, w_pair: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, epilogue: Epilogue) !gpu.Tensor {
    const n = try std.math.mul(usize, rows, depth);
    const outputs = try std.math.mul(usize, rows, cols);
    const packing = x.dtype == .f32;
    const residual: ?gpu.Tensor = switch (epilogue) {
        .residual, .tile_residual => |t| t,
        else => null,
    };
    if (rows == 0 or cols == 0 or depth == 0 or depth > 4736 or cols > 4736 or
        x.count != (if (packing) n else pairCount(n)) or (!packing and x.dtype != .f16) or
        weight.dtype != .f32 or w_pair.dtype != .f16 or bias.dtype != .f32 or
        (epilogue == .tile_residual and rows != 5184) or
        (residual != null and (residual.?.dtype != .f32 or residual.?.count != outputs))) return error.SamPairProjectionShape;
    const output = if (epilogue == .gelu_pair) try g.allocDevice(pairCount(outputs), .f16) else try g.allocDevice(outputs, .f32);
    errdefer output.deinit();
    const x_pair = if (packing) try g.allocDevice(pairCount(n), .f16) else x;
    defer if (packing) x_pair.deinit();
    const plan = try g.allocDevice(16, .f32);
    defer plan.deinit();
    const needs_scratch = epilogue == .residual or epilogue == .tile_residual;
    const scratch = if (needs_scratch) try g.allocDevice(outputs, .f32) else null;
    defer if (scratch) |t| t.deinit();
    const shape = [3]u32{ @intCast(std.math.cast(i32, rows) orelse return error.SamPairProjectionShape), @intCast(cols), @intCast(depth) };
    const code: kernel.Epilogue = switch (epilogue) {
        .none => .none,
        .gelu_pair => .gelu_pair,
        .residual => .residual,
        .tile_residual => .tile_residual,
    };
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    if (g.operator_observer != null) g.operator_shape = .{ rows, cols, depth };
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var helpers: [7]gpu.mtl.Pipeline = undefined;
    for (kernel.helper_names, 0..) |name, i| helpers[i] = try g.pipeline(name);
    const main = try g.pipeline(kernel.mainName(code));
    const pass = g.command();
    kernel.encode(gpu.mtl, pass.enc, helpers, main, [_]gpu.Tensor{ x, w_pair, weight, bias, output, x_pair, plan, residual orelse output, scratch orelse output }, shape, packing, code);
    try g.finish(pass.cb, pass.enc);
    return output;
}
