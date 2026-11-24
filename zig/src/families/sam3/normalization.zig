const std = @import("std");
const gpu = @import("tensor_metal");

pub const WideResult = struct { out: gpu.Tensor, flag: gpu.Tensor };

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32) !gpu.Tensor {
    return encode(g, x, weight, bias, width, eps, null);
}

// Normalizes into FP32 while emitting the 1-word wide-domain flag; tiled puts rows in the trunk's tile order.
pub fn runWide(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32, tiled: bool) !WideResult {
    if (tiled and x.count != 5184 * width) return error.SamNormalizationShape;
    if (width == 0 or width > 4096 or x.count == 0 or x.count % width != 0 or
        weight.count != width or bias.count != width or x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        !std.math.isFinite(eps) or eps <= 0) return error.SamNormalizationShape;
    const rows = std.math.cast(u32, x.count / width) orelse return error.SamNormalizationShape;
    const out = try g.allocDevice(x.count, .f32);
    errdefer out.deinit();
    const flag = try g.allocDevice(1, .f32);
    errdefer flag.deinit();
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const clear_pipe = try g.pipeline("sam_norm_wide_clear");
    const norm_pipe = try g.pipeline("sam_norm_compensated_wide");
    const pass = g.command();
    pass.enc.setPipeline(clear_pipe);
    pass.enc.setBuffer(flag.buffer, flag.offset, 0);
    pass.enc.dispatchGroups(gpu.mtl.Size.of(1, 1, 1), gpu.mtl.Size.of(32, 1, 1));
    pass.enc.barrier();
    pass.enc.setPipeline(norm_pipe);
    for ([_]gpu.Tensor{ x, weight, bias, out, flag }, 0..) |value, i| pass.enc.setBuffer(value.buffer, value.offset, i);
    pass.enc.setValue([2]u32{ rows, @intCast(width) }, 5);
    pass.enc.setValue(eps, 6);
    pass.enc.setValue(@as(u32, @intFromBool(tiled)), 7);
    pass.enc.dispatchGroups(gpu.mtl.Size.of((@as(usize, rows) + 3) / 4, 1, 1), gpu.mtl.Size.of(128, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return .{ .out = out, .flag = flag };
}

// Normalizes into the planes buffer of a dense_planes projection; tiled puts rows in the trunk's tile order.
pub fn runPlanes(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32, tiled: bool) !gpu.Tensor {
    if (tiled and x.count != 5184 * width) return error.SamNormalizationShape;
    return encode(g, x, weight, bias, width, eps, tiled);
}

// Normalizes into the 2-plane pair buffer of a dense_pair projection; tiled puts rows in tile order.
pub fn runPair(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32, tiled: bool) !gpu.Tensor {
    if (tiled and x.count != 5184 * width) return error.SamNormalizationShape;
    if (width == 0 or width > 4096 or x.count == 0 or x.count % width != 0 or
        weight.count != width or bias.count != width or x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        !std.math.isFinite(eps) or eps <= 0) return error.SamNormalizationShape;
    const count = std.math.cast(u32, x.count) orelse return error.SamNormalizationShape;
    const rows = std.math.cast(u32, x.count / width) orelse return error.SamNormalizationShape;
    const out = try g.allocDevice(@import("dense_pair.zig").pairCount(x.count), .f16);
    errdefer out.deinit();
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const pass = g.command();
    pass.enc.setPipeline(try g.pipeline("sam_norm_pair_clear"));
    pass.enc.setBuffer(out.buffer, out.offset, 0);
    pass.enc.setValue(count, 1);
    pass.enc.dispatchThreads(gpu.mtl.Size.of(8, 1, 1), gpu.mtl.Size.of(8, 1, 1));
    pass.enc.barrier();
    pass.enc.setPipeline(try g.pipeline("sam_norm_compensated_pair"));
    for ([_]gpu.Tensor{ x, weight, bias, out }, 0..) |value, i| pass.enc.setBuffer(value.buffer, value.offset, i);
    pass.enc.setValue([2]u32{ rows, @intCast(width) }, 4);
    pass.enc.setValue(eps, 5);
    pass.enc.setValue(@as(u32, @intFromBool(tiled)), 6);
    pass.enc.dispatchGroups(gpu.mtl.Size.of((@as(usize, rows) + 3) / 4, 1, 1), gpu.mtl.Size.of(128, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return out;
}

fn encode(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, width: usize, eps: f32, planes: ?bool) !gpu.Tensor {
    if (width == 0 or width > 4096 or x.count == 0 or x.count % width != 0 or
        weight.count != width or bias.count != width or x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        !std.math.isFinite(eps) or eps <= 0) return error.SamNormalizationShape;
    const count = std.math.cast(u32, x.count) orelse return error.SamNormalizationShape;
    const out = if (planes != null) try g.allocDevice(@import("dense_planes.zig").planesCount(x.count), .f16) else try g.allocDevice(x.count, .f32);
    errdefer out.deinit();
    const rows = std.math.cast(u32, x.count / width) orelse return error.SamNormalizationShape;
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const pass = g.command();
    if (planes != null) {
        pass.enc.setPipeline(try g.pipeline("sam_norm_planes_clear"));
        pass.enc.setBuffer(out.buffer, out.offset, 0);
        pass.enc.setValue(count, 1);
        pass.enc.dispatchThreads(gpu.mtl.Size.of(8, 1, 1), gpu.mtl.Size.of(8, 1, 1));
        pass.enc.barrier();
    }
    pass.enc.setPipeline(try g.pipeline(if (planes != null) "sam_norm_compensated_planes" else "sam_norm_compensated"));
    for ([_]gpu.Tensor{ x, weight, bias, out }, 0..) |value, i| pass.enc.setBuffer(value.buffer, value.offset, i);
    pass.enc.setValue([2]u32{ rows, @intCast(width) }, 4);
    pass.enc.setValue(eps, 5);
    if (planes) |tiled| pass.enc.setValue(@as(u32, @intFromBool(tiled)), 6);
    pass.enc.dispatchGroups(gpu.mtl.Size.of((@as(usize, rows) + 3) / 4, 1, 1), gpu.mtl.Size.of(128, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return out;
}
