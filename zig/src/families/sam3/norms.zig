const std = @import("std");
const gpu = @import("tensor_metal");
const runtime = @import("runtime.zig");
const kernel_abi = @import("kernel_abi.zig");
const normalization = @import("normalization.zig");

const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const Routes = runtime.Routes;
const FeatureMap = runtime.FeatureMap;

pub const NormRoute = enum { plain, trunk };

const compensated_width = 1024;
const compensated_eps: f32 = 1e-5;
const group_threads = 256;

fn compensated(route: NormRoute, routes: Routes, width: usize, eps: f32) bool {
    return route == .trunk and routes.exactDevice() and width == compensated_width and eps == compensated_eps;
}

/// Layer normalisation of rows of the width of `leaf.weight`, with learned scale and bias.
pub fn layerNorm(s: Scope, leaf: []const u8, x: Tensor, eps: f32, route: NormRoute) !Tensor {
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 1) return error.SamNormShape;
    const width = shape.dim(0);
    if (width == 0 or x.count % width != 0) return error.SamNormShape;
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    if (weight.count != width or bias.count != width) return error.SamNormShape;
    const rt = s.rt;
    if (compensated(route, rt.routes, width, eps)) return rt.hold(try normalization.run(rt.gpu, x, weight, bias, width, eps));
    return rt.hold(try gpu.ops.norm(rt.gpu, x, weight, bias, width, false, eps));
}

const planes_rows = 5184;

fn planesRoute(routes: Routes, width: usize, eps: f32, count: usize, consumer_planes: bool) bool {
    return compensated(.trunk, routes, width, eps) and count == planes_rows * compensated_width and consumer_planes;
}

pub const WideNorm = struct { tensor: Tensor, flag: Tensor };

/// Trunk block norm writing FP32 + a 1-word wide-domain flag (in tile order when `tiled`), else null.
pub fn trunkWide(s: Scope, leaf: []const u8, x: Tensor, eps: f32, tiled: bool) !?WideNorm {
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 1) return error.SamNormShape;
    const width = shape.dim(0);
    if (x.dtype != .f32 or !compensated(.trunk, s.rt.routes, width, eps)) return null;
    if (tiled and x.count != planes_rows * compensated_width) return null;
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    if (weight.count != width or bias.count != width) return error.SamNormShape;
    const rt = s.rt;
    const res = try normalization.runWide(rt.gpu, x, weight, bias, width, eps, tiled);
    const out = rt.hold(res.out) catch |err| {
        res.flag.deinit();
        return err;
    };
    const flag = try rt.hold(res.flag);
    return .{ .tensor = out, .flag = flag };
}

/// Route N-planes: trunk block norm as FP16 planes for a planes consumer (tile order when `tiled`), else null.
pub fn trunkPlanes(s: Scope, leaf: []const u8, x: Tensor, eps: f32, consumer_planes: bool, tiled: bool) !?Tensor {
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 1) return error.SamNormShape;
    const width = shape.dim(0);
    if (x.dtype != .f32 or !planesRoute(s.rt.routes, width, eps, x.count, consumer_planes)) return null;
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    if (weight.count != width or bias.count != width) return error.SamNormShape;
    const rt = s.rt;
    return try rt.hold(try normalization.runPlanes(rt.gpu, x, weight, bias, width, eps, tiled));
}

/// Trunk block norm into a 2-plane pair buffer for a dense_pair consumer (tile order when `tiled`), else null.
pub fn trunkPair(s: Scope, leaf: []const u8, x: Tensor, eps: f32, tiled: bool) !?Tensor {
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 1) return error.SamNormShape;
    const width = shape.dim(0);
    if (x.dtype != .f32 or !planesRoute(s.rt.routes, width, eps, x.count, true)) return null;
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    if (weight.count != width or bias.count != width) return error.SamNormShape;
    const rt = s.rt;
    return try rt.hold(try normalization.runPair(rt.gpu, x, weight, bias, width, eps, tiled));
}

/// Group normalisation over all pixels and each group's channels, population statistics, ε = 1e-5 in the kernel.
pub fn groupNorm(s: Scope, leaf: []const u8, x: FeatureMap, groups: usize) !FeatureMap {
    const channels = x.channels;
    if (groups == 0 or channels == 0 or channels % groups != 0) return error.SamNormShape;
    const pixels = std.math.mul(usize, x.height, x.width) catch return error.SamNormShape;
    const count = std.math.mul(usize, pixels, channels) catch return error.SamNormShape;
    if (x.tensor.count != count or x.tensor.dtype != .f32) return error.SamNormShape;
    const p = try s.sub(leaf);
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    if (weight.count != channels or bias.count != channels) return error.SamNormShape;
    const constants = [3]u32{
        std.math.cast(u32, pixels) orelse return error.SamNormShape,
        std.math.cast(u32, channels) orelse return error.SamNormShape,
        std.math.cast(u32, groups) orelse return error.SamNormShape,
    };
    const rt = s.rt;
    const out = try rt.deviceTensor(count, .f32);
    const size = gpu.mtl.Size;
    try rt.gpu.dispatch(kernel_abi.group_norm, &.{ x.tensor, weight, bias, out }, constants, size.of(groups * group_threads, 1, 1), size.of(group_threads, 1, 1));
    return FeatureMap.of(out, x.height, x.width, channels);
}

test "SAM compensated layer norm only on the trunk route, FP32 with tensor units, width 1024 and eps 1e-5" {
    const on = Routes{ .precision = .fp32, .tensor_units = true };
    try std.testing.expect(compensated(.trunk, on, 1024, 1e-5));
    try std.testing.expect(!compensated(.plain, on, 1024, 1e-5));
    try std.testing.expect(!compensated(.trunk, .{ .precision = .fp32 }, 1024, 1e-5));
    try std.testing.expect(!compensated(.trunk, .{ .precision = .mixed, .tensor_units = true }, 1024, 1e-5));
    try std.testing.expect(!compensated(.trunk, on, 256, 1e-5));
    try std.testing.expect(!compensated(.trunk, on, 1024, 1e-6));
}

test "SAM trunk norm writes planes only on the compensated route, the trunk shape and a planes consumer" {
    const on = Routes{ .precision = .fp32, .tensor_units = true };
    const count = 5184 * 1024;
    try std.testing.expect(planesRoute(on, 1024, 1e-5, count, true));
    try std.testing.expect(!planesRoute(on, 1024, 1e-5, count, false));
    try std.testing.expect(!planesRoute(on, 1024, 1e-5, count - 1024, true));
    try std.testing.expect(!planesRoute(on, 1024, 1e-6, count, true));
    try std.testing.expect(!planesRoute(.{ .precision = .fp32 }, 1024, 1e-5, count, true));
    try std.testing.expect(!planesRoute(.{ .precision = .mixed, .tensor_units = true }, 1024, 1e-5, count, true));
}
