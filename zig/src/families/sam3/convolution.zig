const std = @import("std");
const gpu = @import("tensor_metal");
const abi = @import("kernel_abi.zig");
const runtime = @import("runtime.zig");
const dense_device = @import("dense_device.zig");
const projection = @import("projection.zig");
const elementwise = @import("elementwise.zig");
const Runtime = runtime.Runtime;
const Routes = runtime.Routes;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Size = gpu.mtl.Size;
const group = Size.of(256, 1, 1);

pub const Placement = struct { stride: usize = 1, pad: usize = 0 };

const dense_limit = 4736;
const deconv_out_limit = 1184;
const chunk_bytes = 33554432;

fn word(value: usize) !u32 {
    return std.math.cast(u32, value) orelse error.SamConvShape;
}

fn outputSide(side: usize, kernel: usize, placement: Placement) !usize {
    if (kernel == 0 or placement.stride == 0 or side + 2 * placement.pad < kernel) return error.SamConvShape;
    return (side + 2 * placement.pad - kernel) / placement.stride + 1;
}

// Output pixels per chunk keep one unfolded chunk near 32 MiB.
fn chunkPixels(width: usize) usize {
    return @min(16384, @max(64, chunk_bytes / (4 * width) / 64 * 64));
}

fn denseConv(routes: Routes, out: usize, width: usize) bool {
    return routes.exactDevice() and out <= dense_limit and width <= dense_limit;
}

fn denseDeconv(routes: Routes, out: usize, in: usize) bool {
    return routes.exactDevice() and out <= deconv_out_limit and in <= dense_limit;
}

fn packedDeconv(routes: Routes, out: usize, in: usize) bool {
    return routes.precision == .mixed or denseDeconv(routes, out, in);
}

fn shape(in_h: usize, in_w: usize, in_c: usize, out_c: usize, kernel: usize, out_h: usize, out_w: usize, placement: Placement, first: usize, rows: usize) !abi.ConvShape {
    return .{
        .in_h = try word(in_h),
        .in_w = try word(in_w),
        .in_c = try word(in_c),
        .out_c = try word(out_c),
        .kernel = try word(kernel),
        .out_h = try word(out_h),
        .out_w = try word(out_w),
        .stride = try word(placement.stride),
        .pad = try word(placement.pad),
        .first_row = try word(first),
        .row_count = try word(rows),
    };
}

fn biasOrZeros(scope: Scope, count: usize) !Tensor {
    if (scope.has("bias")) {
        const bias = try scope.param("bias");
        if (bias.count != count) return error.SamConvShape;
        return bias;
    }
    return elementwise.hostFill(scope.rt, count, 0);
}

// Packed weights are written by a kernel, so they may be created inside an open batch.
fn packedWeight(scope: Scope, prefix: []const u8, kernel: []const u8, dims: [3]usize) !Tensor {
    const rt = scope.rt;
    const name = try rt.print("{s}.{s}", .{ prefix, try scope.key("weight") });
    if (rt.constant(name)) |existing| return existing;
    const raw = try scope.param("weight");
    const count = dims[0] * dims[1] * dims[2];
    if (raw.count != count) return error.SamConvShape;
    var fresh = try rt.newConstant(count, .f32);
    errdefer fresh.deinit();
    const params = [3]u32{ try word(dims[0]), try word(dims[1]), try word(dims[2]) };
    try rt.gpu.dispatch(kernel, &.{ raw, fresh }, params, Size.of(count, 1, 1), group);
    fresh.tensor_operand_exact = raw.tensor_operand_exact;
    return rt.adoptConstant(name, fresh);
}

const Batch = struct {
    g: *gpu.Gpu,
    opened: bool,

    fn open(g: *gpu.Gpu) !Batch {
        const opened = g.batch == null;
        if (opened) try g.beginBatch();
        return .{ .g = g, .opened = opened };
    }

    fn close(self: Batch) !void {
        if (self.opened and self.g.batch != null) try self.g.endBatch();
    }

    fn abandon(self: Batch) void {
        if (self.opened and self.g.batch != null) self.g.endBatch() catch {};
    }
};

pub fn conv(s: Scope, leaf: []const u8, x: FeatureMap, placement: Placement) !FeatureMap {
    const rt = s.rt;
    const scope = try s.sub(leaf);
    const info = try scope.info("weight");
    if (info.rank != 4 or info.shape[2] != info.shape[3] or info.shape[1] != x.channels) return error.SamConvShape;
    if (x.tensor.dtype != .f32 or x.tensor.count != x.height * x.width * x.channels) return error.SamConvShape;
    const out = info.shape[0];
    const in = info.shape[1];
    const kernel = info.shape[2];
    const out_h = try outputSide(x.height, kernel, placement);
    const out_w = try outputSide(x.width, kernel, placement);
    const width = kernel * kernel * in;
    const pixels = out_h * out_w;
    const batch = try Batch.open(rt.gpu);
    errdefer batch.abandon();
    const bias = try biasOrZeros(scope, out);
    const weight = try packedWeight(scope, "conv", abi.conv_pack, .{ out, in, kernel * kernel });
    const output = try rt.deviceTensor(try std.math.mul(usize, pixels, out), .f32);
    const direct = kernel == 1 and placement.stride == 1 and placement.pad == 0;
    const use_dense = denseConv(rt.routes, out, width);
    const wide_flags: ?Tensor = if (use_dense and info.dtype != .f16) try projection.wideFlags(scope, weight, bias, out, width) else null;
    const chunk = chunkPixels(width);
    var first: usize = 0;
    while (first < pixels) : (first += chunk) {
        const rows = @min(chunk, pixels - first);
        const start = rt.mark();
        const columns = if (direct) runtime.part(x.tensor, first * in, rows * in) else blk: {
            const unfolded = try rt.deviceTensor(rows * width, .f32);
            const params = try shape(x.height, x.width, in, out, kernel, out_h, out_w, placement, first, rows);
            try rt.gpu.dispatch(abi.unfold, &.{ x.tensor, unfolded }, params, Size.of(unfolded.count, 1, 1), group);
            break :blk unfolded;
        };
        const dst = runtime.part(output, first * out, rows * out);
        if (wide_flags) |flags| {
            const layout: dense_device.Layout = if (kernel == 3) .wide_compensated else .wide;
            try dense_device.intoCached(rt.gpu, columns, weight, bias, flags, dst, rows, out, width, layout);
        } else if (use_dense) {
            const layout: dense_device.Layout = if (kernel == 3) .nk_compensated else .nk;
            try dense_device.into(rt.gpu, columns, weight, bias, dst, rows, out, width, layout);
        } else {
            try rt.gpu.matmulInto(columns, weight, dst, rows, out, width);
            const extent = [2]u32{ try word(dst.count), try word(out) };
            try rt.gpu.dispatch(abi.bias_rows, &.{ dst, bias }, extent, Size.of(dst.count, 1, 1), group);
        }
        rt.rewind(start, &.{});
    }
    try batch.close();
    return FeatureMap.of(output, out_h, out_w, out);
}

pub fn upsample2(s: Scope, leaf: []const u8, x: FeatureMap) !FeatureMap {
    const rt = s.rt;
    const scope = try s.sub(leaf);
    const info = try scope.info("weight");
    if (info.rank != 4 or info.shape[0] != x.channels or info.shape[2] != 2 or info.shape[3] != 2) return error.SamConvShape;
    if (x.tensor.dtype != .f32 or x.tensor.count != x.height * x.width * x.channels) return error.SamConvShape;
    const in = info.shape[0];
    const out = info.shape[1];
    const pixels = x.height * x.width;
    const params = try shape(x.height, x.width, in, out, 2, 2 * x.height, 2 * x.width, .{ .stride = 2 }, 0, 0);
    if (!packedDeconv(rt.routes, out, in)) {
        const raw = try scope.param("weight");
        const bias = try biasOrZeros(scope, out);
        const output = try rt.deviceTensor(try std.math.mul(usize, 4 * pixels, out), .f32);
        try rt.gpu.dispatch("im_sam_deconv", &.{ x.tensor, raw, bias, output }, params, Size.of(pixels * out, 1, 1), group);
        return FeatureMap.of(output, 2 * x.height, 2 * x.width, out);
    }
    const batch = try Batch.open(rt.gpu);
    errdefer batch.abandon();
    const bias = try biasOrZeros(scope, out);
    const weight = try packedWeight(scope, "deconv", abi.deconv_pack, .{ out, in, 4 });
    const output = try rt.deviceTensor(try std.math.mul(usize, 4 * pixels, out), .f32);
    const start = rt.mark();
    const product = if (denseDeconv(rt.routes, out, in)) blk: {
        const zeros = try elementwise.hostFill(rt, 4 * out, 0);
        if (info.dtype != .f16) {
            if (rt.observer != null) try rt.observe(try rt.print("{s}.deconv.input", .{scope.path}), x.tensor, &.{ pixels, in });
            const flags = try projection.wideFlags(scope, weight, zeros, 4 * out, in);
            break :blk try rt.hold(try dense_device.runCached(rt.gpu, x.tensor, weight, zeros, flags, pixels, 4 * out, in, .wide));
        }
        const into = try rt.deviceTensor(pixels * 4 * out, .f32);
        try dense_device.into(rt.gpu, x.tensor, weight, zeros, into, pixels, 4 * out, in, .nk);
        break :blk into;
    } else try rt.hold(try rt.gpu.matmul(x.tensor, weight, pixels, 4 * out, in));
    try rt.gpu.dispatch(abi.deconv_scatter, &.{ product, bias, output }, params, Size.of(output.count, 1, 1), group);
    rt.rewind(start, &.{});
    try batch.close();
    return FeatureMap.of(output, 2 * x.height, 2 * x.width, out);
}

pub fn depthwise(s: Scope, leaf: []const u8, x: FeatureMap, pad: usize) !FeatureMap {
    const rt = s.rt;
    const scope = try s.sub(leaf);
    const info = try scope.info("weight");
    const channels = x.channels;
    if (info.rank != 4 or info.shape[0] != channels or info.shape[1] != 1 or info.shape[2] != info.shape[3] or
        2 * pad + 1 != info.shape[2]) return error.SamConvShape;
    if (x.tensor.dtype != .f32 or x.tensor.count != x.height * x.width * channels) return error.SamConvShape;
    const kernel = info.shape[2];
    const weight = try scope.param("weight");
    const bias = try biasOrZeros(scope, channels);
    const output = try rt.deviceTensor(x.tensor.count, .f32);
    const params = try shape(x.height, x.width, channels, channels, kernel, x.height, x.width, .{ .pad = pad }, 0, 0);
    try rt.gpu.dispatch(abi.depthwise, &.{ x.tensor, weight, bias, output }, params, Size.of(output.count, 1, 1), group);
    return FeatureMap.of(output, x.height, x.width, channels);
}

test "SAM convolution output sides follow the integer rule" {
    try std.testing.expectEqual(@as(usize, 72), try outputSide(1008, 14, .{ .stride = 14 }));
    try std.testing.expectEqual(@as(usize, 576), try outputSide(1152, 3, .{ .stride = 2, .pad = 1 }));
    try std.testing.expectEqual(@as(usize, 72), try outputSide(72, 3, .{ .pad = 1 }));
    try std.testing.expectEqual(@as(usize, 288), try outputSide(288, 1, .{}));
    try std.testing.expectError(error.SamConvShape, outputSide(2, 3, .{}));
}

test "SAM convolution chunks keep the unfolded columns near 32 MiB" {
    try std.testing.expectEqual(@as(usize, 3584), chunkPixels(3 * 3 * 256));
    try std.testing.expectEqual(@as(usize, 8192), chunkPixels(1024));
    try std.testing.expectEqual(@as(usize, 14208), chunkPixels(14 * 14 * 3));
    try std.testing.expectEqual(@as(usize, 16384), chunkPixels(16));
    try std.testing.expectEqual(@as(usize, 192), chunkPixels(3 * 3 * 4736));
}

test "SAM convolution device routes stop at the kept kernel limits" {
    const device = Routes{ .precision = .fp32, .tensor_units = true };
    try std.testing.expect(denseConv(device, 4736, 4736));
    try std.testing.expect(!denseConv(device, 4737, 16));
    try std.testing.expect(!denseConv(device, 16, 4737));
    try std.testing.expect(!denseConv(.{ .precision = .fp32 }, 16, 16));
    try std.testing.expect(!denseConv(.{ .precision = .mixed, .tensor_units = true }, 16, 16));
    try std.testing.expect(packedDeconv(device, 1184, 4736));
    try std.testing.expect(!packedDeconv(device, 1185, 16));
    try std.testing.expect(!packedDeconv(device, 16, 4737));
    try std.testing.expect(!denseDeconv(.{ .precision = .mixed, .tensor_units = true }, 16, 16));
    try std.testing.expect(packedDeconv(.{ .precision = .mixed, .tensor_units = true }, 4096, 8192));
    try std.testing.expect(!packedDeconv(.{ .precision = .fp32 }, 16, 16));
}
