const std = @import("std");
const gpu = @import("tensor_metal");
const abi = @import("kernel_abi.zig");
const runtime = @import("runtime.zig");
const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const DType = runtime.DType;
const Size = gpu.mtl.Size;
const group = Size.of(256, 1, 1);

fn word(value: usize) !u32 {
    return std.math.cast(u32, value) orelse error.SamElementwiseShape;
}

fn requireF32(x: Tensor) !void {
    if (x.dtype != .f32) return error.SamElementwiseType;
}

// Equal counts combine element by element (width null); otherwise y is one row broadcast over x.
fn broadcastWidth(x_count: usize, y_count: usize) !?usize {
    if (x_count == y_count) return null;
    if (y_count == 0 or x_count % y_count != 0) return error.SamElementwiseShape;
    return y_count;
}

fn point(rt: *Runtime, x: Tensor, y: Tensor, same: u32, broadcast: u32) !Tensor {
    if (x.dtype != y.dtype) return error.SamElementwiseType;
    const width = try broadcastWidth(x.count, y.count);
    return rt.hold(try gpu.ops.point(rt.gpu, x, y, if (width == null) same else broadcast, y.count));
}

fn unary(rt: *Runtime, name: []const u8, x: Tensor, params: anytype) !Tensor {
    try requireF32(x);
    const out = try rt.deviceTensor(x.count, .f32);
    try rt.gpu.dispatch(name, &.{ x, out }, params, Size.of(x.count, 1, 1), group);
    return out;
}

pub fn add(rt: *Runtime, x: Tensor, y: Tensor) !Tensor {
    return point(rt, x, y, 0, 1);
}

pub fn mul(rt: *Runtime, x: Tensor, y: Tensor) !Tensor {
    return point(rt, x, y, 2, 3);
}

pub fn scale(rt: *Runtime, x: Tensor, factor: f32) !Tensor {
    return unary(rt, abi.scale, x, abi.Scale{ .count = try word(x.count), .factor = factor });
}

pub fn relu(rt: *Runtime, x: Tensor) !Tensor {
    return unary(rt, abi.relu, x, try word(x.count));
}

pub fn sigmoid(rt: *Runtime, x: Tensor) !Tensor {
    return unary(rt, abi.sigmoid, x, try word(x.count));
}

pub fn gelu(rt: *Runtime, x: Tensor) !Tensor {
    return switch (x.dtype) {
        .f32 => unary(rt, "im_sam_gelu", x, try word(x.count)),
        .f16 => rt.hold(try gpu.ops.gelu(rt.gpu, x, true)),
        else => error.SamElementwiseType,
    };
}

pub fn cast(rt: *Runtime, x: Tensor, dtype: DType) !Tensor {
    return rt.hold(try gpu.ops.cast(rt.gpu, x, dtype));
}

pub fn concat(rt: *Runtime, parts: []const Tensor) !Tensor {
    if (parts.len == 0) return error.SamElementwiseShape;
    var total: usize = 0;
    for (parts) |item| {
        try requireF32(item);
        total = try std.math.add(usize, total, item.count);
    }
    const out = try rt.deviceTensor(total, .f32);
    var offset: usize = 0;
    for (parts) |item| {
        const into = runtime.part(out, offset, item.count);
        try rt.gpu.dispatch(abi.copy, &.{ item, into }, try word(item.count), Size.of(item.count, 1, 1), group);
        offset += item.count;
    }
    return out;
}

pub fn columns(rt: *Runtime, x: Tensor, rows: usize, width: usize, first: usize, count: usize) !Tensor {
    try requireF32(x);
    if (rows == 0 or count == 0 or x.count != try std.math.mul(usize, rows, width) or first + count > width)
        return error.SamElementwiseShape;
    const out = try rt.deviceTensor(rows * count, .f32);
    const shape = [4]u32{ try word(rows), try word(width), try word(first), try word(count) };
    try rt.gpu.dispatch(abi.columns, &.{ x, out }, shape, Size.of(out.count, 1, 1), group);
    return out;
}

pub fn transpose(rt: *Runtime, x: Tensor, rows: usize, cols: usize) !Tensor {
    try requireF32(x);
    if (x.count != try std.math.mul(usize, rows, cols)) return error.SamElementwiseShape;
    const out = try rt.deviceTensor(x.count, .f32);
    try rt.gpu.dispatch(abi.transpose, &.{ x, out }, [2]u32{ try word(rows), try word(cols) }, Size.of(x.count, 1, 1), group);
    return out;
}

pub fn fillChannels(rt: *Runtime, x: Tensor, pixels: usize, channels: usize, first: usize, count: usize, bits: u16) !void {
    try requireF32(x);
    if (x.count != try std.math.mul(usize, pixels, channels) or first + count > channels or count > 16)
        return error.SamElementwiseShape;
    if (count == 0) return;
    const fill = abi.ChannelFill{
        .pixels = try word(pixels),
        .channels = try word(channels),
        .first = try word(first),
        .count = try word(count),
        .bits = bits,
    };
    try rt.gpu.dispatch(abi.channel_fill, &.{x}, fill, Size.of(pixels * count, 1, 1), group);
}

pub fn dedicatedCopy(rt: *Runtime, x: Tensor) !Tensor {
    try requireF32(x);
    const out = try rt.dedicatedTensor(x.count, .f32);
    try rt.gpu.dispatch(abi.copy, &.{ x, out }, try word(x.count), Size.of(x.count, 1, 1), group);
    return out;
}

pub fn hostFill(rt: *Runtime, count: usize, value: f32) !Tensor {
    const out = try rt.hostTensor(count);
    @memset(out.slice(f32, count), value);
    return out;
}

test "SAM elementwise operands combine equal counts or broadcast one dividing row" {
    try std.testing.expectEqual(@as(?usize, null), try broadcastWidth(12, 12));
    try std.testing.expectEqual(@as(?usize, 4), try broadcastWidth(12, 4));
    try std.testing.expectEqual(@as(?usize, 256), try broadcastWidth(5184 * 256, 256));
    try std.testing.expectError(error.SamElementwiseShape, broadcastWidth(12, 5));
    try std.testing.expectError(error.SamElementwiseShape, broadcastWidth(12, 0));
}
