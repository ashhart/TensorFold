const std = @import("std");
const runtime = @import("runtime.zig");
const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;

// f32(2π), the constant every code below multiplies by.
const two_pi: f32 = @bitCast(@as(u32, 0x40C90FDB));
const base: f32 = 10000;

fn float(value: usize) f32 {
    return @floatFromInt(value);
}

fn coordinate(cell: usize, extent: usize) f32 {
    return float(cell + 1) / float(extent);
}

pub fn sineGrid(rt: *Runtime, height: usize, width: usize, channels: usize) !Tensor {
    if (height < 2 or width < 2 or channels == 0 or channels % 4 != 0) return error.SamEncodingShape;
    std.debug.assert(rt.gpu.batch == null);
    const name = try rt.print("sine.{d}.{d}.{d}", .{ height, width, channels });
    if (rt.constant(name)) |found| return found;
    const count = try std.math.mul(usize, try std.math.mul(usize, height, width), channels);
    const host = try rt.hostTensor(count);
    fillSine(host.slice(f32, count), height, width, channels);
    return rt.keepConstant(name, host);
}

fn fillSine(out: []f32, height: usize, width: usize, channels: usize) void {
    const half = channels / 2;
    for (0..height) |y| for (0..width) |x| {
        const row = out[(y * width + x) * channels ..][0..channels];
        sinePairs(row[0..half], coordinate(y, height));
        sinePairs(row[half..], coordinate(x, width));
    };
}

fn sinePairs(out: []f32, value: f32) void {
    for (0..out.len / 2) |k| {
        const exponent = float(2 * k) / float(out.len);
        const phase = (value * two_pi) / std.math.pow(f32, base, exponent);
        out[2 * k] = @sin(phase);
        out[2 * k + 1] = @cos(phase);
    }
}

pub const FourierBasis = struct {
    matrix: Tensor,
    width: usize,

    pub fn load(rt: *Runtime, key: []const u8) !FourierBasis {
        std.debug.assert(rt.gpu.batch == null);
        const shape = try rt.info(key);
        if (shape.rank != 2 or shape.dim(0) != 2 or shape.dim(1) == 0) return error.SamEncodingShape;
        return .{ .matrix = try rt.param(key), .width = 2 * shape.dim(1) };
    }

    pub fn encode(self: FourierBasis, point: [2]f32, out: []f32) void {
        std.debug.assert(out.len == self.width and self.matrix.count == self.width);
        fourier(self.matrix.slice(f32, self.width), point, out);
    }
};

fn fourier(matrix: []const f32, point: [2]f32, out: []f32) void {
    const half = out.len / 2;
    const ux = 2 * point[0] - 1;
    const uy = 2 * point[1] - 1;
    for (0..half) |j| {
        const t = ux * matrix[j] + uy * matrix[half + j];
        const phase = two_pi * t;
        out[j] = @sin(phase);
        out[half + j] = @cos(phase);
    }
}

pub fn fourierGrid(rt: *Runtime, matrix_key: []const u8) !Tensor {
    std.debug.assert(rt.gpu.batch == null);
    const name = try rt.print("fourier.{s}", .{matrix_key});
    if (rt.constant(name)) |found| return found;
    const basis = try FourierBasis.load(rt, matrix_key);
    const side = rt.config.featureSide();
    const count = try std.math.mul(usize, side * side, basis.width);
    const host = try rt.hostTensor(count);
    const values = host.slice(f32, count);
    for (0..side) |y| for (0..side) |x| {
        const point = [2]f32{ (float(x) + 0.5) / float(side), (float(y) + 0.5) / float(side) };
        basis.encode(point, values[(y * side + x) * basis.width ..][0..basis.width]);
    };
    return rt.keepConstant(name, host);
}

pub fn timeCode(out: []f32, offset: usize, horizon: usize) void {
    std.debug.assert(out.len % 2 == 0);
    const half = out.len / 2;
    const divisor: usize = if (horizon >= 2) horizon - 1 else 1;
    const t = float(offset) / float(divisor);
    for (0..half) |i| {
        const exponent = float(2 * (i / 2)) / float(half);
        const phase = t / std.math.pow(f32, base, exponent);
        out[i] = @sin(phase);
        out[half + i] = @cos(phase);
    }
}

pub fn boxCode(rt: *Runtime, boxes: Tensor) !Tensor {
    std.debug.assert(rt.gpu.batch == null);
    if (boxes.dtype != .f32 or boxes.count == 0 or boxes.count % 4 != 0) return error.SamEncodingShape;
    const width = rt.config.hidden / 2;
    const count = try std.math.mul(usize, boxes.count, width);
    const host = try rt.hostTensor(count);
    boxRows(boxes.slice(f32, boxes.count), width, host.slice(f32, count));
    return host;
}

// Rows per box follow (cy, cx, w, h) while boxes are stored as (cx, cy, w, h).
const box_order = [4]usize{ 1, 0, 2, 3 };

fn boxRows(boxes: []const f32, width: usize, out: []f32) void {
    for (0..boxes.len / 4) |b| for (box_order, 0..) |source, m| {
        const row = out[(4 * b + m) * width ..][0..width];
        const rho = boxes[4 * b + source] * two_pi;
        for (0..width / 2) |k| {
            const exponent = float(2 * k) / float(width);
            const phase = rho / std.math.pow(f32, base, exponent);
            row[2 * k] = @sin(phase);
            row[2 * k + 1] = @cos(phase);
        }
    };
}

const testing = std.testing;

test "SAM sine grid coordinates are exact at quarter cells and the last cell" {
    try testing.expectEqual(@as(f32, 0.25), coordinate(17, 72));
    try testing.expectEqual(@as(f32, 0.75), coordinate(53, 72));
    try testing.expectEqual(@as(f32, 1), coordinate(71, 72));
    try testing.expectEqual(@as(f32, 1), coordinate(287, 288));
    var grid: [72 * 2 * 8]f32 = undefined;
    fillSine(&grid, 72, 2, 8);
    const row = grid[(17 * 2 + 1) * 8 ..][0..8];
    try testing.expectApproxEqAbs(@as(f32, 1), row[0], 0x1p-24);
    try testing.expectEqual(@cos(@as(f32, 0.25) * two_pi), row[1]);
    const phase = (@as(f32, 0.25) * two_pi) / std.math.pow(f32, base, @as(f32, 2) / 4);
    try testing.expectEqual(@sin(phase), row[2]);
    try testing.expectEqual(@cos(phase), row[3]);
    try testing.expectEqual(@sin(@as(f32, 1) * two_pi), row[4]);
    try testing.expectEqual(@cos(@as(f32, 1) * two_pi), row[5]);
}

test "SAM Fourier code matches an FP64 evaluation" {
    const matrix = [4]f32{ 0.1, -0.05, 0.03, 0.12 };
    const points = [_][2]f32{ .{ 0.3, 0.7 }, .{ 0.9, 0.2 }, .{ 0.5, 0.5 } };
    for (points) |point| {
        var out: [4]f32 = undefined;
        fourier(&matrix, point, &out);
        for (0..2) |j| {
            const ux = 2 * @as(f64, point[0]) - 1;
            const uy = 2 * @as(f64, point[1]) - 1;
            const phase = 2 * std.math.pi * (ux * matrix[j] + uy * matrix[2 + j]);
            try testing.expectApproxEqAbs(@sin(phase), @as(f64, out[j]), 4 * 0x1p-23);
            try testing.expectApproxEqAbs(@cos(phase), @as(f64, out[2 + j]), 4 * 0x1p-23);
        }
    }
}

test "SAM time code places sines before cosines over the horizon" {
    var out: [256]f32 = undefined;
    timeCode(&out, 0, 16);
    for (out[0..128]) |value| try testing.expectEqual(@as(f32, 0), value);
    for (out[128..]) |value| try testing.expectEqual(@as(f32, 1), value);
    timeCode(&out, 15, 16);
    try testing.expectEqual(@sin(@as(f32, 1)), out[0]);
    try testing.expectEqual(@sin(@as(f32, 1)), out[1]);
    try testing.expectEqual(@cos(@as(f32, 1)), out[128]);
    try testing.expectEqual(@sin(@as(f32, 1) / std.math.pow(f32, base, @as(f32, 2) / 128)), out[2]);
    var single: [256]f32 = undefined;
    for ([_]usize{ 0, 1 }) |horizon| {
        timeCode(&single, 3, horizon);
        try testing.expectEqual(@sin(@as(f32, 3)), single[0]);
        try testing.expectEqual(@cos(@as(f32, 3)), single[128]);
    }
}

test "SAM box code rows follow cy, cx, w, h and match FP64" {
    const boxes = [_]f32{ 0.5, 0.25, 0.125, 0.375, 0.0625, 0.9, 0.3, 0.2 };
    var out: [8 * 128]f32 = undefined;
    boxRows(&boxes, 128, &out);
    for (0..2) |b| for (box_order, 0..) |source, m| {
        const row = out[(4 * b + m) * 128 ..][0..128];
        try testing.expectEqual(@sin(boxes[4 * b + source] * two_pi), row[0]);
        for (0..64) |k| {
            const exponent = @as(f64, @floatFromInt(2 * k)) / 128;
            const phase = @as(f64, boxes[4 * b + source]) * 2 * std.math.pi / std.math.pow(f64, 10000, exponent);
            try testing.expectApproxEqAbs(@sin(phase), @as(f64, row[2 * k]), 0x1p-20);
            try testing.expectApproxEqAbs(@cos(phase), @as(f64, row[2 * k + 1]), 0x1p-20);
        }
    };
}
