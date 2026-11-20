const std = @import("std");
const gpu = @import("tensor_metal");
const abi = @import("kernel_abi.zig");
const runtime = @import("runtime.zig");
const prompts = @import("prompts.zig");
const mask = @import("mask.zig");
const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Size = gpu.mtl.Size;

pub const Filter = enum { bilinear, nearest };
pub const TileDirection = enum { into_tiles, from_tiles };

fn word(value: usize) !u32 {
    return std.math.cast(u32, value) orelse error.SamSpatialShape;
}

pub fn resample(rt: *Runtime, x: FeatureMap, height: usize, width: usize, filter: Filter) !FeatureMap {
    if (x.tensor.dtype != .f32) return error.SamSpatialType;
    if (height == 0 or width == 0 or x.height == 0 or x.width == 0 or x.channels == 0 or
        x.tensor.count != try std.math.mul(usize, try std.math.mul(usize, x.height, x.width), x.channels))
        return error.SamSpatialShape;
    const count = try std.math.mul(usize, try std.math.mul(usize, height, width), x.channels);
    const shape = [4]u32{ try word(x.height), try word(x.width), try word(x.channels), @intFromBool(filter == .bilinear) };
    const size = [2]u32{ try word(height), try word(width) };
    const ratio = [2]f32{
        @as(f32, @floatFromInt(x.height)) / @as(f32, @floatFromInt(height)),
        @as(f32, @floatFromInt(x.width)) / @as(f32, @floatFromInt(width)),
    };
    const out = try rt.deviceTensor(count, .f32);
    const g = rt.gpu;
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    const pipe = try g.pipeline(abi.resample);
    const pass = g.command();
    pass.enc.setPipeline(pipe);
    pass.enc.setBuffer(x.tensor.buffer, x.tensor.offset, 0);
    pass.enc.setBuffer(out.buffer, out.offset, 1);
    pass.enc.setValue(shape, 2);
    pass.enc.setValue(size, 3);
    pass.enc.setValue(ratio, 4);
    pass.enc.dispatchThreads(Size.of(count, 1, 1), Size.of(256, 1, 1));
    try g.finish(pass.cb, pass.enc);
    return FeatureMap.of(out, height, width, x.channels);
}

pub fn tiles(rt: *Runtime, x: Tensor, channels: usize, direction: TileDirection) !Tensor {
    const grid = rt.config.featureSide();
    const tile = rt.config.tile_side;
    if (x.dtype != .f32) return error.SamSpatialType;
    if (channels == 0 or tile == 0 or grid % tile != 0 or x.count != try std.math.mul(usize, grid * grid, channels))
        return error.SamSpatialShape;
    const out = try rt.deviceTensor(x.count, .f32);
    const params = abi.Tiles{
        .channels = try word(channels),
        .grid = try word(grid),
        .tile = try word(tile),
        .direction = @intFromBool(direction == .from_tiles),
    };
    try rt.gpu.dispatch(abi.tiles, &.{ x, out }, params, Size.of(x.count, 1, 1), Size.of(256, 1, 1));
    return out;
}

// Tiles are enumerated column by column; rows inside a tile are row-major because rotary positions use them.
pub fn gridRow(grid: usize, tile: usize, tiled_row: usize) usize {
    const across = grid / tile;
    const area = tile * tile;
    const index = tiled_row / area;
    const local = tiled_row % area;
    const tx = index / across;
    const ty = index % across;
    return (ty * tile + local / tile) * grid + tx * tile + local % tile;
}

fn tiledRow(grid: usize, tile: usize, grid_row: usize) usize {
    const gy = grid_row / grid;
    const gx = grid_row % grid;
    const index = gx / tile * (grid / tile) + gy / tile;
    return index * tile * tile + gy % tile * tile + gx % tile;
}

pub fn sourceMask(rt: *Runtime, allocator: std.mem.Allocator, logits: FeatureMap, size: prompts.Size, score: f32) !mask.Mask {
    std.debug.assert(rt.gpu.batch == null);
    if (logits.channels != 1) return error.SamSpatialShape;
    const pixels = try size.pixels();
    const start = rt.mark();
    defer rt.rewind(start, &.{});
    const resampled = try resample(rt, logits, size.height, size.width, .bilinear);
    return mask.fromLogits(allocator, resampled.tensor.slice(f32, pixels), size, score);
}

/// Host antialiased shrink of a one-channel map: a triangle filter stretched by the scale, columns then rows.
pub fn shrinkAntialiased(allocator: std.mem.Allocator, x: []const f32, height: usize, width: usize, out_height: usize, out_width: usize, out: []f32) !void {
    if (x.len != height * width or out.len != out_height * out_width or out_height == 0 or out_width == 0 or
        out_height > height or out_width > width) return error.SamSpatialShape;
    const rows = try allocator.alloc(f32, height * out_width);
    defer allocator.free(rows);
    for (0..out_width) |j| {
        const span = triangle(j, width, out_width);
        for (0..height) |i| {
            var sum: f32 = 0;
            for (span.first..span.end) |k| sum += x[i * width + k] * triangleWeight(k, span);
            rows[i * out_width + j] = sum;
        }
    }
    for (0..out_height) |i| {
        const span = triangle(i, height, out_height);
        for (0..out_width) |j| {
            var sum: f32 = 0;
            for (span.first..span.end) |k| sum += rows[k * out_width + j] * triangleWeight(k, span);
            out[i * out_width + j] = sum;
        }
    }
}

const Span = struct { first: usize, end: usize, centre: f32, scale: f32, total: f32 };

fn triangle(dst: usize, in: usize, out: usize) Span {
    const scale = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
    const centre = scale * (@as(f32, @floatFromInt(dst)) + 0.5);
    const first: usize = @intFromFloat(@max(0, @floor(centre - scale + 0.5)));
    const end: usize = @intFromFloat(@min(@as(f32, @floatFromInt(in)), @floor(centre + scale + 0.5)));
    var span = Span{ .first = first, .end = end, .centre = centre, .scale = scale, .total = 0 };
    for (first..end) |k| span.total += triangleRaw(k, span);
    return span;
}

fn triangleRaw(k: usize, span: Span) f32 {
    return @max(0, 1 - @abs((@as(f32, @floatFromInt(k)) - span.centre + 0.5) / span.scale));
}

fn triangleWeight(k: usize, span: Span) f32 {
    return triangleRaw(k, span) / span.total;
}

const Tap = struct { first: usize, second: usize, fraction: f32 };

fn bilinearTap(dst: usize, in: usize, out: usize) Tap {
    const ratio = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
    const source = @max(0, @mulAdd(f32, @as(f32, @floatFromInt(dst)) + 0.5, ratio, -0.5));
    const first = @min(@as(usize, @intFromFloat(source)), in - 1);
    return .{ .first = first, .second = @min(first + 1, in - 1), .fraction = source - @as(f32, @floatFromInt(first)) };
}

fn nearestTap(dst: usize, in: usize, out: usize) usize {
    return dst * in / out;
}

test "SAM tile map is a bijection with the stated inverse and column-by-column tile order" {
    var seen: [5184]bool = @splat(false);
    for (0..5184) |row| {
        const g = gridRow(72, 24, row);
        try std.testing.expect(g < 5184 and !seen[g]);
        seen[g] = true;
        try std.testing.expectEqual(row, tiledRow(72, 24, g));
    }
    try std.testing.expectEqual(@as(usize, 0), gridRow(72, 24, 0));
    try std.testing.expectEqual(@as(usize, 1), gridRow(72, 24, 1));
    try std.testing.expectEqual(@as(usize, 72), gridRow(72, 24, 24));
    try std.testing.expectEqual(@as(usize, 73), gridRow(72, 24, 25));
    try std.testing.expectEqual(@as(usize, 1728), gridRow(72, 24, 576));
    try std.testing.expectEqual(@as(usize, 24), gridRow(72, 24, 3 * 576));
}

test "SAM antialiased shrink by four uses eight taps, renormalised at the edges" {
    const span = triangle(5, 1008, 252);
    try std.testing.expectEqual(@as(usize, 18), span.first);
    try std.testing.expectEqual(@as(usize, 26), span.end);
    try std.testing.expectEqual(@as(f32, 0.125 / 4.0), triangleWeight(18, span));
    try std.testing.expectEqual(@as(f32, 0.875 / 4.0), triangleWeight(21, span));
    const edge = triangle(0, 1008, 252);
    try std.testing.expectEqual(@as(usize, 0), edge.first);
    try std.testing.expectEqual(@as(f32, 3.5), edge.total);
    var x: [64]f32 = @splat(-10);
    var out: [4]f32 = undefined;
    try shrinkAntialiased(std.testing.allocator, &x, 8, 8, 2, 2, &out);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, -10), v, 1e-5);
}

test "SAM resample coordinates follow the centre rule and the floor rule" {
    const cases = [_]struct { dst: usize, tap: Tap }{
        .{ .dst = 0, .tap = .{ .first = 0, .second = 1, .fraction = 0 } },
        .{ .dst = 1, .tap = .{ .first = 0, .second = 1, .fraction = 0 } },
        .{ .dst = 2, .tap = .{ .first = 0, .second = 1, .fraction = 0.125 } },
        .{ .dst = 1151, .tap = .{ .first = 287, .second = 287, .fraction = 0.375 } },
    };
    for (cases) |case| try std.testing.expectEqual(case.tap, bilinearTap(case.dst, 288, 1152));
    for (0..288) |dst| try std.testing.expectEqual(dst / 4, nearestTap(dst, 72, 288));
}
