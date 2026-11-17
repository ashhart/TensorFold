const std = @import("std");
const ops = @import("ops.zig");

pub const Fourier = struct {
    matrix: ops.Tensor,
    width: usize,

    pub fn init(runtime: *ops.Runtime, key: []const u8) !Fourier {
        const geometry = try runtime.store.get(key);
        const channels = runtime.config.hidden;
        if (channels == 0 or channels % 2 != 0 or geometry.rank != 2 or
            geometry.dim(0) != 2 or geometry.dim(1) != channels / 2) return error.SamFourierShape;
        return .{ .matrix = try runtime.weight(key), .width = channels };
    }

    pub fn write(self: Fourier, coordinate: [2]f32, row: []f32) void {
        std.debug.assert(row.len == self.width);
        const bands = self.width / 2;
        const centred = [2]f32{ 2 * coordinate[0] - 1, 2 * coordinate[1] - 1 };
        const sine = row[0..bands];
        const cosine = row[bands..];
        for (sine, cosine, 0..) |*s, *c, band| {
            const projection = centred[0] * self.matrix.get(band) + centred[1] * self.matrix.get(bands + band);
            const phase = (2 * std.math.pi) * projection;
            s.* = @sin(phase);
            c.* = @cos(phase);
        }
    }
};

pub fn grid(runtime: *ops.Runtime, matrix_key: []const u8) !ops.Tensor {
    const key = try runtime.name("position.grid.{s}", .{matrix_key});
    if (runtime.constant(key)) |value| return value;
    const basis = try Fourier.init(runtime, matrix_key);
    const side = runtime.config.featureSide();
    const output = try runtime.alloc(side * side * basis.width);
    const values = output.slice(f32, output.count);
    const extent: f32 = @floatFromInt(side);
    for (0..side) |y| for (0..side) |x| {
        const coordinate = [2]f32{ (@as(f32, @floatFromInt(x)) + 0.5) / extent, (@as(f32, @floatFromInt(y)) + 0.5) / extent };
        basis.write(coordinate, values[(y * side + x) * basis.width ..][0..basis.width]);
    };
    return runtime.remember(key, output);
}

fn spatialCoordinate(cell: usize, extent: usize) f32 {
    const numerator: f32 = @floatFromInt(cell + 1);
    const denominator = @as(f32, @floatFromInt(extent)) + 1e-6;
    return numerator / denominator;
}

pub fn spatial(runtime: *ops.Runtime, height: usize, width: usize, channels: usize) !ops.Tensor {
    if (height < 2 or width < 2 or channels == 0 or channels % 4 != 0) return error.SamPositionShape;
    const key = try runtime.name("position.spatial.{d}.{d}.{d}", .{ height, width, channels });
    if (runtime.constant(key)) |value| return value;
    const output = try runtime.alloc(height * width * channels);
    const values = output.slice(f32, output.count);
    const axis_width = channels / 2;
    for (0..height) |y| for (0..width) |x| {
        const row = values[(y * width + x) * channels ..][0..channels];
        const axes = [2]f32{ spatialCoordinate(y, height), spatialCoordinate(x, width) };
        for (axes, 0..) |coordinate, axis| {
            for (0..axis_width / 2) |band| {
                const exponent = @as(f32, @floatFromInt(2 * band)) / @as(f32, @floatFromInt(axis_width));
                const phase = coordinate * (2 * std.math.pi) / std.math.pow(f32, 10000, exponent);
                row[axis * axis_width + 2 * band] = @sin(phase);
                row[axis * axis_width + 2 * band + 1] = @cos(phase);
            }
        }
    };
    return runtime.remember(key, output);
}

test "SAM spatial positions begin at the first cell and reach the normalized extent" {
    const phase = spatialCoordinate(0, 72) * (2 * std.math.pi);
    try std.testing.expectApproxEqAbs(@as(f32, 0.08715574154023614), @sin(phase), 2e-8);
    try std.testing.expectEqual(@as(f32, 0.5), spatialCoordinate(35, 72));
    try std.testing.expectEqual(@as(f32, 1), spatialCoordinate(71, 72));
    try std.testing.expectEqual(@as(f32, 1), spatialCoordinate(143, 144));
}
