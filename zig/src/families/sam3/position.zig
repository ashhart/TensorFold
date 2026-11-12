const std = @import("std");
const ops = @import("ops.zig");

pub const Fourier = struct {
    matrix: ops.Tensor,
    width: usize,

    pub fn init(r: *ops.Runtime, key: []const u8) !Fourier {
        const matrix = try r.weight(key);
        const width = r.config.hidden;
        if (width % 2 != 0 or matrix.count != width) return error.SamFourierShape;
        return .{ .matrix = matrix, .width = width };
    }

    pub fn write(self: Fourier, point: [2]f32, target: []f32) void {
        std.debug.assert(target.len == self.width);
        const count = self.width / 2;
        const x = point[0] * 2 - 1;
        const y = point[1] * 2 - 1;
        for (0..count) |frequency| {
            const angle = (2 * std.math.pi) * (x * self.matrix.get(frequency) + y * self.matrix.get(count + frequency));
            target[frequency] = @sin(angle);
            target[count + frequency] = @cos(angle);
        }
    }
};

pub fn grid(r: *ops.Runtime, key: []const u8) !ops.Tensor {
    const cached_key = try r.name("position.grid.{s}", .{key});
    if (r.constant(cached_key)) |cached| return cached;
    const mapping = try Fourier.init(r, key);
    const side = r.config.featureSide();
    const result = try r.alloc(side * side * mapping.width);
    const step: f32 = @floatFromInt(side);
    for (0..side * side) |pixel| {
        const xy = [2]f32{ (@as(f32, @floatFromInt(pixel % side)) + 0.5) / step, (@as(f32, @floatFromInt(pixel / side)) + 0.5) / step };
        mapping.write(xy, result.slice(f32, result.count)[pixel * mapping.width ..][0..mapping.width]);
    }
    return r.remember(cached_key, result);
}

pub fn spatial(r: *ops.Runtime, height: usize, width: usize, channels: usize) !ops.Tensor {
    if (height < 2 or width < 2 or channels == 0 or channels % 4 != 0) return error.SamPositionShape;
    const key = try r.name("position.spatial.{d}.{d}.{d}", .{ height, width, channels });
    if (r.constant(key)) |cached| return cached;
    const output = try r.alloc(height * width * channels);
    const half = channels / 2;
    for (0..height * width) |pixel| {
        const coordinates = [2]f32{
            @as(f32, @floatFromInt(pixel / width)) / (@as(f32, @floatFromInt(height - 1)) + 1e-6),
            @as(f32, @floatFromInt(pixel % width)) / (@as(f32, @floatFromInt(width - 1)) + 1e-6),
        };
        for (0..channels) |channel| {
            const exponent = @as(f32, @floatFromInt((channel % half) / 2 * 2)) / @as(f32, @floatFromInt(half));
            const phase = coordinates[channel / half] * (2 * std.math.pi) / std.math.pow(f32, 10000, exponent);
            output.slice(f32, output.count)[pixel * channels + channel] = if (channel % 2 == 0) @sin(phase) else @cos(phase);
        }
    }
    return r.remember(key, output);
}
