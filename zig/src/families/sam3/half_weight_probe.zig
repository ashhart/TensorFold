const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");

pub const Probe = struct {
    weight: gpu.Tensor,

    pub fn init(g: *gpu.Gpu, file: *const st.File) !Probe {
        const source = file.get("weight") orelse return error.MissingProjectionWeight;
        if (source.dtype != .f32 or source.rank != 2) return error.HalfWeightShape;
        const weight = try g.alloc(source.numel(), .f16);
        errdefer weight.deinit();
        for (weight.slice(f16, weight.count), 0..) |*target, i| {
            const value: f32 = @bitCast(std.mem.readInt(u32, source.bytes[i * 4 ..][0..4], .little));
            target.* = @floatCast(value);
            if (!std.math.isFinite(value) or @as(u32, @bitCast(@as(f32, target.*))) != @as(u32, @bitCast(value)))
                return error.WeightNotExactlyHalf;
        }
        return .{ .weight = weight };
    }

    pub fn deinit(self: Probe) void {
        self.weight.deinit();
    }

    pub fn run(self: Probe, g: *gpu.Gpu, x: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
        return @import("dense_device.zig").run(g, x, self.weight, bias, rows, cols, depth, .nk_half);
    }
};
