//! A channel's u8 value to the tower's float as each image processor computes it: one 256-entry table a channel.
const std = @import("std");

pub const Mode = enum {
    /// transformers' PIL processors: x * rescale in f64 to f32, then (x - mean) / std; ToTensor's x / 255 too
    unfused,
    /// torchvision's (transformers' fast processors): (x - f32(mean / rescale)) / f32(std / rescale) in f32
    fused,
    /// transformers' PIL processors: x * rescale in f64 to f32, then (x - mean) / std; also ToTensor's x / 255
    fused_f32,
    /// GLM5-Next's images (0.6.6): x * f32(rescale) in f32, then (x - mean) / std
    rescale_f32,
};

pub const Table = [3][256]f32;

/// Every value's float for `mode`; `rescale` is the processor's factor (1 / 255).
pub fn table(mode: Mode, rescale: f64, mean: [3]f32, sd: [3]f32) Table {
    @setEvalBranchQuota(20000);
    var t: Table = undefined;
    const by: f32 = @floatCast(1.0 / rescale);
    for (&t, 0..) |*row, ch| for (row, 0..) |*v, i| {
        const x: f32 = @floatFromInt(i);
        v.* = switch (mode) {
            .unfused => (@as(f32, @floatCast(@as(f64, @floatFromInt(i)) * rescale)) - mean[ch]) / sd[ch],
            .fused => (x - @as(f32, @floatCast(@as(f64, mean[ch]) / rescale))) / @as(f32, @floatCast(@as(f64, sd[ch]) / rescale)),
            .fused_f32 => (x - mean[ch] * by) / (sd[ch] * by),
            .rescale_f32 => (x * @as(f32, @floatCast(rescale)) - mean[ch]) / sd[ch],
        };
    };
    return t;
}
