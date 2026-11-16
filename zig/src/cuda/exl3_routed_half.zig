//! Routed EXL3 rows whose checkpoint rounds each FP16 input-times-scale product before its FP32 rotation.
const std = @import("std");
const launch_ = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;

pub const Shape = struct {
    pairs: usize,
    width: usize,
    input_slots: usize,
    experts: usize,

    pub fn validate(s: Shape) !void {
        if (s.width == 0 or s.width % 128 != 0 or s.experts == 0) return error.Exl3Shape;
        for ([_]usize{ s.pairs, s.width, s.input_slots, s.experts }) |n|
            if (n > std.math.maxInt(i32)) return error.Exl3Shape;
        _ = try std.math.mul(usize, s.pairs, s.width);
    }
};

/// Invalid picks leave their output unchanged; input_slots zero selects one input row per routed pair.
pub fn run(f: Function, stream: Stream, x: u64, picks: u64, suh: u64, out: u64, shape: Shape) !void {
    try shape.validate();
    if (shape.pairs == 0) return;
    var args: launch_.Args = .{};
    inline for (.{ x, picks, suh, out }) |p| args.add(p);
    inline for (.{ shape.width, shape.input_slots, shape.experts }) |n| args.add(@as(i32, @intCast(n)));
    try launch_.launch(f, .{ .grid = .{ .x = @intCast(shape.pairs), .y = @intCast(shape.width / 128) }, .block = .{ .x = 32 } }, stream, &args);
}

test "routed half rotation rejects ragged or overflowing geometry before launch" {
    try (Shape{ .pairs = 2048, .width = 4608, .input_slots = 8, .experts = 192 }).validate();
    try (Shape{ .pairs = 4, .width = 2304, .input_slots = 0, .experts = 384 }).validate();
    for ([_]usize{ 0, 1, 127, 129 }) |width|
        try std.testing.expectError(error.Exl3Shape, (Shape{ .pairs = 8, .width = width, .input_slots = 8, .experts = 192 }).validate());
    try std.testing.expectError(error.Exl3Shape, (Shape{ .pairs = 8, .width = 128, .input_slots = 8, .experts = 0 }).validate());
    try std.testing.expectError(error.Exl3Shape, (Shape{ .pairs = @as(usize, std.math.maxInt(i32)) + 1, .width = 128, .input_slots = 8, .experts = 1 }).validate());
}
