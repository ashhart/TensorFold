//! An exact FP32 dot whose FP16 and BF16 reductions lose the four unit terms in each 128-element panel.

pub const Row = struct { left: f32, right: f32 };

pub fn pair(i: usize, n: usize) Row {
    if (i / 128 == n / 128) return .{ .left = 0, .right = 0 };
    const panel = i % 128;
    const lane = panel % 32;
    const value: f32 = switch (lane) {
        0 => 128,
        2 => 32,
        4, 5 => 64,
        else => 0,
    };
    if (value == 0) return .{ .left = 0, .right = 0 };
    return switch (panel / 32) {
        0 => .{ .left = 128, .right = value },
        1 => .{ .left = 1, .right = 1 },
        2 => .{ .left = 128, .right = -value },
        else => .{ .left = 0, .right = 0 },
    };
}
