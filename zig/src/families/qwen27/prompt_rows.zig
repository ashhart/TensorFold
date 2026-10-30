//! Prompt frame capacity follows the chip and an optional validated row count.
const std = @import("std");

pub fn requested(generation: u32, value: ?[]const u8) u32 {
    const default: u32 = if (generation == 17) 2048 else 512;
    return override(value) orelse default;
}

/// Only validated values override the memory-aware default.
pub fn override(value: ?[]const u8) ?u32 {
    const text = value orelse return null;
    const n = std.fmt.parseInt(u32, text, 10) catch return null;
    return if (n >= 16 and n <= @import("gpu_frame.zig").max_rows and n % 16 == 0) n else null;
}
