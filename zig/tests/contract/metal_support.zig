//! Tensor-only contract probes report an unchecked family and leave the supported probes running.
const std = @import("std");

pub fn tensor(available: bool, name: []const u8) bool {
    if (available) return true;
    std.debug.print("UNSUPPORTED {s}: tensor units required; coverage unchecked\n", .{name});
    return false;
}

/// Gemma's kind zero uses the SIMD-capable affine library; only its fused variants require tensor units.
pub fn gemma(available: bool, kind: u8, name: []const u8) bool {
    return kind == 0 or tensor(available, name);
}

test "Metal support: Gemma dense remains checked without tensor units, independent of names" {
    for ([_]bool{ false, true }) |available| {
        for ([_]u8{ 0, 1, 2, 3 }) |kind| {
            for ([_][]const u8{ "rounded dense", "fused" }) |name|
                try std.testing.expectEqual(available or kind == 0, gemma(available, kind, name));
        }
    }
}
