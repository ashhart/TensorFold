//! Gate 5's sources for the SAM3 kernels registered by the layout, arithmetic, rotary, rows, flash and trunk suites.
const std = @import("std");

const Owner = struct { prefix: []const u8, tensor: bool, file: type };

const owners = [_]Owner{
    .{ .prefix = "sam3 layout ", .tensor = false, .file = @import("metal_sam3_layout.zig") },
    .{ .prefix = "sam3 arithmetic ", .tensor = false, .file = @import("metal_sam3_arithmetic.zig") },
    .{ .prefix = "sam3 rotary ", .tensor = false, .file = @import("metal_sam3_rotary.zig") },
    .{ .prefix = "sam3 rows attention ", .tensor = false, .file = @import("metal_sam3_rows.zig") },
    .{ .prefix = "sam3 flash ", .tensor = true, .file = @import("metal_sam3_flash.zig") },
    .{ .prefix = "sam3 trunk tile ", .tensor = true, .file = @import("metal_sam3_trunk.zig") },
    .{ .prefix = "sam3 tensor ops ", .tensor = true, .file = @import("metal_sam3_tensor_ops.zig") },
    .{ .prefix = "sam3 core attention ", .tensor = true, .file = @import("metal_sam3_core_attention.zig") },
};

pub fn owns(name: []const u8) bool {
    inline for (owners) |o| if (std.mem.startsWith(u8, name, o.prefix)) return true;
    return false;
}

/// Tensor-op launches build only where the engine's SAM library does (the M5's tensor units).
pub fn tensorOnly(name: []const u8) bool {
    inline for (owners) |o| if (o.tensor and std.mem.startsWith(u8, name, o.prefix)) return true;
    return false;
}

/// The source a record's pipeline compiles here, and before the M5 (empty: tensor-op source, no M1/M2 build).
pub fn text(a: std.mem.Allocator, name: []const u8) ![2][]const u8 {
    const control = std.mem.indexOf(u8, name, " control ") != null;
    inline for (owners) |o| if (std.mem.startsWith(u8, name, o.prefix)) {
        const here = try o.file.limitText(a, name, control);
        return .{ here, if (o.tensor) "" else here };
    };
    return error.UnknownLaunch;
}
