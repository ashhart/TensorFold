//! A captured launch may overlap only with the synchronization its compiled variant expects.
const std = @import("std");

pub const Const = struct { int: ?i64 = null, f32: ?u32 = null };

/// Kernels without a PDL constexpr keep ordinary stream ordering.
pub fn validate(name: []const u8, launch_pdl: bool, consts: std.json.ArrayHashMap(Const)) !void {
    if (consts.map.get("PDL")) |c| {
        if (c.f32 == null) if (c.int) |value| {
            if (value == @intFromBool(launch_pdl)) return;
        };
    } else if (!launch_pdl) return;
    std.log.warn("aot: kernel {s}: launch pdl={} requires constexpr PDL={d}", .{
        name, launch_pdl, @intFromBool(launch_pdl),
    });
    return error.PdlMismatch;
}
