//! macOS: the speech families in the Metal engines (Drama 3 and ZONOS3), and with a test step their own targets.
const std = @import("std");

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engines: *std.Build.Module, lanes: *std.Build.Module, metal_std: []const u8, test_step: ?*std.Build.Step) void {
    @import("drama3_metal.zig").addNative(b, target, optimize, metal, engines, metal_std);
    @import("zonos3_metal.zig").addNative(b, target, optimize, metal, engines, lanes, metal_std);
    const step = test_step orelse return;
    @import("drama3_metal.zig").targets(b, target, optimize, metal, metal_std, step);
    @import("zonos3_metal.zig").targets(b, target, optimize, metal, lanes, metal_std, step);
}
