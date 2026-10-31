//! DeepSeek-V4.1's gate-5 launchOf branches, tried in the order metal_limits held them.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const limits = @import("contract").limits;
const Launch = @import("metal_limits.zig").Launch;

/// The dsv41 launch's sources, or null when the name belongs to another family.
pub fn launchOf(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch, fun: []const u8) !?Launch {
    _ = device;
    if (std.mem.startsWith(u8, d.name, "dsv41 fp8 mma ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.fp8_mma, .pre_m5 = sources.dsv41.fp8_mma };
    if (std.mem.startsWith(u8, d.name, "dsv41 fp8 rows ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.fp8_rows, .pre_m5 = sources.dsv41.fp8_rows };
    if (std.mem.startsWith(u8, d.name, "dsv41 experts ")) {
        const body = try @import("tensorfold").dsv41_metal.experts.source(a, "", &.{});
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    return null;
}
