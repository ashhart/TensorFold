//! DeepSeek-V4.1's gate-5 launchOf branches, tried in the order metal_limits held them.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const limits = @import("contract").limits;
const Launch = @import("metal_limits.zig").Launch;

/// The dsv41 launch's sources, or null when the name belongs to another family.
pub fn launchOf(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch, fun: []const u8) !?Launch {
    _ = device;
    if (std.mem.startsWith(u8, d.name, "dsv41 xmma ")) {
        const b = try tf.dsv41_metal.xmma.source(a, "", &.{});
        return .{ .decl = d, .function = fun, .here = b, .pre_m5 = b };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 fp8 mma ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.fp8_mma, .pre_m5 = sources.dsv41.fp8_mma };
    if (std.mem.startsWith(u8, d.name, "dsv41 attn ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.csa2_attn, .pre_m5 = sources.dsv41.csa2_attn };
    if (std.mem.startsWith(u8, d.name, "dsv41 experts ")) {
        const body = try tf.dsv41_metal.experts.source(a, "", &.{});
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "dsv41 glue_tmp ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.glue_tmp, .pre_m5 = sources.dsv41.glue_tmp };
    if (std.mem.startsWith(u8, d.name, "dsv41 glue ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.glue, .pre_m5 = sources.dsv41.glue };
    if (std.mem.startsWith(u8, d.name, "dsv41 mv ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.mv, .pre_m5 = sources.dsv41.mv };
    if (std.mem.startsWith(u8, d.name, "dsv41 fp8 rows ")) return .{ .decl = d, .function = fun, .here = sources.dsv41.fp8_rows, .pre_m5 = sources.dsv41.fp8_rows };
    return null;
}
