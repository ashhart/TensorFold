//! EXL3 libraries built with the same codebook and warp definitions as the engine's selected dot pipelines.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const limits = @import("contract").limits;
const Launch = @import("metal_limits.zig").Launch;

pub fn launchOf(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch, fun: []const u8) !?Launch {
    if (!std.mem.startsWith(u8, d.name, "exl3-metal ")) return null;
    if (std.mem.eql(u8, fun, "tf_exl3_matmul"))
        return .{ .decl = d, .function = fun, .here = sources.core_exl3_projection, .pre_m5 = sources.core_exl3_projection, .precise = true };
    const prefix = if (std.mem.eql(u8, fun, "tf_exl3_matmul_prefill_nax31")) "" else "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n";
    const body = try tf.frags.source(device, a, sources.core_exl3_projection_nax);
    const older = try tf.frags.precision.source(a, false, sources.core_exl3_projection_nax, sources.nax, false);
    return .{
        .decl = d,
        .function = fun,
        .here = try std.mem.concat(a, u8, &.{ prefix, body }),
        .pre_m5 = try std.mem.concat(a, u8, &.{ prefix, older }),
    };
}
