//! Device admission policy for registered Metal limit launches.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");

/// The pre-M5 text of a source that includes nax.h (frags.source's branch for GPUs without tensor units).
pub fn preM5(a: std.mem.Allocator, head: []const u8, text: []const u8) ![]u8 {
    const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", sources.nax);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ head, "#define TF_SIMD_FRAGS 1\n", body });
}

pub fn hasMixed(s: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, s, part) != null;
}

pub fn supported(device: mtl.Device, d: @import("contract").limits.MetalLaunch) bool {
    if (std.mem.eql(u8, d.name, "metal inkling2 vision ik2_gemm_bf16_nt_rows")) return !device.tensorUnits();
    if (std.mem.startsWith(u8, d.name, "q27 cache ")) return device.tensorUnits();
    if (std.mem.startsWith(u8, d.name, "qwen27 prompt ")) return device.tensorUnits() == (std.mem.indexOf(u8, d.name, " m5 ") != null);
    if (std.mem.startsWith(u8, d.name, "q27 stack qk ")) return @import("metal_support.zig").tensor(device.tensorUnits(), d.name);
    if (std.mem.startsWith(u8, d.name, "lfm2 masked prompt "))
        return @import("metal_support.zig").tensor(tf.lfm2_moe.source.optionalAvailable(device), d.name);
    if (std.mem.startsWith(u8, d.name, "hy4 units "))
        return @import("metal_support.zig").tensor(device.tensorUnits(), d.name);
    if (@import("metal_dsv41_limits.zig").tensorOnly(d.name))
        return @import("metal_support.zig").tensor(device.tensorUnits(), d.name);
    if (std.mem.startsWith(u8, d.name, "lane projection ") and std.mem.indexOf(u8, d.name, " reg ") != null)
        return @import("metal_support.zig").tensor(device.tensorUnits(), d.name);
    return !std.mem.startsWith(u8, d.name, "qwen27 q8 ") or @import("metal_support.zig").tensor(device.tensorUnits(), d.name);
}
