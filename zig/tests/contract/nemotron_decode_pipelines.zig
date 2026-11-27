//! Legacy diagnostic partials use the same generated source and per-generation build as serving.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const rig = @import("rig.zig");
pub fn partial(a: std.mem.Allocator, device: mtl.Device, tiles: usize) !mtl.Pipeline {
    var buf: [32]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "attn_partial_{d}", .{tiles});
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) {
        const text = if (device.tensorUnits() or std.mem.indexOf(u8, k.source, "partial_direct") == null) try a.dupe(u8, k.source) else try tf.nemotron.simd_attention.forGeneration(a, device.generation(), k.source);
        defer a.free(text);
        return rig.pipeline(device, text, k.function);
    };
    return error.UnknownLegacyPartial;
}
