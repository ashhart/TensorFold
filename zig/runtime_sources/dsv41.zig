//! DeepSeek's complete libraries reuse the nested expert include assembler.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources").dsv41;
const Set = @import("set.zig").Set;
const xm = @import("../src/families/deepseek_v41/metal/experts_mma.zig");
pub fn add(s: *Set, device: mtl.Device) !void {
    inline for (.{ "fp8_mma", "fp8_rows", "fp8_tile", "csa2_attn", "topk512", "glue", "glue_tmp", "fused", "fold", "compress", "mv", "cand", "index", "engram", "hc" }) |name| {
        const text = @field(ks, name);
        if (comptime std.mem.eql(u8, name, "csa2_attn") or std.mem.eql(u8, name, "fused") or std.mem.eql(u8, name, "compress") or std.mem.eql(u8, name, "cand") or std.mem.eql(u8, name, "index"))
            try s.rawFrom("deepseek-v41", name, &.{ ks.segs_h, text }, text)
        else
            try s.raw("deepseek-v41", name, text);
    }
    try s.rawFrom("deepseek-v41", "experts", &.{ ks.experts_kimi, ks.experts }, try @import("../src/families/deepseek_v41/metal/experts.zig").source(s.a, "", &.{}));
    try s.rawFrom("deepseek-v41", "expert-mma", &.{ ks.experts_kimi, ks.experts, ks.xmma }, try xm.source(s.a, "", &.{}, false));
    if (expertUnits(device)) try s.rawFrom("deepseek-v41", "expert-mma-units", &.{ ks.experts_kimi, ks.experts, ks.xmma, ks.xunits }, try xm.source(s.a, "", &.{}, true));
    for ([_][2]u32{ .{ 256, 8 }, .{ 128, 4 } }) |shape|
        try s.rawFrom("deepseek-v41", "route", &.{ks.route}, try @import("../src/families/deepseek_v41/metal/pipelines.zig").routeSource(s.a, shape[0], shape[1]));
}

pub fn expertUnits(device: mtl.Device) bool {
    if (@import("builtin").target.os.tag == .macos) return xm.units(device);
    return device.tensorUnits() and device.generation() == 17;
}
