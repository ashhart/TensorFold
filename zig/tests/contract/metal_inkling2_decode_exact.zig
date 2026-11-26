//! Exhaustive actual GPU decoder against main's serial ldexp/BF16 reference.
const std = @import("std");
const mtl = @import("metal");
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const host_plan = @import("inkling2_experts_plan");
fn hr(b: rig_.Ref) host.Ref {
    return .{ .buf = b.buf, .off = b.off };
}
fn bfBits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

pub fn check(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipes: *const host.Pipelines) !?[]const u8 {
    var r: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer r.deinit();
    var w_all: [65536]u8 = undefined;
    var s_all: [256 * 8]u8 = undefined;
    for (0..256) |row| {
        for (0..256) |col| w_all[row * 256 + col] = @intCast(col);
        for (0..8) |blk_i| s_all[row * 8 + blk_i] = @intCast(row);
    }
    const rw = try r.input(&w_all);
    const rs = try r.input(&s_all);
    const ro = try r.output(65536 * 2);
    const pass = r.begin();
    try host.mxfp8Quantize(pipes, pass.e, hr(rw), hr(rs), hr(ro), 256, 256);
    try r.finish(pass);
    const got = std.mem.bytesAsSlice(u16, try r.outputBytes());
    for (0..256) |scale| for (0..256) |val| {
        const g_bits = got[scale * 256 + val];
        const want_bits = bfBits(host_plan.mxWeight(@intCast(val), @intCast(scale)));
        const both_nan = (g_bits & 0x7fff) > 0x7f80 and (want_bits & 0x7fff) > 0x7f80;
        if (!both_nan and g_bits != want_bits) return "65536-pair (e4m3, ue8m0) GPU decode differs in BF16 bits";
    };
    return null;
}
