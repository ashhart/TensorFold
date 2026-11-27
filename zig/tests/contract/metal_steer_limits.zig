//! Metal limit checks for the specialized Qwen steering source.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");

/// Compare serving's Q27_XS source on this GPU and before the M5.
pub fn steerServed(a: std.mem.Allocator, device: mtl.Device, l: anytype) !usize {
    if (!std.mem.startsWith(u8, l.decl.name, "qwen27 norm steer ")) return 0;
    const q = tf.qwen27.quant_gpu;
    const xs = device.tensorUnits();
    const here = try q.specializedSource(a, try q.steerKey(contract.steer.hidden, if (xs) 64 else 0));
    defer a.free(here);
    const pre = try q.specializedSource(a, try q.steerKey(contract.steer.hidden, 0));
    defer a.free(pre);
    const same = std.mem.eql(u8, l.here, here) and std.mem.eql(u8, l.pre_m5, pre);
    std.debug.print("{s} limits {s}: the serving source here (Q27_XS {d}) and before the M5 (Q27_XS 0)\n", .{ if (same) "PASS" else "FAIL", l.decl.name, @intFromBool(xs) });
    return @intFromBool(!same);
}
