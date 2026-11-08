//! Shared source assembly for both Gemma image checkpoint layouts.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
pub fn source(device: mtl.Device, gpa: std.mem.Allocator, out_hidden: usize) ![]u8 {
    const body = try frags.source(device, gpa, ks.gemma4_vision);
    defer gpa.free(body);
    const src = try std.fmt.allocPrint(gpa, "#define G4V_OUT {d}\n{s}", .{ out_hidden, body });
    return src;
}
