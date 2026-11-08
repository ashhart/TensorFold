//! LFM2 prompt libraries prepend their common base before the adapted attention body.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
pub fn prompt(device: mtl.Device, a: std.mem.Allocator, masked: bool) ![]u8 {
    const body = try frags.source(device, a, if (masked) sources.lfm2_prompt_masked.source else sources.lfm2_attn_nax);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ sources.lfm2, "\n", body });
}
