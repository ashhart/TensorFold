//! Source-only developer contract build; its graph has no offline shader compiler.
const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    @import("zig/build/masked_values.zig").steps(b, target, .ReleaseSafe);
    @import("zig/build/sys.zig").wire(b);
    @import("zig/build/bf16_round.zig").wire(b, null);
}
