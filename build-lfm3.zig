//! Standalone LFM3 checkpoint checks while native inference remains under development.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const checks = b.step("test", "LFM3 checkpoint admission without a GPU");
    @import("zig/build/lfm3_host.zig").steps(b, checks);
    b.default_step.dependOn(checks);
}
