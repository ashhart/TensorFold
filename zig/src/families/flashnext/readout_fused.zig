//! TF_FLASHNEXT_READOUT_FUSED: a CUDA prompt pass's BF16 read-out on two fused tiles; other backends refuse it.
const std = @import("std");

const name = "TF_FLASHNEXT_READOUT_FUSED";

/// The setting as the startup check accepted it: unset, blank or 0 is off.
pub fn on() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, name, false) catch false;
}

/// A backend or read-out without the fused tiles stops the start with the setting on rather than taking the chain.
pub fn refuse(what: []const u8, fused: bool) !void {
    if (!fused) return;
    if (!@import("builtin").is_test) std.log.err("{s}=1: Flash Next on {s} has no fused read-out tiles; unset it or set 0", .{ name, what });
    return error.InvalidEnvironment;
}

test "fused read-out: on stops Metal, HIP and an affine CUDA read-out, and off starts them" {
    try std.testing.expectError(error.InvalidEnvironment, refuse("Metal", true));
    try refuse("HIP", false);
    const metal = "try @import(\"readout_fused.zig\").refuse(\"Metal\", @import(\"readout_fused.zig\").on());";
    const hip = "try @import(\"../readout_fused.zig\").refuse(\"HIP\", @import(\"../readout_fused.zig\").on());";
    const affine = "try @import(\"readout_fused.zig\").refuse(\"CUDA with an MLX affine read-out\", @import(\"readout_fused.zig\").on());";
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("engine_load.zig"), metal) != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("hip/load.zig"), hip) != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_prompt_wt.zig"), affine) != null);
}
