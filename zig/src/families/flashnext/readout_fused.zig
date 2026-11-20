//! TF_FLASHNEXT_READOUT_FUSED: a CUDA prompt pass's BF16 read-out on two fused tiles; other backends refuse it.
const std = @import("std");

const name = "TF_FLASHNEXT_READOUT_FUSED";

/// The setting as the startup check accepted it: unset or blank is on, 0 off.
pub fn on() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, name, true) catch true;
}

/// Whether the setting's text asks for the fused tiles: an explicit 1, never unset, blank or 0.
pub fn requested() bool {
    const raw = @import("sys").environment.rawText(name) orelse return false;
    return std.mem.trim(u8, raw, " \t\r\n").len != 0 and on();
}

/// A backend or read-out without the fused tiles stops the start when the setting asks for them.
pub fn refuse(what: []const u8, asked: bool) !void {
    if (!asked) return;
    if (!@import("builtin").is_test) std.log.err("{s}=1: Flash Next on {s} has no fused read-out tiles; unset it or set 0", .{ name, what });
    return error.InvalidEnvironment;
}

test "fused read-out: an explicit 1 stops Metal, HIP and an affine CUDA read-out, and unset or 0 starts them" {
    const env = @import("sys").env;
    _ = env.unsetenv(name);
    defer _ = env.unsetenv(name);
    try std.testing.expect(on());
    try std.testing.expect(!requested());
    try refuse("Metal", requested());
    _ = env.setenv(name, "", 1);
    try std.testing.expect(!requested());
    try refuse("HIP", requested());
    _ = env.setenv(name, " \t\r\n", 1);
    try std.testing.expect(!requested());
    _ = env.setenv(name, "0", 1);
    try std.testing.expect(!on());
    try refuse("Metal", requested());
    _ = env.setenv(name, "1", 1);
    try std.testing.expect(on());
    try std.testing.expect(requested());
    try std.testing.expectError(error.InvalidEnvironment, refuse("CUDA with an MLX affine read-out", requested()));
    _ = env.setenv(name, "2", 1);
    try std.testing.expect(on());
    const metal = "try @import(\"readout_fused.zig\").refuse(\"Metal\", @import(\"readout_fused.zig\").requested());";
    const hip = "try @import(\"../readout_fused.zig\").refuse(\"HIP\", @import(\"../readout_fused.zig\").requested());";
    const affine = "try @import(\"readout_fused.zig\").refuse(\"CUDA with an MLX affine read-out\", @import(\"readout_fused.zig\").requested());";
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("engine_load.zig"), metal) != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("hip/load.zig"), hip) != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_prompt_wt.zig"), affine) != null);
}
