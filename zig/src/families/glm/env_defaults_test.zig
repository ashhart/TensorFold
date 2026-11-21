//! The documented TF_GLM_* defaults a CUDA rank reads from the environment, pinned without a device.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");

fn absent(_: [:0]const u8) ?[]const u8 {
    return null;
}

fn denseZero(key: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, key, "TF_GLM_DENSE_BITS")) "0" else null;
}

fn denseFive(key: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, key, "TF_GLM_DENSE_BITS")) "5" else null;
}

test "env defaults: TF_GLM_PREFILL_REDUCE unset splits each rank's own rows without a ring" {
    var problem: []const u8 = "";
    const s = try ranks.Settings.read(&absent, &problem);
    try std.testing.expect(s.split and !s.ring);
}

test "env defaults: TF_GLM_DENSE_BITS unset keeps bf16, 0 spells bf16, and 5 is refused by name" {
    var problem: []const u8 = "";
    try std.testing.expectEqual(@as(u8, 16), (try ranks.Settings.read(&absent, &problem)).dense_bits);
    try std.testing.expectEqual(@as(u8, 16), (try ranks.Settings.read(&denseZero, &problem)).dense_bits);
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(&denseFive, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DENSE_BITS: 4, 6, 8 or 16", problem);
}
