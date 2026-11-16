//! Width 12's settings: named, or a GB10's default; ranks at 12 and 16 agree because they read the same values.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const hashBytes = ranks.hashBytes;

fn fake(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) return "4";
    if (std.mem.eql(u8, name, "TF_GLM_HEAD_BITS")) return "8";
    return null;
}

fn exact(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) return "12";
    return null;
}

test "width 12: named or a GB10's default, and ranks at 12 and 16 agree because they read the same values" {
    var problem: []const u8 = "";
    const named = try Settings.read(&exact, &problem);
    try std.testing.expect(named.dense_bits == 12 and named.head_bits == 16 and named.bits_named);
    const unset: Settings = .{};
    try std.testing.expect(!unset.bits_named);
    const gb10 = unset.exactBits(121);
    try std.testing.expect(gb10.dense_bits == 12 and gb10.head_bits == 12);
    const other = unset.exactBits(120);
    try std.testing.expect(other.dense_bits == 16 and other.head_bits == 16);
    try std.testing.expectEqual(@as(u8, 12), named.exactBits(121).dense_bits); // named bits stand
    try std.testing.expectEqual(@as(u8, 16), named.exactBits(121).head_bits);
    const lossy = try Settings.read(&fake, &problem);
    try std.testing.expect(lossy.exactBits(121).dense_bits == 4 and lossy.exactBits(121).head_bits == 8);
    var off = unset;
    off.tiled = false;
    try std.testing.expectEqual(@as(u8, 16), off.exactBits(121).dense_bits); // no glm_mm, no tiles to hold
    const h: [32]u8 = @splat(1);
    const a = Fingerprint.of(2, 0, gb10, 8192, true, h, h, hashBytes("/c"));
    const b = Fingerprint.of(2, 1, unset, 8192, true, h, h, hashBytes("/c"));
    try std.testing.expectEqual(a.bits, b.bits);
    try std.testing.expectEqual(@as(u8, 4), ranks.sameValues(4));
}
