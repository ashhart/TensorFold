//! TF_GLM_PAIR_EXPERTS in cuda_ranks: read from the environment and agreed by every rank.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;

test "TF_GLM_PAIR_EXPERTS: off by default, 0 or 1, and every rank must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_EXPERTS")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_EXPERTS")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_EXPERTS")) "2" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).pair_experts);
    const on = try Settings.read(Reader.on, &problem);
    try std.testing.expect(on.pair_experts);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).pair_experts);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PAIR_EXPERTS: 0 or 1", problem);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0)), Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(Mismatch{ .field = .pair_experts, .rank = 1 }, mismatch(&all).?);
}
