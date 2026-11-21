//! TF_GLM_KV_SPLIT in cuda_ranks: read from the environment, carried in the fingerprint, and agreed by every rank.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const mismatch = ranks.mismatch;
const hashBytes = ranks.hashBytes;

fn splitOn(name: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, name, "TF_GLM_KV_SPLIT")) "1" else null;
}

fn splitBad(name: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, name, "TF_GLM_KV_SPLIT")) "yes" else null;
}

fn splitWord(name: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, name, "TF_GLM_KV_SPLIT")) "off" else null;
}

test "TF_GLM_KV_SPLIT: off by default, 0, off, 1 or on, refused otherwise by name, and every rank must agree" {
    var problem: []const u8 = "";
    try std.testing.expectError(error.BadSetting, Settings.read(&splitBad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_KV_SPLIT: 0, off, 1 or on", problem);
    try std.testing.expect(!(try Settings.read(&splitWord, &problem)).kv_split);
    const h: [32]u8 = @splat(0);
    const unset = struct {
        fn get(_: [:0]const u8) ?[]const u8 {
            return null;
        }
    }.get;
    const all = [_]Fingerprint{
        Fingerprint.of(2, 0, try Settings.read(&unset, &problem), 8192, true, h, h, hashBytes("/m")),
        Fingerprint.of(2, 1, try Settings.read(&splitOn, &problem), 8192, true, h, h, hashBytes("/m")),
    };
    const m = mismatch(&all) orelse return error.RanksAgree; // a rank that ignores the switch would serve unsplit
    try std.testing.expectEqualStrings("kv_split", @tagName(m.field));
    try std.testing.expectEqual(@as(usize, 1), m.rank);
}
