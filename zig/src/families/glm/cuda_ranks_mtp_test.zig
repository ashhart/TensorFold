//! MTP absorption's default-off setting, malformed-value refusal and TP fingerprint agreement.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;

fn fake(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "gather";
    return null;
}

fn enabled(name: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, name, "TF_GLM_MTP_LAST_ROW")) "1" else fake(name);
}

fn malformed(name: [:0]const u8) ?[]const u8 {
    return if (std.mem.eql(u8, name, "TF_GLM_MTP_LAST_ROW")) "yes" else null;
}

test "MTP last-row settings: default off, explicit on, malformed refused and ranks agree" {
    var problem: []const u8 = "";
    const off = try Settings.read(&fake, &problem);
    try std.testing.expect(!off.mtp_last_row);
    const on = try Settings.read(&enabled, &problem);
    try std.testing.expect(on.mtp_last_row);
    try std.testing.expectError(error.BadSetting, Settings.read(&malformed, &problem));
    try std.testing.expectEqualStrings("TF_GLM_MTP_LAST_ROW: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, off, 8192, true, h, h, h), Fingerprint.of(2, 1, on, 8192, true, h, h, h) };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .mtp, .rank = 1 }, ranks.mismatch(&all).?);
    all[0] = Fingerprint.of(2, 0, on, 8192, true, h, h, h);
    try std.testing.expect(ranks.mismatch(&all) == null);
}
