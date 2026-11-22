//! TF_GLM_KEEP_FOLD parsing and the rank-agreement field it controls.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;
const hashBytes = ranks.hashBytes;

test "TF_GLM_KEEP_FOLD: default off, exact 0 or 1, and rank disagreement is named" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_KEEP_FOLD")) value else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(&Env.get, &problem)).keep_fold);
    Env.value = "1";
    const enabled = try Settings.read(&Env.get, &problem);
    try std.testing.expect(enabled.keep_fold);
    Env.value = "0";
    try std.testing.expect(!(try Settings.read(&Env.get, &problem)).keep_fold);
    Env.value = "yes";
    try std.testing.expectError(error.BadSetting, Settings.read(&Env.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_KEEP_FOLD: 0 or 1", problem);

    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, enabled, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
    };
    try std.testing.expectEqual(Mismatch{ .field = .keep_fold, .rank = 1 }, mismatch(&all).?);
}
