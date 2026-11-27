//! GLM opt-in settings and the fields every rank must agree on before loading.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");

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
    try std.testing.expect(!(try ranks.Settings.read(Reader.absent, &problem)).pair_experts);
    const on = try ranks.Settings.read(Reader.on, &problem);
    try std.testing.expect(on.pair_experts);
    try std.testing.expect(!(try ranks.Settings.read(Reader.off, &problem)).pair_experts);
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PAIR_EXPERTS: 0 or 1", problem);
    const all = [_]ranks.Fingerprint{ ranks.Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0)), ranks.Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .pair_experts, .rank = 1 }, ranks.mismatch(&all).?);
}

test "GLM DSA columns: settings are default-off, parsed exactly and agreed before loading" {
    const Reader = struct {
        fn at(key: [:0]const u8, value: ?[]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DSA_COLUMNS")) value else null;
        }
        fn absent(key: [:0]const u8) ?[]const u8 {
            return at(key, null);
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return at(key, "1");
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return at(key, "0");
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return at(key, "yes");
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(Reader.absent, &problem)).dsa_columns);
    const on = try ranks.Settings.read(Reader.on, &problem);
    try std.testing.expect(on.dsa_columns);
    try std.testing.expect(!(try ranks.Settings.read(Reader.off, &problem)).dsa_columns);
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DSA_COLUMNS: 0 or 1", problem);
    const all = [_]ranks.Fingerprint{ ranks.Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0)), ranks.Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .dsa_columns, .rank = 1 }, ranks.mismatch(&all).?);
}
