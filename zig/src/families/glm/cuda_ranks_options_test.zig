//! cuda_ranks' pair-expert agreement and wire-piece option checks.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;
const wirePieces = ranks.wirePieces;
const toolPieces = ranks.toolPieces;

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

test "wire pieces: a blank override is unset; the tools' form takes its world from TF_GLM_WORLD" {
    const Reader = struct {
        fn at(key: [:0]const u8, world: ?[]const u8, pieces: ?[]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_CHUNK_SOLO")) return "4096";
            if (std.mem.eql(u8, key, "TF_GLM_ROCE_MULTI")) return "1";
            if (std.mem.eql(u8, key, "TF_GLM_WORLD")) return world;
            if (std.mem.eql(u8, key, "TF_GLM_OVERLAP_PIECES")) return pieces;
            return null;
        }
        fn blank(key: [:0]const u8) ?[]const u8 {
            return at(key, null, " ");
        }
        fn three(key: [:0]const u8) ?[]const u8 {
            return at(key, "3", null);
        }
        fn none(key: [:0]const u8) ?[]const u8 {
            return at(key, null, null);
        }
    };
    var s: Settings = .{ .solo = true, .roce_multi = true, .pieces = 3 };
    wirePieces(&s, 3, Reader.blank);
    try std.testing.expectEqual(@as(u32, 8), s.pieces);
    s.pieces = 3;
    toolPieces(&s, Reader.three);
    try std.testing.expectEqual(@as(u32, 8), s.pieces);
    s.pieces = 3;
    toolPieces(&s, Reader.none);
    try std.testing.expectEqual(@as(u32, 3), s.pieces);
}
