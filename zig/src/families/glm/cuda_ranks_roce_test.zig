//! TF_ROCE and TF_GLM_DRAFT_SPLIT: the RoCE gather and the drafter split at three ranks or more, and how ranks agree.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;
const hashBytes = ranks.hashBytes;
const Fabric = @import("cuda_ranks_test.zig").Fabric;

test "TF_ROCE: two ranks default on and 0 turns them off; three or more need 1" {
    const roce = cuda.roce;
    for ([_]?[]const u8{ null, "1", "yes" }) |v| try std.testing.expect(roce.asked(2, v));
    try std.testing.expect(!roce.asked(2, "0"));
    for ([_]usize{ 3, 4, 8 }) |world| {
        for ([_]?[]const u8{ null, "0", "yes", "" }) |v| try std.testing.expect(!roce.asked(world, v));
        try std.testing.expect(roce.asked(world, "1"));
    }
}

test "TF_ROCE=1 is in the ranks' agreement: one rank with it and one without refuse each other" {
    var all = [_]Fingerprint{
        Fingerprint.of(4, 0, .{ .roce_wide = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(4, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(4, 2, .{ .roce_wide = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(4, 3, .{ .roce_wide = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
    };
    try std.testing.expectEqual(Mismatch{ .field = .roce_wide, .rank = 1 }, mismatch(&all).?);
}

test "TF_ROCE reads into the settings: only 1 asks the wide path" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_ROCE")) value else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "0", "yes" }) |v| {
        Env.value = v;
        try std.testing.expect(!(try Settings.read(&Env.get, &problem)).roce_wide);
    }
    Env.value = "1";
    try std.testing.expect((try Settings.read(&Env.get, &problem)).roce_wide);
}

test "TF_GLM_ROCE_MULTI=1 leaves the group's own RoCE path alone at two ranks and takes it over from three" {
    try std.testing.expect(ranks.ownRoce(.{ .roce_multi = true }, 2)); // the wire opens at three ranks or more only
    for ([_]usize{ 3, 4 }) |world| try std.testing.expect(!ranks.ownRoce(.{ .roce_multi = true }, world));
    for ([_]usize{ 2, 3, 4 }) |world| try std.testing.expect(ranks.ownRoce(.{}, world));
}

test "two ranks agree whatever TF_ROCE says: only three or more compare it" {
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{ .roce_wide = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
    };
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
}

test "RoCE all-gather on this fabric: a mesh of four reaches every rank; a ring of four leaves the far rank out" {
    const Reach = struct {
        fn reaches(_: void, rank: usize, peer: usize) bool {
            Fabric.rank = rank;
            return Fabric.reaches(@intCast(peer));
        }
    };
    // behind a switch every rank is a peer's one RDMA write away: the RoCE path (Group.openRoce) comes up on all four
    Fabric.world = 4;
    Fabric.switched = true;
    for (0..4) |r| try std.testing.expectEqual(@as(?usize, null), cuda.roce.firstUnreachable(r, 4, {}, &Reach.reaches));
    // a ring of four without a switch: each far rank has no path, so the probe fails and every rank keeps NCCL
    Fabric.switched = false;
    for (0..4) |r| {
        const far: ?usize = (r + 2) % 4;
        try std.testing.expectEqual(far, cuda.roce.firstUnreachable(r, 4, {}, &Reach.reaches));
    }
    // two ranks on one cable are each other's neighbours: the two-rank path as before
    Fabric.world = 2;
    for (0..2) |r| try std.testing.expectEqual(@as(?usize, null), cuda.roce.firstUnreachable(r, 2, {}, &Reach.reaches));
}

test "TF_GLM_DRAFT_SPLIT: off by default, 1 or 0, refused otherwise" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DRAFT_SPLIT")) value else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "0" }) |v| {
        Env.value = v;
        try std.testing.expect(!(try Settings.read(&Env.get, &problem)).draft_split);
    }
    Env.value = "1";
    const on = try Settings.read(&Env.get, &problem);
    try std.testing.expect(on.draft_split and !on.roce_wide);
    Env.value = "yes";
    try std.testing.expectError(error.BadSetting, Settings.read(&Env.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DRAFT_SPLIT: 0 or 1", problem);
}

test "three ranks or more split the drafter MLP with TF_ROCE=1 or TF_GLM_DRAFT_SPLIT=1, where shares are 256-aligned" {
    const df = @import("dflash.zig");
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 8 }) |world| {
        const fits = world >= 2 and df.FF % world == 0 and (df.FF / world) % 256 == 0;
        try std.testing.expectEqual(world == 2, df.mlpSplit(world, (Settings{}).draftWide()).split);
        for ([_]Settings{ .{ .draft_split = true }, .{ .roce_wide = true } }) |s| {
            const m = df.mlpSplit(world, s.draftWide());
            try std.testing.expectEqual(fits, m.split);
            try std.testing.expectEqual(if (fits) df.FF / world else df.FF, m.ff);
        }
    }
}

test "TF_GLM_DRAFT_SPLIT=1 is in the ranks' agreement at three ranks or more, and two ranks agree whatever it says" {
    var all = [_]Fingerprint{
        Fingerprint.of(3, 0, .{ .draft_split = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(3, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(3, 2, .{ .draft_split = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
    };
    try std.testing.expectEqual(Mismatch{ .field = .draft_split, .rank = 1 }, mismatch(&all).?);
    var two = [_]Fingerprint{
        Fingerprint.of(2, 0, .{ .draft_split = true }, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
    };
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&two));
}

test "GLM fused RoCE hc_post: strict opt-in parsing and rank fingerprint agreement" {
    const Reader = struct {
        fn value(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_HC_POST_FUSED")) "1" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_HC_POST_FUSED")) "yes" else null;
        }
    };
    var problem: []const u8 = "";
    const on = try Settings.read(Reader.value, &problem);
    try std.testing.expect(on.hc_post_fused);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_HC_POST_FUSED: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, on, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    };
    try std.testing.expectEqual(Mismatch{ .field = .hc_post_fused, .rank = 1 }, mismatch(&all).?);
}
