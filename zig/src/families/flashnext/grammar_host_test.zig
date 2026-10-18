//! Exhausted grammar rows must never emit a forbidden token.
const std = @import("std");
const lanes = @import("lanes");
const draw = @import("grammar.zig").draw;
const registry = @import("contract").registry;
const t = std.testing;

test "Flash Next contracts: exhausted masked rows fail for greedy and sampled requests" {
    for ([_]u16{ 0xff80, 0x7fc0 }) |bad| {
        const row = [_]u16{ 0x4120, bad, 0x4100, 0x40e0 };
        for ([_]?lanes.Sampling{ null, .{ .seed = 9, .temperature = 0.7 } }) |s|
            try t.expectError(error.NoAllowedToken, draw(t.allocator, &row, &.{2}, s, 7));
    }
}

test "Flash Next contracts: finite masked draws stay allowed and greedy ties choose the lower id" {
    const row = [_]u16{ 0x4120, 0x40e0, 0x4100, 0x40e0 };
    try t.expectEqual(@as(u32, 1), try draw(t.allocator, &row, &.{10}, null, 0));
    for (0..32) |i| {
        const picked = try draw(t.allocator, &row, &.{10}, .{ .seed = 3, .temperature = 0.7 }, @intCast(i));
        try t.expect(picked == 1 or picked == 3);
    }
}

test "Flash Next contracts: both vision launches are registered pending measured limits" {
    const records = if (@hasDecl(registry, "all_metal")) registry.all_metal else registry.metal;
    for ([_][]const u8{ "flashnext vision fz_mask_rows", "flashnext vision fn_attn_prep_mrope" }) |name| {
        var found = false;
        for (records) |r| if (std.mem.eql(u8, name, r.name)) {
            found = true;
            try t.expectEqual(@as(u32, 256), r.threads);
            try t.expectEqual(@as(u32, 0), r.m1);
            try t.expectEqual(@as(u32, 0), r.m2);
            try t.expect(try registry.checkMetal(t.allocator, &.{r}, true) > 0);
        };
        try t.expect(found);
    }
}
