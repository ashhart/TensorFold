//! One host-checkable source-planted fault for each Inkling 2 Mini fused six-projection tier.
const std = @import("std");

pub const tiers = [_]u32{ 1, 2, 4, 8, 16, 32 };

const Edit = struct { from: []const u8, to: []const u8 };

fn edit(tier: u32) Edit {
    return switch (tier) {
        1 => .{ .from = "TF_INK2_MV6(1, wq)", .to = "TF_INK2_MV6(1, wg)" },
        2 => .{ .from = "TF_INK2_MV6(2, wq)", .to = "TF_INK2_MV6(2, wg)" },
        4 => .{ .from = "TF_INK2_MV6(4, wq)", .to = "TF_INK2_MV6(4, wg)" },
        8 => .{ .from = "TF_INK2_MV6(8, wq)", .to = "TF_INK2_MV6(8, wg)" },
        16 => .{ .from = "TF_INK2_MV6(16, wq)", .to = "TF_INK2_MV6(16, wg)" },
        32 => .{ .from = "TF_INK2_MV6(32, wq)", .to = "TF_INK2_MV6(32, wg)" },
        else => unreachable,
    };
}

pub fn name(tier: u32) []const u8 {
    return switch (tier) {
        1 => "control: metal inkling2 dense mv6_1 tf_ink2_mv6_1",
        2 => "control: metal inkling2 dense mv6_2 tf_ink2_mv6_2",
        4 => "control: metal inkling2 dense mv6_4 tf_ink2_mv6_4",
        8 => "control: metal inkling2 dense mv6_8 tf_ink2_mv6_8",
        16 => "control: metal inkling2 dense mv6_16 tf_ink2_mv6_16",
        32 => "control: metal inkling2 dense mv6_32 tf_ink2_mv6_32",
        else => unreachable,
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, tier: u32) ![]u8 {
    const e = edit(tier);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

test "every dense_fuse control applies once and changes only its discriminating anchor" {
    const production = @import("inkling2_metal").dense_fuse;
    const a = std.testing.allocator;
    for (tiers) |tier| {
        const e = edit(tier);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, tier);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, tier));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", tier));
    }
}

test "a repeated dense_fuse control anchor is ambiguous and refuses to prove a fault" {
    const production = @import("inkling2_metal").dense_fuse;
    const a = std.testing.allocator;
    for (tiers) |tier| {
        const twice = try std.mem.concat(a, u8, &.{ production, production });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, source(a, twice, tier));
    }
}
