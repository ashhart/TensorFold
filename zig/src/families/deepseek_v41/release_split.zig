//! The release projections' fixed K split: a pure function of a projection's K and its rank's N, never of rows.
const std = @import("std");

/// Columns a decode CTA of the projection kernels covers (two warps of 32).
pub const cta_cols = 64;
/// GB10's SMs: a split stops once the CTAs of one launch fill them.
pub const sms = 48;
/// The most slices, and the least K a slice keeps.
pub const most = 8;
pub const least_k = 512;

/// Slices of K for a projection of K inputs and N outputs: 1, 2, 4 or 8, each a whole number of 32-wide groups.
pub fn split(k: usize, n: usize) u32 {
    const ctas = (n + cta_cols - 1) / cta_cols;
    var s: usize = 1;
    while (s < most and ctas * s < sms and (k / 32) % (2 * s) == 0 and k / (2 * s) >= least_k) s *= 2;
    return @intCast(s);
}

/// The world-4 projections the release serves, with the split each takes.
pub const Served = struct { name: []const u8, k: usize, n: usize, s: u32 };
pub const served = [_]Served{
    .{ .name = "wq_a", .k = 5120, .n = 1280, .s = 4 },
    .{ .name = "wkv", .k = 5120, .n = 512, .s = 8 },
    .{ .name = "wq_b", .k = 1280, .n = 8192, .s = 1 },
    .{ .name = "ix_wq_b", .k = 1280, .n = 4096, .s = 1 },
    .{ .name = "wo_a group", .k = 4096, .n = 1024, .s = 4 },
    .{ .name = "wo_b", .k = 2048, .n = 5120, .s = 1 },
    .{ .name = "shared w1/w3", .k = 5120, .n = 576, .s = 8 },
    .{ .name = "shared w2", .k = 576, .n = 5120, .s = 1 },
    .{ .name = "engram wkv", .k = 6144, .n = 25600, .s = 1 },
    .{ .name = "dspark main_proj", .k = 15360, .n = 5120, .s = 1 },
    .{ .name = "head ranks 0-1", .k = 5120, .n = 32384, .s = 1 },
    .{ .name = "head ranks 2-3", .k = 5120, .n = 32256, .s = 1 },
    .{ .name = "compressor ratio 2", .k = 5120, .n = 512, .s = 8 },
    .{ .name = "ix_wk", .k = 512, .n = 128, .s = 1 },
};

test "release split: the served world-4 table, whole groups a slice, independent of rows" {
    for (served) |p| {
        const s = split(p.k, p.n);
        std.testing.expectEqual(p.s, s) catch |err| {
            std.debug.print("{s}: K {d} N {d} splits {d}\n", .{ p.name, p.k, p.n, s });
            return err;
        };
        try std.testing.expect(s == 1 or s == 2 or s == 4 or s == 8);
        try std.testing.expectEqual(@as(usize, 0), (p.k / 32) % s);
    }
    try std.testing.expectEqual(@as(u32, 1), split(32, 32));
    try std.testing.expectEqual(@as(u32, 1), split(576, 32));
    try std.testing.expectEqual(@as(u32, 8), split(8192, 32));
}
