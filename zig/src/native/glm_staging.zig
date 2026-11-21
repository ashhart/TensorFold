//! The GLM Metal host's rows-store staging permit: the family's whole image, not the store's 1 GiB default (#4114).
const std = @import("std");

/// The rows store's own default permit, and the floor under a priced one.
const floor: u64 = 1 << 30;

/// The permit for a store capped at `cap`: the whole image `image_bytes` of the deepest save, never under the default.
pub fn stagingPermit(cap: u64, image_bytes: u64, env_gib: ?u64) u64 {
    const priced = @max(image_bytes, floor);
    // `env_gib` stands in for the priced figure; a GiB count whose GiB product leaves a u64 is not a price either.
    const want = if (env_gib) |g| std.math.mul(u64, g, floor) catch priced else priced;
    return @min(cap, want);
}

test "glm staging: a short image keeps the store's 1 GiB default" {
    try std.testing.expectEqual(floor, stagingPermit(8 << 30, 12 << 20, null));
}

test "glm staging: a deep image prices its whole image" {
    const deep: u64 = 3 << 30; // a long window's rows, keys, indexer rows and pooled blocks together
    try std.testing.expectEqual(deep, stagingPermit(8 << 30, deep, null));
}

test "glm staging: a priced image never passes the store budget" {
    try std.testing.expectEqual(@as(u64, 2 << 30), stagingPermit(2 << 30, 6 << 30, null));
}

test "glm staging: the operator's GiB stands in for the priced figure" {
    try std.testing.expectEqual(@as(u64, 3 << 30), stagingPermit(8 << 30, 12 << 20, 3));
    try std.testing.expectEqual(@as(u64, 5 << 30), stagingPermit(8 << 30, 3 << 30, 5));
}

test "glm staging: an override above the budget clamps to the budget" {
    try std.testing.expectEqual(@as(u64, 2 << 30), stagingPermit(2 << 30, 12 << 20, 9));
    try std.testing.expectEqual(@as(u64, 8 << 30), stagingPermit(8 << 30, 12 << 20, 8));
}

test "glm staging: an override whose GiB product leaves a u64 keeps the priced figure" {
    // a wrapped 0 would skip every save, so the priced image stands instead: 1 << 40 GiB does not fit a u64
    try std.testing.expectEqual(@as(u64, 3 << 30), stagingPermit(8 << 30, 3 << 30, 1 << 40));
    try std.testing.expectEqual(@as(u64, 1 << 30), stagingPermit(1 << 30, 0, 1 << 40));
}
