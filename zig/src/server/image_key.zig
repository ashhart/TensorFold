//! An image prompt's cache key: placeholder rows as ids from each image's digest; restores compare whole digests.
const std = @import("std");
const api = @import("engine_api");

/// Ids the tokenizer never makes: the top bit set.
const marked: u32 = 0x8000_0000;

/// `ids` with placeholder run k (ids in `pads`) as ids from image k's digest; unpaired runs key by all the images.
pub fn cacheKey(a: std.mem.Allocator, ids: []const u32, images: []const api.Image, pads: []const u32) ![]u32 {
    const key = try a.dupe(u32, ids);
    errdefer a.free(key);
    var runs: usize = 0;
    for (ids, 0..) |t, i| {
        if (isPad(t, pads) and (i == 0 or !isPad(ids[i - 1], pads))) runs += 1;
    }
    const seeds = try a.alloc(u64, images.len);
    defer a.free(seeds);
    var all = std.hash.Wyhash.init(0x51ce_7a92);
    for (images, seeds) |im, *seed| {
        const d = im.digest();
        all.update(&d);
        seed.* = std.mem.readInt(u64, d[0..8], .little);
    }
    const together = all.final();
    var run: usize = 0;
    var row: u64 = 0;
    var h: u64 = together;
    for (key, 0..) |*t, i| {
        if (!isPad(ids[i], pads)) continue;
        if (i == 0 or !isPad(ids[i - 1], pads)) {
            h = if (runs == images.len) seeds[run] else together;
            run += 1;
            row = 0;
        }
        t.* = marked | @as(u32, @truncate(std.hash.Wyhash.hash(h, std.mem.asBytes(&row)) & 0x7fff_ffff));
        row += 1;
    }
    return key;
}

fn isPad(t: u32, pads: []const u32) bool {
    return std.mem.indexOfScalar(u32, pads, t) != null;
}

test "two images under the same placeholder ids key apart; the text around them keys as itself" {
    const a = std.testing.allocator;
    const px_a: [4]f32 = .{ 0.1, 0.2, 0.3, 0.4 };
    const px_b: [4]f32 = .{ 0.1, 0.2, 0.3, 0.5 };
    const ids = [_]u32{ 7, 52, 9, 9, 53, 8 };
    const ka = try cacheKey(a, &ids, &.{.{ .patches = &px_a, .grid = .{ 1, 2, 2 } }}, &.{9});
    defer a.free(ka);
    const kb = try cacheKey(a, &ids, &.{.{ .patches = &px_b, .grid = .{ 1, 2, 2 } }}, &.{9});
    defer a.free(kb);
    const ka2 = try cacheKey(a, &ids, &.{.{ .patches = &px_a, .grid = .{ 1, 2, 2 } }}, &.{9});
    defer a.free(ka2);
    try std.testing.expectEqualSlices(u32, &.{ 7, 52 }, ka[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 53, 8 }, ka[4..6]);
    try std.testing.expect(ka[2] >= marked and ka[2] != ka[3]);
    try std.testing.expect(ka[2] != kb[2] and ka[3] != kb[3]);
    try std.testing.expectEqualSlices(u32, ka, ka2);
    // the same pixels in another grid are another image
    const kg = try cacheKey(a, &ids, &.{.{ .patches = &px_a, .grid = .{ 1, 1, 4 } }}, &.{9});
    defer a.free(kg);
    try std.testing.expect(kg[2] != ka[2]);
}

test "a second image's change keys only its own run apart; runs that do not pair with images key by all of them" {
    const a = std.testing.allocator;
    const p1: [2]f32 = .{ 1, 2 };
    const p2: [2]f32 = .{ 3, 4 };
    const p3: [2]f32 = .{ 3, 5 };
    const ids = [_]u32{ 52, 9, 9, 53, 7, 52, 9, 53 };
    const k12 = try cacheKey(a, &ids, &.{ .{ .patches = &p1, .grid = .{ 1, 2, 2 } }, .{ .patches = &p2, .grid = .{ 1, 2, 2 } } }, &.{9});
    defer a.free(k12);
    const k13 = try cacheKey(a, &ids, &.{ .{ .patches = &p1, .grid = .{ 1, 2, 2 } }, .{ .patches = &p3, .grid = .{ 1, 2, 2 } } }, &.{9});
    defer a.free(k13);
    try std.testing.expectEqualSlices(u32, k12[0..6], k13[0..6]);
    try std.testing.expect(k12[6] != k13[6]);
    const one = try cacheKey(a, &ids, &.{.{ .patches = &p1, .grid = .{ 1, 2, 2 } }}, &.{9});
    defer a.free(one);
    try std.testing.expect(one[1] != k12[1]); // two runs, one image: neither run keys as image 1 alone
}

test "two different images behind one placeholder key apart" {
    const a = std.testing.allocator;
    const ids = [_]u32{ 7, 52, 9, 53, 8 };
    var seen: [64]u32 = undefined;
    // 64 one-row images that differ in one patch value each: 64 different keys at the placeholder
    for (&seen, 0..) |*k, i| {
        const px: [2]f32 = .{ 0.5, @floatFromInt(i) };
        const key = try cacheKey(a, &ids, &.{.{ .patches = &px, .grid = .{ 1, 1, 1 } }}, &.{9});
        defer a.free(key);
        try std.testing.expectEqualSlices(u32, &.{ 7, 52 }, key[0..2]);
        k.* = key[2];
        for (seen[0..i]) |o| try std.testing.expect(o != k.*);
    }
}
