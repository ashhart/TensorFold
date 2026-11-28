//! A salted prompt-cache key: its first words masked by the salt, so only the same salt's requests share states.
const std = @import("std");

/// Key words a salt masks: the first carries a disk-exclusion marker, the other three retain all 32 input bits.
pub const words = 4;
pub const Error = std.mem.Allocator.Error || error{UnsupportedCacheSaltPrefix};

/// Refuse a high-bit first word; later words use full-width XOR, preserving image words and the key's length.
pub fn salted(a: std.mem.Allocator, base: []const u32, salt: u64) Error![]u32 {
    if (base.len > 0 and base[0] & 0x8000_0000 != 0) return error.UnsupportedCacheSaltPrefix;
    const out = try a.dupe(u32, base);
    for (out[0..@min(words, out.len)], 0..) |*w, i| w.* = if (i == 0)
        0x8000_0000 | ((w.* ^ mask(salt, i)) & 0x7fff_ffff)
    else
        w.* ^ mask(salt, i);
    return out;
}

/// The salt's mask for word `i`: a mask, not a hash, so one salt never maps two text tokens to the same word.
fn mask(salt: u64, i: usize) u32 {
    return @truncate(std.hash.Wyhash.hash(salt, std.mem.asBytes(&@as(u64, i))));
}

test "one salt keeps every text token apart, as an unsalted key does; two salts differ in every masked word" {
    const t = std.testing;
    var base: [words]u32 = undefined;
    var seen = std.AutoHashMap(u32, void).init(t.allocator);
    defer seen.deinit();
    for (0..1 << 16) |token| { // the same token at every position, under one salt: a different word for each token
        base = @splat(@intCast(token));
        const k = try salted(t.allocator, &base, 0x5a17);
        defer t.allocator.free(k);
        try t.expect(k[0] & 0x8000_0000 != 0);
        try t.expect(!(try seen.getOrPut(k[0])).found_existing);
    }
    base = .{ 1, 2, 3, 4 };
    const one = try salted(t.allocator, &base, 1);
    defer t.allocator.free(one);
    const two = try salted(t.allocator, &base, 2);
    defer t.allocator.free(two);
    for (one, two) |p, q| try t.expect(p != q);
}

test "one salt keeps bit 31 meaningful or refuses a masked image prefix" {
    const t = std.testing;
    const text = [_]u32{ 5, 5, 5, 5, 5 };
    const key = try salted(t.allocator, &text, 0x5a17);
    defer t.allocator.free(key);
    for (0..words) |i| {
        var image = text;
        image[i] |= 0x8000_0000;
        if (salted(t.allocator, &image, 0x5a17)) |other| {
            defer t.allocator.free(other);
            errdefer std.debug.print("bit-31 collision at masked word {d}\n", .{i});
            try t.expect(!std.mem.eql(u32, key, other));
        } else |err| try t.expectEqual(@as(anyerror, error.UnsupportedCacheSaltPrefix), @as(anyerror, err));
    }
    var later = text;
    later[words] |= 0x8000_0000;
    const tail = try salted(t.allocator, &later, 0x5a17);
    defer t.allocator.free(tail);
    try t.expectEqual(later[words], tail[words]);
    try t.expect(!std.mem.eql(u32, key, tail));
}

test {
    _ = @import("cache_salt_disk_test.zig");
}
