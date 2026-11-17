//! A salted prompt-cache key: its first words masked by the salt, so only the same salt's requests share states.
const std = @import("std");

/// Key words a salt masks: four 31-bit masks must all agree before two salts' keys share a prefix.
pub const words = 4;

/// `base` with its first words high-bit and masked by `salt`; such words never go to disk (cache_images.diskSafe).
pub fn salted(a: std.mem.Allocator, base: []const u32, salt: u64) std.mem.Allocator.Error![]u32 {
    const out = try a.dupe(u32, base);
    for (out[0..@min(words, out.len)], 0..) |*w, i| w.* = 0x8000_0000 | ((w.* ^ mask(salt, i)) & 0x7fff_ffff);
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
        for (k) |w| try t.expect(w & 0x8000_0000 != 0);
        try t.expect(!(try seen.getOrPut(k[0])).found_existing);
    }
    base = .{ 1, 2, 3, 4 };
    const one = try salted(t.allocator, &base, 1);
    defer t.allocator.free(one);
    const two = try salted(t.allocator, &base, 2);
    defer t.allocator.free(two);
    for (one, two) |p, q| try t.expect(p != q);
}

test {
    _ = @import("cache_salt_disk_test.zig");
}
