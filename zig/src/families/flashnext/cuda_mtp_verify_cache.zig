const std = @import("std");

pub const Cache = struct {
    generation: u64 = 0,
    ready: ?u64 = null,
    rows: usize = 0,
    position: usize = 0,

    pub fn invalidate(cache: *Cache) void {
        cache.generation +%= 1;
        cache.ready = null;
    }

    pub fn publish(cache: *Cache, generation: u64, rows: usize, position: usize) void {
        if (generation != cache.generation) return;
        cache.rows = rows;
        cache.position = position;
        cache.ready = generation;
    }

    pub fn take(cache: *Cache, rows: usize, position: usize, greedy: bool) bool {
        defer cache.invalidate();
        return greedy and cache.ready == cache.generation and cache.rows == rows and cache.position == position;
    }
};

test "exl3: verification records require completion in the current generation and are consumed once" {
    var cache: Cache = .{};
    try std.testing.expect(!cache.take(4, 10, true));
    const generation = cache.generation;
    cache.publish(generation, 4, 10);
    try std.testing.expect(cache.take(4, 10, true));
    try std.testing.expect(!cache.take(4, 10, true));
    cache.publish(generation, 4, 10);
    try std.testing.expect(!cache.take(4, 10, true));
}

test "exl3: sampled draws, changed windows, rollback and reuse invalidate verification records" {
    var cache: Cache = .{};
    cache.publish(cache.generation, 4, 10);
    try std.testing.expect(!cache.take(4, 10, false));
    try std.testing.expect(!cache.take(4, 10, true));
    cache.publish(cache.generation, 4, 10);
    try std.testing.expect(!cache.take(3, 10, true));
    cache.publish(cache.generation, 4, 10);
    try std.testing.expect(!cache.take(4, 11, true));
    const prior = cache.generation;
    cache.invalidate();
    cache.publish(prior, 4, 10);
    try std.testing.expect(!cache.take(4, 10, true));
    cache.generation = std.math.maxInt(u64);
    cache.publish(cache.generation, 4, 10);
    cache.invalidate();
    try std.testing.expect(!cache.take(4, 10, true));
}

test "exl3: verification generations belong to each draft independently" {
    var first: Cache = .{};
    var second: Cache = .{};
    first.publish(first.generation, 4, 10);
    second.publish(second.generation, 8, 20);
    first.invalidate();
    try std.testing.expect(!first.take(4, 10, true));
    try std.testing.expect(second.take(8, 20, true));
}
