//! Rolling draft cache construction and complete allocation ownership.
const std = @import("std");
const cfg = @import("config.zig");
pub fn Pair(comptime Buffer: type) type {
    return struct { keys: Buffer, values: Buffer };
}
/// `layers` rolling K/V caches of the window plus one block, as a stream's drafter context holds them.
pub fn newCaches(comptime Buffer: type, a: std.mem.Allocator, device: anytype, c: cfg.Config, layers: usize) ![]Pair(Buffer) {
    const Cache = Pair(Buffer);
    const caches = try a.alloc(Cache, layers);
    var made: usize = 0;
    errdefer a.free(caches);
    errdefer for (caches[0..made]) |cache| {
        cache.keys.deinit();
        cache.values.deinit();
    };
    for (caches) |*cache| {
        const bytes = @as(usize, c.window + 16) * c.kvWidth() * 2;
        const keys = try device.buffer(bytes, @import("metal").ResourceOptions.shared | @import("metal").ResourceOptions.untracked);
        const values = device.buffer(bytes, @import("metal").ResourceOptions.shared | @import("metal").ResourceOptions.untracked) catch |err| {
            keys.deinit();
            return err;
        };
        cache.* = .{ .keys = keys, .values = values };
        made += 1;
    }
    return caches;
}
pub fn freeCaches(comptime Buffer: type, a: std.mem.Allocator, caches: []Pair(Buffer)) void {
    for (caches) |cache| {
        cache.keys.deinit();
        cache.values.deinit();
    }
    a.free(caches);
}
