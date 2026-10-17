//! Model-free GLM expiry tests, including the real pool, slot invalidation, and cache ownership.
const std = @import("std");

test "GLM expiry: a counted deferred restore failure logs once without token text" {
    const pc = @import("core/prompt_cache.zig");
    const Fake = @import("core/prompt_cache_fake.zig").Fake;
    const a = std.testing.allocator;
    var fake: Fake = .{ .gpa = a };
    var store = pc.Store.init(a, fake.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    _ = fake.pass(&store, &prompt, try store.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    const lookup = try store.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{});
    const Log = struct {
        count: usize = 0,
        at: u32 = 0,
        reason: ?anyerror = null,
        fn write(ctx: *anyopaque, at: u32, reason: anyerror) void {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            l.count += 1;
            l.at = at;
            l.reason = reason;
        }
    };
    var log: Log = .{};
    store.restore_log = .{ .ctx = &log, .write = Log.write };
    store.resumed(lookup.entry.?, &prompt, false);
    try std.testing.expectEqual(@as(u64, 1), store.counts.failed);
    try std.testing.expectEqual(@as(usize, 1), log.count);
    try std.testing.expectEqual(@as(u32, 4), log.at);
    try std.testing.expectEqual(error.RestoreRejected, log.reason.?);
}

test {
    _ = @import("families/glm/expiry_test.zig");
}
