//! Drama Metal's real address registry is exercised without shader or model loading.
const std = @import("std");
const mtl = @import("metal");
const Gpu = @import("src/families/drama3/metal/gpu.zig").Gpu;
const Warm = @import("src/families/drama3/metal/residency.zig").Warm;
const a = std.testing.allocator;

test "residency owner Drama Metal: the real address registry follows state, codec scratch, aliases and allocation failures" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var gpu: Gpu = .{ .gpa = a, .device = record.device(), .queue = record.queue(), .lib = undefined, .pipes = undefined, .bufs = .empty, .owned = .empty, .indirect = .empty };
        defer {
            gpu.releaseFrom(1);
            gpu.bufs.deinit(a);
            gpu.owned.deinit(a);
            gpu.indirect.deinit(a);
        }
        try gpu.bufs.append(a, undefined);
        try gpu.owned.append(a, false);
        const shard = try record.device().buffer(256, 0);
        defer shard.deinit();
        _ = try gpu.register(shard, false);
        while (gpu.bufs.items.len < gpu.bufs.capacity) _ = try gpu.alloc(64);
        gpu.bufs = .fromOwnedSlice(try gpu.bufs.toOwnedSlice(a));
        gpu.owned = .fromOwnedSlice(try gpu.owned.toOwnedSlice(a));
        var warm: Warm = .{};
        try warm.hold(&gpu, std.testing.io);
        defer warm.drop(&gpu);
        try record.checkInitial();
        try record.check();
        mtl.keepalive.Target.tick(&warm.target);
        try std.testing.expect(record.copies > 0 and record.uses > 0);
        for (0..2) |point| {
            var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = point });
            gpu.gpa = failing.allocator();
            const refused = gpu.alloc(128);
            gpu.gpa = a;
            try std.testing.expectError(error.OutOfMemory, refused);
            try record.check();
        }
        record.fail_at = record.allocations.items.len;
        try std.testing.expectError(error.InjectedBufferFailure, gpu.alloc(128));
        record.fail_at = null;
        try record.check();
        const first = gpu.bufs.items.len;
        _ = try gpu.alloc(128);
        warm.resident.?.held.flush();
        try record.check();
        gpu.releaseFrom(first);
        try record.check();
        _ = try gpu.register(shard, false);
        warm.resident.?.held.flush();
        try record.check();
        gpu.releaseFrom(first);
        try record.check();
        warm.drop(&gpu);
        record.refuse_set = true;
        try warm.hold(&gpu, std.testing.io);
        try std.testing.expect(warm.resident == null and warm.target.scratch == null);
        try record.check();
    }
    try record.allFreed();
}
