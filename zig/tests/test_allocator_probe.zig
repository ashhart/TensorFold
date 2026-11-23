//! The macOS allocator policy exercised on the host, with balanced worker allocations and one controlled leak.
const std = @import("std");
const options = @import("allocator_probe");

test "test allocator: zero-frame threaded allocations keep leak detection" {
    try std.testing.expectEqual(@as(usize, 0), std.testing.allocator_instance.stack_trace_size);
    const Worker = struct {
        fn run() void {
            for (0..64) |_| {
                const bytes = std.testing.allocator.alloc(u8, 32) catch @panic("allocation failed");
                @memset(bytes, 0xa5);
                std.testing.allocator.free(bytes);
            }
        }
    };
    {
        var workers: [4]std.Thread = undefined;
        var started: usize = 0;
        defer for (workers[0..started]) |worker| worker.join();
        for (&workers) |*worker| {
            worker.* = try std.Thread.spawn(.{}, Worker.run, .{});
            started += 1;
        }
    }
    if (options.leak) {
        const lost = try std.testing.allocator.create(u64);
        lost.* = 42;
        std.mem.doNotOptimizeAway(lost);
    }
}
