//! Host policy cost only; this fake provider performs no device allocation, copy, synchronization or free.
const std = @import("std");
const builtin = @import("builtin");
const Workspace = @import("recurrent").Workspace;
const Resource = struct { id: usize };
const Provider = struct {
    calls: usize = 0,
    pub fn allocate(p: *Provider, _: usize) !Resource {
        p.calls += 1;
        return .{ .id = p.calls };
    }
    pub fn dispose(_: *Provider, _: *Resource) void {}
};
pub fn main(init: std.process.Init) !void {
    const Pool = Workspace(Resource);
    const iterations = 10000;
    for ([_]usize{ 1, 8, 64, 128 }) |streams| {
        var provider: Provider = .{};
        var pool = Pool.init(std.heap.page_allocator, 1);
        var leases: [256]Pool.Lease = undefined;
        for (leases[0 .. 2 * streams], 0..) |*lease, i| lease.* = try pool.acquire(1, if (i % 2 == 0) 1024 else 16384, &provider);
        const began = std.Io.Clock.awake.now(init.io);
        var guard: usize = 0;
        for (0..iterations) |_| {
            for (leases[0 .. 2 * streams]) |lease| {
                guard +%= (try pool.view(1, lease)).id;
                try pool.markUsed(1, lease);
            }
            try pool.complete(1);
            try pool.collect(1, &provider);
            std.mem.doNotOptimizeAway(pool.entries.items);
        }
        const ns = began.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        for (leases[0 .. 2 * streams]) |lease| try pool.release(1, lease);
        try pool.deinit(1, &provider);
        std.debug.print("{{\"arch\":\"{s}\",\"host_policy_only\":true,\"streams\":{d},\"frames\":{d},\"total_ns\":{d},\"ns_per_frame\":{d:.2},\"guard\":{d}}}\n", .{ @tagName(builtin.cpu.arch), streams, iterations, ns, @as(f64, @floatFromInt(ns)) / iterations, guard });
    }
}
