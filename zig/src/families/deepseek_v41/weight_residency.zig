//! Keep checkpoint residency in bounded sets; macOS may evict an entire set at a time.
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("../../core/checkpoint_metal.zig");

// One of Metal's 32 queue slots remains reserved for execution buffers and descriptors.
const max_sets = 31;
pub const Groups = struct {
    pub const Refresh = struct {
        gpa: std.mem.Allocator,
        sets: [max_sets]mtl.ResidencySet,
        count: usize,
        ticker: *@import("../../core/keepalive.zig").Keepalive,

        pub fn start(gpa: std.mem.Allocator, io: std.Io, groups: *const Groups) !?*Refresh {
            const mode = std.c.getenv("TF_DSV41_RESIDENCY_REFRESH") orelse return null;
            if (!std.mem.eql(u8, std.mem.span(mode), "1")) return null;
            if (groups.count == 0) return error.ResidencyRefreshRequiresSplitSets;
            const r = try gpa.create(Refresh);
            errdefer gpa.destroy(r);
            r.* = .{ .gpa = gpa, .sets = groups.sets, .count = groups.count, .ticker = undefined };
            r.ticker = try @import("../../core/keepalive.zig").Keepalive.start(gpa, io, .{ .ctx = r, .tick = tick }, std.math.maxInt(i64));
            return r;
        }

        fn tick(ptr: *anyopaque) void {
            const r: *Refresh = @ptrCast(@alignCast(ptr));
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            // Only immutable weight sets; execution descriptors may be added concurrently to the other set.
            for (r.sets[0..r.count]) |set| set.requestResidency();
        }

        pub fn stop(r: *Refresh) void {
            r.ticker.stop();
            r.gpa.destroy(r);
        }
    };

    sets: [max_sets]mtl.ResidencySet = undefined,
    count: usize = 0,

    pub fn init(device: mtl.Device, queue: mtl.Queue, cp: *const checkpoint.Checkpoint, previous: mtl.ResidencySet) !Groups {
        var result: Groups = .{};
        const mode = std.c.getenv("TF_DSV41_SPLIT_RESIDENCY") orelse return result;
        if (!std.mem.eql(u8, std.mem.span(mode), "1")) return result;
        errdefer result.deinit(queue);
        var plan = Plan{ .limit = @max(device.maxWorkingSet() / 20, 64 << 20) };
        for (cp.shards.items) |shard| {
            const index = plan.add(shard.buffer.length());
            if (index == result.count) {
                result.sets[index] = try device.residencySet(cp.shards.items.len);
                result.count += 1;
                queue.addResidencySet(result.sets[index]);
            }
            result.sets[index].add(shard.buffer);
        }
        for (result.sets[0..result.count]) |set| {
            set.commit();
            set.requestResidency();
        }
        // Remove only after every new set exists, preserving the old ownership on allocation failure.
        for (cp.shards.items) |shard| previous.remove(shard.buffer);
        previous.commit();
        return result;
    }

    pub fn deinit(g: *Groups, queue: mtl.Queue) void {
        for (g.sets[0..g.count]) |set| {
            queue.removeResidencySet(set);
            set.deinit();
        }
        g.count = 0;
    }

    pub fn bytes(g: *const Groups) usize {
        var total: usize = 0;
        for (g.sets[0..g.count]) |set| total += set.allocatedSize();
        return total;
    }
};

const Plan = struct {
    sizes: [max_sets]usize = @splat(0),
    count: usize = 0,
    limit: usize,

    fn add(p: *Plan, bytes: usize) usize {
        var smallest: usize = 0;
        for (p.sizes[0..p.count], 0..) |size, index| {
            if (size <= p.limit and bytes <= p.limit - size) {
                p.sizes[index] += bytes;
                return index;
            }
            if (size < p.sizes[smallest]) smallest = index;
        }
        if (p.count < max_sets) {
            const index = p.count;
            p.count += 1;
            p.sizes[index] = bytes;
            return index;
        }
        p.sizes[smallest] += bytes;
        return smallest;
    }
};

test "TP residency keeps every byte while bounding sets and reserving an execution slot" {
    var p = Plan{ .limit = 10 };
    for (0..41) |_| _ = p.add(5);
    try std.testing.expectEqual(@as(usize, 21), p.count);
    var total: usize = 0;
    for (p.sizes[0..p.count]) |n| {
        try std.testing.expect(n <= 10);
        total += n;
    }
    try std.testing.expectEqual(@as(usize, 205), total);
    for (0..100) |_| _ = p.add(50);
    try std.testing.expectEqual(@as(usize, 31), p.count);
    total = 0;
    for (p.sizes[0..p.count]) |n| total += n;
    try std.testing.expectEqual(@as(usize, 5205), total);
}
