//! The cluster on the fabric's own in-process endpoints: membership on its pipes, exact sums on its channel, rounds on its words.
const std = @import("std");
const fabric = @import("fabric");
const node = @import("node.zig");
const membership = @import("membership.zig");
const exchange = @import("exchange.zig");
const barrier = @import("barrier.zig");
const canon = @import("canon.zig");
const fabric_link = @import("fabric_link.zig");

const gpa = std.testing.allocator;

fn cluster(n: u32, layout: fabric_link.Layout) !*fabric.fake.Cluster {
    return fabric.fake.Cluster.init(gpa, n, layout.total, .immediate, null);
}

test "four ranks join over the fabric's pipes, elect the lowest id and agree on one view" {
    const layout = fabric_link.Layout.init(4, 4096, 64 << 10);
    const c = try cluster(4, layout);
    defer c.deinit();
    var links: [4]fabric_link.Link = undefined;
    var ms: [4]membership.Membership = undefined;
    for (&links, &ms, 0..) |*l, *m, i| {
        l.* = try fabric_link.Link.init(gpa, c.endpoint(@intCast(i)), layout);
        m.* = membership.Membership.init(gpa, .{}, .{ .id = 500 - 10 * @as(u64, @intCast(i)), .name = .of("r") }, 1);
    }
    defer for (&links, &ms) |*l, *m| {
        m.deinit();
        l.deinit();
    };
    var now: u64 = 0;
    while (now < 600 * std.time.ns_per_ms) : (now += 10 * std.time.ns_per_ms) {
        for (&links, &ms) |*l, *m| try m.step(now, l.transport(), null);
    }
    for (&ms) |*m| {
        try std.testing.expect(m.settled());
        try std.testing.expectEqual(@as(u64, 470), m.leader);
        try std.testing.expectEqual(@as(usize, 4), m.ids.items.len);
    }
}

test "exact sums and round words on the fabric's channel: the same bits as one rank, and lockstep" {
    const layout = fabric_link.Layout.init(4, 64 << 10, 4096);
    const c = try cluster(4, layout);
    defer c.deinit();
    const units = 8;
    const width = 6;
    var data: [units][width]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(9);
    for (&data) |*s| for (s) |*v| {
        v.* = (prng.random().float(f32) - 0.5) * std.math.pow(f32, 10, @floatFromInt(prng.random().intRangeAtMost(i32, -3, 7)));
    };
    var solo: [width]f32 = undefined;
    {
        const one = try exchange.Mem.init(gpa, 1);
        defer one.deinit();
        var all: [units][]const f32 = undefined;
        for (&all, 0..) |*s, i| s.* = &data[i];
        try exchange.allReduce(gpa, one.endpoint(0), units, .{ .begin = 0, .end = units }, &all, 1, width, &solo);
    }
    const Run = struct {
        fn go(l: *fabric_link.Link, d: *const [units][width]f32, out: *[width]f32) void {
            const x = l.steps();
            var ranges: [4]canon.Range = undefined;
            canon.partition(units, &.{ 3, 1, 1, 3 }, &ranges);
            const mine = ranges[x.rank()];
            var views: [units][]const f32 = undefined;
            for (mine.begin..mine.end, 0..) |s, j| views[j] = &d[s];
            exchange.allReduce(gpa, x, units, mine, views[0..mine.len()], 1, width, out) catch unreachable;
            var lock: barrier.Lockstep = .{ .w = l.roundWords() };
            for (1..20) |r| {
                lock.mayWrite(r) catch unreachable;
                lock.arrive(r) catch unreachable;
            }
            lock.wait(19) catch unreachable;
        }
    };
    var links: [4]fabric_link.Link = undefined;
    for (&links, 0..) |*l, i| l.* = try fabric_link.Link.init(gpa, c.endpoint(@intCast(i)), layout);
    defer for (&links) |*l| l.deinit();
    var outs: [4][width]f32 = undefined;
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Run.go, .{ &links[i], &data, &outs[i] });
    for (threads) |t| t.join();
    for (outs) |o| try std.testing.expectEqualSlices(u32, @ptrCast(&solo), @ptrCast(&o));
}

test "Thunderbolt's two-sided links: four ranks join on the control link and agree on one view" {
    const sr = try fabric.sendrecv_fake.Cluster.init(gpa, 4, false);
    defer sr.deinit();
    const data = try fabric.sendrecv_fake.Cluster.init(gpa, 4, false);
    defer data.deinit();
    var sides: [4]fabric_link.TwoSided = undefined;
    var ms: [4]membership.Membership = undefined;
    for (&sides, &ms, 0..) |*t, *m, i| {
        t.* = fabric_link.TwoSided.init(data.link(@intCast(i)), sr.link(@intCast(i)));
        m.* = membership.Membership.init(gpa, .{}, .{ .id = 500 - 10 * @as(u64, @intCast(i)), .name = .of("r") }, 1);
    }
    defer for (&ms) |*m| m.deinit();
    var now: u64 = 0;
    while (now < 600 * std.time.ns_per_ms) : (now += 10 * std.time.ns_per_ms) {
        for (&sides, &ms) |*t, *m| try m.step(now, t.transport(), null);
    }
    for (&ms) |*m| {
        try std.testing.expect(m.settled());
        try std.testing.expectEqual(@as(u64, 470), m.leader);
    }
}
