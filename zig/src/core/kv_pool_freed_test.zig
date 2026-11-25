//! A tier frees frames only when the pool makes their pages reusable, including after delayed readers finish.
const std = @import("std");
const kv = @import("kv_pool.zig");
const t = std.testing;

const Seen = struct {
    calls: usize = 0,
    fn call(ctx: *anyopaque, id: u32) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        std.debug.assert(id == 0);
        s.calls += 1;
    }
};

test "pool freed hook: lease, store and run release notify once after the last holder" {
    for (0..3) |mode| {
        var p = try kv.Pool.init(t.allocator, 1, 64);
        defer p.deinit();
        var seen: Seen = .{};
        p.freed = .{ .ctx = &seen, .call = Seen.call };
        var l: kv.Lease = .{};
        try p.promise(&l, 1);
        if (mode == 2) {
            const id = p.claimRun(&l, 1).?;
            p.releaseRun(id, 1);
            p.finish(&l);
        } else {
            const id = try p.take(&l);
            if (mode == 1) p.retain(id);
            p.finish(&l);
            if (mode == 1) {
                try t.expectEqual(@as(usize, 0), seen.calls);
                p.release(id);
            }
        }
        try t.expectEqual(@as(usize, 1), seen.calls);
        try t.expect(p.idle());
    }
}

test "pool freed hook: a retired pinned frame survives until its reader settles" {
    var host: [64]u8 = @splat(0);
    var tier: kv.Tier = undefined;
    try tier.init(t.allocator, &host, 1, 64);
    defer tier.deinit();
    var l: kv.tier.Lease = .{};
    defer tier.finish(&l);
    try tier.reserve(&l, 1, 1);
    try tier.cover(&l, 1, null);
    const id = l.pages.ids.items[0];
    const fill = try tier.pin(&l, id);
    const ticket: kv.tier.Ticket = .{ .frame = fill.frame, .gen = fill.gen };
    try kv.retirement.retire(&tier.pool, &l.pages, 0, 7);
    try t.expect(tier.valid(id, ticket));
    try t.expectEqual(@as(usize, 0), tier.nfree);
    try kv.retirement.settle(&tier.pool, 6);
    try t.expect(tier.valid(id, ticket));
    try tier.unpin(&l, id);
    try kv.retirement.settle(&tier.pool, 7);
    try t.expect(!tier.valid(id, ticket));
    try t.expectEqual(@as(usize, 1), tier.nfree);
    try kv.retirement.settle(&tier.pool, 7);
    try t.expectEqual(@as(usize, 1), tier.nfree);
}
