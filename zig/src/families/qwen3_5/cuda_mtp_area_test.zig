//! MTP's ring and state in a pool lane's run: the area's layout, borrowed on open, given back on release.
const std = @import("std");
const lanes = @import("lanes");
const mtp = @import("cuda_mtp.zig");
const E = @import("cuda_engine.zig").Engine;
const t = std.testing;

/// The target's lanes: stream 0 sits on a pool lane whose run holds the area at `base`, stream 1 on a reserved window.
const Runs = struct {
    base: u64 = 0x7000_0000,
    on_pool: *lanes.Stream,
    fn area(p: *anyopaque, s: *lanes.Stream) ?u64 {
        const r: *Runs = @ptrCast(@alignCast(p));
        return if (s == r.on_pool) r.base else null;
    }
};

test "MTP on a pool lane borrows its run's area: ring, then 256-aligned state, given back with the lane" {
    var e: E = undefined;
    e.d = undefined;
    e.g.kv_heads = 2;
    e.g.hidden = 2048;
    var streams: [2]lanes.Stream = undefined;
    var runs = Runs{ .on_pool = &streams[0] };
    var d: mtp.Mtp = undefined;
    d.gpa = t.allocator;
    d.e = &e;
    d.streams = .empty;
    d.pool = .empty;
    d.ring_of = .{ .ctx = &runs, .area = Runs.area };
    defer d.streams.deinit(t.allocator);
    defer d.pool.deinit(t.allocator);
    // the reserved window's stream takes the drafter's own context
    var own = mtp.Ctx{ .ring = .{ .d = undefined, .ptr = 0x1000, .len = 1 }, .last = .{ .d = undefined, .ptr = 0x2000, .len = 1 } };
    try d.pool.append(t.allocator, &own);
    // with lane runs, the drafter keeps one context of its own however many streams are admitted
    try d.reserve(8);
    try t.expectEqual(@as(usize, 1), d.pool.items.len);
    for ([_]usize{ 1, 3, 4096, 32768 }) |capacity| {
        d.capacity = capacity;
        const ring = 2 * capacity * d.kvWidth() * 2;
        const drafter = d.drafter();
        try drafter.open(&streams[0]);
        try t.expectError(error.StreamOpen, drafter.open(&streams[0]));
        const ctx = d.streams.get(&streams[0]).?;
        try t.expect(ctx.ring.borrowed and ctx.last.borrowed);
        try t.expectEqual(runs.base, ctx.ring.ptr);
        try t.expectEqual(ring, ctx.ring.len);
        try t.expectEqual(runs.base + std.mem.alignForward(usize, ring, 256), ctx.last.ptr);
        try t.expect(ctx.ring.ptr + ctx.ring.len <= ctx.last.ptr);
        try t.expectEqual(@as(u64, 0), ctx.last.ptr % 256);
        try t.expectEqual(e.g.hidden * 2, ctx.last.len);
        // the area's last byte is the state's: areaBytes is the run's ring part, nothing past it
        try t.expectEqual(runs.base + d.areaBytes(), ctx.last.ptr + ctx.last.len);
        try drafter.open(&streams[1]);
        try t.expectEqual(&own, d.streams.get(&streams[1]).?);
        try t.expectEqual(@as(usize, 0), d.pool.items.len);
        // release frees the borrowed context (else a leak), leaves the run alone, returns the owned one to the pool
        drafter.release(&streams[0]);
        drafter.release(&streams[1]);
        try t.expectEqual(@as(usize, 0), d.streams.count());
        try t.expectEqual(@as(usize, 1), d.pool.items.len);
        try t.expectEqual(&own, d.pool.items[0]);
    }
}
