//! MTP's ring and states in a pool lane's run: the area's layout, borrowed on open, given back on release.
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

test "MTP on a pool lane borrows its run's area: ring, then 256-aligned states, given back with the lane" {
    var e: E = undefined;
    e.d = undefined;
    e.g.kv_heads = 2;
    e.g.hidden = 2048;
    e.g.intermediate = 128;
    e.g.layers = 1;
    e.g.q_heads = 2;
    e.g.vocab = 256;
    e.g.experts = 0;
    var streams: [2]lanes.Stream = undefined;
    var runs = Runs{ .on_pool = &streams[0] };
    var d: mtp.Mtp = undefined;
    d.gpa = t.allocator;
    d.e = &e;
    d.streams = .empty;
    d.pool = .empty;
    d.ring_of = .{ .ctx = &runs, .area = Runs.area };
    d.capacity = 32768;
    try d.initializeMemorySizes();
    defer d.streams.deinit(t.allocator);
    defer d.pool.deinit(t.allocator);
    // the reserved window's stream takes the drafter's own context
    var own = mtp.Ctx{ .ring = .{ .d = undefined, .ptr = 0x1000, .len = 1 }, .last = .{ .d = undefined, .ptr = 0x2000, .len = 1 }, .step = .{ .d = undefined, .ptr = 0x3000, .len = 1 }, .held_dev = .{ .d = undefined, .ptr = 0x4000, .len = 1 } };
    try d.pool.append(t.allocator, &own);
    // with lane runs, the drafter keeps one context of its own however many streams are admitted
    try d.reserve(8);
    try t.expectEqual(@as(usize, 1), d.pool.items.len);
    for ([_]usize{ 1, 3, 4096, 32768 }) |capacity| {
        d.capacity = capacity;
        try d.initializeMemorySizes();
        const ring = 2 * capacity * d.kvWidth() * 2;
        const drafter = d.drafter();
        try drafter.open(&streams[0]);
        try t.expectError(error.StreamOpen, drafter.open(&streams[0]));
        const ctx = d.streams.get(&streams[0]).?;
        try t.expect(ctx.ring.borrowed and ctx.last.borrowed and ctx.step.borrowed and ctx.held_dev.borrowed);
        try t.expectEqual(runs.base, ctx.ring.ptr);
        try t.expectEqual(ring, ctx.ring.len);
        try t.expectEqual(runs.base + std.mem.alignForward(usize, ring, 256), ctx.last.ptr);
        try t.expect(ctx.ring.ptr + ctx.ring.len <= ctx.last.ptr);
        try t.expectEqual(@as(u64, 0), ctx.last.ptr % 256);
        try t.expectEqual(e.g.hidden * 2, ctx.last.len);
        // the batched chain's step state follows, 256-aligned, the same size
        try t.expect(ctx.last.ptr + ctx.last.len <= ctx.step.ptr);
        try t.expectEqual(@as(u64, 0), ctx.step.ptr % 256);
        try t.expectEqual(e.g.hidden * 2, ctx.step.len);
        try t.expect(ctx.step.ptr + ctx.step.len <= ctx.held_dev.ptr);
        try t.expectEqual(mtp.Mtp.max_chain_tokens * @sizeOf(u32), ctx.held_dev.len);
        try t.expectEqual(runs.base + d.areaBytes(), ctx.held_dev.ptr + ctx.held_dev.len);
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

test "MTP context lifecycle: release returns an interrupted stream's context reset for reuse" {
    var e: E = undefined;
    e.d = undefined;
    var d: mtp.Mtp = undefined;
    d.gpa = t.allocator;
    d.e = &e;
    d.streams = .empty;
    d.pool = .empty;
    defer {
        var it = d.streams.valueIterator();
        while (it.next()) |ctx| t.allocator.destroy(ctx.*);
        d.streams.deinit(t.allocator);
        for (d.pool.items) |ctx| t.allocator.destroy(ctx);
        d.pool.deinit(t.allocator);
    }

    const ring = mtp.Ctx{ .ring = .{ .d = undefined, .ptr = 0x1000, .len = 8192 }, .last = .{ .d = undefined, .ptr = 0x4000, .len = 4096 }, .step = .{ .d = undefined, .ptr = 0x6000, .len = 4096 }, .held_dev = .{ .d = undefined, .ptr = 0x8000, .len = 64 } };
    const owned = try t.allocator.create(mtp.Ctx);
    owned.* = ring;
    try d.pool.append(t.allocator, owned);

    var interrupted = try lanes.Stream.init(t.allocator, .{ .id = "cancelled", .prompt = &.{1}, .max_new = 8 });
    defer interrupted.deinit(t.allocator);
    var resumed = try lanes.Stream.init(t.allocator, .{ .id = "resumed", .prompt = &.{2}, .max_new = 8 });
    defer resumed.deinit(t.allocator);

    const drafter = d.drafter();
    try drafter.open(&interrupted);
    const first = d.streams.get(&interrupted).?;
    try t.expectEqual(owned, first);
    first.begin = 17;
    first.end = 29;
    first.cands = @splat(0xCAFE);
    first.held = @splat(0xBABE);
    first.held_n = 4;

    // Cancellation, failure, and normal completion all reach this drafter release hook.
    drafter.release(&interrupted);
    try t.expectEqual(@as(usize, 0), d.streams.count());
    try t.expectEqual(@as(usize, 1), d.pool.items.len);
    try t.expectEqual(owned, d.pool.items[0]);

    try drafter.open(&resumed);
    const second = d.streams.get(&resumed).?;
    try t.expectEqual(owned, second);
    try t.expectEqual(@as(u64, 0), second.begin);
    try t.expectEqual(@as(u64, 0), second.end);
    try t.expectEqual(@as(@TypeOf(second.cands), @splat(0)), second.cands);
    try t.expectEqual(@as(usize, 0), second.held_n);
    try t.expectEqual(ring.ring.ptr, second.ring.ptr);
    try t.expectEqual(ring.ring.len, second.ring.len);
    try t.expectEqual(ring.last.ptr, second.last.ptr);
    try t.expectEqual(ring.step.ptr, second.step.ptr);
    try t.expectEqual(ring.held_dev.ptr, second.held_dev.ptr);

    drafter.release(&resumed);
    try t.expectEqual(@as(usize, 0), d.streams.count());
    try t.expectEqual(@as(usize, 1), d.pool.items.len);
}

test "MTP on a pool lane: admission includes the target tap ring exactly once" {
    var e: E = undefined;
    e.d = undefined;
    e.scratch_rows = 4096;
    e.g.hidden = 5120;
    e.g.intermediate = 17408;
    e.g.layers = 64;
    e.g.q_heads = 24;
    e.g.vocab = 248320;
    e.g.kv_heads = 4;
    e.g.experts = 0;
    var d: mtp.Mtp = undefined;
    d.e = &e;
    d.capacity = 6144;
    d.tap = .{63};
    try d.initializeMemorySizes();
    const hidden_bytes = e.g.hidden * 2;
    const tap_bytes = 6144 * hidden_bytes;
    const expected = d.ring_bytes + 2 * hidden_bytes + mtp.max_lanes * 4 + tap_bytes;
    try t.expectEqual(@as(usize, 60 * 1024 * 1024), tap_bytes);
    try t.expectEqual(expected, d.stream_bytes);
    const first = d.stream_bytes;
    try d.initializeMemorySizes();
    try t.expectEqual(first, d.stream_bytes);
    const target_baseline: usize = 1024 * 1024;
    try t.expectEqual(target_baseline + expected, target_baseline + d.stream_bytes);
}
