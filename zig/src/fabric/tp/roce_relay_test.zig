//! Relay topology, arithmetic, custody and bounded named faults without a device or RDMA library.
const std = @import("std");
const t = std.testing;
const relay = @import("roce_relay.zig");
const guard = @import("roce_multi_guard.zig");
const consumers = @import("roce_relay_collectives.zig");
const Harness = @import("roce_relay_fixture.zig").Harness;

fn stress(comptime verbs: bool) !void {
    var h: Harness = undefined;
    try h.init(4096);
    defer h.deinit();
    var rng = std.Random.DefaultPrng.init(0x1971_303);
    const random = rng.random();
    for (1..100001) |s| {
        const seq: u32 = @intCast(s);
        for (0..4) |r| try h.stage(@intCast(r), seq, 16 * random.intRangeAtMost(u32, 1, 64));
        var done: [4]bool = @splat(false);
        var rounds: u32 = 0;
        while (!std.mem.allEqual(bool, &done, true)) : (rounds += 1) {
            if (rounds > 128) return error.Stalled;
            const start = random.intRangeLessThan(u32, 0, 4);
            for (0..4) |k| {
                const r = (start + @as(u32, @intCast(k))) % 4;
                if (random.boolean()) try h.tick(verbs, r);
                if (!done[r] and h.ready(r, seq)) {
                    const mode: consumers.Mode = @fromBackingInt(@intCast(seq % 4));
                    var n: u32 = h.sizes[0];
                    for (h.sizes) |size| n = @min(n, size);
                    var actual: [4096]u8 = undefined;
                    var expected: [4096]u8 = undefined;
                    const len = n * @as(usize, if (mode == .gather) 4 else 1);
                    try consumers.consume(h.fabric.regions[r], h.fabric.layout, r, seq, n, mode, actual[0..len]);
                    try consumers.consume(h.memory[r], h.fabric.layout, r, seq, n, mode, expected[0..len]);
                    try t.expectEqualSlices(u8, expected[0..len], actual[0..len]);
                    try h.check(r, seq);
                    done[r] = true;
                }
            }
        }
    }
    for (0..4) |r| {
        for (0..16) |_| try h.tick(verbs, @intCast(r));
        try t.expect(h.fabric.wires[r].idle());
    }
    try t.expectEqual(@as(usize, 0), h.fabric.bad);
}
test "relay: 100000 random variable-size collectives match existing mesh posts, every word, on deferred fake verbs" {
    try stress(true);
}
test "relay: 100000 random variable-size collectives match existing mesh posts, every word, on shared-memory wire" {
    try stress(false);
}
test "relay: exhaustive 16-byte segment arithmetic through the engine write cap" {
    const l = guard.Layout.of(4, guard.max_write);
    var n: u32 = 16;
    while (n <= guard.max_write) : (n += 16) {
        const a = try relay.half(n, 0);
        const b = try relay.half(n, 1);
        try t.expectEqual(@as(u32, 0), a.offset);
        try t.expectEqual(a.bytes, b.offset);
        try t.expectEqual(n, a.bytes + b.bytes);
        try t.expect(a.bytes % 16 == 0 and b.bytes % 16 == 0 and a.bytes >= b.bytes);
        for (0..4) |me| for (0..2) |d| for (0..2) |slot| {
            const r: u32 = @intCast(me);
            const dir: u32 = @intCast(d);
            for ([_]u32{ r, relay.neighbour(r, 1 - dir) }) |src| {
                const p = try relay.plan(l, r, .{ .dst = relay.neighbour(r, dir), .source = src, .seq = @intCast(slot), .direction = dir, .total = n });
                const base = if (src == r) l.sendAt(@intCast(slot)) else l.recvAt(src, @intCast(slot));
                try t.expect(p.local >= base and p.local + p.len <= base + n);
                const remote = l.recvAt(src, @intCast(slot));
                try t.expect(p.remote >= remote and p.remote + p.len <= remote + n);
                try t.expect(p.flag >= l.flags and p.flag + 8 <= l.send);
            }
        };
    }
    try t.expectError(error.BadRelay, relay.plan(l, 0, .{ .dst = 2, .source = 0, .seq = 1, .direction = 0, .total = 16 }));
    try t.expectError(error.BadRelay, relay.half(0, 0));
    try t.expectError(error.BadRelay, relay.half(17, 0));
    try t.expectError(error.BadRelay, relay.half(guard.max_write + 16, 0));
}

test "relay sums: four ranks accumulate fp32 in rank order and round bf16 once" {
    const l = guard.Layout.of(4, 16);
    const region = try t.allocator.alloc(u8, l.len);
    defer t.allocator.free(region);
    @memset(region, 0);
    const values = [_]f32{ 1.0, 0x1p-8, 0x1p-8, 0 };
    for (0..4) |me| {
        for (values, 0..) |v, src| {
            const at = if (src == me) l.sendAt(1) else l.recvAt(@intCast(src), 1);
            std.mem.writeInt(u16, region[at..][0..2], @import("bf16_round").bits(v), .little);
        }
        var out: [4]u8 = undefined;
        try consumers.consume(region, l, @intCast(me), 1, 4, .sum_bf16, &out);
        try t.expectEqual(@as(u16, 0x3f81), std.mem.readInt(u16, out[0..2], .little));
        const ordered = [_]f32{ 0x1p40, -0x1p40, 1, 2 };
        for (ordered, 0..) |v, src| {
            const at = if (src == me) l.sendAt(1) else l.recvAt(@intCast(src), 1);
            std.mem.writeInt(u32, region[at..][0..4], @bitCast(v), .little);
        }
        try consumers.consume(region, l, @intCast(me), 1, 4, .sum_f32, &out);
        try t.expectEqual(@as(u32, @bitCast(@as(f32, 3))), std.mem.readInt(u32, &out, .little));
    }
}

test "relay: a held reader prevents the next producer from reusing either live slot" {
    var h: Harness = undefined;
    try h.init(4096);
    defer h.deinit();
    for (0..4) |r| try h.stage(@intCast(r), 1, 4096);
    for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
    try t.expect(h.ready(1, 1));
    // Rank 1 is still reading its final piece of sequence 1. Other ranks have consumed it and ask for sequence 2.
    for ([_]u32{ 0, 2, 3 }) |r| {
        try h.check(r, 1);
        try h.stage(r, 2, 16);
    }
    for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
    try t.expectError(error.ConsumerStillReading, h.stage(0, 3, 4096));
    try h.check(1, 1);
    try h.stage(1, 2, 16);
    for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
    for (0..4) |r| try h.check(@intCast(r), 2);
    for (0..4) |r| try h.stage(@intCast(r), 3, 4096);
    for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
    for (0..4) |r| try h.check(@intCast(r), 3);
}

test "relay: dropped flag, late rank and wrong sequence fail by rank and sequence within the bound" {
    for (0..3) |failure| {
        var h: Harness = undefined;
        try h.init(4096);
        defer h.deinit();
        if (failure != 1) try h.stage(3, 1, 48);
        for (0..3) |r| try h.stage(@intCast(r), 1, 48);
        const at = @intFromPtr(h.fabric.regions[0].ptr) + h.fabric.layout.flagAt(3, 1, 0);
        if (failure == 0) h.fabric.drop_at = at;
        if (failure == 2) h.fabric.skew_at = at;
        h.states[0].timeout_ns = 100;
        var failed: ?anyerror = null;
        for (0..200) |_| {
            for (0..4) |r| h.tick(true, @intCast(r)) catch |e| {
                if (r == 0) failed = e;
            };
            if (failed != null) break;
        }
        try t.expectEqual(@as(?anyerror, if (failure == 2) error.WrongSequence else error.Timeout), failed);
        try t.expectEqual(@as(u32, 3), h.states[0].fault.rank);
        try t.expectEqual(@as(u32, 1), h.states[0].fault.sequence);
        var line: [100]u8 = undefined;
        try t.expect(std.mem.startsWith(u8, h.states[0].fault.describe(&line), "rank 3 sequence 1"));
        try t.expect(h.ticks < 800);
    }
}

test "relay topology: mixed opt-ins, broken opposite and sibling records refuse before any QP changes" {
    const Fabric = @import("roce_multi_fake.zig").Fabric;
    var f: Fabric = undefined;
    try f.init(4, 2, 4096);
    defer f.deinit();
    f.ring();
    var good: [4]@import("roce_multi_verbs.zig").Record = undefined;
    for (0..4) |r| {
        f.wires[r].relay = true;
        good[r] = f.wires[r].record();
    }
    for (0..5) |defect| {
        var records = good;
        switch (defect) {
            0 => records[3].magic -%= 1,
            1 => records[3].n_hca = 1,
            2 => records[3].qpn[0][0] = 0,
            3 => records[3].gid[0] = records[1].gid[0],
            4 => records[0].region += 4096,
            else => unreachable,
        }
        try t.expectError(error.BadRecord, f.wires[0].connect(&records));
        for (f.dest[0..4]) |hs| for (hs) |ps| for (ps) |dst| try t.expect(dst == null);
    }
    for (0..4) |r| try f.wires[r].connect(&good);
    try t.expectEqual(@as(usize, 0), f.bad);
}

test "relay: real shared-memory proxy threads and host gathers preserve each rank's varying shard lengths" {
    const Roce = @import("roce_multi.zig").Roce;
    var h: Harness = undefined;
    try h.init(4096);
    defer h.deinit();
    const Worker = struct {
        rank: u32,
        roce: Roce,
        err: ?anyerror = null,
        fn producer(w: *@This()) void {
            w.body() catch |e| {
                w.err = e;
            };
        }
        fn body(w: *@This()) !void {
            var in: [1024]u8 = undefined;
            var out: [4096]u8 = undefined;
            for (1..1001) |s| {
                var lens: [4]u32 = undefined;
                for (&lens, 0..) |*n, src| n.* = @intCast((s * 37 + src * 113) % 1025);
                for (&in, 0..) |*b, i| b.* = @truncate(s * 19 + w.rank * 73 + i);
                @memset(&out, 0xd7);
                try w.roce.hostGatherV(&in, &out, &lens);
                for (lens, 0..) |n, src| for (out[src * in.len ..][0..n], 0..) |b, i| {
                    try t.expectEqual(@as(u8, @truncate(s * 19 + src * 73 + i)), b);
                };
            }
        }
    };
    var workers: [4]Worker = undefined;
    var proxies: [4]std.Thread = undefined;
    var producers: [4]std.Thread = undefined;
    for (&workers, 0..) |*w, r| {
        w.* = .{ .rank = @intCast(r), .roce = .{ .me = @intCast(r), .world = 4, .relay = true, .layout = h.fabric.layout, .slot_bytes = 4096, .timeout_ns = 10 * std.time.ns_per_s, .region = h.fabric.regions[r], .mapping = h.fabric.regions[r], .wire = .{ .shm = h.shm[r] }, .cpu = null } };
        w.roce.running.store(true, .release);
        proxies[r] = try std.Thread.spawn(.{}, @import("roce_relay_proxy.zig").run, .{&w.roce});
    }
    defer {
        for (&workers) |*w| w.roce.running.store(false, .release);
        for (proxies) |p| p.join();
    }
    for (&producers, &workers) |*p, *w| p.* = try std.Thread.spawn(.{}, Worker.producer, .{w});
    for (producers) |p| p.join();
    for (workers) |w| try t.expectEqual(@as(?anyerror, null), w.err);
}

test "relay: full decode boundaries and sequence wrap retain both half lengths" {
    var h: Harness = undefined;
    try h.init(guard.max_write);
    defer h.deinit();
    const start: u32 = 0xffff_fffe;
    for (0..4) |r| {
        h.doorbell[r] = start;
        h.consumed[r] = start;
        h.states[r].last = start;
        h.states[r].started = true;
        h.states[r].sent = .{ true, true };
        for (0..4) |src| for (0..2) |slot| for (0..2) |dir| {
            const at = h.fabric.layout.flagAt(@intCast(src), @intCast(slot), @intCast(dir));
            h.states[r].word(at).store(start, .release);
        };
    }
    const sizes = [_]u32{ 16, 48, 4096, 14336, 16384, 65536, 262144, 1048576 };
    for (sizes, 0..) |n, i| {
        const seq = start +% @as(u32, @intCast(i + 1));
        for (0..4) |r| try h.stage(@intCast(r), seq, n);
        for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
        for (0..4) |r| {
            try h.check(@intCast(r), seq);
            try t.expect(h.fabric.wires[r].peak.load(.monotonic) <= 2 * guard.max_write);
        }
    }
}

test "relay custody: consumed s-2 and every prior flag are required before a slot can post again" {
    var h: Harness = undefined;
    try h.init(4096);
    defer h.deinit();
    for (1..3) |s| {
        const seq: u32 = @intCast(s);
        for (0..4) |r| try h.stage(@intCast(r), seq, 16);
        for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
        for (0..4) |r| try h.check(@intCast(r), seq);
    }
    const posted = h.fabric.wires[0].posted;
    _ = try h.states[0].progress(&h.fabric.wires[0], 3, 0, 16, h.ticks);
    try t.expectEqual(posted, h.fabric.wires[0].posted);
    try t.expect(h.states[0].fault.stage == .custody);
    const at = h.fabric.layout.flagAt(2, 0, 1);
    h.states[0].word(at).store(0, .release);
    _ = try h.states[0].progress(&h.fabric.wires[0], 3, 2, 16, h.ticks);
    try t.expectEqual(posted, h.fabric.wires[0].posted);
    try t.expectEqual(@as(u32, 2), h.states[0].fault.rank);
    h.states[0].word(at).store(2, .release);
    _ = try h.states[0].progress(&h.fabric.wires[0], 3, 2, 16, h.ticks);
    try t.expect(h.fabric.wires[0].posted > posted);
}

test "relay custody: a fresh flag over stale payload is rejected by the every-word checker" {
    var h: Harness = undefined;
    try h.init(4096);
    defer h.deinit();
    h.fabric.omit_payload = true;
    for (0..4) |r| try h.stage(@intCast(r), 1, 4096);
    for (0..32) |_| for (0..4) |r| try h.tick(true, @intCast(r));
    try t.expect(h.ready(0, 1));
    try t.expectError(error.StaleWord, h.check(0, 1));
}
