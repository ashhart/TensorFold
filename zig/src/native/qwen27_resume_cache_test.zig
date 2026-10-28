//! Real cache lookups and slot-standing rules retain a useful fallback across an interleaved conversation.
const std = @import("std");
const pc = @import("engine_api").prompt_cache;
const rules = @import("qwen27_resume.zig");
const t = std.testing;

const Family = struct {
    at: u32 = 0,
    sum: u64 = 0,
    spare_bytes: u64 = 0,
    const State = struct { at: u32, sum: u64 };
    fn bytes(_: *anyopaque, at: u32) u64 {
        // Rounded fixture matches the supplied short, checkpoint and history charges without allocating device buffers.
        const raw = (146 << 20) + @as(u64, at) * (64 << 10);
        return std.mem.alignForward(u64, raw + raw / 8, 32 << 20);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f: *Family = @ptrCast(@alignCast(ptr));
        if (f.at != at) return error.NotAtMark;
        const s = try t.allocator.create(State);
        s.* = .{ .at = at, .sum = f.sum };
        return s;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        const f: *Family = @ptrCast(@alignCast(ptr));
        const s: *const State = @ptrCast(@alignCast(saved));
        f.* = .{ .at = s.at, .sum = s.sum };
    }
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        t.allocator.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn spare(ptr: *anyopaque) u64 {
        return @as(*Family, @ptrCast(@alignCast(ptr))).spare_bytes;
    }
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .spare = spare } };
    }
    fn pass(f: *Family, s: *pc.Store, ids: []const u32, plan: pc.Plan, checkpoint: u32) u64 {
        if (plan.from == 0) f.* = .{};
        for (ids[plan.from..]) |token| {
            f.sum = f.sum *% 31 +% token;
            f.at += 1;
            if (std.mem.indexOfScalar(u32, plan.marks, f.at) != null or (f.at == checkpoint and rules.checkpointFits(s, ids, f.at, plan.marks, &.{}))) _ = s.keep(ids, f.at, null, &.{}, &.{});
        }
        return f.sum;
    }
};

fn fresh(ids: []const u32) u64 {
    var sum: u64 = 0;
    for (ids) |id| sum = sum *% 31 +% id;
    return sum;
}

fn point(from: u32, ids: []const u32, history: u32, marks: []const u32) u32 {
    return rules.cachePoint(true, from, 2048, ids.len, history, marks, &.{}, &.{});
}

test "qwen27 resume: A then B then A restores 2048 with a cache too small for the history" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var family: Family = .{};
    var store = pc.Store.init(t.allocator, family.snapshots(), .{ .warm = .on, .chunk = 2048, .chunk_marks = false }, 819 << 20);
    defer store.deinit();
    const first = try a.alloc(u32, 16000);
    const other = try a.alloc(u32, 18000);
    const next = try a.alloc(u32, first.len + 12);
    for (first, 0..) |*id, i| id.* = @intCast(i + 1);
    for (other, 0..) |*id, i| id.* = if (i < 3) @intCast(i + 1) else @intCast(100000 + i);
    @memcpy(next[0..first.len], first);
    @memset(next[first.len..], 7);
    const p = try store.beginRewind(a, first, 15993, 3, &.{}, &.{}, null, &.{});
    try t.expectEqualSlices(u32, &.{3}, p.marks);
    try t.expectEqual(fresh(first), family.pass(&store, first, p, point(p.from, first, 15993, p.marks)));
    try t.expect(!store.keep(first, 15993, null, &.{}, &.{}));
    const saved = store.find(next, &.{}, &.{}).?.saved;
    const b = try store.beginRewind(a, other, 17993, 3, &.{}, &.{}, null, &.{});
    try t.expectEqual(@as(u32, 3), b.from);
    try t.expectEqual(fresh(other), family.pass(&store, other, b, point(b.from, other, 17993, b.marks)));
    try t.expectEqual(saved, store.find(next, &.{}, &.{}).?.saved);
    try t.expectEqual(@as(u64, 0), store.counts.evicted);
    const resumed = try store.beginRewind(a, next, 16005, 15999, &.{}, &.{}, null, &.{});
    try t.expectEqual(@as(u32, 2048), resumed.from);
    try t.expectEqual(fresh(next), family.pass(&store, next, resumed, point(resumed.from, next, 16005, resumed.marks)));
    try t.expect(store.held <= store.budget);
}

test "qwen27 resume: the cache checkpoint leaves the live slot available for an immediate next turn" {
    var st: rules.Standing = .{};
    defer st.deinit(t.allocator);
    var context: [16000]u32 = undefined;
    for (&context, 0..) |*id, i| id.* = @intCast(i + 1);
    st.leave(t.allocator, .{ .at = 16000, .prompt_len = 16000, .context_len = 16000, .fill_at = 16000, .ended = true }, &context, &.{});
    const next = context ++ [_]u32{ 9, 10 };
    try t.expectEqual(@as(u32, 2048), point(0, &context, 15993, &.{3}));
    try t.expectEqual(rules.Resume{ .here = 16000 }, st.plan(&next, &.{}, 512));
    try t.expectEqual(@as(rules.Resume, .none), st.plan(&.{ 1, 2, 3, 999, 1000 }, &.{}, 3));
}

test "qwen27 resume: fallback respects physical cuts, spans and already planned history states" {
    try t.expectEqual(@as(u32, 0), rules.cachePoint(false, 0, 512, 16000, 15993, &.{3}, &.{}, &.{}));
    try t.expectEqual(@as(u32, 0), rules.cachePoint(true, 0, 512, 16000, 15993, &.{15993}, &.{}, &.{}));
    try t.expectEqual(@as(u32, 0), rules.cachePoint(true, 0, 512, 100, 93, &.{}, &.{}, &.{}));
    try t.expectEqual(@as(u32, 0), rules.cachePoint(true, 15900, 512, 16000, 15993, &.{}, &.{}, &.{}));
    try t.expectEqual(@as(u32, 256), rules.cachePoint(true, 0, 512, 16000, 15993, &.{3}, &.{256}, &.{}));
    try t.expectEqual(@as(u32, 128), rules.cachePoint(true, 0, 512, 16000, 15993, &.{3}, &.{}, &.{.{ 128, 200 }}));
}

fn interleaved(first: []const u32, other: []const u32, next: []const u32, budget: u64, fallback: bool) !u32 {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var family: Family = .{};
    var store = pc.Store.init(t.allocator, family.snapshots(), .{ .warm = .on, .chunk = 2048, .chunk_marks = false }, budget);
    defer store.deinit();
    const history: u32 = @intCast(first.len - 7);
    const p = try store.beginRewind(a, first, history, 3, &.{}, &.{}, null, &.{});
    try t.expectEqual(fresh(first), family.pass(&store, first, p, if (fallback) point(p.from, first, history, p.marks) else 0));
    _ = store.keep(first, @intCast(first.len), null, &.{}, &.{});
    if (!fallback) {
        // Main's warm pass yields its first chunk when B arrives; B's warm pass finishes before A returns.
        var kept: u32 = 0;
        for (store.entries.items) |e| if (store.wouldServe(next, &.{}, &.{}, e.tokens, e.decode_spans, e.at)) {
            kept = @max(kept, e.at);
        };
        family.at = if (kept > 3) @intCast(@min(first.len, @as(usize, kept) + 2048)) else 2048;
        family.sum = fresh(first[0..family.at]);
        _ = store.keep(first, family.at, null, &.{}, &.{});
    }
    const bh: u32 = @intCast(other.len - 7);
    const b = try store.beginRewind(a, other, bh, 3, &.{}, &.{}, null, &.{});
    try t.expectEqual(fresh(other), family.pass(&store, other, b, if (fallback) point(b.from, other, bh, b.marks) else 0));
    _ = store.keep(other, @intCast(other.len), null, &.{}, &.{});
    const resumed = try store.beginRewind(a, next, @intCast(next.len - 7), @intCast(first.len - 1), &.{}, &.{}, null, &.{});
    try t.expectEqual(fresh(next), family.pass(&store, next, resumed, 0));
    try t.expect(store.held + store.spare() <= budget);
    return resumed.from;
}

test "qwen27 resume: returning A is no worse than main across every rounded budget boundary" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 16000, 32000 }) |len| {
        const first = try a.alloc(u32, len);
        const other = try a.alloc(u32, len + 2000);
        const next = try a.alloc(u32, len + 92);
        for (first, 0..) |*id, i| id.* = @intCast(i + 1);
        for (other, 0..) |*id, i| id.* = if (i < 3) @intCast(i + 1) else @intCast(100000 + i);
        @memcpy(next[0..len], first);
        @memset(next[len..], 7);
        // All charges and sums are multiples of 32 MiB; adjacent bytes cover each admission class through 16 GiB.
        for (0..513) |step| for ([_]u64{ (@as(u64, step) << 25) -| 1, @as(u64, step) << 25, (@as(u64, step) << 25) + 1 }) |budget| {
            const base = try interleaved(first, other, next, budget, false);
            const new = try interleaved(first, other, next, budget, true);
            errdefer std.debug.print("prompt {d}, budget {d}, main {d}, fallback {d}\n", .{ len, budget, base, new });
            try t.expect(new >= base);
        };
    }
}

test "qwen27 resume: fallback admission reserves future marks, spare buffers and state slots" {
    var family: Family = .{};
    var store = pc.Store.init(t.allocator, family.snapshots(), .{}, 512 << 20);
    defer store.deinit();
    var ids: [5000]u32 = @splat(1);
    try t.expect(rules.checkpointFits(&store, &ids, 2048, &.{}, &.{}));
    try t.expect(!rules.checkpointFits(&store, &ids, 2048, &.{4096}, &.{}));
    store.budget = 320 << 20;
    family.spare_bytes = 32 << 20;
    try t.expect(!rules.checkpointFits(&store, &ids, 2048, &.{}, &.{}));
    family.spare_bytes = 0;
    try t.expect(rules.checkpointFits(&store, &ids, 2048, &.{}, &.{}));
    store.budget = 512 << 20;
    store.most = 0;
    try t.expect(!rules.checkpointFits(&store, &ids, 2048, &.{}, &.{}));
    store.most = 1;
    try t.expect(!rules.checkpointFits(&store, &ids, 512, &.{1024}, &.{}));
    store.hold = true;
    try t.expect(!rules.checkpointFits(&store, &ids, 2048, &.{}, &.{}));
    store.hold = false;
    family.at = 2048;
    family.sum = fresh(ids[0..2048]);
    try t.expect(store.keep(&ids, 2048, null, &.{}, &.{}));
    const clock = store.clock;
    try t.expect(!rules.checkpointFits(&store, &ids, 2048, &.{}, &.{}));
    try t.expectEqual(clock, store.clock);
    try t.expect(!rules.checkpointFits(&store, &ids, 2051, &.{}, &.{}));
    try t.expectEqual(@as(u64, 0), store.counts.evicted);
    try t.expectEqual(@as(u64, 192 << 20), Family.bytes(&family, 3));
    try t.expectEqual(@as(u64, 320 << 20), Family.bytes(&family, 2048));
    try t.expectEqual(@as(u64, 1280 << 20), Family.bytes(&family, 15650));
    store.budget = 544 << 20;
    store.most = 2;
    try t.expect(rules.checkpointFits(&store, &ids, 512, &.{2048}, &.{}));
    try t.expect(!rules.checkpointFits(&store, &ids, 512, &.{2048}, &.{.{ 1000, 2000 }}));
    const other_key: [5000]u32 = @splat(2);
    try t.expect(!rules.checkpointFits(&store, &other_key, 512, &.{2048}, &.{}));
}

test "qwen27 resume: a 2048 checkpoint stays reachable beside 512 row decoding chunks" {
    for ([_]usize{ 0, 3 }) |from| {
        const checkpoint = rules.cachePoint(true, from, 2048, 16000, 15993, &.{3}, &.{}, &.{});
        try t.expectEqual(@as(u32, @intCast(from + 2048)), checkpoint);
        var at = from;
        while (at < checkpoint) {
            const end = rules.chunkEnd(at, 512, 16000, &.{}, &.{});
            at = @min(end, checkpoint);
        }
        try t.expectEqual(@as(usize, checkpoint), at);
    }
}
