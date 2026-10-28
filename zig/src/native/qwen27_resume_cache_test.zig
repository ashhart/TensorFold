//! Real cache lookups and slot-standing rules retain a useful fallback across an interleaved conversation.
const std = @import("std");
const pc = @import("engine_api").prompt_cache;
const rules = @import("qwen27_resume.zig");
const t = std.testing;

const Family = struct {
    at: u32 = 0,
    sum: u64 = 0,
    const State = struct { at: u32, sum: u64 };
    fn bytes(_: *anyopaque, at: u32) u64 {
        return (192 << 20) + @as(u64, at) * (64 << 10);
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
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn pass(f: *Family, s: *pc.Store, ids: []const u32, plan: pc.Plan, checkpoint: u32) u64 {
        if (plan.from == 0) f.* = .{};
        for (ids[plan.from..]) |token| {
            f.sum = f.sum *% 31 +% token;
            f.at += 1;
            if (f.at == checkpoint or std.mem.indexOfScalar(u32, plan.marks, f.at) != null) _ = s.keep(ids, f.at, null, &.{}, &.{});
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
    return rules.cachePoint(true, from, 512, ids.len, history, marks, &.{}, &.{});
}

test "qwen27 resume: A then B then A restores 512 with a cache too small for the history" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var family: Family = .{};
    var store = pc.Store.init(t.allocator, family.snapshots(), .{ .warm = .on, .chunk = 512, .chunk_marks = false }, 819 << 20);
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
    const b = try store.beginRewind(a, other, 17993, 3, &.{}, &.{}, null, &.{});
    try t.expectEqual(@as(u32, 3), b.from);
    try t.expectEqual(fresh(other), family.pass(&store, other, b, point(b.from, other, 17993, b.marks)));
    const resumed = try store.beginRewind(a, next, 16005, 15999, &.{}, &.{}, null, &.{});
    try t.expectEqual(@as(u32, 512), resumed.from);
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
    try t.expectEqual(@as(u32, 512), point(0, &context, 15993, &.{3}));
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
