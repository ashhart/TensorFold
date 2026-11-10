//! State pieces on any pages of a lease: admitted and counted, fragmentation, failures, the load report's pad.
const std = @import("std");
const kv = @import("kv_pool.zig");
const t = std.testing;

const mib: u64 = 1 << 20;

/// A run of 10 pages, or the same state as 37 pieces on 10 pages (page 4 MiB, 64 KiB a token).
const run: kv.Layout = .{ .fixed_bytes = 64 * mib, .token_bytes = 64 << 10, .page_tokens = 64, .snap_bytes = 0, .run_bytes = 10 * 4 * mib - 1 };
const pieced: kv.Layout = .{ .fixed_bytes = 64 * mib, .token_bytes = 64 << 10, .page_tokens = 64, .snap_bytes = 0, .pieces = .{ .n = 37, .pages = 10, .bytes = 36 * mib } };

test "pieces count as their pages wherever a run counted: need, plan, holds" {
    try t.expectEqual(@as(usize, 10), pieced.statePages());
    try t.expectEqual(@as(usize, 10), run.statePages());
    try t.expectEqual(run.need(2048, 4096, 16), pieced.need(2048, 4096, 16));
    const g: u64 = 1 << 30;
    for ([_]kv.Budget{
        .{ .room = 40 * g, .engine = g, .lanes = 4, .window = 16384 + 16 },
        .{ .room = 2 * g + 2 * (64 * mib + 11 * 4 * mib) + mib, .engine = 2 * g, .lanes = 8 },
        .{ .room = 40 * g, .engine = g, .lanes = 4, .cache = 2 * g, .window = 8192, .engine_paged = true },
    }) |b| {
        try t.expectEqual(try kv.plan(run, b), try kv.plan(pieced, b));
        try t.expectEqual(kv.holds(run, b), kv.holds(pieced, b));
    }
    var fewer = pieced;
    fewer.pieces.?.pages = 9;
    try t.expectEqual(run.need(1, 0, 0) - 1, fewer.need(1, 0, 0));
}

test "a lane's fixed part counts at admission: more of it outside the pool admits fewer streams" {
    const g: u64 = 1 << 30;
    const b: kv.Budget = .{ .room = 2 * g + 3 * (64 * mib + 11 * 4 * mib) + mib, .engine = 2 * g, .lanes = 8 };
    try t.expectEqual(@as(u32, 4), (try kv.plan(pieced, b)).lanes);
    var wide = pieced;
    wide.fixed_bytes += 48 * mib;
    wide.outside = 48 * mib;
    try t.expectEqual(@as(u32, 3), (try kv.plan(wide, b)).lanes);
}

test "takeState holds any free pages from the promise; finish gives them back with the table's" {
    var p = try kv.Pool.init(t.allocator, 16, 4096);
    defer p.deinit();
    const load = p.census();
    var l: kv.Lease = .{};
    try p.promise(&l, 7);
    try p.takeState(&l, 4);
    try p.cover(&l, 2, null);
    try t.expectEqual(@as(usize, 4), l.state.items.len);
    try t.expectEqual(@as(usize, 1), l.promised);
    try t.expectEqual(@as(usize, 6), p.census().leased);
    try t.expectEqual(@as(usize, 0), p.reclaimable()); // a state page is never the store's to reclaim
    for (l.state.items) |id| try t.expectEqual(@as(u32, 1), p.holders(id));
    try t.expectError(error.PastReservation, p.takeState(&l, 2));
    try t.expectEqual(@as(usize, 1), l.promised);
    p.finish(&l);
    try t.expectEqual(load, p.census());
    try t.expect(p.idle());
}

test "fragmentation: no two free pages side by side, a run is refused, the same state as pieces admits" {
    var p = try kv.Pool.init(t.allocator, 8, 4096);
    defer p.deinit();
    var all: [8]u32 = undefined;
    try t.expect(p.hold(&all));
    for ([_]u32{ 1, 3, 5, 7 }) |id| try p.drop(id); // the store keeps every other page
    var a: kv.Lease = .{};
    try p.promise(&a, 4);
    try t.expectEqual(@as(?u32, null), p.claimRun(&a, 2)); // today's run: LaneMemory
    try p.takeState(&a, 4);
    try t.expectEqual(@as(usize, 0), p.available());
    var got = a.state.items[0..4].*;
    std.mem.sort(u32, &got, {}, std.sort.asc(u32));
    try t.expectEqualSlices(u32, &.{ 1, 3, 5, 7 }, &got);
    p.finish(&a);
    for ([_]u32{ 0, 2, 4, 6 }) |id| try p.drop(id);
    try t.expect(p.idle());
}

test "LEAK: a failed state allocation, a cancel after open and a failed table write leave the pool as at load" {
    var p = try kv.Pool.init(t.allocator, 16, 4096);
    defer p.deinit();
    const load = p.census();
    var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const keep = p.gpa;
    p.gpa = fail.allocator();
    var l: kv.Lease = .{};
    try p.promise(&l, 6);
    try t.expectError(error.OutOfMemory, p.takeState(&l, 4));
    try t.expectEqual(@as(usize, 6), l.promised); // nothing taken
    p.finish(&l);
    p.gpa = keep;
    try t.expectEqual(load, p.census());
    try p.promise(&l, 6);
    try p.takeState(&l, 4);
    p.finish(&l); // cancelled after open
    try t.expectEqual(load, p.census());
    const Bad = struct {
        fn write(_: *anyopaque, _: []const u32, _: usize) anyerror!void {
            return error.Upload;
        }
    };
    var ctx: u8 = 0;
    try p.promise(&l, 6);
    try p.takeState(&l, 4);
    try t.expectError(error.Upload, p.cover(&l, 2, .{ .ctx = &ctx, .write = Bad.write }));
    try t.expectEqual(@as(usize, 0), l.ids.items.len);
    p.finish(&l);
    try t.expectEqual(load, p.census());
    try t.expect(p.idle());
}

test "the load report shows a lane's pieces, their pages and pad, and its fixed part outside the pool" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var l = pieced;
    l.outside = 48 * mib;
    l.fixed_bytes += 48 * mib;
    const shown = try kv.groups.line(arena.allocator(), l.shape());
    const want = "KV page groups: one page 4.00 MiB, 64 rows of the full group; full: layers not named, 64.00 KiB a token, " ++
        "64 rows a page, 0 B padding (0.0 %); state, in the pool as pieces: 37 pieces, 36.00 MiB on 10 pages a lane, " ++
        "4.00 MiB padding (10.0 %); a lane's fixed part, outside the pool and counted at admission: 112.00 MiB, " ++
        "48.00 MiB of it buffers wider than a page";
    try t.expectEqualStrings(want, shown);
    const plain = try kv.groups.line(arena.allocator(), run.shape()); // no pieces, nothing outside: today's line
    try t.expect(std.mem.indexOf(u8, plain, "pieces") == null and std.mem.indexOf(u8, plain, "fixed part") == null);
}

test "takeState takes the highest free ids and keeps the free stack's order, so table pages still come from the bottom" {
    var p = try kv.Pool.init(t.allocator, 12, 4096);
    defer p.deinit();
    var a: kv.Lease = .{};
    try p.promise(&a, 5);
    try p.takeState(&a, 3);
    try t.expectEqualSlices(u32, &.{ 11, 10, 9 }, a.state.items);
    var b: kv.Lease = .{};
    try p.promise(&b, 4);
    try p.takeState(&b, 2);
    try t.expectEqualSlices(u32, &.{ 8, 7 }, b.state.items);
    const Ok = struct {
        fn write(_: *anyopaque, _: []const u32, _: usize) anyerror!void {}
    };
    var ctx: u8 = 0;
    try p.cover(&a, 2, .{ .ctx = &ctx, .write = Ok.write });
    try p.cover(&b, 2, .{ .ctx = &ctx, .write = Ok.write });
    try t.expectEqualSlices(u32, &.{ 0, 1 }, a.ids.items);
    try t.expectEqualSlices(u32, &.{ 2, 3 }, b.ids.items);
    p.finish(&a);
    p.finish(&b);
    try t.expect(p.idle());
}
