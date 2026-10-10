//! The NVMe tier on real files: whole and delta files round-trip bit for bit, split ranks, damage.

const std = @import("std");
const testing = std.testing;
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Family = pool_mod.Family;
const HostPool = @import("pagestore.zig").HostPool;
const disk = @import("disk.zig");

const fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584, .split = true },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584, .split = true },
};

/// Fills a page of every family with bytes that name (family, logical page, byte).
fn fill(h: *HostPool, p: *const Pool, logical: u32, pg: u32, salt: u8) void {
    for (fams, 0..) |f, fi| {
        if (f.split and !p.owns(pg)) continue;
        const v = h.pageOf(@intCast(fi), p.familyPage(f, pg));
        for (v, 0..) |*b, i| b.* = @truncate(i *% 131 +% logical *% 7 +% fi *% 29 +% salt);
    }
}

fn familyPages(gpa: std.mem.Allocator, p: *const Pool, pages: []const u32, first: u32, owned_only: bool) ![][]u32 {
    const out = try gpa.alloc([]u32, fams.len);
    for (fams, 0..) |f, fi| {
        var l: std.ArrayList(u32) = .empty;
        for (pages[first..], first..) |pg, k| {
            if (f.split and p.world > 1 and owned_only and k % p.world != p.rank) continue;
            try l.append(gpa, p.familyPage(f, pg));
        }
        out[fi] = try l.toOwnedSlice(gpa);
    }
    return out;
}

fn freePages(gpa: std.mem.Allocator, x: [][]u32) void {
    for (x) |l| gpa.free(l);
    gpa.free(x);
}

fn roundTrip(world: u32, rank: u32) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    var p = try Pool.init(gpa, .{ .families = &fams, .page = 256 }, 64 * 256, world, rank);
    defer p.deinit();
    var h = try HostPool.init(gpa, &p);
    defer h.deinit();
    // small chunks so pool segments span many chunks; few staging buffers so the pump waits on the lanes
    const d = try disk.Disk.open(gpa, io, .{ .root = root, .compat = @splat(5), .rank = rank, .world = world, .chunk = 3 * 74_752, .staging = 3, .lanes = 3, .min_tokens = 1 });
    defer d.close();
    const a = try p.newSlot(32 * 256);
    const pos: u64 = 9 * 256 + 100;
    try a.ensure(pos);
    for (a.mapped(), 0..) |pg, k| fill(&h, &p, @intCast(k), pg, 1);
    const ids = try gpa.alloc(i32, pos);
    defer gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast(i * 3 + 1);
    var blob: [5000]u8 = undefined;
    for (&blob, 0..) |*x, i| x.* = @truncate(9 - i % 5);
    const fp = try familyPages(gpa, &p, a.mapped(), 0, true);
    defer freePages(gpa, fp);
    const src: disk.WriteSrc = .{ .key = @splat(1), .tag = 4, .page = 256, .pos = pos, .npages = a.len, .ids = ids, .store = h.store(), .families = &fams, .pages = fp, .blobs = &.{.{ .name = "rings", .bytes = &blob }} };
    const size = try d.writeFile(&src);
    _ = try d.register(&src, size);
    // a newer turn stacks a delta on it: pages [9, 14) only (page 9 was partial in the base)
    const pos2: u64 = 13 * 256 + 7;
    try a.ensure(pos2);
    for (a.mapped()[9..], 9..) |pg, k| fill(&h, &p, @intCast(k), pg, 2);
    const ids2 = try gpa.alloc(i32, pos2);
    defer gpa.free(ids2);
    for (ids2, 0..) |*t, i| t.* = @intCast(i * 3 + 1);
    const fp2 = try familyPages(gpa, &p, a.mapped(), 9, true);
    defer freePages(gpa, fp2);
    const src2: disk.WriteSrc = .{ .key = @splat(2), .base = @splat(1), .tag = 4, .page = 256, .pos = pos2, .first_page = 9, .npages = a.len, .ids = ids2[9 * 256 ..], .store = h.store(), .families = &fams, .pages = fp2, .blobs = &.{.{ .name = "rings", .bytes = blob[0..77] }} };
    const size2 = try d.writeFile(&src2);
    _ = try d.register(&src2, size2);
    try testing.expect(size2 < size);
    try testing.expectError(error.HasDependents, d.remove(@splat(1)));
    // restore the delta onto other pages through its chain
    const b = try p.newSlot(32 * 256);
    try b.ensure(pos2);
    var ext: std.ArrayList(disk.Extent) = .empty;
    defer ext.deinit(gpa);
    try d.extents(@splat(2), &ext);
    try testing.expectEqual(@as(usize, 2), ext.items.len);
    try testing.expectEqual(@as(u32, 9), ext.items[0].hi);
    const tp = try familyPages(gpa, &p, b.mapped(), 0, false);
    defer freePages(gpa, tp);
    const sink: disk.ReadSink = .{ .store = h.store(), .families = &fams, .page = 256, .world = world, .rank = rank, .pages = tp };
    try testing.expect(try d.readFile(ext.items[0], &sink, false) == null);
    var got = (try d.readFile(ext.items[1], &sink, true)).?;
    defer got.deinit(gpa);
    try testing.expectEqualSlices(u8, blob[0..77], got.blobs[0].bytes);
    for (a.mapped(), b.mapped()) |pa, pb| for (fams, 0..) |f, fi| {
        if (f.split and !p.owns(pa)) continue;
        try testing.expectEqualSlices(u8, h.pageOf(@intCast(fi), p.familyPage(f, pa)), h.pageOf(@intCast(fi), p.familyPage(f, pb)));
    };
    // damage one chunk: the read refuses it before or as its rows land
    var nb: [48]u8 = undefined;
    const key2: [16]u8 = @splat(2);
    const name = try std.fmt.bufPrint(&nb, "{x}.tfs", .{&key2});
    const f = try d.dir.openFile(io, name, .{ .mode = .read_write });
    var one = [_]u8{0};
    _ = try f.readPositional(io, &.{&one}, 3 * 4096 + 100);
    one[0] ^= 0xff;
    try f.writePositionalAll(io, &one, 3 * 4096 + 100);
    f.close(io);
    try testing.expectError(error.ChecksumMismatch, d.readFile(ext.items[1], &sink, true));
}

test "whole and delta files round-trip bit for bit (replicated)" {
    try roundTrip(1, 0);
}

test "split ranks write and restore their own pages (rank 0 and 1 of 2)" {
    try roundTrip(2, 0);
    try roundTrip(2, 1);
}
