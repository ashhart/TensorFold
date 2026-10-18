//! The union plan against the sort-and-search plan it replaced (kept here as the oracle): byte-identical blocks.

const std = @import("std");
const split = @import("split.zig");
const testing = std.testing;

fn refRange(gpa: std.mem.Allocator, in: split.UnionIn, a: u32, b: u32, out: *std.ArrayList(split.Block)) !void {
    const part = in.sel[@as(usize, a) * in.k .. @as(usize, b) * in.k];
    var uniq: std.ArrayList(u32) = .empty;
    defer uniq.deinit(gpa);
    try uniq.ensureTotalCapacity(gpa, part.len);
    for (part) |t| if (t >= 0) uniq.appendAssumeCapacity(@intCast(t));
    std.mem.sort(u32, uniq.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (uniq.items, 0..) |t, i| if (i == 0 or t != uniq.items[i - 1]) {
        uniq.items[n] = t;
        n += 1;
    };
    uniq.shrinkRetainingCapacity(n);
    // stable partition by owner: each owner's rows stay sorted, so a row's index in its list is a binary search
    var sizes: [split.max_world]u32 = @splat(0);
    for (uniq.items) |t| sizes[split.ownerOf(t, in.psh, in.world)] += 1;
    var m: u32 = 1;
    for (sizes[0..in.world]) |sz| m = @max(m, sz);
    if (@as(u64, in.world + 1) * m * in.row_bytes > in.cap and b - a > 1) {
        const h = a + (b - a) / 2;
        try refRange(gpa, in, a, h, out);
        return refRange(gpa, in, h, b, out);
    }
    const lists = try gpa.alloc(u32, uniq.items.len);
    defer gpa.free(lists);
    var at: [split.max_world]u32 = undefined;
    var acc: u32 = 0;
    for (0..in.world) |r| {
        at[r] = acc;
        acc += sizes[r];
    }
    const starts = at;
    for (uniq.items) |t| {
        const r = split.ownerOf(t, in.psh, in.world);
        lists[at[r]] = t;
        at[r] += 1;
    }
    var blk: split.Block = .{ .a = a, .b = b, .m = m, .phys = try gpa.alloc(u32, m), .tokens = try gpa.alloc(i32, part.len) };
    errdefer blk.deinit(gpa);
    const mask = (@as(u32, 1) << @intCast(in.psh)) - 1;
    const mine = lists[starts[in.rank]..][0..sizes[in.rank]];
    for (blk.phys, 0..) |*p, i| p.* = if (i < mine.len) (in.table[mine[i] >> @intCast(in.psh)] << @intCast(in.psh)) | (mine[i] & mask) else 0; // padding: any stored row
    for (part, blk.tokens) |t, *o| {
        if (t < 0) {
            o.* = -1;
            continue;
        }
        const u: u32 = @intCast(t);
        const r = split.ownerOf(u, in.psh, in.world);
        const list = lists[starts[r]..][0..sizes[r]];
        const idx = std.sort.binarySearch(u32, list, u, orderU32).?;
        o.* = @intCast(r * m + idx);
    }
    try out.append(gpa, blk);
}

fn orderU32(a: u32, b: u32) std.math.Order {
    return std.math.order(a, b);
}

/// A segment's selections over `kv_rows` stored rows: k random rows a row (repeats allowed), -1 past them.
fn selections(a: std.mem.Allocator, rnd: std.Random, rows: u32, k: u32, kv_rows: u32) ![]i32 {
    const sel = try a.alloc(i32, @as(usize, rows) * k);
    for (0..rows) |i| {
        const row = sel[i * k ..][0..k];
        const have = @min(k, kv_rows);
        for (row[0..have]) |*x| x.* = @intCast(rnd.uintLessThan(u32, kv_rows));
        if (kv_rows <= k) for (row[0..have], 0..) |*x, j| {
            x.* = @intCast(j);
        };
        @memset(row[have..], -1);
        if (i % 7 == 3) row[k - 1] = -1; // a short row
    }
    return sel;
}

fn expectSame(in: split.UnionIn, want_blocks: ?usize) !void {
    var got: std.ArrayList(split.Block) = .empty;
    defer {
        for (got.items) |*b| b.deinit(testing.allocator);
        got.deinit(testing.allocator);
    }
    var ref: std.ArrayList(split.Block) = .empty;
    defer {
        for (ref.items) |*b| b.deinit(testing.allocator);
        ref.deinit(testing.allocator);
    }
    try split.planUnion(testing.allocator, in, &got);
    try refRange(testing.allocator, in, 0, @intCast(in.sel.len / in.k), &ref);
    if (want_blocks) |n| try testing.expectEqual(n, ref.items.len);
    try testing.expectEqual(ref.items.len, got.items.len);
    for (ref.items, got.items) |r, g| {
        try testing.expectEqual(r.a, g.a);
        try testing.expectEqual(r.b, g.b);
        try testing.expectEqual(r.m, g.m);
        try testing.expectEqualSlices(u32, r.phys, g.phys);
        try testing.expectEqualSlices(i32, r.tokens, g.tokens);
    }
}

test "union plan == the sort-and-search plan: 7K, 184K and 1M contexts (1M capped into 8 blocks), both ranks, W 2 and 3" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    const psh = 6; // 256-token pages at ratio 4: 64 rows a page
    const cases = [_]struct { ctx: u32, world: u32, blocks: ?usize }{
        .{ .ctx = 7000, .world = 2, .blocks = 1 },
        .{ .ctx = 184000, .world = 2, .blocks = 1 },
        .{ .ctx = 1000000, .world = 2, .blocks = 8 },
        .{ .ctx = 184000, .world = 3, .blocks = null },
    };
    for (cases) |c| {
        const kv_rows = c.ctx / 4;
        const sel = try selections(a, rnd, 2048, 512, kv_rows);
        // the split table: each logical page's local page (a shuffled, sparse slot)
        const table = try a.alloc(u32, c.ctx / 256 + 1);
        for (table, 0..) |*t, i| t.* = @intCast(i / c.world);
        rnd.shuffle(u32, table);
        for (0..c.world) |rank| try expectSame(.{ .sel = sel, .k = 512, .table = table, .psh = psh, .world = c.world, .rank = @intCast(rank), .row_bytes = 584, .cap = 128 << 20 }, c.blocks);
    }
    // a tight cap: many blocks, down to one row
    const sel = try selections(a, rnd, 64, 512, 30000);
    const table = try a.alloc(u32, 30000 / 64 + 1);
    for (table, 0..) |*t, i| t.* = @intCast(i);
    try expectSame(.{ .sel = sel, .k = 512, .table = table, .psh = psh, .world = 2, .rank = 1, .row_bytes = 584, .cap = 3 * 300 * 584 }, null);
}
