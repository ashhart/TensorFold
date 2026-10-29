//! Read distinct PLE rows in table order, then copy duplicates after the parked readers have joined.
const std = @import("std");
const pool = @import("cuda_ple_pool.zig");

pub fn run(g: anytype, ids: []const i64, out: []u16) !void {
    const order = try g.gpa.alloc(usize, ids.len);
    defer g.gpa.free(order);
    const heads = try g.gpa.alloc(usize, ids.len);
    defer g.gpa.free(heads);
    for (order, 0..) |*v, i| v.* = i;
    const Sort = struct {
        fn less(keys: []const i64, a: usize, b: usize) bool {
            return keys[a] < keys[b] or (keys[a] == keys[b] and a < b);
        }
    };
    std.sort.pdq(usize, order, ids, Sort.less);
    var count: usize = 0;
    for (order, 0..) |index, i| {
        if (i == 0 or ids[index] != ids[order[i - 1]]) {
            heads[count] = index;
            count += 1;
        }
    }
    const Split = struct {
        reader: @TypeOf(g),
        ids: []const i64,
        out: []u16,
        heads: []const usize,
        per: usize,
        fn part(ctx: *const anyopaque, part_index: usize) anyerror!void {
            const self: *const @This() = @ptrCast(@alignCast(ctx));
            const lo = @min(self.heads.len, part_index * self.per);
            const hi = @min(self.heads.len, lo + self.per);
            for (self.heads[lo..hi]) |index| try self.reader.fill(self.ids, self.out, index, index + 1);
        }
    };
    const parts = if (g.pool == null) 1 else @max(1, @min(pool.max_parts, count / 4));
    const split: Split = .{ .reader = g, .ids = ids, .out = out, .heads = heads[0..count], .per = (count + parts - 1) / parts };
    if (g.pool) |p| try p.run(parts, .{ .ctx = &split, .run = Split.part }) else try Split.part(&split, 0);
    const width = g.hash.dh;
    var first: usize = 0;
    for (order, 0..) |index, i| {
        if (i == 0 or ids[index] != ids[order[i - 1]]) {
            first = index;
        } else @memcpy(out[index * width ..][0..width], out[first * width ..][0..width]);
    }
}

test "PLE distinct gather: each unique table row is read once and duplicates wait for every reader" {
    const Reader = struct {
        gpa: std.mem.Allocator = std.testing.allocator,
        pool: ?*pool.Pool,
        hash: struct { dh: usize = 2 } = .{},
        hits: [32]std.atomic.Value(u32) = @splat(.init(0)),
        pub fn fill(self: *@This(), ids: []const i64, out: []u16, lo: usize, hi: usize) !void {
            for (lo..hi) |i| {
                const id: usize = @intCast(ids[i]);
                if (self.hits[id].fetchAdd(1, .monotonic) != 0) return error.DuplicateRead;
                out[2 * i] = @intCast(id * 17);
                out[2 * i + 1] = @intCast(id * 19);
            }
        }
    };
    var reader: Reader = .{ .pool = try pool.Pool.create(std.testing.allocator, std.testing.io) };
    defer reader.pool.?.destroy();
    var ids: [257]i64 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast((i * 13) % 32);
    var out: [2 * ids.len + 2]u16 = @splat(0xbeef);
    try run(&reader, &ids, out[1 .. out.len - 1]);
    for (ids, 0..) |id, i| {
        try std.testing.expectEqual(@as(u16, @intCast(id * 17)), out[1 + 2 * i]);
        try std.testing.expectEqual(@as(u16, @intCast(id * 19)), out[2 + 2 * i]);
    }
    for (&reader.hits) |*n| try std.testing.expectEqual(@as(u32, 1), n.load(.monotonic));
    try std.testing.expectEqual(@as(u16, 0xbeef), out[0]);
    try std.testing.expectEqual(@as(u16, 0xbeef), out[out.len - 1]);
}
