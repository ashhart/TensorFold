//! A kept prompt state as a family's live parts in order, walked onto kept bytes (a buffer or pool pages) by one hook.
const std = @import("std");
const kv = @import("kv_pool.zig");

/// The unit kept states lease from a pool of their own (GLM's, Liquid's): a state holds whole pages.
pub const page: usize = 2 << 20;

/// Where kept byte o lies: `base` + o (no `ids`: one buffer), else `base` + ids[o / size] * size + o % size.
pub const Kept = struct { ids: []const u32 = &.{}, size: usize = page, base: u64 = 0 };

/// `bytes` rounded up to whole pages.
pub fn rounded(bytes: usize) !usize {
    return (try std.math.add(usize, bytes, page - 1)) & ~(page - 1);
}

/// A lease of the whole pages `bytes` take from `pool`, each one covered now: a keep or restore allocates nothing.
pub fn lease(pool: *kv.Pool, bytes: usize) !kv.Lease {
    const n = (try rounded(bytes)) / page;
    if (n == 0 or n > pool.available()) return error.NoSnapshotRoom;
    var l: kv.Lease = .{};
    errdefer pool.finish(&l);
    try pool.promise(&l, n);
    try pool.cover(&l, n, null);
    return l;
}

/// Parts over live addresses of type `Live`: `at(live, n)` is the address n bytes on (a pointer add, a buffer's Ref).
pub fn Walk(comptime Live: type, comptime at: fn (Live, usize) Live) type {
    return struct {
        /// `len` bytes at `live`, or on a sequence's pages: `span` bytes a page, `page` bytes apart, from byte `from`.
        pub const Part = struct { live: Live, len: usize, span: usize = 0, page: u64 = 0, from: usize = 0 };

        /// Where a part's bytes from `done` on lie (`ids`: the sequence's pages in row order), as many as lie in a row.
        pub fn piece(ids: []const u32, part: Part, done: usize) Part {
            if (part.span == 0) return .{ .live = at(part.live, done), .len = part.len - done };
            const o = part.from + done;
            const off = o % part.span;
            const by: usize = @intCast(@as(u64, ids[o / part.span]) * part.page + off);
            return .{ .live = at(part.live, by), .len = @min(part.span - off, part.len - done) };
        }

        /// The bytes `parts` hold, packed back to back.
        pub fn total(parts: []const Part) usize {
            var n: usize = 0;
            for (parts) |p| n += p.len;
            return n;
        }

        /// Each part in order against kept bytes from 0: f(ctx, live, kept, n), split where a live or kept page ends.
        pub fn each(
            parts: []const Part,
            ids: []const u32,
            kept: Kept,
            ctx: anytype,
            comptime f: fn (@TypeOf(ctx), Live, u64, usize) anyerror!void,
        ) !void {
            var off: usize = 0;
            for (parts) |part| {
                var done: usize = 0;
                while (done < part.len) {
                    const o = off + done;
                    const live = piece(ids, part, done);
                    if (kept.ids.len == 0) {
                        try f(ctx, live.live, kept.base + o, live.len);
                        done += live.len;
                        continue;
                    }
                    if (o / kept.size >= kept.ids.len) return error.PastKept;
                    const n = @min(live.len, kept.size - o % kept.size);
                    try f(ctx, live.live, kept.base + @as(u64, kept.ids[o / kept.size]) * kept.size + o % kept.size, n);
                    done += n;
                }
                off += part.len;
            }
        }
    };
}

fn add(a: u64, n: usize) u64 {
    return a + n;
}

/// Device pointers (CUDA, HIP): the hook copies between `live` and `kept` as addresses.
pub const Device = Walk(u64, add);

const testing = std.testing;

const Runs = struct {
    got: *std.ArrayList([3]u64),
    fn add(r: Runs, live: u64, kept: u64, n: usize) anyerror!void {
        try r.got.append(testing.allocator, .{ live, kept, n });
    }
};

test "parts walk onto one buffer whole, onto kept pages split at each page's end, live pages split at a span's" {
    var got: std.ArrayList([3]u64) = .empty;
    defer got.deinit(testing.allocator);
    const ids = [_]u32{ 3, 0, 2 }; // the live sequence's pages in row order
    const parts = [_]Device.Part{
        .{ .live = 1000, .len = 30 },
        .{ .live = 1 << 20, .len = 250, .span = 100, .page = 4096, .from = 20 }, // rows from byte 20 of its first span
    };
    try testing.expectEqual(@as(usize, 280), Device.total(&parts));
    try Device.each(&parts, &ids, .{ .base = 7 }, Runs{ .got = &got }, Runs.add);
    const flat = [_][3]u64{
        .{ 1000, 7, 30 },
        .{ (1 << 20) + 3 * 4096 + 20, 37, 80 },
        .{ (1 << 20) + 0 * 4096, 117, 100 },
        .{ (1 << 20) + 2 * 4096, 217, 70 },
    };
    try testing.expectEqualSlices([3]u64, &flat, got.items);
    got.clearRetainingCapacity();
    // kept pages of 64 bytes: ids 9, 4, 5, 1, 8 at base 1 << 30
    const kept: Kept = .{ .ids = &.{ 9, 4, 5, 1, 8 }, .size = 64, .base = 1 << 30 };
    try Device.each(&parts, &ids, kept, Runs{ .got = &got }, Runs.add);
    var live_done: u64 = 0;
    for (got.items) |r| {
        try testing.expect(r[2] > 0 and r[2] <= 64);
        const o = live_done;
        try testing.expectEqual((1 << 30) + @as(u64, kept.ids[o / 64]) * 64 + o % 64, r[1]);
        live_done += r[2];
    }
    try testing.expectEqual(@as(u64, 280), live_done);
    try testing.expectEqual([3]u64{ 1000, (1 << 30) + 9 * 64, 30 }, got.items[0]);
    try testing.expectEqual([3]u64{ (1 << 20) + 3 * 4096 + 20, (1 << 30) + 9 * 64 + 30, 34 }, got.items[1]);
    const short: Kept = .{ .ids = &.{ 9, 4 }, .size = 64 };
    try testing.expectError(error.PastKept, Device.each(&parts, &ids, short, Runs{ .got = &got }, Runs.add));
}

test "a walk over a buffer-and-offset address (Metal's Ref) resolves the same pieces as over pointers" {
    const Ref = struct {
        buf: u8,
        off: usize = 0,
        fn at(r: @This(), by: usize) @This() {
            return .{ .buf = r.buf, .off = r.off + by };
        }
    };
    const W = Walk(Ref, Ref.at);
    const parts = [_]W.Part{ .{ .live = .{ .buf = 1 }, .len = 8 }, .{ .live = .{ .buf = 2, .off = 4 }, .len = 12 } };
    const Got = struct {
        got: *std.ArrayList([4]u64),
        fn add(g: @This(), live: Ref, kept: u64, n: usize) anyerror!void {
            try g.got.append(testing.allocator, .{ live.buf, live.off, kept, n });
        }
    };
    var got: std.ArrayList([4]u64) = .empty;
    defer got.deinit(testing.allocator);
    try W.each(&parts, &.{}, .{ .base = 100 }, Got{ .got = &got }, Got.add);
    try testing.expectEqualSlices([4]u64, &.{ .{ 1, 0, 100, 8 }, .{ 2, 4, 108, 12 } }, got.items);
}

test "a state's lease takes whole pages, refuses none or more than are free, and gives them all back" {
    var pool = try kv.Pool.init(testing.allocator, 3, page);
    defer pool.deinit();
    try testing.expectEqual(@as(usize, 2 * page), try rounded(page + 1));
    try testing.expectError(error.Overflow, rounded(std.math.maxInt(usize)));
    var l = try lease(&pool, page + 1);
    try testing.expectEqual(@as(usize, 2), l.ids.items.len);
    try testing.expectEqual(@as(usize, 1), pool.available());
    try testing.expectError(error.NoSnapshotRoom, lease(&pool, 2 * page));
    try testing.expectError(error.NoSnapshotRoom, lease(&pool, 0));
    pool.finish(&l);
    try testing.expectEqual(@as(usize, 3), pool.available());
}
