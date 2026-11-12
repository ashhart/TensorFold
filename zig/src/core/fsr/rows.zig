//! PLE rows read from immutable tensor spans, deduplicated and coalesced without a resident table.
const std = @import("std");
const reader = @import("reader.zig");
pub const Shard = struct { rows: usize, parts: [3]reader.Span };
pub const Table = struct {
    a: std.mem.Allocator,
    shards: []const Shard,
    row_bytes: [3]usize,
    unique: []u32,
    inverse: []u32,
    slabs: [3][]u8,
    used: usize = 0,
    requests: usize = 0,
    reads: usize = 0,
    bytes: usize = 0,
    const max_read = 1 << 20;

    pub fn init(a: std.mem.Allocator, shards: []const Shard, maximum: usize, slabs: [3][]u8) !Table {
        if (shards.len == 0 or maximum == 0) return error.InvalidRowTable;
        var row_bytes: [3]usize = undefined;
        var total: usize = 0;
        for (shards, 0..) |shard, si| {
            if (shard.rows == 0) return error.InvalidRowTable;
            total = try std.math.add(usize, total, shard.rows);
            for (shard.parts, 0..) |span, j| {
                if (span.length == 0 or span.length % shard.rows != 0) return error.InvalidRowTable;
                const width = span.length / shard.rows;
                if (si == 0) row_bytes[j] = width else if (width != row_bytes[j]) return error.InconsistentRowWidth;
                if (width > max_read) return error.InvalidRowTable;
            }
        }
        if (total > std.math.maxInt(u32)) return error.InvalidRowTable;
        for (slabs, row_bytes) |slab, width| if (slab.len < try std.math.mul(usize, maximum, width)) return error.InvalidRowSlab;
        const unique = try a.alloc(u32, maximum);
        errdefer a.free(unique);
        const inverse = try a.alloc(u32, maximum);
        return .{ .a = a, .shards = shards, .row_bytes = row_bytes, .unique = unique, .inverse = inverse, .slabs = slabs };
    }

    pub fn deinit(t: *Table) void {
        t.a.free(t.unique);
        t.a.free(t.inverse);
    }

    fn locate(t: *const Table, id: u32) !struct { shard: usize, row: usize } {
        var row: usize = id;
        for (t.shards, 0..) |shard, si| {
            if (row < shard.rows) return .{ .shard = si, .row = row };
            row -= shard.rows;
        }
        return error.InvalidPleRow;
    }

    /// Slabs are not GPU-readable until this succeeds. Caller drains the previous row consumers before reuse.
    pub fn gather(t: *Table, source: anytype, ids: []const u32) !void {
        t.used = 0;
        t.requests = 0;
        t.reads = 0;
        t.bytes = 0;
        if (ids.len == 0 or ids.len > t.unique.len) return error.InvalidRowCount;
        for (ids) |id| _ = try t.locate(id);
        @memcpy(t.unique[0..ids.len], ids);
        std.mem.sort(u32, t.unique[0..ids.len], {}, std.sort.asc(u32));
        var count: usize = 0;
        for (0..ids.len) |i| {
            const id = t.unique[i];
            if (count == 0 or id != t.unique[count - 1]) {
                t.unique[count] = id;
                count += 1;
            }
        }
        for (ids, t.inverse[0..ids.len]) |id, *inverse| {
            var lo: usize = 0;
            var hi = count;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (t.unique[mid] < id) lo = mid + 1 else hi = mid;
            }
            inverse.* = @intCast(lo);
        }
        var requests: [reader.width]reader.Request = undefined;
        var sizes: [reader.width]u64 = undefined;
        var queued: usize = 0;
        var stats: reader.Stats = .{};
        for (0..3) |part| {
            const width = t.row_bytes[part];
            var at: usize = 0;
            while (at < count) {
                const first = try t.locate(t.unique[at]);
                var stop = at + 1;
                while (stop < count and (stop - at + 1) * width <= max_read) : (stop += 1) {
                    if (t.unique[stop] != t.unique[stop - 1] + 1) break;
                    const next = try t.locate(t.unique[stop]);
                    if (next.shard != first.shard) break;
                }
                var span = t.shards[first.shard].parts[part];
                span.offset = try std.math.add(u64, span.offset, try std.math.mul(u64, first.row, width));
                span.length = (stop - at) * width;
                const destination = t.slabs[part][at * width .. stop * width];
                const size = try source.fileSize(span.file);
                try reader.validate(span, size, destination.len);
                requests[queued] = .{ .span = span, .destination = destination };
                sizes[queued] = size;
                queued += 1;
                if (queued == reader.width) {
                    try drain(source, requests[0..queued], sizes[0..queued], &stats);
                    queued = 0;
                }
                t.reads += 1;
                t.bytes += span.length;
                at = stop;
            }
        }
        try drain(source, requests[0..queued], sizes[0..queued], &stats);
        t.requests = ids.len;
        t.used = count;
    }

    /// A batch of reads in flight together (readMany, else readAt in turn), then each file's size is rechecked.
    fn drain(source: anytype, requests: []const reader.Request, sizes: []const u64, stats: *reader.Stats) !void {
        if (requests.len == 0) return;
        try reader.flush(source, requests, stats);
        for (requests, sizes) |request, size| if (try source.fileSize(request.span.file) != size) return error.CheckpointChanged;
    }
};

test "FSR PLE rows preserve component bytes, duplicate order and shard edges without full tables" {
    const Fake = struct {
        data: []const u8,
        pub fn fileSize(f: @This(), _: usize) !u64 {
            return f.data.len;
        }
        pub fn readAt(f: @This(), _: usize, dst: []u8, offset: u64) !usize {
            @memcpy(dst, f.data[@intCast(offset)..][0..dst.len]);
            return dst.len;
        }
    };
    var data: [36]u8 = undefined;
    for (&data, 0..) |*v, i| v.* = @intCast(i);
    const shards = [_]Shard{
        .{ .rows = 3, .parts = .{ .{ .file = 0, .offset = 0, .length = 12 }, .{ .file = 0, .offset = 12, .length = 6 }, .{ .file = 0, .offset = 18, .length = 6 } } },
        .{ .rows = 1, .parts = .{ .{ .file = 0, .offset = 24, .length = 4 }, .{ .file = 0, .offset = 28, .length = 2 }, .{ .file = 0, .offset = 30, .length = 2 } } },
    };
    var w: [16]u8 = undefined;
    var s: [8]u8 = undefined;
    var b: [8]u8 = undefined;
    var t = try Table.init(std.testing.allocator, &shards, 4, .{ &w, &s, &b });
    defer t.deinit();
    const f: Fake = .{ .data = &data };
    try t.gather(f, &.{ 3, 1, 2, 1 });
    try std.testing.expectEqualSlices(u32, &.{ 2, 0, 1, 0 }, t.inverse[0..4]);
    try std.testing.expectEqual(@as(usize, 3), t.used);
    try std.testing.expectEqual(@as(usize, 6), t.reads);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 26, 27 }, w[0..12]);
    try std.testing.expectEqualSlices(u8, &.{ 14, 15, 16, 17, 28, 29 }, s[0..6]);
    try std.testing.expectEqualSlices(u8, &.{ 20, 21, 22, 23, 30, 31 }, b[0..6]);
    try std.testing.expectError(error.InvalidPleRow, t.gather(f, &.{4}));
    try std.testing.expectEqual(@as(usize, 0), t.used);
    try std.testing.expectEqual(@as(usize, 0), t.reads);
    var bad = shards;
    bad[1].parts[0].length = 8;
    try std.testing.expectError(error.InconsistentRowWidth, Table.init(std.testing.allocator, &bad, 4, .{ &w, &s, &b }));
}

test "FSR PLE rows reach the source's readMany in batches of at most reader.width" {
    const Fake = struct {
        data: [256]u8,
        batches: usize = 0,
        widest: usize = 0,
        singles: usize = 0,
        pub fn fileSize(f: *const @This(), _: usize) !u64 {
            return f.data.len;
        }
        pub fn readAt(f: *@This(), _: usize, dst: []u8, offset: u64) !usize {
            f.singles += 1;
            @memcpy(dst, f.data[@intCast(offset)..][0..dst.len]);
            return dst.len;
        }
        pub fn readMany(f: *@This(), requests: []const reader.Request, _: *reader.Stats) !void {
            f.batches += 1;
            f.widest = @max(f.widest, requests.len);
            for (requests) |q| @memcpy(q.destination, f.data[@intCast(q.span.offset)..][0..q.destination.len]);
        }
    };
    var fake: Fake = .{ .data = undefined };
    for (&fake.data, 0..) |*v, i| v.* = @intCast(i);
    const shards = [_]Shard{.{ .rows = 64, .parts = .{
        .{ .file = 0, .offset = 0, .length = 128 },
        .{ .file = 0, .offset = 128, .length = 64 },
        .{ .file = 0, .offset = 192, .length = 64 },
    } }};
    var w: [32]u8 = undefined;
    var s: [16]u8 = undefined;
    var b: [16]u8 = undefined;
    var t = try Table.init(std.testing.allocator, &shards, 16, .{ &w, &s, &b });
    defer t.deinit();
    try t.gather(&fake, &.{ 0, 2, 4, 6, 8, 10, 12, 14 }); // apart, so each id is its own run: 8 runs x 3 parts
    try std.testing.expectEqual(@as(usize, 24), t.reads);
    try std.testing.expectEqual(@as(usize, 3), fake.batches);
    try std.testing.expectEqual(@as(usize, reader.width), fake.widest);
    try std.testing.expectEqual(@as(usize, 0), fake.singles);
    for (0..8) |k| {
        try std.testing.expectEqualSlices(u8, fake.data[4 * k ..][0..2], w[2 * k ..][0..2]);
        try std.testing.expectEqual(fake.data[128 + 2 * k], s[k]);
        try std.testing.expectEqual(fake.data[192 + 2 * k], b[k]);
    }
}
