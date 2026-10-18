//! engram_rows.zig against reference Engram host fixtures: bf16 table, packed shard via sync, threads and io_uring.

const std = @import("std");
const testing = std.testing;
const er = @import("engram_rows.zig");
const eh = @import("engram_host.zig");

const ref = @embedFile("fixtures/engram-rows-ref.txt");

/// engram_fixture.record: byte j of global row r.
fn recordByte(r: u64, j: u64) u8 {
    const h = (r * 2654435761 + j * 40503) % (1 << 32);
    return if (j < 256) @truncate(h >> 7) else @intCast(100 + (h >> 9) % 40);
}

const Fx = struct { lo: u64, rows: u64, total: u64, sets: [3]struct { idx: []i64, sha: [32]u8 } };

fn parseFx(a: std.mem.Allocator, lut_sha: *[32]u8) !Fx {
    var fx: Fx = undefined;
    var set: usize = 0;
    var it = std.mem.tokenizeScalar(u8, ref, '\n');
    while (it.next()) |line| {
        var w = std.mem.tokenizeScalar(u8, line, ' ');
        const kind = w.next().?;
        if (std.mem.eql(u8, kind, "lut")) {
            _ = try std.fmt.hexToBytes(lut_sha, w.next().?);
        } else if (std.mem.eql(u8, kind, "shard")) {
            fx.lo = try std.fmt.parseInt(u64, w.next().?, 10);
            fx.rows = try std.fmt.parseInt(u64, w.next().?, 10);
            fx.total = try std.fmt.parseInt(u64, w.next().?, 10);
        } else {
            var ids: std.ArrayList(i64) = .empty;
            var c = std.mem.tokenizeScalar(u8, w.next().?, ',');
            while (c.next()) |x| try ids.append(a, try std.fmt.parseInt(i64, x, 10));
            fx.sets[set].idx = try ids.toOwnedSlice(a);
            _ = try std.fmt.hexToBytes(&fx.sets[set].sha, w.next().?);
            set += 1;
        }
    }
    return fx;
}

/// Writes the fixture's shard (the packer's layout) as engram-l1-r0of2.bin under `dir`.
fn writeShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, fx: Fx) !void {
    return writeShardAs(a, io, dir, fx, 1);
}

fn writeShardAs(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, fx: Fx, layer: u64) !void {
    const n: usize = @intCast(er.header_bytes + fx.rows * er.row_bytes);
    const b = try a.alloc(u8, n);
    defer a.free(b);
    @memset(b, 0);
    for ([_]u64{ er.magic, layer, fx.lo, fx.lo + fx.rows, fx.total, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    for (0..fx.rows) |k| for (0..er.row_bytes) |j| {
        b[er.header_bytes + k * er.row_bytes + j] = recordByte(fx.lo + k, j);
    };
    var nb: [32]u8 = undefined;
    try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "engram-l{d}-r0of2.bin", .{layer}), .data = b });
}

fn sha(out: []const u16) [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(out), &d, .{});
    return d;
}

/// How a test reads: serially on the calling thread, by the reader threads (no io_uring), or by io_uring.
const Mode = enum { sync, pool, uring, aio, aio_poll };
const modes = [_]Mode{ .sync, .pool, .uring, .aio, .aio_poll };

fn openRows(a: std.mem.Allocator, io: std.Io, tmp: *testing.TmpDir, mode: Mode) !*er.Rows {
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const r = try er.Rows.open(a, root, 0, 2, &.{1}, 4096);
    if (mode == .aio or mode == .aio_poll) {
        try r.startAio(); // TF_DSV41_ENGRAM_AIO's reader: native AIO, no threads
        r.aio.?.poll = mode == .aio_poll; // + TF_DSV41_ENGRAM_POLL: waits polled
        return r;
    }
    if (r.aio) |*x| { // TF_DSV41_ENGRAM_AIO=1 in the environment: the other modes without it
        _ = x;
        r.aio.?.deinit();
        r.aio = null;
    }
    if (mode != .uring) if (r.ring) |*g| {
        g.deinit();
        r.ring = null;
    };
    if (mode == .pool and r.pool == null) try r.startPool(8);
    if (mode == .sync) r.stopPool(); // no io_uring here: open started the pool
    return r;
}

test "engram rows: a packed shard read as the reference ShardRows.rows(...).to(bfloat16), synchronously, by the reader threads and by io_uring" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, io, tmp.dir, fx);
    for (modes) |mode| {
        const r = try openRows(a, io, &tmp, mode);
        defer r.close();
        try testing.expectEqual(fx.lo, r.tables[0].h.lo);
        for (fx.sets) |s| {
            const out = try a.alloc(u16, s.idx.len * 256);
            defer a.free(out);
            try r.rows(0, s.idx, 256, out);
            try testing.expectEqual(s.sha, sha(out));
            // again: every record from the cache now, the same bits
            const hits = r.stats.hits;
            try r.rows(0, s.idx, 256, out);
            try testing.expectEqual(s.sha, sha(out));
            try testing.expect(r.stats.hits > hits);
        }
        try testing.expectError(error.EngramRowOutsideShard, r.rows(0, &.{@intCast(fx.lo - 1)}, 256, try arena.allocator().alloc(u16, 256)));
    }
}

test "engram rows: issued ahead, then waited for; bulk reads past the cache; eviction keeps the bits" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, io, tmp.dir, fx);
    for (modes) |mode| {
        const r = try openRows(a, io, &tmp, mode);
        defer r.close();
        const s = fx.sets[1];
        try r.issue(0, s.idx);
        try testing.expectEqual(@as(u64, s.idx.len), r.stats.issued);
        try r.issue(0, s.idx); // nothing new: every row is in flight or cached
        try testing.expectEqual(@as(u64, s.idx.len), r.stats.issued);
        const out = try a.alloc(u16, s.idx.len * 256);
        defer a.free(out);
        try r.rows(0, s.idx, 256, out);
        try testing.expectEqual(s.sha, sha(out));
        try testing.expectEqual(@as(u64, 0), r.stats.misses);
        // every row of the shard once (a prompt's bulk read: past cap / 4, read through), then the sets again
        const all = try a.alloc(i64, fx.rows);
        defer a.free(all);
        for (all, 0..) |*x, k| x.* = @intCast(fx.lo + fx.rows - 1 - k);
        const big = try a.alloc(u16, all.len * 256);
        defer a.free(big);
        try r.rows(0, all, 256, big);
        for (all, 0..) |x, k| for (0..256) |j| {
            const row: u64 = @intCast(x);
            const v = recordByte(row, j);
            const sc = recordByte(row, 256 + j / 32);
            try testing.expectEqual(r.lut[@as(usize, sc) << 8 | v], big[k * 256 + j]);
        };
        for (fx.sets) |q| {
            const o = try a.alloc(u16, q.idx.len * 256);
            defer a.free(o);
            try r.rows(0, q.idx, 256, o);
            try testing.expectEqual(q.sha, sha(o));
        }
        try testing.expect(r.tables[0].used <= 4096);
    }
}

test "engram rows: a prompt segment's bulk reads issued ahead wait through one unrelated read, then serve it" {
    // TF_DSV41_PREFETCH_AHEAD: segment k + 1's read issued before k reads; k must not drop it, k + 1 finds every row
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, io, tmp.dir, fx);
    const all = try a.alloc(i64, fx.rows);
    defer a.free(all);
    for (all, 0..) |*x, k| x.* = @intCast(fx.lo + k);
    try testing.expect(all.len > 4096 / 4); // a bulk batch: read through, not cached
    const big = try a.alloc(u16, all.len * 256);
    defer a.free(big);
    for (modes) |mode| {
        const r = try openRows(a, io, &tmp, mode);
        defer r.close();
        try r.issue(0, all);
        const s = fx.sets[0];
        const o = try a.alloc(u16, s.idx.len * 256);
        defer a.free(o);
        try r.rows(0, s.idx, 256, o); // segment k: waits for the rows it shares with the batch, the batch stays
        try testing.expectEqual(s.sha, sha(o));
        const reads = r.stats.reads;
        try r.rows(0, all, 256, big); // segment k + 1: every row from the batch issued ahead
        try testing.expectEqual(@as(u64, 0), r.stats.misses);
        try testing.expectEqual(reads, r.stats.reads);
        for (all, 0..) |x, k| for (0..256) |j| {
            const row: u64 = @intCast(x);
            try testing.expectEqual(r.lut[@as(usize, recordByte(row, 256 + j / 32)) << 8 | recordByte(row, j)], big[k * 256 + j]);
        };
        // a wrong guess: issued, then two unrelated reads of its table, then it is gone (read again on demand)
        try r.issue(0, all);
        try r.rows(0, s.idx, 256, o);
        try r.rows(0, s.idx, 256, o);
        const misses = r.stats.misses;
        try r.rows(0, all, 256, big);
        try testing.expect(r.stats.misses > misses);
        for (r.batches.items) |b| try testing.expect(!b.in_use);
    }
}

test "engram rows: a prompt segment ahead on two tables survives the segment's layer-1 then layer-14 reads" {
    // both layers' batches of k + 1 issued, k reads 1 then 14, k + 1 reads 1 (settle) then 14: no miss on either table
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShardAs(a, io, tmp.dir, fx, 1);
    try writeShardAs(a, io, tmp.dir, fx, 14);
    const all = try a.alloc(i64, fx.rows);
    defer a.free(all);
    for (all, 0..) |*x, k| x.* = @intCast(fx.lo + k);
    const big = try a.alloc(u16, all.len * 256);
    defer a.free(big);
    const s = fx.sets[0];
    const o = try a.alloc(u16, s.idx.len * 256);
    defer a.free(o);
    for (modes) |mode| {
        const root = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(root);
        const r = try er.Rows.open(a, root, 0, 2, &.{ 1, 14 }, 4096);
        defer r.close();
        if (mode != .uring) if (r.ring) |*g| {
            g.deinit();
            r.ring = null;
        };
        if (mode == .pool and r.pool == null) try r.startPool(8);
        if (mode == .sync) r.stopPool();
        try r.issue(0, all);
        try r.issue(1, all);
        for (0..2) |li| {
            try r.rows(li, s.idx, 256, o); // segment k: a few shared rows of each table (waits for the batch)
            try testing.expectEqual(s.sha, sha(o));
        }
        const reads = r.stats.reads;
        for (0..2) |li| {
            try r.rows(li, all, 256, big); // segment k + 1: layer 1, then layer 14, all from the batches issued ahead
            try testing.expectEqual(@as(u64, 0), r.stats.misses);
            try testing.expectEqual(reads, r.stats.reads);
            for (all, 0..) |x, k| for (0..256) |j| {
                const row: u64 = @intCast(x);
                try testing.expectEqual(r.lut[@as(usize, recordByte(row, 256 + j / 32)) << 8 | recordByte(row, j)], big[k * 256 + j]);
            };
        }
        for (r.batches.items) |b| try testing.expect(!b.in_use);
    }
}

test "engram rows: the reader threads with many batches in flight (issued, missed, waited for out of order)" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, io, tmp.dir, fx);
    const r = try openRows(a, io, &tmp, .pool);
    defer r.close();
    var rng = std.Random.DefaultPrng.init(4101);
    const rand = rng.random();
    const sets = try a.alloc([]i64, 48);
    defer {
        for (sets) |q| a.free(q);
        a.free(sets);
    }
    for (sets) |*q| {
        q.* = try a.alloc(i64, 1 + rand.uintLessThan(usize, 300));
        for (q.*) |*x| x.* = @intCast(fx.lo + rand.uintLessThan(u64, fx.rows));
    }
    // every set issued ahead, then read in reverse (later batches waited for first), each read checked value by value
    for (sets) |q| try r.issue(0, q);
    var k = sets.len;
    while (k > 0) {
        k -= 1;
        const q = sets[k];
        const out = try a.alloc(u16, q.len * 256);
        defer a.free(out);
        try r.rows(0, q, 256, out);
        for (q, 0..) |x, i| for (0..256) |j| {
            const row: u64 = @intCast(x);
            try testing.expectEqual(r.lut[@as(usize, recordByte(row, 256 + j / 32)) << 8 | recordByte(row, j)], out[i * 256 + j]);
        };
    }
    try testing.expect(r.stats.issued > 0);
}

test "engram rows: shard headers" {
    var b: [48]u8 = undefined;
    for ([_]u64{ er.magic, 14, 10, 20, 100, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    const h = try er.Header.parse(&b, er.header_bytes + 10 * er.row_bytes);
    try testing.expectEqual(@as(u64, 14), h.layer);
    try testing.expectError(error.BadEngramShard, er.Header.parse(&b, er.header_bytes + 9 * er.row_bytes));
    b[0] ^= 1;
    try testing.expectError(error.NotEngramShard, er.Header.parse(&b, er.header_bytes + 10 * er.row_bytes));
}

test "engram lookback: the tail, then the kept rows of the window in flight" {
    var buf: [16]u32 = undefined;
    // nothing in flight: the next window starts at the slot's position
    try testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, eh.lookbackAt(&.{ 7, 8, 9 }, 50, null, 50, 3, &buf).?);
    try testing.expect(eh.lookbackAt(&.{ 7, 8, 9 }, 50, null, 51, 3, &buf) == null);
    // a window [a, d1, d2] at 50 not kept yet; the next starts at 52: a and d1 were kept
    try testing.expectEqualSlices(u32, &.{ 9, 100, 101 }, eh.lookbackAt(&.{ 7, 8, 9 }, 50, &.{ 100, 101, 102 }, 52, 3, &buf).?);
    try testing.expectEqualSlices(u32, &.{ 100, 101, 102 }, eh.lookbackAt(&.{ 7, 8, 9 }, 50, &.{ 100, 101, 102 }, 53, 3, &buf).?);
    try testing.expect(eh.lookbackAt(&.{ 7, 8, 9 }, 50, &.{ 100, 101, 102 }, 54, 3, &buf) == null);
    // the sequence start: fewer than keep_n
    try testing.expectEqualSlices(u32, &.{5}, eh.lookbackAt(&.{}, 0, &.{ 5, 6 }, 1, 3, &buf).?);
}

/// Slots of the record map that hold neither a key nor a tombstone: a lookup that misses probes until it finds one.
fn freeSlots(r: *const er.Rows) usize {
    const m = &r.tables[0].map;
    const meta = m.metadata orelse return 0;
    var n: usize = 0;
    for (meta[0..m.capacity()]) |x| n += @intFromBool(x.isFree());
    return n;
}

test "engram rows: a long run of bulk reads and evictions leaves the record map free slots (no probe of the whole map)" {
    // bulk reads past cap / 4 (inserted, then dropped) and evictions, as a long uncached prefill does
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const fx: Fx = .{ .lo = 1000, .rows = 20000, .total = 40000, .sets = undefined };
    try writeShard(a, io, tmp.dir, fx);
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const r = try er.Rows.open(a, root, 0, 2, &.{1}, 4096);
    defer r.close();
    const ids = try a.alloc(i64, 1500);
    defer a.free(ids);
    const out = try a.alloc(u16, ids.len * 256);
    defer a.free(out);
    var rng = std.Random.DefaultPrng.init(7);
    // the worst sample: free slots against the map's slots (it grows meanwhile)
    var worst: [2]usize = .{ 1, 1 };
    for (0..300) |i| {
        // a prompt segment's bulk read (past cap / 4: read through, its keys dropped), then a decode window's few
        const n: usize = if (i % 2 == 0) ids.len else 64;
        for (ids[0..n]) |*x| x.* = @intCast(fx.lo + rng.random().uintLessThan(u64, fx.rows));
        try r.rows(0, ids[0..n], 256, out[0 .. n * 256]);
        const all = r.tables[0].map.capacity();
        const free = freeSlots(r);
        if (free * worst[1] < worst[0] * all) worst = .{ free, all };
    }
    errdefer std.debug.print("record map: {d} free of {d} slots at the worst\n", .{ worst[0], worst[1] });
    try testing.expect(worst[0] * 10 >= worst[1]);
}
