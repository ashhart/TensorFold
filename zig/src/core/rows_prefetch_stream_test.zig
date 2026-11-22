//! Bounded prefetch streams: byte identity across ring turns, fixed ownership and terminal errors.
const std = @import("std");
const platform = @import("sys");
const rows = @import("rows_store.zig");
const prefetch = @import("rows_prefetch.zig");
const Fake = @import("rows_fake.zig").Fake;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const gpa = std.testing.allocator;
const t = std.testing;

fn byteAt(i: usize) u8 {
    return @truncate((i * 131 + i / 251 + 17) ^ (i >> 9));
}

fn pattern(buf: []u8, base: usize) void {
    for (buf, 0..) |*b, i| b.* = byteAt(base + i);
}

const Stream = struct {
    r: *rows.Rows,
    f: *prefetch.Prefetch,
    rd: ?*prefetch.Read,
    name: rows.chain.Name,
    len: usize,

    fn open(dir: []const u8, len: usize, queued: bool) !Stream {
        var fake: Fake = .{ .gpa = gpa };
        const r = try rows.Rows.open(gpa, fake.snapshots(), .{ .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 });
        errdefer r.deinit();
        const f = try prefetch.make(r, t.io, .{});
        var name: rows.chain.Name = undefined;
        for (&name, 0..) |*b, i| b.* = @truncate(i * 7 + 3);
        var w = try rows.file.Writer.open(dir, &rows.chain.hex(name), ".blk");
        errdefer w.abort();
        var buf: [64 << 10]u8 = undefined;
        var done: usize = 0;
        while (done < len) {
            const n = @min(buf.len, len - done);
            pattern(buf[0..n], done);
            try w.append(buf[0..n]);
            done += n;
        }
        _ = try w.finish(.{ .kind = @backingInt(rows.file.Kind.block), .payload = 0, .payload_hash = 0, .name = name, .block = r.keying.block }, &.{}, false);

        const pieces = try gpa.alloc(prefetch.Piece, 1);
        errdefer gpa.free(pieces);
        pieces[0] = .{ .name = name, .kind = .block, .at = 0, .len = len, .first = 0, .end = 0, .off = 0 };
        const rd = try gpa.create(prefetch.Read);
        errdefer gpa.destroy(rd);
        const first = try gpa.alignedAlloc(u8, .fromByteUnits(rows.file.alignment), prefetch.chunk_bytes);
        errdefer gpa.free(first);
        const second = try gpa.alignedAlloc(u8, .fromByteUnits(rows.file.alignment), prefetch.chunk_bytes);
        errdefer gpa.free(second);
        rd.* = .{ .name = name, .plan = .{ .pieces = pieces, .bytes = len }, .since = 0, .slots = .{ .{ .buf = first }, .{ .buf = second } }, .phase = if (queued) .queued else .reading, .producer_active = !queued };
        f.mutex.lockUncancelable(f.io);
        f.reads.append(f.gpa, rd) catch |err| {
            f.mutex.unlock(f.io);
            return err;
        };
        f.held_reads = 1;
        f.held_bytes = 2 * prefetch.chunk_bytes;
        f.counts.made = 1;
        f.mutex.unlock(f.io);
        const detached = f.take(name, &rd.plan);
        return .{ .r = r, .f = f, .rd = detached, .name = name, .len = len };
    }

    fn produce(s: *Stream) void {
        const err: ?anyerror = blk: {
            prefetch.readFiles(s.f, s.rd.?) catch |e| break :blk e;
            break :blk null;
        };
        const f = s.f;
        const rd = s.rd.?;
        f.mutex.lockUncancelable(f.io);
        rd.err = err;
        rd.ns = 1;
        rd.phase = if (err == null) .ready else .failed;
        rd.producer_active = false;
        if (err != null) f.counts.failed += 1;
        f.cond.broadcast(f.io);
        f.mutex.unlock(f.io);
    }

    fn phase(s: *Stream) prefetch.Phase {
        s.f.mutex.lockUncancelable(s.f.io);
        defer s.f.mutex.unlock(s.f.io);
        return s.rd.?.phase;
    }

    fn close(s: *Stream) void {
        if (s.rd) |rd| s.f.finish(rd);
        s.rd = null;
        s.r.deinit();
    }
};

test "prefetch stream: multi-chunk identity, fixed buffers and readiness before recycling" {
    var path: [128]u8 = undefined;
    const dir = std.fmt.bufPrint(&path, "/tmp/tf-prefetch-stream-{d}", .{platform.c.getpid()}) catch unreachable;
    defer rmTree(dir);
    var s = try Stream.open(dir, 2 * prefetch.chunk_bytes + 16_384 + 17, false);
    defer s.close();
    const snap = s.f.snapshot();
    try t.expectEqual(@as(u64, 2 * prefetch.chunk_bytes), snap.held_bytes);
    try t.expectEqual(@as(usize, 1), snap.held_reads);
    const producer = try std.Thread.spawn(.{}, Stream.produce, .{&s});
    defer producer.join();
    var at: usize = 0;
    var chunks: usize = 0;
    while (try s.f.next(s.rd.?)) |chunk| {
        try t.expectEqual(@as(usize, 0), chunk.piece);
        try t.expectEqual(@as(u64, at), chunk.off);
        try t.expect(chunk.bytes.len <= prefetch.chunk_bytes);
        for (chunk.bytes, 0..) |b, i| try t.expectEqual(byteAt(at + i), b);
        at += chunk.bytes.len;
        chunks += 1;
        s.f.recycle(s.rd.?, chunk);
        if (chunks == 1) {
            while (s.phase() == .reading) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
            try t.expectEqual(prefetch.Phase.ready, s.phase());
        }
    }
    try t.expectEqual(s.len, at);
    try t.expectEqual(@as(usize, 3), chunks);
}

test "prefetch stream: a corrupt trailer hash reaches a detached consumer; source stays present" {
    var path: [128]u8 = undefined;
    const dir = std.fmt.bufPrint(&path, "/tmp/tf-prefetch-stream-fail-{d}", .{platform.c.getpid()}) catch unreachable;
    defer rmTree(dir);
    var s = try Stream.open(dir, prefetch.chunk_bytes + 93, false);
    defer s.close();
    var name_buf: [1200]u8 = undefined;
    const source = try rows.file.path(&name_buf, dir, &rows.chain.hex(s.name), ".blk", false);
    const fd = platform.c.open(source, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    try t.expect(fd >= 0);
    try t.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, 100));
    _ = platform.c.close(fd);
    const producer = try std.Thread.spawn(.{}, Stream.produce, .{&s});
    defer producer.join();
    var failed = false;
    while (true) {
        const chunk = s.f.next(s.rd.?) catch |err| {
            try t.expectEqual(error.RowsHash, err);
            failed = true;
            break;
        };
        if (chunk) |c| {
            s.f.recycle(s.rd.?, c);
            continue;
        }
        break;
    }
    try t.expect(failed);
    s.f.finish(s.rd.?);
    s.rd = null;
    var opened = try rows.file.open(source, .block, s.name);
    opened.close();
    try t.expectEqual(@as(u64, 0), s.f.snapshot().held_bytes);
}

test "prefetch stream: a queued read falls back and releases its slots without stranding restore" {
    var path: [128]u8 = undefined;
    const dir = std.fmt.bufPrint(&path, "/tmp/tf-prefetch-stream-queued-{d}", .{platform.c.getpid()}) catch unreachable;
    defer rmTree(dir);
    var s = try Stream.open(dir, 93, true);
    defer s.close();
    try t.expectEqual(@as(?*prefetch.Read, null), s.rd);
    const snap = s.f.snapshot();
    try t.expectEqual(@as(u64, 0), snap.held_bytes);
    try t.expectEqual(@as(usize, 0), snap.held_reads);
    var name_buf: [1200]u8 = undefined;
    const source = try rows.file.path(&name_buf, dir, &rows.chain.hex(s.name), ".blk", false);
    var opened = try rows.file.open(source, .block, s.name);
    opened.close();
}
