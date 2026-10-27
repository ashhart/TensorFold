//! Tests of the paged snapshot: invalid geometry refused before allocation, flush order before host reads.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const prompt = flash.prompt;
const kvp = flash.exl3.kvpage;

test "exl3: invalid paged snapshot geometry rejects before allocation, CUDA access or TP peer dispatch" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var driver: cuda.Driver = undefined;
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.driver = &driver;
    engine.stream = &stream;
    var session: flash.rank_engine.Session = undefined;
    session.gpa = failing.allocator();
    session.engine = &engine;
    session.next_object = 77;
    var pager: kvp.Paged = undefined;
    pager.cfg = .{ .window = 0 };
    pager.cap = 200_016;
    var seq: prompt.Seq = undefined;
    seq.invalidated = false;
    seq.paged = &pager;
    var draft: flash.mtp.Draft = undefined;
    for ([_]usize{ 0, 1, 3, 4, 2051, 2052, 4096, 200_000 }) |at| {
        seq.fed = at;
        seq.cap = 200_016;
        seq.history = .{ 81, 93 };
        try std.testing.expectError(error.KvPageConfig, session.save(&seq, null));
        try std.testing.expectError(error.KvPageConfig, session.save(&seq, &draft));
        try std.testing.expectEqual(at, seq.fed);
        try std.testing.expectEqual([2]i64{ 81, 93 }, seq.history);
        try std.testing.expectEqual(@as(u64, 77), session.next_object);
    }
    try std.testing.expectEqual(@as(usize, 0), failing.alloc_index);
    try std.testing.expect(!failing.has_induced_failure);
}

test "exl3: paged flush orders all pending BF16 window rows before host snapshot reads" {
    const Fake = struct {
        const Copy = struct { dst: u64, src: u64, len: usize };
        var pending: [24]Copy = undefined;
        var count: usize = 0;

        fn copy(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(count < pending.len);
            pending[count] = .{ .dst = dst, .src = src, .len = len };
            count += 1;
            return cuda.abi.success;
        }

        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            for (pending[0..count]) |item| {
                const dst: [*]u8 = @ptrFromInt(item.dst);
                const src: [*]const u8 = @ptrFromInt(item.src);
                @memcpy(dst[0..item.len], src[0..item.len]);
            }
            count = 0;
            return cuda.abi.success;
        }

        fn pattern(ai: usize, side: usize, position: usize) u8 {
            return @intCast(ai * 7 + side * 3 + position);
        }
    };
    Fake.count = 0;
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    driver.api.cuStreamSynchronize = Fake.sync;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var pg: kvp.Paged = undefined;
    pg.d = &driver;
    pg.cfg = .{ .window = 2052, .rows = 1, .slots = 1 };
    pg.cap = 32;
    pg.wr = pg.cfg.window + pg.cfg.rows + 4;
    const layout = try kvp.SnapshotLayout.init(pg.cap, pg.cfg, 20);
    pg.nblocks = layout.nblocks;
    pg.dev.len = layout.device;
    pg.tabs.len = layout.tables;
    pg.rebases = @splat(7);
    pg.written = @splat(12);
    const a = std.testing.allocator;
    const device = try a.alloc(u8, pg.devBytes());
    defer a.free(device);
    const host = try a.alignedAlloc(u8, .@"16", 24 * pg.cap * kvp.row_bytes);
    defer a.free(host);
    @memset(host, 0xFF);
    pg.dev.ptr = @intFromPtr(device.ptr);
    pg.host = .{ .d = &driver, .bytes = @alignCast(host) };
    pg.host_dev = @intFromPtr(host.ptr);
    const fed: usize = 20;
    for (0..12) |ai| {
        pg.win0[ai] = ai % 3 * 4;
        for (0..2) |side| {
            const canonical = pg.hostMut(ai, side == 1);
            for (0..pg.win0[ai]) |position| @memset(canonical[position * kvp.row_bytes ..][0..kvp.row_bytes], Fake.pattern(ai, side, position));
            const window: [*]u8 = @ptrFromInt(pg.devRows(ai, side == 1));
            for (pg.win0[ai]..fed) |position| @memset(window[(position - pg.win0[ai]) * kvp.row_bytes ..][0..kvp.row_bytes], Fake.pattern(ai, side, position));
        }
    }
    const origins = pg.win0;
    try pg.flush(stream, fed);
    try std.testing.expectEqual(@as(usize, 24), Fake.count);
    for (0..12) |ai| for (0..2) |side| {
        try std.testing.expectEqual(@as(u8, 0xFF), pg.hostRows(ai, side == 1)[pg.win0[ai] * kvp.row_bytes]);
    };
    try stream.synchronize();
    try std.testing.expectEqual(@as(usize, 0), Fake.count);
    for (0..12) |ai| for (0..2) |side| {
        const canonical = pg.hostRows(ai, side == 1);
        for (0..fed) |position| for (canonical[position * kvp.row_bytes ..][0..kvp.row_bytes]) |byte| {
            try std.testing.expectEqual(Fake.pattern(ai, side, position), byte);
        };
        for (canonical[fed * kvp.row_bytes ..]) |byte| try std.testing.expectEqual(@as(u8, 0xFF), byte);
    };
    try std.testing.expectEqual(origins, pg.win0);
    try std.testing.expectEqual(@as([12]u64, @splat(7)), pg.rebases);
    try std.testing.expectEqual(@as([12]u64, @splat(12)), pg.written);
}
