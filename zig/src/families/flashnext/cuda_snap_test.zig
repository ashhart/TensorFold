const std = @import("std");
const cuda = @import("cuda");
const snap = @import("cuda_snap.zig");
const pager = @import("exl3/kv/cuda_exl3_kvpage.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const fixture_helpers = @import("cuda_snap_test_fixture.zig");
const Fake = fixture_helpers.Fake;
const Fixture = fixture_helpers.Fixture;

test {
    _ = @import("cuda_snap_shared_test.zig");
}

test "exl3: paged snapshot byte admission is checked and includes fixed payload and header" {
    const cfg: pager.Config = .{};
    const layout = try pager.SnapshotLayout.init(200_256, cfg, 200_000);
    try std.testing.expect(try snap.pagedBytes(200_256, cfg, 200_000, true) > layout.total + @sizeOf(snap.Snap));
    try std.testing.expectError(error.Overflow, snap.pagedBytes(std.math.maxInt(usize), cfg, 1, false));
    try std.testing.expectError(error.KvPageRows, snap.pagedBytes(3, cfg, 4, false));
    var image: pager.Snapshot = .{ .layout = layout, .win0 = @splat(196_000), .rebases = @splat(3), .written = @splat(8192) };
    try image.validate();
    image.layout.total -= 1;
    try std.testing.expectError(error.KvPageSnapshotLayout, image.validate());
}

test "exl3: paged snapshot restores distinct allocations and all state without source ownership" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var source: ?*Fixture = try Fixture.init(&driver);
    defer if (source) |fixture| fixture.deinit();
    const destination = try Fixture.init(&driver);
    defer destination.deinit();
    source.?.seq.fed = 4099;
    source.?.pg.win0 = @splat(2048);
    source.?.pg.win0[1] = 2052;
    source.?.pg.rebases = @splat(8);
    source.?.pg.written = @splat(2052);
    const controls: [*]u32 = @ptrFromInt(source.?.pg.tabs.ptr + 12 * 2 * source.?.pg.nblocks * 4);
    for (0..12) |layer| {
        controls[layer * 8] = 73;
        controls[layer * 8 + 6] = @intCast(17 + layer);
    }
    const saved = try snap.save(std.testing.allocator, &driver, stream, &source.?.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    try std.testing.expectEqual(saved.buf.len + @sizeOf(snap.Snap), try snap.pagedBytes(source.?.seq.cap, source.?.pg.cfg, saved.at, false));
    const expected = try std.testing.allocator.dupe(u8, saved.host.?.bytes[0..saved.buf.len]);
    defer std.testing.allocator.free(expected);
    source.?.deinit();
    source = null;
    const device_address = destination.pg.dev.ptr;
    for (0..2) |_| {
        try snap.restore(stream, &destination.seq, null, saved);
        try std.testing.expectEqual(device_address, destination.pg.dev.ptr);
        try std.testing.expectEqual(saved.history, destination.seq.history);
        try std.testing.expectEqual(saved.at, destination.seq.fed);
        try std.testing.expect(!destination.pg.graph_prepared and !destination.seq.invalidated);
        const again = try snap.save(std.testing.allocator, &driver, stream, &destination.seq, null);
        defer snap.drop(std.testing.allocator, stream, again);
        try std.testing.expectEqualDeep(saved.paged.?, again.paged.?);
        try std.testing.expectEqualSlices(u8, expected, again.host.?.bytes[0..again.buf.len]);
    }
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: paged snapshot submission failures drain and a partial restore stays invalid" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 4;
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    for (1..125) |failure| {
        Fake.submitted = 0;
        Fake.fail_at = failure;
        fixture.seq.invalidated = false;
        const previous = fixture.seq.fed;
        try std.testing.expectError(error.NotReady, snap.restore(stream, &fixture.seq, null, saved));
        try std.testing.expect(fixture.seq.invalidated);
        try std.testing.expectEqual(previous, fixture.seq.fed);
        try std.testing.expectEqual(@as(usize, 0), Fake.pending);
    }
    Fake.fail_at = null;
    fixture.seq.invalidated = false;
    var incompatible = saved.*;
    incompatible.version = 2;
    const submissions = Fake.submitted;
    try std.testing.expectError(error.SnapshotVersion, snap.restore(stream, &fixture.seq, null, &incompatible));
    try std.testing.expectEqual(submissions, Fake.submitted);
    try std.testing.expect(!fixture.seq.invalidated);
    fixture.pg.cfg.slots += 1;
    try std.testing.expectError(error.KvPageSnapshotLayout, snap.restore(stream, &fixture.seq, null, saved));
    try std.testing.expectEqual(submissions, Fake.submitted);
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: paged snapshot preserves MTP head mark and rejects stale verification publication" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 3;
    const allocator = fixture.arena.allocator();
    var head: @import("cuda_mtp_draft.zig").Draft = undefined;
    head.cap = fixture.seq.cap;
    head.kc = try Fixture.buffer(allocator, &driver, head.cap * prompt.kv_row_bytes, 0x80);
    head.vc = try Fixture.buffer(allocator, &driver, head.cap * prompt.kv_row_bytes, 0x7F);
    head.ikc = try Fixture.buffer(allocator, &driver, head.cap * wt.index_dim * 2, 0xFC);
    head.pooled = try Fixture.buffer(allocator, &driver, prompt.pooledBytes(head.cap), 0xA5);
    head.streams = try Fixture.buffer(allocator, &driver, 2 * wt.wide * 2, 0xFF);
    head.prompt_last = head.streams.ptr + wt.wide * 2;
    head.len = 3;
    head.drafted = 0;
    head.open = false;
    head.verify_cache = .{};
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, &head);
    defer snap.drop(std.testing.allocator, stream, saved);
    head.drafted = 1;
    try std.testing.expectError(error.NotAtMark, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, &head));
    head.drafted = 0;
    head.open = true;
    try std.testing.expectError(error.NotAtMark, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, &head));
    head.open = false;
    const generation = head.verify_cache.generation;
    head.verify_cache.publish(generation, 3, 3);
    try snap.restore(stream, &fixture.seq, &head, saved);
    head.verify_cache.publish(generation, 3, 3);
    try std.testing.expect(!head.verify_cache.take(3, 3, true));
    try std.testing.expectEqual(@as(usize, 3), head.len);
    try std.testing.expect(head.drafted == 0 and head.rows == 0 and !head.open);
    try std.testing.expectEqual(head.streams.ptr, head.prompt_last);
}

test "exl3: paged restore stages data before publishing sequence metadata" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 3;
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    fixture.seq.fed = 7;
    fixture.seq.history = .{ 111, 222 };
    try snap.restoreData(stream, &fixture.seq, null, saved);
    try std.testing.expectEqual(@as(usize, 7), fixture.seq.fed);
    try std.testing.expectEqual([2]i64{ 111, 222 }, fixture.seq.history);
    try std.testing.expect(fixture.seq.invalidated);
    snap.publishRestore(&fixture.seq, null, saved);
    try std.testing.expectEqual(@as(usize, 3), fixture.seq.fed);
    try std.testing.expectEqual(saved.history, fixture.seq.history);
    try std.testing.expect(!fixture.seq.invalidated);
}

test "exl3: paged save allocation, mapping and completion errors never publish or free pending copies" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 3;
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, snap.save(allocator.allocator(), &driver, stream, &fixture.seq, null));
    Fake.fail_alloc = true;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
    Fake.fail_alloc = false;
    Fake.fail_mapping = true;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
    Fake.fail_mapping = false;
    try std.testing.expect(!fixture.seq.invalidated);
    Fake.sync_calls = 0;
    Fake.fail_sync_at = 3;
    try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
    try std.testing.expect(fixture.seq.invalidated);
    try std.testing.expectEqual(@as(usize, 0), Fake.pending);
    try std.testing.expect(!Fake.freed_pending);
    for (Fake.allocations) |allocation| try std.testing.expect(allocation == null);
}

test "exl3: paged snapshot preflight sync failures drain and invalidate the sequence" {
    for (1..3) |failure_at| {
        Fake.reset();
        var driver = Fake.driver();
        const stream: cuda.Stream = .{ .d = &driver, .handle = null };
        const fixture = try Fixture.init(&driver);
        defer fixture.deinit();
        fixture.seq.fed = 3;
        Fake.fail_sync_at = failure_at;
        try std.testing.expectError(error.NotReady, snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null));
        try std.testing.expect(fixture.seq.invalidated);
        try std.testing.expectEqual(@as(usize, 0), Fake.pending);
        try std.testing.expect(!Fake.freed_pending);
    }
}

test "exl3: TP2 paged restore prepares both ranks and commits only after copy acknowledgements" {
    const wire = @import("cuda_rank_wire.zig");
    const ranks = @import("cuda_rank_engine.zig");
    Fake.reset();
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const fixture = try Fixture.init(&driver);
    defer fixture.deinit();
    fixture.seq.fed = 4;
    fixture.seq.rank_id = 9;
    const saved = try snap.save(std.testing.allocator, &driver, stream, &fixture.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    saved.rank_id = 13;
    var engine: prompt.Engine = undefined;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.round_graph = .{};
    for ([_]bool{ false, true }) |reject| {
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer _ = std.c.close(pair[0]);
        defer _ = std.c.close(pair[1]);
        const peer: wire.Link = .{ .fd = pair[1] };
        var session: ranks.Session = .{ .gpa = std.testing.allocator, .rank = 0, .engine = &engine, .link = .{ .fd = pair[0] }, .collective = undefined };
        session.collective.comm = null;
        session.collective.failed = false;
        fixture.seq.fed = 8;
        fixture.seq.invalidated = false;
        for (1..if (reject) @as(usize, 2) else 4) |sequence| try peer.sendHeader(.{ .kind = .ack, .sequence = sequence, .object = 9, .flags = if (reject) 2 else 0 }, wire.deadline(1000));
        const before = Fake.submitted;
        if (reject) {
            try std.testing.expectError(error.RemoteRank, session.restore(&fixture.seq, null, saved));
            try std.testing.expectEqual(before, Fake.submitted);
            try std.testing.expectEqual(@as(usize, 8), fixture.seq.fed);
            try std.testing.expect(fixture.seq.invalidated and !session.available.load(.acquire));
        } else {
            try session.restore(&fixture.seq, null, saved);
            try std.testing.expectEqual(saved.at, fixture.seq.fed);
            for ([_]wire.Kind{ .restore_check, .restore, .restore_commit }, 1..) |kind, sequence| {
                const command = try peer.receiveHeader(wire.deadline(1000));
                try std.testing.expectEqual(kind, command.kind);
                try std.testing.expectEqual(sequence, command.sequence);
                try std.testing.expectEqual(@as(u64, 13), command.arg);
            }
        }
    }
}
