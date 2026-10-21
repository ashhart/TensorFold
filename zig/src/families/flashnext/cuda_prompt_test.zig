const std = @import("std");
const cuda = @import("cuda");
const kvpage = @import("cuda_exl3_kvpage.zig");
const weights = @import("cuda_prompt_wt.zig");

pub fn graphLease(comptime Engine: type, comptime Seq: type, comptime operations: anytype) !void {
    const fixture = @import("cuda_exl3_shared_kv_test.zig");
    var driver = fixture.Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try kvpage.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 8192, .lanes = 2, .window = 2052, .rows = 1, .slots = 3 });
    defer arena.destroy() catch unreachable;
    fixture.Fake.install(arena);
    const first = try arena.register(8192);
    defer arena.release(first, stream) catch unreachable;
    const second = try arena.register(8192);
    defer arena.release(second, stream) catch unreachable;
    const cfg: kvpage.Config = .{ .window = 2052, .rows = 1, .slots = 3, .shared_lanes = 2 };
    var status_first: [weights.attn_n * 8 * 4]u8 align(16) = undefined;
    var status_second: [weights.attn_n * 8 * 4]u8 align(16) = undefined;
    var pager_first: kvpage.Paged = undefined;
    var pager_second: kvpage.Paged = undefined;
    for ([_]*kvpage.Paged{ &pager_first, &pager_second }, [_]kvpage.shared_contract.Sequence{ first, second }, [_][]align(16) u8{ &status_first, &status_second }) |pg, handle, status| {
        pg.d = &driver;
        pg.cfg = cfg;
        pg.shared_arena = arena;
        pg.shared_sequence = handle;
        pg.status_host = .{ .d = &driver, .bytes = status };
    }
    var sequence_first: Seq = undefined;
    var sequence_second: Seq = undefined;
    sequence_first.paged = &pager_first;
    sequence_second.paged = &pager_second;
    sequence_first.invalidated = false;
    sequence_second.invalidated = false;
    const seqs = [_]*Seq{ &sequence_first, &sequence_second };
    var engine: Engine = undefined;
    engine.page_config = cfg;
    engine.shared_arena = arena;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.collective = null;
    var other_driver = fixture.Fake.driver();
    var wrong_stream: cuda.Stream = .{ .d = &other_driver, .handle = null };
    engine.stream = &wrong_stream;
    {
        defer if (arena.active) |active| arena.cancel(active.lease) catch unreachable;
        const sync_before = fixture.Fake.sync_calls;
        try std.testing.expectError(error.SharedKvDriver, operations.begin(&engine, &seqs));
        try std.testing.expect(arena.active == null);
        try std.testing.expectEqual(sync_before, fixture.Fake.sync_calls);
        try std.testing.expect(!sequence_first.invalidated and !sequence_second.invalidated);
    }
    const recovered = try arena.acquireMany(&.{ first, second });
    try arena.cancel(recovered);
    engine.stream = &stream;
    const lease = try operations.begin(&engine, &seqs);
    try pager_first.validateSharedLease();
    try pager_second.validateSharedLease();
    const graph: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(1) };
    const before = fixture.Fake.sync_calls;
    try graph.launchOn(stream);
    try std.testing.expect(fixture.Fake.graph_pending);
    try std.testing.expectError(error.SharedKvBusy, arena.acquire(first));
    try operations.finish(&engine, &seqs);
    try std.testing.expect(!fixture.Fake.graph_pending);
    try std.testing.expectEqual(before + 1, fixture.Fake.sync_calls);
    operations.end(&engine, lease);
    try std.testing.expectEqual(before + 1, fixture.Fake.sync_calls);
    try std.testing.expectEqual(@as(usize, 1), fixture.Fake.graph_calls);
    try std.testing.expectError(error.SharedKvLease, pager_first.validateSharedLease());
    const failing = try operations.begin(&engine, &seqs);
    fixture.Fake.fail_graph = true;
    try std.testing.expectError(error.NotReady, graph.launchOn(stream));
    operations.fail(&engine, failing, &seqs);
    try std.testing.expect(sequence_first.invalidated and sequence_second.invalidated);
    try std.testing.expect(!fixture.Fake.graph_pending);
    try std.testing.expect(arena.active == null);
    try std.testing.expectError(error.SharedKvSequenceFailed, arena.acquire(first));
    try std.testing.expectError(error.SharedKvSequenceFailed, arena.acquire(second));
}

pub fn failedFill(comptime makeBuffers: anytype) !void {
    const Fake = struct {
        live: *usize,
        pub fn free(buffer: *@This()) void {
            buffer.live.* -= 1;
        }
    };
    const Src = struct {
        live: *usize,
        fills: *usize,
        fail_at: usize,
        pub fn make(source: @This()) !Fake {
            source.live.* += 1;
            return .{ .live = source.live };
        }
        pub fn fill(source: @This(), _: *Fake) !void {
            source.fills.* += 1;
            if (source.fills.* == source.fail_at) return error.FillFailed;
        }
    };
    for ([_]usize{ 1, 2, 4 }) |fail_at| {
        var live: usize = 0;
        var fills: usize = 0;
        try std.testing.expectError(error.FillFailed, makeBuffers(Fake, std.testing.allocator, 4, Src{ .live = &live, .fills = &fills, .fail_at = fail_at }));
        try std.testing.expectEqual(@as(usize, 0), live);
    }
    var live: usize = 0;
    var fills: usize = 0;
    const all = try makeBuffers(Fake, std.testing.allocator, 3, Src{ .live = &live, .fills = &fills, .fail_at = 0 });
    defer std.testing.allocator.free(all);
    try std.testing.expectEqual(@as(usize, 3), live);
}

pub fn layerIndices() !void {
    try std.testing.expect(weights.linearLayer(0));
    try std.testing.expect(!weights.linearLayer(3));
    try std.testing.expect(!weights.linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), weights.linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), weights.linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), weights.linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), weights.attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), weights.attnIndex(47));
}

pub fn sequenceBytes(comptime sequenceSize: anytype) !void {
    const per_token = weights.attn_n * (2 * weights.kv_heads * weights.head_dim * 2 + weights.index_dim * 2);
    try std.testing.expectEqual(per_token, sequenceSize(6) - sequenceSize(5));
    try std.testing.expectEqual(per_token + weights.attn_n * weights.index_dim * 2, sequenceSize(5) - sequenceSize(4));
    try std.testing.expect(sequenceSize(weights.window) > weights.window * per_token);
}
