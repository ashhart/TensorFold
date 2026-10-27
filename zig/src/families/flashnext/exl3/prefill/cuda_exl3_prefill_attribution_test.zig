const std = @import("std");
const cuda = @import("cuda");
const attribution = @import("cuda_exl3_prefill_attribution.zig");
const ring_module = @import("cuda_exl3_prefill_ring.zig");
const pool_module = @import("../offload/cuda_exl3_pool.zig");
const telemetry = @import("../model/exl3_telemetry.zig");
const State = std.meta.Child(@FieldType(ring_module.Ring, "state"));
const allocator = std.testing.allocator;

const Calls = struct {
    copies: usize = 0,
    payload_bytes: usize = 0,
    records: usize = 0,
    queries: usize = 0,
    waits: usize = 0,
    syncs: usize = 0,
};

const Fixture = struct {
    driver: cuda.Driver,
    ring: *ring_module.Ring,
    pool: *pool_module.Pool,
    payload: [8]u8,
    sources: [512][3][]const u8,
    counts: [48][512]u32,
    addresses: [2]u64,
    offsets: [2][3]u32,
    rooms: [2][3]u32,
    states: [2]State,
    owners: [2]u32,
    epochs: [2]u64,

    var calls: Calls = .{};
    var copy_result: cuda.abi.Result = cuda.abi.success;

    fn copy(destination: u64, _: ?*const anyopaque, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        calls.copies += 1;
        if (destination < 0x9000) calls.payload_bytes += bytes;
        return copy_result;
    }

    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        calls.records += 1;
        return cuda.abi.success;
    }

    fn query(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        calls.queries += 1;
        return cuda.abi.error_not_ready;
    }

    fn wait(_: cuda.abi.Stream, _: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        calls.waits += 1;
        return cuda.abi.success;
    }

    fn sync(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        calls.syncs += 1;
        return cuda.abi.success;
    }

    fn init(enabled: bool) !*Fixture {
        const fixture = try allocator.create(Fixture);
        errdefer allocator.destroy(fixture);
        fixture.* = undefined;
        fixture.ring = try allocator.create(ring_module.Ring);
        errdefer allocator.destroy(fixture.ring);
        fixture.pool = try allocator.create(pool_module.Pool);
        errdefer allocator.destroy(fixture.pool);
        fixture.pool.* = undefined;
        fixture.pool.opt = .{ .layer = 0, .lines = 2 };
        fixture.pool.src = .{ .routed = &fixture.sources };
        fixture.pool.counts = &fixture.counts;
        fixture.pool.lines.tables.ptr = 0x9000;
        fixture.payload = @splat(7);
        for (&fixture.sources, 0..) |*source, index| {
            const length = index % 4 + 1;
            source.* = .{ fixture.payload[0..length], fixture.payload[0 .. length + 1], fixture.payload[0 .. length + 2] };
        }
        for (&fixture.counts) |*row| @memset(row, 0);
        fixture.addresses = .{ 0x1000, 0x2000 };
        fixture.offsets = @splat(.{ 0, 8, 16 });
        fixture.rooms = @splat(.{ 8, 8, 8 });
        fixture.states = @splat(.free);
        fixture.owners = @splat(0);
        fixture.epochs = @splat(0);
        fixture.driver = undefined;
        fixture.driver.api.cuMemcpyHtoDAsync_v2 = copy;
        fixture.driver.api.cuEventRecord = record;
        fixture.driver.api.cuEventQuery = query;
        fixture.driver.api.cuStreamWaitEvent = wait;
        fixture.driver.api.cuEventSynchronize = sync;
        const ring = fixture.ring;
        ring.* = undefined;
        ring.gpa = allocator;
        ring.d = &fixture.driver;
        ring.pools = @splat(fixture.pool);
        ring.addr = &fixture.addresses;
        ring.off = &fixture.offsets;
        ring.room_of = &fixture.rooms;
        ring.state = &fixture.states;
        ring.owner = &fixture.owners;
        ring.epoch_of = &fixture.epochs;
        ring.next = 0;
        ring.pending_n = 0;
        ring.epoch = 0;
        ring.copy_ordered_epoch = 0;
        ring.batches = 0;
        ring.open_layer = null;
        ring.open_n = 0;
        ring.own_lo = 0;
        ring.own_hi = 512;
        ring.copy = .{ .d = &fixture.driver, .handle = @ptrFromInt(1) };
        for (&ring.cev, 0..) |*event, index| event.* = .{ .d = &fixture.driver, .handle = @ptrFromInt(index + 1) };
        for (&ring.rev, 0..) |*event, index| event.* = .{ .d = &fixture.driver, .handle = @ptrFromInt(index + 1) };
        ring.tlay = cuda.exl3_pool.Tables.init(513);
        const shadow = try allocator.alignedAlloc(u8, .@"16", 48 * ring.tlay.bytes);
        errdefer allocator.free(shadow);
        @memset(shadow, 0);
        ring.shadow = .{ .d = &fixture.driver, .bytes = shadow };
        const snapshots = try allocator.alignedAlloc(u8, .@"16", 64 * ring.tlay.k2[0]);
        errdefer allocator.free(snapshots);
        @memset(snapshots, 0);
        ring.snaps = .{ .d = &fixture.driver, .bytes = snapshots };
        for (&ring.slot) |*row| @memset(row, -1);
        for (&ring.counts) |*row| @memset(row, 0);
        ring.stats = .{};
        ring.attribution = try attribution.State.init(allocator, 2, enabled);
        calls = .{};
        copy_result = cuda.abi.success;
        return fixture;
    }

    fn deinit(fixture: *Fixture) void {
        if (fixture.ring.attribution) |*state| state.deinit(allocator);
        allocator.free(fixture.ring.snaps.bytes);
        allocator.free(fixture.ring.shadow.bytes);
        allocator.destroy(fixture.pool);
        allocator.destroy(fixture.ring);
        allocator.destroy(fixture);
    }

    fn checksum(fixture: *Fixture, unused_bytes: u64) !void {
        var sum: attribution.Counts = .{};
        for (fixture.ring.attribution.?.totals) |counts| try telemetry.add(attribution.Counts, &sum, counts);
        try std.testing.expectEqual(fixture.ring.stats.copies, sum.copies);
        try std.testing.expectEqual(fixture.ring.stats.dma_requested_bytes, sum.requested_bytes);
        try std.testing.expectEqual(fixture.ring.stats.wasted, sum.unused_copies);
        try std.testing.expectEqual(unused_bytes, sum.unused_requested_bytes);
    }

    fn snapshot(fixture: *Fixture) telemetry.Snapshot {
        const stats = fixture.ring.stats;
        var experts: telemetry.Experts = .{ .ring_dma_copies = stats.copies, .ring_dma_requested_bytes = stats.dma_requested_bytes, .ring_unused_copies = stats.wasted };
        attribution.publish(fixture.ring.attribution.?.totals, &experts);
        return .{ .rank = .{ .rank = 1, .world = 2 }, .scope = 1, .ring_owner = 1, .experts = experts, .kv = &.{} };
    }
};

test "exl3: prefill attribution counts actual copies once across held retries full rings drop and slot reuse" {
    const fixture = try Fixture.init(true);
    defer fixture.deinit();
    const ring = fixture.ring;
    try std.testing.expect(try ring.load(0, 0, .initial_prefetch));
    try std.testing.expect(try ring.load(1, 1, .cross_layer_prefetch));
    try fixture.checksum(0);
    try std.testing.expectEqual(@as(usize, 15), Fixture.calls.payload_bytes);
    const before_retry = ring.attribution.?.totals;
    const calls_before_retry = Fixture.calls;
    try std.testing.expect(!try ring.load(2, 2, .demand));
    try std.testing.expect(try ring.load(0, 0, .demand));
    try std.testing.expectEqualDeep(before_retry, ring.attribution.?.totals);
    try std.testing.expectEqualDeep(calls_before_retry, Fixture.calls);
    try std.testing.expectEqual(@as(i32, 0), ring.slot[0][0]);
    try std.testing.expectEqual(@as(i32, 1), ring.slot[1][1]);
    const used: [512]u32 = @splat(0);
    ring.drop(0, &used);
    ring.drop(0, &used);
    try fixture.checksum(6);
    try std.testing.expect(try ring.load(0, 0, .cross_layer_prefetch));
    try fixture.checksum(6);
    try std.testing.expect(try ring.load(2, 2, .demand));
    try std.testing.expectEqual(@as(i32, -1), ring.slot[0][0]);
    try std.testing.expectEqual(@as(i32, 1), ring.slot[1][1]);
    try fixture.checksum(6);
    try ring.newChunk();
    try fixture.checksum(27);
    try ring.newChunk();
    try fixture.checksum(27);
    const totals = ring.attribution.?.totals;
    try std.testing.expectEqualDeep(attribution.Counts{ .copies = 1, .requested_bytes = 12, .unused_copies = 1, .unused_requested_bytes = 12 }, totals[0]);
    try std.testing.expectEqualDeep(attribution.Counts{ .copies = 1, .requested_bytes = 6, .unused_copies = 1, .unused_requested_bytes = 6 }, totals[1]);
    try std.testing.expectEqualDeep(attribution.Counts{ .copies = 1, .requested_bytes = 9, .unused_copies = 1, .unused_requested_bytes = 9 }, totals[2]);
    try std.testing.expectEqualDeep(telemetry.Experts{
        .ring_dma_copies = 3,
        .ring_dma_requested_bytes = 27,
        .ring_unused_copies = 3,
        .ring_demand_copies = 1,
        .ring_demand_requested_bytes = 12,
        .ring_demand_unused_copies = 1,
        .ring_demand_unused_requested_bytes = 12,
        .ring_initial_prefetch_copies = 1,
        .ring_initial_prefetch_requested_bytes = 6,
        .ring_initial_prefetch_unused_copies = 1,
        .ring_initial_prefetch_unused_requested_bytes = 6,
        .ring_cross_layer_prefetch_copies = 1,
        .ring_cross_layer_prefetch_requested_bytes = 9,
        .ring_cross_layer_prefetch_unused_copies = 1,
        .ring_cross_layer_prefetch_unused_requested_bytes = 9,
    }, fixture.snapshot().experts);
    try std.testing.expectEqual(@as(usize, 0), Fixture.calls.syncs);
}

test "exl3: prefill attribution retains outstanding provenance across snapshots releases and later chunk reset" {
    const fixture = try Fixture.init(true);
    defer fixture.deinit();
    const ring = fixture.ring;
    try std.testing.expect(try ring.load(0, 0, .initial_prefetch));
    try std.testing.expect(try ring.load(0, 1, .cross_layer_prefetch));
    const before = fixture.snapshot();
    try ring.gate(ring.copy, 0, &.{ 0, 1 });
    try ring.release(ring.copy, 0, &.{0});
    const used: [512]u32 = @splat(0);
    ring.drop(0, &used);
    try fixture.checksum(9);
    try std.testing.expect(try ring.load(0, 0, .demand));
    try ring.newChunk();
    try fixture.checksum(9);
    const after = fixture.snapshot();
    const report = try telemetry.Report.between(before, after, .{ .phase = .prefill, .model_rows = 2048, .steps = 1 });
    try std.testing.expectEqual(@as(u64, 0), report.experts.ring_dma_copies);
    try std.testing.expectEqual(@as(u64, 0), report.experts.ring_dma_requested_bytes);
    try std.testing.expectEqual(@as(u64, 1), report.experts.ring_cross_layer_prefetch_unused_copies);
    try std.testing.expectEqual(@as(u64, 9), report.experts.ring_cross_layer_prefetch_unused_requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), report.experts.ring_initial_prefetch_unused_copies);
    const text = try report.formatAlloc(allocator);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "ring_cross_layer_prefetch_unused_requested_bytes=9") != null);
    try std.testing.expect(try ring.load(2, 2, .demand));
    try fixture.checksum(9);
    try std.testing.expectEqual(@as(usize, 1), Fixture.calls.queries);
    try std.testing.expectEqual(@as(usize, 2), Fixture.calls.waits);
    try std.testing.expectEqual(@as(usize, 0), Fixture.calls.syncs);
}

test "exl3: prefill attribution preserves prefetch ownership and first-copy source during cross-layer topups" {
    const fixture = try Fixture.init(true);
    defer fixture.deinit();
    const ring = fixture.ring;
    ring.own_lo = 1;
    ring.own_hi = 3;
    ring.counts[4][0] = 100;
    ring.counts[4][2] = 20;
    ring.counts[4][1] = 10;
    try std.testing.expectEqual(@as(usize, 1), try ring.prefetch(4, 1, 0, .initial_prefetch));
    try std.testing.expectEqual(@as(usize, 1), try ring.prefetch(4, 2, 0, .cross_layer_prefetch));
    try std.testing.expectEqual(@as(usize, 0), try ring.prefetch(4, 2, 0, .cross_layer_prefetch));
    try std.testing.expect(!ring.isLoaded(4, 0));
    try fixture.checksum(0);
    try std.testing.expectEqual(@as(u64, 1), ring.attribution.?.totals[1].copies);
    try std.testing.expectEqual(@as(u64, 1), ring.attribution.?.totals[2].copies);
    try ring.newChunk();
    try fixture.checksum(21);
}

test "exl3: disabled prefill attribution allocates nothing and bookkeeping adds no driver operations" {
    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
    try std.testing.expect((try attribution.State.init(failing.allocator(), 1000, false)) == null);
    try std.testing.expectError(error.OutOfMemory, attribution.State.init(failing.allocator(), 1000, true));
    var previous_calls: Calls = undefined;
    var previous_stats: ring_module.Stats = undefined;
    var previous_image: u64 = 0;
    for ([_]bool{ false, true }) |enabled| {
        const fixture = try Fixture.init(enabled);
        defer fixture.deinit();
        const ring = fixture.ring;
        try std.testing.expect(try ring.load(0, 0, .initial_prefetch));
        try ring.gate(ring.copy, 0, &.{0});
        try ring.release(ring.copy, 0, &.{0});
        try std.testing.expect(try ring.load(1, 1, .cross_layer_prefetch));
        try ring.newChunk();
        const image = std.hash.Wyhash.hash(0, ring.shadow.bytes);
        if (enabled) {
            try std.testing.expectEqualDeep(previous_calls, Fixture.calls);
            try std.testing.expectEqualDeep(previous_stats, ring.stats);
            try std.testing.expectEqual(previous_image, image);
            try fixture.checksum(9);
        } else {
            try std.testing.expect(ring.attribution == null);
            previous_calls = Fixture.calls;
            previous_stats = ring.stats;
            previous_image = image;
        }
    }
}

test "exl3: prefill attribution ignores failed transfers without suppressing retries" {
    const fixture = try Fixture.init(true);
    defer fixture.deinit();
    const ring = fixture.ring;
    Fixture.copy_result = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, ring.load(0, 0, .initial_prefetch));
    try std.testing.expect(!ring.isLoaded(0, 0));
    try fixture.checksum(0);
    Fixture.copy_result = cuda.abi.success;
    try std.testing.expect(try ring.load(0, 0, .demand));
    try fixture.checksum(0);
    try std.testing.expectEqual(@as(u64, 1), ring.attribution.?.totals[0].copies);
    try std.testing.expectEqual(@as(u64, 0), ring.attribution.?.totals[1].copies);
}
