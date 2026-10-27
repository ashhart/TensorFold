const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("../../cuda_prompt.zig");
const pager = @import("../kv/cuda_exl3_kvpage.zig");
const recycle = @import("cuda_exl3_recycle.zig");
const graphs = @import("../graph/cuda_exl3_graph.zig");
const kv = @import("../kv/cuda_exl3_kv.zig");
const fixture = @import("../../cuda_snap_test_fixture.zig");
const shared_fixture = @import("cuda_exl3_shared_kv_test.zig");
const allocator = std.testing.allocator;

const Recording = struct {
    var api: cuda.abi.Api = undefined;
    var allocations: usize = 0;
    var fills: usize = 0;
    var fail_fill_at: ?usize = null;
    var host_allocations: [64]?[]align(16) u8 = @splat(null);

    fn hostAlloc(output: *?*anyopaque, len: usize, _: c_uint) callconv(.c) c_int {
        for (&host_allocations) |*slot| if (slot.* == null) {
            const bytes = allocator.alignedAlloc(u8, .@"16", len) catch return 2;
            slot.* = bytes;
            output.* = bytes.ptr;
            return 0;
        };
        return 2;
    }

    fn hostFree(address: ?*anyopaque) callconv(.c) c_int {
        fixture.Fake.freed_pending = fixture.Fake.freed_pending or fixture.Fake.pending != 0;
        for (&host_allocations) |*slot| if (slot.*) |bytes| if (@intFromPtr(bytes.ptr) == @intFromPtr(address)) {
            allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        return 1;
    }

    fn alloc(output: *u64, len: usize) callconv(.c) c_int {
        allocations += 1;
        return api.cuMemAlloc_v2(output, len);
    }

    fn fill(address: u64, value: u8, len: usize, stream: cuda.abi.Stream) callconv(.c) c_int {
        fills += 1;
        if (fail_fill_at == fills) return cuda.abi.error_not_ready;
        return api.cuMemsetD8Async(address, value, len, stream);
    }

    fn unload(_: cuda.abi.Module) callconv(.c) c_int {
        return 0;
    }

    fn driver() cuda.Driver {
        fixture.Fake.reset();
        var driver_ = fixture.Fake.driver();
        api = driver_.api;
        allocations = 0;
        fills = 0;
        fail_fill_at = null;
        driver_.api.cuMemAlloc_v2 = alloc;
        driver_.api.cuMemsetD8Async = fill;
        driver_.api.cuModuleUnload = unload;
        driver_.api.cuMemHostAlloc = hostAlloc;
        driver_.api.cuMemFreeHost = hostFree;
        return driver_;
    }
};

fn engine(driver: *cuda.Driver, stream: *cuda.Stream, config: ?pager.Config, arena: ?*pager.shared_device.Arena) prompt.Engine {
    var result: prompt.Engine = undefined;
    result.gpa = allocator;
    result.driver = driver;
    result.stream = stream;
    result.page_config = config;
    result.shared_arena = arena;
    result.chunk = 1;
    result.eos = 42;
    result.round_graph = .{};
    return result;
}

fn buffer(driver: *const cuda.Driver, len: usize) !cuda.DeviceBuffer {
    const result = try cuda.DeviceBuffer.alloc(driver, len);
    try result.fill8(0x7f, null);
    return result;
}

fn buffers(driver: *const cuda.Driver) ![]cuda.DeviceBuffer {
    const result = try allocator.alloc(cuda.DeviceBuffer, 1);
    result[0] = try buffer(driver, 16);
    return result;
}

fn sequence(owner: *prompt.Engine, cap: usize) !prompt.Seq {
    const driver = owner.driver;
    var pg: ?*pager.Paged = null;
    if (owner.page_config) |cfg| {
        if (owner.shared_arena) |arena| {
            pg = try pager.Paged.initShared(allocator, driver, cap, cfg, arena);
        } else {
            const layout = try pager.SnapshotLayout.init(cap, cfg, 0);
            const store = try allocator.create(pager.Paged);
            const host = try cuda.HostBuffer.allocMapped(driver, 24 * cap * pager.row_bytes);
            store.* = .{
                .gpa = allocator,
                .d = driver,
                .k = undefined,
                .cfg = cfg,
                .cap = cap,
                .wr = layout.wr,
                .nblocks = layout.nblocks,
                .host = host,
                .host_dev = try host.device(),
                .status_host = try cuda.HostBuffer.alloc(driver, 12 * 8 * 4),
                .dev = try buffer(driver, layout.device),
                .tabs = try buffer(driver, layout.tables),
                .work = try buffer(driver, 16),
            };
            store.k.module = .{ .d = driver, .handle = null };
            pg = store;
        }
    }
    var result: prompt.Seq = .{
        .recycle_owner = owner,
        .conv = try buffers(driver),
        .state = try buffers(driver),
        .kc = try buffers(driver),
        .vc = try buffers(driver),
        .ikc = try buffers(driver),
        .pooled = try buffers(driver),
        .tail = try buffer(driver, 16),
        .history = .{ 1, 2 },
        .fed = cap,
        .rank_id = 99,
        .untaught = true,
        .logits = try allocator.alloc(u8, 16),
        .cap = cap,
        .paged = pg,
    };
    if (kv.of(owner).ikc_host) {
        result.ikc[0].free();
        result.ikc_host = try cuda.HostBuffer.allocMapped(driver, 16);
        result.ikc[0] = .{ .d = driver, .ptr = try result.ikc_host.?.device(), .len = 16, .borrowed = true };
    }
    return result;
}

fn zero(bytes: []const u8) !void {
    for (bytes) |value| try std.testing.expectEqual(@as(u8, 0), value);
}

test "exl3: paged reuse keeps allocations with fresh incarnations and a complete BF16 reset" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    for ([_]bool{ false, true }) |shared| for ([_]bool{ false, true }) |host_index| {
        var driver = Recording.driver();
        var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
        const cfg: pager.Config = .{ .window = 2052, .rows = 1, .slots = 3, .shared_lanes = if (shared) 2 else 0 };
        const arena = if (shared) try pager.shared_device.Arena.initStorage(allocator, &driver, .{ .cap = 16, .lanes = 2, .window = 2052, .rows = 1, .slots = 3 }) else null;
        defer if (arena) |active| active.destroy() catch unreachable;
        if (arena) |active| shared_fixture.Fake.install(active);
        var owner = engine(&driver, &stream, cfg, arena);
        defer kv.forget(allocator, &owner);
        if (host_index) try kv.enable(&owner, false, false, true);
        defer recycle.drain(&owner);
        var current = try sequence(&owner, 16);
        const pg = current.paged.?;
        try std.testing.expectError(error.KvPageNotRetired, pg.resetForReuse(stream));
        const addresses = [_]u64{ current.tail.ptr, pg.host_dev, pg.dev.ptr, pg.tabs.ptr, pg.work.ptr };
        const count = Recording.allocations;
        var write: [16]u8 = @splat(0x5a);
        for (0..8) |_| {
            current.fed = 16;
            current.history = .{ 1, 2 };
            current.rank_id = 99;
            current.untaught = true;
            pg.win0 = @splat(4);
            pg.rebases = @splat(9);
            pg.written = @splat(7);
            pg.graph_prepared = true;
            @memset(pg.host.bytes, 0x6a);
            @memset(pg.status_host.bytes, 0xff);
            @memset(current.logits, 0xff);
            const key = pg.graphKey();
            const identity = pg.telemetry_id;
            const incarnation = pg.incarnation;
            try driver.check(driver.api.cuMemcpyDtoDAsync_v2(pg.host_dev, @intFromPtr(&write), write.len, stream.handle), "queue prior request write");
            current.deinit(allocator);
            try std.testing.expectEqual(@as(usize, 0), fixture.Fake.pending);
            try std.testing.expectEqualSlices(u8, &write, pg.host.bytes[0..write.len]);
            try std.testing.expect(pg.retired_for_reuse);
            try std.testing.expectError(error.KvPageRetired, pg.retireForReuse());
            current = try owner.newSeq(16);
            try std.testing.expectEqual(addresses, [_]u64{ current.tail.ptr, pg.host_dev, pg.dev.ptr, pg.tabs.ptr, pg.work.ptr });
            try std.testing.expectEqual(count, Recording.allocations);
            try std.testing.expect(pg.telemetry_id != identity and pg.incarnation != incarnation);
            try std.testing.expect(!std.meta.eql(key, pg.graphKey()));
            try std.testing.expectEqual(@as(usize, 0), current.fed);
            try std.testing.expectEqual([2]i64{ 42, 42 }, current.history);
            try std.testing.expectEqual(@as(u64, 0), current.rank_id);
            try std.testing.expect(!current.untaught and !current.invalidated and !pg.graph_prepared);
            try std.testing.expect(!pg.retired_for_reuse);
            try std.testing.expectEqual(@as(u64, 0), pg.telemetry_revision);
            try std.testing.expectEqual(@as(usize, 0), pg.win0[0]);
            try std.testing.expectEqual(@as(u64, 0), pg.rebases[0]);
            try std.testing.expectEqual(@as(u64, 0), pg.written[0]);
            try std.testing.expectEqual(@as(u64, 0), (try pg.total()).calls);
            try zero(pg.host.bytes);
            try zero(pg.status_host.bytes);
            try zero(current.logits);
            for ([_][]cuda.DeviceBuffer{ current.conv, current.state, current.kc, current.vc, current.ikc, current.pooled }) |group| for (group) |item| try zero(@as([*]const u8, @ptrFromInt(item.ptr))[0..item.len]);
            try zero(@as([*]const u8, @ptrFromInt(current.tail.ptr))[0..current.tail.len]);
        }
        current.deinit(allocator);
        try std.testing.expect(!fixture.Fake.freed_pending);
    };
}

test "exl3: request allocation retention is bounded to four sequences" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream, null, null);
    defer recycle.drain(&owner);
    const before = cuda.memory.usage(false);
    var requests: [5]prompt.Seq = undefined;
    for (&requests) |*request| request.* = try sequence(&owner, 16);
    for (&requests) |*request| request.deinit(allocator);
    var retained: [4]prompt.Seq = undefined;
    for (&retained) |*request| request.* = (try recycle.takeSeq(&owner, 16)).?;
    try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
    for (&retained) |*request| {
        request.invalidated = true;
        request.deinit(allocator);
    }
    const after = cuda.memory.usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host, after.host);
}

test "exl3: shared recycle releases slots and rejects stale owners while snapshots remain independent" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    const cfg: pager.Config = .{ .window = 2052, .rows = 1, .slots = 3, .shared_lanes = 2 };
    const arena = try pager.shared_device.Arena.initStorage(allocator, &driver, .{ .cap = 16, .lanes = 2, .window = 2052, .rows = 1, .slots = 3 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    var owner = engine(&driver, &stream, cfg, arena);
    defer recycle.drain(&owner);
    var current = try sequence(&owner, 16);
    const pg = current.paged.?;
    const old = pg.shared_sequence;
    const other = try arena.register(16);
    defer arena.release(other, stream) catch unreachable;
    shared_fixture.seed(arena, old, 0, 0, 0, 5, 0x71);
    shared_fixture.seed(arena, other, 0, 1, 1, 6, 0x72);
    const held = try arena.acquire(old);
    try std.testing.expectError(error.SharedKvBusy, pg.retireForReuse());
    var saved = try arena.saveOwnedBlocks(allocator, held);
    defer saved.deinit(allocator);
    try arena.cancel(held);
    fixture.Fake.fail_sync_at = fixture.Fake.sync_calls + 1;
    try std.testing.expectError(error.NotReady, pg.retireForReuse());
    try std.testing.expect(!pg.retired_for_reuse and pg.shared_attached);
    fixture.Fake.fail_sync_at = null;
    const still_live = try arena.acquire(old);
    try arena.cancel(still_live);
    current.deinit(allocator);
    try std.testing.expect(!pg.shared_attached);
    try std.testing.expectError(error.SharedKvSequence, arena.acquire(old));
    try std.testing.expectError(error.SharedKvSequence, pg.snapshot(0));
    const replacement = try arena.register(16);
    try std.testing.expectEqual(old.lane, replacement.lane);
    try std.testing.expect(replacement.generation > old.generation);
    try arena.release(replacement, stream);
    current = try owner.newSeq(16);
    try std.testing.expectEqual(pg, current.paged.?);
    const fresh = current.paged.?.shared_sequence;
    try std.testing.expect(fresh.generation > replacement.generation);
    try std.testing.expectError(error.SharedKvSequence, arena.acquire(old));
    const lease = try arena.acquire(fresh);
    try arena.restoreOwnedBlocks(lease, &saved);
    try arena.complete(lease, stream);
    const restored_lease = try arena.acquire(fresh);
    var restored = try arena.saveOwnedBlocks(allocator, restored_lease);
    defer restored.deinit(allocator);
    try arena.cancel(restored_lease);
    try std.testing.expectEqualSlices(u8, saved.payload, restored.payload);
    try std.testing.expectEqual(old, saved.source);
    try std.testing.expectEqual(fresh, restored.source);
    const other_lease = try arena.acquire(other);
    var other_image = try arena.saveOwnedBlocks(allocator, other_lease);
    defer other_image.deinit(allocator);
    try arena.cancel(other_lease);
    try std.testing.expectEqual(@as(usize, 1), other_image.entries.len);
    try std.testing.expectEqual(@as(u8, 0x72), other_image.payload[0]);
    current.deinit(allocator);
    const blocked = try arena.register(16);
    try std.testing.expectError(error.SharedKvLanes, recycle.takeSeq(&owner, 16));
    try arena.release(blocked, stream);
}

test "exl3: recycler isolates engine shape and driver and drains only its owner's sequences" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream, null, null);
    var other = engine(&driver, &stream, null, null);
    defer recycle.drain(&owner);
    defer recycle.drain(&other);
    var first = try sequence(&owner, 16);
    first.deinit(allocator);
    try std.testing.expect(try recycle.takeSeq(&other, 16) == null);
    const evicted = recycle.stats.seq_evicted;
    try std.testing.expect(try recycle.takeSeq(&owner, 17) == null);
    try std.testing.expectEqual(evicted + 1, recycle.stats.seq_evicted);
    first = try sequence(&owner, 16);
    first.deinit(allocator);
    owner.page_config = .{};
    try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
    try std.testing.expectEqual(evicted + 2, recycle.stats.seq_evicted);
    owner.page_config = null;
    first = try sequence(&owner, 16);
    first.deinit(allocator);
    var wrong_driver: cuda.Driver = undefined;
    owner.driver = &wrong_driver;
    try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
    try std.testing.expectEqual(evicted + 3, recycle.stats.seq_evicted);
    owner.driver = &driver;
    first = try sequence(&owner, 16);
    const address = first.tail.ptr;
    first.deinit(allocator);
    var second = try sequence(&other, 16);
    second.deinit(allocator);
    recycle.drain(&other);
    first = (try recycle.takeSeq(&owner, 16)).?;
    try std.testing.expectEqual(address, first.tail.ptr);
    first.invalidated = true;
    try std.testing.expect(!recycle.giveSeq(&first));
    first.deinit(allocator);
}

test "exl3: private c8 capacity churn evicts retained paging and host indexer allocations under the original limit" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    const cfg: pager.Config = .{ .window = 2052, .rows = 1, .slots = 3 };
    var owner = engine(&driver, &stream, cfg, null);
    try kv.enable(&owner, false, false, true);
    defer kv.forget(allocator, &owner);
    defer recycle.drain(&owner);
    var active: [8]?prompt.Seq = @splat(null);
    defer for (&active) |*entry| if (entry.*) |*request| request.deinit(allocator);
    for (&active) |*request| request.* = try sequence(&owner, 64);
    const admitted = cuda.memory.usage(false);
    const previous_limit = try cuda.memory.setAllocationLimit(admitted.device + admitted.host);
    defer _ = cuda.memory.setAllocationLimit(previous_limit) catch unreachable;
    for (active[0..4]) |*entry| {
        entry.*.?.deinit(allocator);
        entry.* = null;
    }
    const evicted = recycle.stats.seq_evicted;
    for (active[0..4]) |*entry| {
        try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
        entry.* = try sequence(&owner, 16);
    }
    try std.testing.expectEqual(evicted + 4, recycle.stats.seq_evicted);
    const mixed = cuda.memory.usage(false);
    try std.testing.expect(mixed.device + mixed.host <= admitted.device + admitted.host);
    for (active[0..4]) |*entry| {
        entry.*.?.deinit(allocator);
        entry.* = null;
    }
    const count = Recording.allocations;
    for (active[0..4]) |*entry| entry.* = (try recycle.takeSeq(&owner, 16)).?;
    try std.testing.expectEqual(count, Recording.allocations);
    for (active[0..4]) |*entry| {
        entry.*.?.deinit(allocator);
        entry.* = null;
    }
    graphs.force = .off;
    try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
    try std.testing.expectEqual(evicted + 8, recycle.stats.seq_evicted);
}

test "exl3: a partial recycled reset drains and frees its allocations without retention" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    for ([_]bool{ false, true }) |shared| {
        var driver = Recording.driver();
        var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
        const cfg: pager.Config = .{ .window = 2052, .rows = 1, .slots = 3, .shared_lanes = if (shared) 2 else 0 };
        const arena = if (shared) try pager.shared_device.Arena.initStorage(allocator, &driver, .{ .cap = 16, .lanes = 2, .window = 2052, .rows = 1, .slots = 3 }) else null;
        defer if (arena) |active| active.destroy() catch unreachable;
        if (arena) |active| shared_fixture.Fake.install(active);
        var owner = engine(&driver, &stream, cfg, arena);
        defer recycle.drain(&owner);
        const before = cuda.memory.usage(false);
        var current = try sequence(&owner, 16);
        current.deinit(allocator);
        Recording.fail_fill_at = Recording.fills + 3;
        try std.testing.expectError(error.NotReady, recycle.takeSeq(&owner, 16));
        try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
        const after = cuda.memory.usage(false);
        try std.testing.expectEqual(before.device, after.device);
        try std.testing.expectEqual(before.host, after.host);
        try std.testing.expect(!fixture.Fake.freed_pending);
    }
}
