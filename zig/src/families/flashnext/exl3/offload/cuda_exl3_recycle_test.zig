const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("../../cuda_prompt.zig");
const recycle = @import("cuda_exl3_recycle.zig");
const graphs = @import("../graph/cuda_exl3_graph.zig");
const allocator = std.testing.allocator;

test "exl3: settled peer windows cannot retain borrowed pool states" {
    const Draft = @import("../../cuda_mtp_draft.zig").Draft;
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var window = try Draft.init(allocator, &driver, stream, 1, 1, 1);
    defer {
        graphs.force = .off;
        window.deinit();
    }
    var run = try buffer(&driver, window.spare.len * 16);
    defer run.free();
    var states: @TypeOf(window.spare) = undefined;
    for (&states, 0..) |*state, i| state.* = .{ .d = &driver, .ptr = run.ptr + i * 16, .len = 16, .borrowed = true };
    var sequence_view: prompt.Seq = undefined;
    sequence_view.state = &states;
    window.open = true;
    window.settle(&sequence_view);
    defer for (&states) |*state| state.free();
    try std.testing.expect(window.pg.pt == 0);
    const retained = recycle.giveDraft(&window);
    // Recover an incorrectly retained window so this regression fails without a dangling recycler entry.
    if (retained) window = (try recycle.takeDraft(stream, 1, 1, 1)).?;
    try std.testing.expect(!retained);
}

/// A driver of host memory: allocations, fills and the two synchronizations, counting what the recycler allocates.
const Recording = struct {
    var allocations: usize = 0;
    var fills: usize = 0;
    var fail_fill_at: ?usize = null;
    var live: [256]?[]align(16) u8 = @splat(null);

    fn alloc(output: *u64, len: usize) callconv(.c) c_int {
        allocations += 1;
        for (&live) |*slot| if (slot.* == null) {
            const bytes = allocator.alignedAlloc(u8, .@"16", @max(len, 1)) catch return 2;
            slot.* = bytes;
            output.* = @intFromPtr(bytes.ptr);
            return 0;
        };
        return 2;
    }

    fn free(address: u64) callconv(.c) c_int {
        for (&live) |*slot| if (slot.*) |bytes| if (@intFromPtr(bytes.ptr) == address) {
            allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        return 1;
    }

    fn fill(address: u64, value: u8, len: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        fills += 1;
        if (fail_fill_at == fills) return cuda.abi.error_not_ready;
        const target: [*]u8 = @ptrFromInt(address);
        @memset(target[0..len], value);
        return 0;
    }

    fn fillNow(address: u64, value: u8, len: usize) callconv(.c) c_int {
        const target: [*]u8 = @ptrFromInt(address);
        @memset(target[0..len], value);
        return 0;
    }

    fn sync() callconv(.c) c_int {
        return 0;
    }

    fn streamSync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }

    fn driver() cuda.Driver {
        allocations = 0;
        fills = 0;
        fail_fill_at = null;
        var result: cuda.Driver = undefined;
        result.api.cuMemAlloc_v2 = alloc;
        result.api.cuMemFree_v2 = free;
        result.api.cuMemsetD8Async = fill;
        result.api.cuMemsetD8_v2 = fillNow;
        result.api.cuCtxSynchronize = sync;
        result.api.cuStreamSynchronize = streamSync;
        return result;
    }
};

fn engine(driver: *cuda.Driver, stream: *cuda.Stream) prompt.Engine {
    var result: prompt.Engine = undefined;
    result.exl3_prefill = .{};
    result.exl3_chain = .{};
    result.gpa = allocator;
    result.driver = driver;
    result.stream = stream;
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
    return .{
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
    };
}

fn zero(bytes: []const u8) !void {
    for (bytes) |value| try std.testing.expectEqual(@as(u8, 0), value);
}

test "exl3: recycled sequences keep their allocations and come back zeroed" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream);
    defer recycle.drain(&owner);
    var current = try sequence(&owner, 16);
    const addresses = [_]u64{ current.tail.ptr, current.kc[0].ptr, current.state[0].ptr };
    const count = Recording.allocations;
    for (0..8) |_| {
        current.fed = 16;
        current.history = .{ 1, 2 };
        current.rank_id = 99;
        current.untaught = true;
        @memset(current.logits, 0xff);
        current.deinit(allocator);
        current = (try recycle.takeSeq(&owner, 16)).?;
        try std.testing.expectEqual(addresses, [_]u64{ current.tail.ptr, current.kc[0].ptr, current.state[0].ptr });
        try std.testing.expectEqual(count, Recording.allocations);
        try std.testing.expectEqual(@as(usize, 0), current.fed);
        try std.testing.expectEqual([2]i64{ 42, 42 }, current.history);
        try std.testing.expectEqual(@as(u64, 0), current.rank_id);
        try std.testing.expect(!current.untaught);
        try zero(current.logits);
        for ([_][]cuda.DeviceBuffer{ current.conv, current.state, current.kc, current.vc, current.ikc, current.pooled }) |group| for (group) |item| try zero(@as([*]const u8, @ptrFromInt(item.ptr))[0..item.len]);
        try zero(@as([*]const u8, @ptrFromInt(current.tail.ptr))[0..current.tail.len]);
    }
    current.deinit(allocator);
}

test "exl3: request allocation retention is bounded to four sequences" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream);
    defer recycle.drain(&owner);
    const before = cuda.memory.usage(false);
    var requests: [5]prompt.Seq = undefined;
    for (&requests) |*request| request.* = try sequence(&owner, 16);
    for (&requests) |*request| request.deinit(allocator);
    var retained: [4]prompt.Seq = undefined;
    for (&retained) |*request| request.* = (try recycle.takeSeq(&owner, 16)).?;
    try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
    for (&retained) |*request| {
        request.recycle_owner = null;
        request.deinit(allocator);
    }
    const after = cuda.memory.usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host, after.host);
}

test "exl3: recycler isolates engine shape and driver and drains only its owner's sequences" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream);
    var other = engine(&driver, &stream);
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
    var wrong_driver: cuda.Driver = undefined;
    owner.driver = &wrong_driver;
    try std.testing.expect(try recycle.takeSeq(&owner, 16) == null);
    try std.testing.expectEqual(evicted + 2, recycle.stats.seq_evicted);
    owner.driver = &driver;
    first = try sequence(&owner, 16);
    const address = first.tail.ptr;
    first.deinit(allocator);
    var second = try sequence(&other, 16);
    second.deinit(allocator);
    recycle.drain(&other);
    first = (try recycle.takeSeq(&owner, 16)).?;
    try std.testing.expectEqual(address, first.tail.ptr);
    first.recycle_owner = null;
    try std.testing.expect(!recycle.giveSeq(&first));
    first.deinit(allocator);
}

test "exl3: capacity churn evicts retained allocations under the original limit" {
    const previous = graphs.force;
    graphs.force = .graph;
    defer graphs.force = previous;
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream);
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
    var driver = Recording.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(9) };
    var owner = engine(&driver, &stream);
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
}
