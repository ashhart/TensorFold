//! Shared CUDA state ownership at the real keep and metadata boundaries, with a recording driver only.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const a = std.testing.allocator;

test {
    _ = @import("qwen35_recurrent_host.zig");
}

test "Shared verification: all 128 streams share one bounded verification frame" {
    try std.testing.expectEqual(@as(usize, 128), q.engine.max_segs);
    try std.testing.expectEqual(@as(usize, 128), q.engine.most_verify);
    var storage: [q.round.bytes]u8 align(8) = undefined;
    var parts: [128]q.engine.Part = undefined;
    for (&parts) |*p| p.* = .{ .cache = undefined, .rows = 1 };
    _ = try q.round.build(&storage, &parts, true);
    for (q.round.steps(&storage)[0..128], 0..) |step, row| {
        try std.testing.expectEqual(@as(u32, @intCast(row)), step.row);
        try std.testing.expectEqual(@as(i32, -1), step.read_slot);
        try std.testing.expectEqual(@as(i32, 0), step.write_slot);
    }
}

const Driver = struct {
    var copies: usize = 0;
    var launches: usize = 0;
    fn copy(dst: u64, src: u64, bytes: usize) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..bytes], @as([*]const u8, @ptrFromInt(src))[0..bytes]);
        copies += 1;
        return cuda.abi.success;
    }
    fn copyAsync(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return copy(dst, src, bytes);
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        return cuda.abi.success;
    }
};
fn engine(d: *cuda.Driver) q.engine.Engine {
    d.api.cuMemcpyDtoD_v2 = Driver.copy;
    d.api.cuMemcpyDtoDAsync_v2 = Driver.copyAsync;
    d.api.cuLaunchKernel = Driver.launch;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = d;
    e.stream = .{ .d = d, .handle = null };
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 32, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 256, .linear = @splat(true) };
    e.n_taps = 0;
    e.kv8 = false;
    e.geo = .{ .hid = 256, .nk = 1, .nv = 1, .qh = 1, .kvh = 1, .nl = 1 };
    const f: cuda.Function = .{ .d = d, .handle = @ptrFromInt(1) };
    e.keeps = .{ .apply = f, .conv = f, .compact = f };
    return e;
}
fn buffer(d: *cuda.Driver, data: []u8) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = @intFromPtr(data.ptr), .len = data.len, .borrowed = true };
}

test "Shared verification: empty keep restores every byte of an in-place advanced recurrence and conv tail" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    const before = try a.alloc(u8, e.g.stateBytes());
    defer a.free(before);
    const state = try a.alloc(u8, before.len);
    defer a.free(state);
    for (before, 0..) |*x, i| x.* = @truncate(i *% 17 +% 31);
    @memset(state, 0xcc);
    const b = buffer(&d, state);
    var cache: q.engine.Cache = .{ .state = b, .kv = b, .log = b, .cap = 256, .len = 18, .last = .{ .start = 17, .rows = 1 }, .logits = &.{} };
    if (comptime @hasField(q.engine.Cache, "before")) cache.before = buffer(&d, before);
    Driver.copies = 0;
    Driver.launches = 0;
    try cache.keep(&e, &.{});
    try std.testing.expectEqualSlices(u8, before, state);
    try std.testing.expectEqual(@as(usize, 17), cache.len);
    try std.testing.expect(cache.last == null);
    try std.testing.expectEqual(@as(usize, 1), Driver.copies);
    try std.testing.expectEqual(@as(usize, 0), Driver.launches);
}

test "Shared verification: keeping an advanced one-row slot does not apply the recurrence twice" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    const b: cuda.DeviceBuffer = .{ .d = &d, .ptr = 1, .len = e.g.stateBytes(), .borrowed = true };
    var cache: q.engine.Cache = .{ .state = b, .kv = b, .log = b, .cap = 256, .len = 18, .last = .{ .start = 17, .rows = 1 }, .logits = &.{} };
    Driver.launches = 0;
    try cache.keep(&e, &.{0});
    try std.testing.expectEqual(@as(usize, 0), Driver.launches);
    try std.testing.expectEqual(@as(usize, 18), cache.len);
}

test "Shared verification: shared lane matrix reaches larger admitted cohorts" {
    _ = @import("qwen35_rounds_host.zig");
}

test "Shared verification: empty multi-row keep preserves the root and completed single rows settle once" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    const b: cuda.DeviceBuffer = .{ .d = &d, .ptr = 1, .len = e.g.stateBytes(), .borrowed = true };
    var cache: q.engine.Cache = .{ .state = b, .kv = b, .log = b, .cap = 256, .len = 7, .logits = &.{} };
    Driver.launches = 0;
    Driver.copies = 0;
    cache.advanced(64, true);
    try cache.keep(&e, &.{});
    try std.testing.expectEqual(@as(usize, 7), cache.len);
    try std.testing.expectEqual(@as(usize, 0), Driver.copies);
    cache.advanced(1, true);
    try std.testing.expect(cache.last != null);
    try cache.settle(&e);
    try std.testing.expectEqual(@as(usize, 8), cache.len);
    try std.testing.expect(cache.last == null);
    try std.testing.expectEqual(@as(usize, 0), Driver.launches);
    cache.advanced(128, false);
    try std.testing.expect(cache.last == null);
    try std.testing.expectEqual(@as(usize, 136), cache.len);
    try std.testing.expectEqual(@as(usize, 2) * e.g.stateBytes() + e.g.kvBytes(256, false) + e.g.logBytes(64), e.streamBytes(256));
}

const Owners = struct {
    var allocations: usize = 0;
    var clears: usize = 0;
    var fail_alloc: usize = 0;
    var fail_clear: usize = 0;
    var live: [8]bool = @splat(false);
    var pending = false;
    fn alloc(ptr: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocations += 1;
        if (allocations == fail_alloc) return cuda.abi.error_not_ready;
        ptr.* = allocations * 256;
        live[allocations] = true;
        return cuda.abi.success;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        const slot = ptr / 256;
        std.debug.assert(!pending);
        std.debug.assert(live[slot]);
        live[slot] = false;
        return cuda.abi.success;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        clears += 1;
        return if (clears == fail_clear) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn clearAsync(ptr: u64, value: u8, bytes: usize, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        std.debug.assert(stream != null);
        const result = clear(ptr, value, bytes);
        if (result == cuda.abi.success) pending = true;
        return result;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = false;
        return cuda.abi.success;
    }
    fn context() callconv(.c) cuda.abi.Result {
        pending = false;
        return cuda.abi.success;
    }
    fn setup(d: *cuda.Driver, fail_alloc_: usize, fail_clear_: usize) void {
        std.debug.assert(!pending);
        allocations = 0;
        clears = 0;
        fail_alloc = fail_alloc_;
        fail_clear = fail_clear_;
        live = @splat(false);
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = clear;
        d.api.cuMemsetD8Async = clearAsync;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
    }
    fn balanced() !void {
        try std.testing.expect(!pending);
        for (live) |owner| try std.testing.expect(!owner);
    }
};

test "Shared verification: the actual cache constructor unwinds the before bank and every later failure" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    e.stream.handle = @ptrFromInt(17);
    e.rec_scratch = q.engine.recurrent.Pool.init(a, @intFromPtr(&e));
    defer q.engine.recurrent.shutdown(&e);
    e.n_taps = 1;
    e.g.layers = 2;
    e.g.linear[1] = false;
    for (1..6) |fail| {
        Owners.setup(&d, fail, 0);
        try std.testing.expectError(error.NotReady, q.engine.Cache.init(a, &e, 256));
        try Owners.balanced();
    }
    for (1..5) |fail| {
        Owners.setup(&d, 0, fail);
        try std.testing.expectError(error.NotReady, q.engine.Cache.init(a, &e, 256));
        try Owners.balanced();
    }
    Owners.setup(&d, 0, 0);
    var failing: std.testing.FailingAllocator = .init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, q.engine.Cache.init(failing.allocator(), &e, 256));
    try Owners.balanced();
    Owners.setup(&d, 0, 0);
    var cache = try q.engine.Cache.init(a, &e, 256);
    try std.testing.expect(cache.before != null);
    try std.testing.expectEqual(@as(usize, 5), Owners.allocations);
    cache.deinit(a);
    try Owners.balanced();
}
