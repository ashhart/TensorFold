//! Actual CUDA cache allocation and keep calls with address-only driver stubs, never a GPU or model payload.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const a = std.testing.allocator;
const rec = q.engine.recurrent;
const Driver = struct {
    const stride: u64 = 1 << 32;
    var allocs: usize = 0;
    var frees: usize = 0;
    var bytes: usize = 0;
    var live: usize = 0;
    var sizes: [65536]usize = @splat(0);
    var alive: [65536]bool = @splat(false);
    var pending_src: u64 = 0;
    var pending_dst: u64 = 0;
    var copy_bytes: usize = 0;
    var unsafe_frees: usize = 0;
    var syncs: usize = 0;
    fn reset() void {
        std.debug.assert(live == 0);
        allocs = 0;
        frees = 0;
        bytes = 0;
        pending_src = 0;
        pending_dst = 0;
        unsafe_frees = 0;
        syncs = 0;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        std.debug.assert(allocs < sizes.len and n < stride);
        out.* = 4096 + @as(u64, @intCast(allocs)) * stride;
        sizes[allocs] = n;
        alive[allocs] = true;
        allocs += 1;
        bytes += n;
        live += n;
        return cuda.abi.success;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        const i: usize = @intCast((ptr - 4096) / stride);
        std.debug.assert(alive[i]);
        if (ptr == pending_src or ptr == pending_dst) unsafe_frees += 1;
        alive[i] = false;
        live -= sizes[i];
        frees += 1;
        return cuda.abi.success;
    }
    fn fill(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn fillAsync(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        pending_src = 0;
        pending_dst = 0;
        return cuda.abi.success;
    }
    fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending_src = src;
        pending_dst = dst;
        copy_bytes = n;
        return cuda.abi.success;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};
fn engine(e: *q.engine.Engine, d: *cuda.Driver) void {
    d.api.cuMemAlloc_v2 = Driver.alloc;
    d.api.cuMemFree_v2 = Driver.free;
    d.api.cuMemsetD8_v2 = Driver.fill;
    d.api.cuMemsetD8Async = Driver.fillAsync;
    d.api.cuStreamSynchronize = Driver.sync;
    d.api.cuMemcpyDtoDAsync_v2 = Driver.copy;
    d.api.cuLaunchKernel = Driver.launch;
    e.d = d;
    e.stream = .{ .d = d, .handle = @ptrFromInt(17) };
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
    for (&e.g.linear, 0..) |*linear, i| linear.* = i < 64 and (i + 1) % 4 != 0;
    e.n_taps = 0;
    e.kv8 = false;
    e.geo = .{ .hid = 5120, .nk = 16, .nv = 48, .qh = 24, .kvh = 4, .nl = 48 };
    const f: cuda.Function = .{ .d = d, .handle = @ptrFromInt(1) };
    e.keeps = .{ .apply = f, .conv = f, .compact = f };
    e.rec_scratch = rec.Pool.init(a, @intFromPtr(e));
}
fn finish(e: *q.engine.Engine, c: *q.engine.Cache) void {
    _ = Driver.sync(e.stream.handle);
    rec.completed(e);
    c.deinit(a);
    rec.shutdown(e);
}
fn forwardDone(e: *q.engine.Engine, c: *q.engine.Cache, rows: usize) !void {
    try rec.mark(c, false);
    if (rows == 1) try rec.mark(c, true);
    c.advanced(rows, true);
    _ = Driver.sync(e.stream.handle);
    rec.completed(e);
}

test "Shared verification: recurrent scratch is absent from an idle real 27B cache" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 1);
    defer finish(&e, &c);
    try std.testing.expectEqual(@as(usize, 2), Driver.allocs);
    try std.testing.expectEqual(e.g.stateBytes() + e.g.kvBytes(1, false), Driver.bytes);
    try std.testing.expect(c.before == null);
    try std.testing.expectEqual(@as(usize, 0), c.log.ptr);
}

test "Shared verification: all window transitions keep log capacity within the original byte bound" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 128);
    defer finish(&e, &c);
    const bound = e.g.stateBytes() + e.g.logBytes(64);
    try std.testing.expectEqual(@as(usize, 153944064), e.g.stateBytes());
    for (1..65) |old| for (1..65) |rows| {
        try c.reset(&e);
        rec.completed(&e);
        try rec.prepare(&e, &c, old, true);
        try forwardDone(&e, &c, old);
        if (old == 1) try c.keep(&e, &.{}) else try c.keep(&e, &.{0});
        try rec.prepare(&e, &c, rows, true);
        try std.testing.expect(c.log.len >= e.g.logBytes(rows));
        try std.testing.expect((c.before != null) == (rows == 1));
        try std.testing.expect(e.rec_scratch.residentBytes() <= bound);
        const sg = c.seg(&e, c.len, 0, rows);
        try std.testing.expectEqual(c.log.ptr, sg.log);
        try std.testing.expectEqual(if (c.before) |b| b.ptr else 0, sg.before);
        try forwardDone(&e, &c, rows);
        if (rows == 1) {
            try c.settle(&e);
            try rec.prepare(&e, &c, 1, true);
            try std.testing.expectEqual(e.g.logBytes(1), c.log.len);
        }
    };
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
}

test "Shared verification: a real empty keep retains its snapshot until the queued restore completes" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 128);
    defer finish(&e, &c);
    try rec.prepare(&e, &c, 1, true);
    try forwardDone(&e, &c, 1);
    const before = c.before.?.ptr;
    try c.keep(&e, &.{});
    try std.testing.expectEqual(before, Driver.pending_src);
    try std.testing.expectEqual(c.state.ptr, Driver.pending_dst);
    try std.testing.expectEqual(e.g.stateBytes(), Driver.copy_bytes);
    rec.release(&c);
    try e.rec_scratch.collect(@intFromPtr(&e), rec.Provider{ .e = &e });
    try std.testing.expectEqual(e.g.stateBytes(), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    const syncs = Driver.syncs;
    rec.idle(&e);
    try std.testing.expectEqual(syncs + 1, Driver.syncs);
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}

test "Shared verification: allocation refusal retains a queued keep's old log owner" {
    Driver.reset();
    var d: cuda.Driver = undefined;
    var e: q.engine.Engine = undefined;
    engine(&e, &d);
    var c = try q.engine.Cache.init(a, &e, 128);
    defer finish(&e, &c);
    var foreign: q.engine.Engine = undefined;
    var other_driver: cuda.Driver = undefined;
    engine(&foreign, &other_driver);
    defer rec.shutdown(&foreign);
    try std.testing.expectError(error.ForeignRecurrentQueue, rec.prepare(&foreign, &c, 8, true));
    try std.testing.expectError(error.InvalidRecurrentWindow, rec.prepare(&e, &c, 65, true));
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
    try rec.prepare(&e, &c, 64, true);
    try forwardDone(&e, &c, 64);
    try c.keep(&e, &.{0});
    const usage = cuda.memory.usage(false);
    const prior = try cuda.setAllocationLimit(usage.device + usage.host);
    defer _ = cuda.setAllocationLimit(prior) catch unreachable;
    try std.testing.expectError(error.OutOfDeviceMemory, rec.prepare(&e, &c, 32, true));
    try std.testing.expectEqual(e.g.logBytes(64), e.rec_scratch.residentBytes());
    try std.testing.expectEqual(@as(usize, 0), Driver.unsafe_frees);
    _ = Driver.sync(e.stream.handle);
    rec.completed(&e);
    try std.testing.expectEqual(@as(usize, 0), e.rec_scratch.residentBytes());
}
