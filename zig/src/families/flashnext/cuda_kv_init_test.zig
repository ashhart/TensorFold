//! Pool construction owns every destination before queuing fills, and proves completion before error cleanup.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const pages = @import("cuda_kv_lanes.zig");
const E = @import("cuda_prompt.zig").Engine;
const gpa = std.testing.allocator;
const layout: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 64, .page_tokens = 16, .snap_bytes = 0, .run_bytes = 64 };

const Device = struct {
    const stream: cuda.abi.Stream = @ptrFromInt(17);
    var next: u64 = 0x10000;
    var allocs: usize = 0;
    var fail_alloc: usize = 0;
    var fail_wait = false;
    var fail_fill: usize = 0;
    var fills: usize = 0;
    var pending = false;
    var freed_pending: usize = 0;
    var live: usize = 0;
    var context_proofs: usize = 0;
    var wrong_stream: usize = 0;
    fn alloc(out: *u64, len: usize) callconv(.c) c_int {
        allocs += 1;
        if (allocs == fail_alloc) return cuda.abi.error_not_ready;
        out.* = next;
        next += len + 256;
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        if (pending) freed_pending += 1;
        live -= 1;
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize, s: cuda.abi.Stream) callconv(.c) c_int {
        wrong_stream += @intFromBool(s != stream);
        fills += 1;
        pending = true;
        return if (fills == fail_fill) cuda.abi.error_not_ready else 0;
    }
    fn legacyFill(dst: u64, v: u8, n: usize) callconv(.c) c_int {
        return fill(dst, v, n, null);
    }
    fn wait(s: cuda.abi.Stream) callconv(.c) c_int {
        std.debug.assert(s == stream);
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) c_int {
        context_proofs += 1;
        pending = false;
        return 0;
    }
    fn driver() cuda.Driver {
        next = 0x10000;
        allocs = 0;
        fail_alloc = 0;
        fail_wait = false;
        fail_fill = 0;
        fills = 0;
        pending = false;
        freed_pending = 0;
        live = 0;
        context_proofs = 0;
        wrong_stream = 0;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemsetD8_v2 = legacyFill;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        return d;
    }
    fn released() !void {
        try std.testing.expectEqual(@as(usize, 0), freed_pending);
        try std.testing.expectEqual(@as(usize, 0), live);
        try std.testing.expect(!pending);
        try std.testing.expectEqual(@as(usize, 0), wrong_stream);
    }
};

var queue: cuda.Stream = undefined;
fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.driver = d;
    queue = .{ .d = d, .handle = Device.stream };
    e.stream = &queue;
    e.eos = 0;
    return e;
}

fn finish(d: *cuda.Driver, stream: cuda.Stream) void {
    Device.fail_wait = false;
    cuda.completion.drain(d, stream, "the constructor test driver must drain before leaving");
}

fn expectPoolError(expected: anyerror, result: anytype, d: *cuda.Driver, stream: cuda.Stream) !void {
    if (result) |pool| {
        finish(d, stream);
        pool.deinit();
        return error.TestExpectedError;
    } else |err| try std.testing.expectEqual(expected, err);
}

test "kv safety: Flash Next failed final init wait drains before freeing its arena and tables" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream.*);
    Device.fail_wait = true;
    try expectPoolError(error.NotReady, pages.Pool.init(gpa, &e, layout, 64, null, 2, 8), &d, e.stream.*);
    try Device.released();
    try std.testing.expectEqual(@as(usize, 3), Device.fills);
    try std.testing.expectEqual(@as(usize, 1), Device.context_proofs);
}

test "kv safety: Flash Next second table allocation refusal precedes every fill" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream.*);
    Device.fail_alloc = 3;
    try expectPoolError(error.NotReady, pages.Pool.init(gpa, &e, layout, 64, null, 2, 8), &d, e.stream.*);
    try Device.released();
    try std.testing.expectEqual(@as(usize, 0), Device.fills);
}

test "kv safety: Flash Next failed arena or table fill drains before constructor cleanup" {
    for (1..4) |failure| {
        var d = Device.driver();
        var e = engine(&d);
        defer finish(&d, e.stream.*);
        Device.fail_fill = failure;
        Device.fail_wait = true;
        try expectPoolError(error.NotReady, pages.Pool.init(gpa, &e, layout, 64, null, 2, 8), &d, e.stream.*);
        try Device.released();
        try std.testing.expectEqual(@as(usize, 1), Device.context_proofs);
    }
}

test "kv safety: Flash Next page IDs and arena size refuse before allocation" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream.*);
    Device.fail_alloc = 1;
    try expectPoolError(error.BadFlashNextKvPages, pages.Pool.init(gpa, &e, layout, 64, null, 2, @as(usize, std.math.maxInt(c_int)) + 2), &d, e.stream.*);
    var oversized = layout;
    oversized.token_bytes = std.math.maxInt(u64);
    try expectPoolError(error.Overflow, pages.Pool.init(gpa, &e, oversized, 64, null, 2, 8), &d, e.stream.*);
    try std.testing.expectEqual(@as(usize, 0), Device.allocs);
    try Device.released();
}

/// A VMM whose cuMemCreate fails: every piece the arena reaches will not map.
const NoMemory = struct {
    var creates: usize = 0;
    fn create(_: *u64, _: usize, _: *const cuda.arena.Vmm.Prop, _: c_ulonglong) callconv(.c) cuda.abi.Result {
        creates += 1;
        return 2; // CUDA_ERROR_OUT_OF_MEMORY
    }
};

test "kv safety: Flash Next open whose arena piece will not map returns LaneMemory and leaves the pool idle" {
    const arena = @import("cuda_kv_arena.zig");
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream.*);
    const pool = try pages.Pool.init(gpa, &e, layout, 64, null, 2, 8);
    var live = true;
    defer if (live) {
        pool.arena.pieces = null;
        pool.deinit();
    };
    // the arena as pieces mapped on first use, none mapped yet, through the failing VMM
    var mapped = [_]bool{false};
    var s: arena.Arena.Pieces = .{ .gpa = gpa, .vmm = undefined, .d = &d };
    s.vmm.va = pool.arena.buf.ptr;
    s.vmm.va_len = arena.piece_bytes;
    s.vmm.next = s.vmm.va;
    s.vmm.granularity = 2 << 20;
    s.vmm.device = 0;
    s.vmm.prop = .{};
    s.vmm.failed = null;
    s.vmm.api.create = NoMemory.create;
    s.lazy = .{ .base = pool.arena.buf.ptr, .len = pool.arena.buf.len, .piece = arena.piece_bytes, .mapped = &mapped };
    pool.arena.pieces = &s;
    NoMemory.creates = 0;
    const before = pool.pool.census();

    try std.testing.expectError(error.LaneMemory, pool.open(64));
    try std.testing.expectEqual(@as(usize, 1), NoMemory.creates); // the open reached the piece
    try std.testing.expect(!mapped[0]);
    for (pool.lanes) |x| try std.testing.expect(!x.used);
    try std.testing.expectEqualDeep(before, pool.pool.census()); // promised 0, every page free again
    try std.testing.expect(pool.fits(64)); // a free lane, the pages and a run: the run went back too

    pool.arena.pieces = null;
    live = false;
    pool.deinit();
    try Device.released();
}

/// A VMM whose cuMemCreate takes 50 ms, noting the thread of each, and the driver calls the arena's thread makes.
const Slow = struct {
    var on_caller: std.atomic.Value(u32) = .init(0); // pieces created on the test's (the engine's) thread
    var created: std.atomic.Value(u32) = .init(0);
    var caller: std.Thread.Id = 0;
    fn create(h: *u64, _: usize, _: *const cuda.arena.Vmm.Prop, _: c_ulonglong) callconv(.c) cuda.abi.Result {
        std.testing.io.sleep(.fromMilliseconds(50), .awake) catch {};
        if (std.Thread.getCurrentId() == caller) _ = on_caller.fetchAdd(1, .monotonic);
        h.* = created.fetchAdd(1, .monotonic) + 1;
        return 0;
    }
    fn ok64(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn map(_: u64, _: usize, _: usize, _: u64, _: c_ulonglong) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn unmap(_: u64, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn access(_: u64, _: usize, _: *const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn getCurrent(c: *cuda.abi.Context) callconv(.c) cuda.abi.Result {
        c.* = @ptrFromInt(29);
        return 0;
    }
    fn setCurrent(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn streamCreate(s: *cuda.abi.Stream, _: c_uint) callconv(.c) cuda.abi.Result {
        s.* = Device.stream;
        return 0;
    }
    fn streamDestroy(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "kv safety: Flash Next open into unmapped pieces beside a running stream waits for the arena's thread, not on a map" {
    const arena = @import("cuda_kv_arena.zig");
    var d = Device.driver();
    d.api.cuCtxGetCurrent = Slow.getCurrent;
    d.api.cuCtxSetCurrent = Slow.setCurrent;
    d.api.cuStreamCreate = Slow.streamCreate;
    d.api.cuStreamDestroy_v2 = Slow.streamDestroy;
    var e = engine(&d);
    e.round_graph = .{};
    defer finish(&d, e.stream.*);
    // 96 pages of 16 MiB: six pieces; runs of 8 pages from the top; 64 rows take 4 pages, 320 rows 20
    const big: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1 << 20, .page_tokens = 16, .snap_bytes = 0, .run_bytes = 128 << 20 };
    const pool = try pages.Pool.init(gpa, &e, big, 512, null, 2, 96);
    var s: arena.Arena.Pieces = .{ .gpa = gpa, .vmm = undefined, .d = &d, .io = std.testing.io };
    var live = true;
    defer if (live) {
        if (s.ahead) |*x| x.deinit();
        pool.arena.pieces = null;
        pool.deinit();
    };
    s.vmm.va = pool.arena.buf.ptr;
    s.vmm.va_len = 6 * arena.piece_bytes;
    s.vmm.next = s.vmm.va;
    s.vmm.granularity = 2 << 20;
    s.vmm.device = 0;
    s.vmm.prop = .{};
    s.vmm.failed = null;
    s.vmm.api.create = Slow.create;
    s.vmm.api.release = Slow.ok64;
    s.vmm.api.map = Slow.map;
    s.vmm.api.unmap = Slow.unmap;
    s.vmm.api.setAccess = @ptrCast(&Slow.access);
    var mapped: [6]bool = @splat(false);
    s.lazy = .{ .base = pool.arena.buf.ptr, .len = pool.arena.buf.len, .piece = arena.piece_bytes, .mapped = &mapped };
    pool.arena.pieces = &s;
    Slow.caller = std.Thread.getCurrentId();
    Slow.on_caller.store(0, .monotonic);
    Slow.created.store(0, .monotonic);

    // alone: no stream can wait, so nothing is posted and the open maps on its own thread, as before
    try std.testing.expect(!pool.waits(320, false));
    try std.testing.expect(s.ahead == null and std.mem.indexOfScalar(bool, &mapped, true) == null);

    // beside a running stream: pages 0-19 reach pieces 0 and 1, the run (pages 88-95) piece 5; the call returns at
    // once and the arena's thread maps them
    const t0 = std.Io.Clock.awake.now(std.testing.io);
    try std.testing.expect(pool.waits(320, true));
    const took = t0.durationTo(std.Io.Clock.awake.now(std.testing.io)).toNanoseconds();
    try std.testing.expect(took < 25 * std.time.ns_per_ms); // a map takes 50 ms: the caller did not wait on one
    try std.testing.expect(pool.waits(320, true)); // asked again before they are mapped: still waits, posts nothing twice
    s.ahead.?.settle();
    try std.testing.expect(mapped[0] and mapped[1] and !mapped[2] and mapped[5]);
    try std.testing.expect(!pool.waits(320, true)); // mapped: the open goes on
    // what the open ensures (its run, then its pages) maps nothing on the caller's thread now
    const pb = big.pageBytes();
    try pool.arena.ensure(88 * pb, 8 * pb, Device.stream);
    try pool.arena.ensure(0, 20 * pb, Device.stream);
    try std.testing.expectEqual(@as(u32, 0), Slow.on_caller.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 3), Slow.created.load(.monotonic));

    s.ahead.?.deinit();
    s.ahead = null;
    s.lazy.?.unmapAll(.{ .be = s.vmm.backend(), .seek = undefined });
    pool.arena.pieces = null;
    live = false;
    pool.deinit();
    try Device.released();
}
