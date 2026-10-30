//! Pool construction owns every destination before queuing fills, and proves completion before error cleanup.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const pages = @import("cuda_pages.zig");
const caches = @import("cuda_cache.zig");
const E = @import("cuda_engine.zig").Engine;
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

fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = Device.stream };
    e.g.vocab = 8;
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

test "kv safety: Qwen failed final init wait drains before freeing its arena and tables" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream);
    Device.fail_wait = true;
    try expectPoolError(error.NotReady, pages.Pages.init(gpa, &e, layout, 64, 0, 2, 8), &d, e.stream);
    try Device.released();
    try std.testing.expectEqual(@as(usize, 3), Device.fills);
    try std.testing.expectEqual(@as(usize, 1), Device.context_proofs);
}

test "kv safety: Qwen second table allocation refusal precedes every fill" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream);
    Device.fail_alloc = 3;
    try expectPoolError(error.NotReady, pages.Pages.init(gpa, &e, layout, 64, 0, 2, 8), &d, e.stream);
    try Device.released();
    try std.testing.expectEqual(@as(usize, 0), Device.fills);
}

test "kv safety: Qwen initPaged logits allocation failure precedes every fill" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream);
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, caches.Cache.initPaged(failing.allocator(), &e, 3, 16, 0x1000));
    try Device.released();
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(@as(usize, 1), Device.allocs);
    try std.testing.expectEqual(@as(usize, 0), Device.fills);
}

test "kv safety: Qwen failed arena or table fill drains before constructor cleanup" {
    for (1..4) |failure| {
        var d = Device.driver();
        var e = engine(&d);
        defer finish(&d, e.stream);
        Device.fail_fill = failure;
        Device.fail_wait = true;
        try expectPoolError(error.NotReady, pages.Pages.init(gpa, &e, layout, 64, 0, 2, 8), &d, e.stream);
        try Device.released();
        try std.testing.expectEqual(@as(usize, 1), Device.context_proofs);
    }
}

test "kv safety: Qwen page IDs and arena size refuse before allocation" {
    var d = Device.driver();
    var e = engine(&d);
    defer finish(&d, e.stream);
    Device.fail_alloc = 1;
    try expectPoolError(error.BadQwenKvPages, pages.Pages.init(gpa, &e, layout, 64, 0, 2, @as(usize, std.math.maxInt(c_int)) + 2), &d, e.stream);
    var oversized = layout;
    oversized.token_bytes = std.math.maxInt(u64);
    try expectPoolError(error.Overflow, pages.Pages.init(gpa, &e, oversized, 64, 0, 2, 8), &d, e.stream);
    try std.testing.expectEqual(@as(usize, 0), Device.allocs);
    try Device.released();
}
