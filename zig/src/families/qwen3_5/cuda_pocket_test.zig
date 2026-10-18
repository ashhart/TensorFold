//! CUDA ABI failure injection through the production pocket device hooks.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const pk = @import("cuda_pocket.zig");
const dims = 64;
const t = std.testing;
const size = 4096 * 4;
var memory: [16][size]u8 align(16) = undefined;
var live: [16]bool = @splat(false);
var copies: usize = 0;
var fail_copy: usize = 0;
var fail_sync = false;
var fail_read = false;
fn alloc(out: *u64, n: usize) callconv(.c) abi.Result {
    std.debug.assert(n <= size);
    for (&live, 0..) |*on, i| if (!on.*) {
        on.* = true;
        out.* = @intFromPtr(&memory[i]);
        return 0;
    };
    return 2;
}
fn free(ptr: u64) callconv(.c) abi.Result {
    for (&live, 0..) |*on, i| if (@intFromPtr(&memory[i]) == ptr) {
        std.debug.assert(on.*);
        on.* = false;
        return 0;
    };
    unreachable;
}
fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
    var ptr: u64 = 0;
    const result = alloc(&ptr, n);
    out.* = @ptrFromInt(ptr);
    return result;
}
fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
    return free(@intFromPtr(ptr.?));
}
fn copy(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
    copies += 1;
    if (copies == fail_copy) return abi.error_not_ready;
    @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
    return 0;
}
fn asyncCopy(dst: u64, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return copy(dst, src, n);
}
fn read(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) abi.Result {
    if (fail_read) return abi.error_not_ready;
    @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    return 0;
}
fn sync(_: abi.Stream) callconv(.c) abi.Result {
    return if (fail_sync) abi.error_not_ready else 0;
}
fn driver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = alloc;
    d.api.cuMemFree_v2 = free;
    d.api.cuMemHostAlloc = hostAlloc;
    d.api.cuMemFreeHost = hostFree;
    d.api.cuMemcpyHtoD_v2 = copy;
    d.api.cuMemcpyHtoDAsync_v2 = asyncCopy;
    d.api.cuMemcpyDtoH_v2 = read;
    d.api.cuStreamSynchronize = sync;
    return d;
}
test "Trainer admission: every failed pocket copy or wait leaves the entire live pocket unchanged" {
    for ([_]bool{ false, true }) |existing| for (0..4) |failure| {
        live = @splat(false);
        copies = 0;
        fail_copy = 0;
        fail_sync = false;
        var d = driver();
        const s: cuda.Stream = .{ .d = &d, .handle = null };
        var p = try pk.Pockets.init(t.allocator, &d, dims, 8);
        p.k = .{ .own = undefined, .rows = undefined };
        const old: [dims]f32 = @splat(0.125);
        const proposed: [dims]f32 = @splat(7);
        if (existing) try p.set(s, 3, &old, &old, &old, 1);
        copies = 0;
        fail_copy = failure;
        fail_sync = failure == 0;
        try t.expectError(error.NotReady, p.set(s, 3, &proposed, &proposed, &proposed, 1));
        fail_copy = 0;
        fail_sync = false;
        if (existing) {
            var g: [dims]f32 = undefined;
            var u: [dims]f32 = undefined;
            var v: [dims]f32 = undefined;
            const got = p.read(s, 3, &g, &u, &v);
            const used = if (@typeInfo(@TypeOf(got)) == .error_union) try got else got;
            try t.expectEqual(@as(usize, 1), used);
            try t.expectEqualSlices(f32, &old, &g);
            try t.expectEqualSlices(f32, &old, &u);
            try t.expectEqualSlices(f32, &old, &v);
        } else try t.expect(p.layers[3] == null);
        p.k = null;
        p.deinit();
        try t.expect(std.mem.allEqual(bool, &live, false));
    };
}
test "Trainer admission: a nonempty pocket read failure is an error, never zero neurons" {
    live = @splat(false);
    copies = 0;
    fail_sync = false;
    fail_read = false;
    var d = driver();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var p = try pk.Pockets.init(t.allocator, &d, dims, 8);
    p.k = .{ .own = undefined, .rows = undefined };
    const old: [dims]f32 = @splat(0.125);
    try p.set(s, 3, &old, &old, &old, 1);
    defer {
        p.k = null;
        p.deinit();
    }
    var g: [dims]f32 = undefined;
    var u: [dims]f32 = undefined;
    var v: [dims]f32 = undefined;
    fail_read = true;
    const got = p.read(s, 3, &g, &u, &v);
    if (@typeInfo(@TypeOf(got)) == .error_union) try t.expectError(error.NotReady, got) else try t.expect(false);
    fail_read = false;
}

test "Dense learning: dense row scales preserve untaught and taught streams and publish only after a successful copy" {
    live = @splat(false);
    copies = 0;
    fail_copy = 0;
    fail_sync = false;
    var d = driver();
    const E = @import("cuda_engine.zig");
    var e: E.Engine = undefined;
    e.stream = .{ .d = &d, .handle = null };
    e.learning = null;
    e.taught = 0;
    e.pockets = try pk.Pockets.init(t.allocator, &d, dims, 8);
    e.pockets.k = .{ .own = undefined, .rows = undefined };
    defer {
        e.pockets.k = null;
        e.pockets.deinit();
    }
    const values: [dims]f32 = @splat(1);
    try e.pockets.set(e.stream, 3, &values, &values, &values, 1);
    var learned: E.Cache = undefined;
    var original: E.Cache = undefined;
    learned.untaught = false;
    original.untaught = true;
    const parts = [_]E.Part{ .{ .cache = &learned, .rows = 2 }, .{ .cache = &original, .rows = 3 } };
    const extra = @import("cuda_forward_extra.zig");
    try extra.stageScales(&e, &parts, 5);
    const ptr = e.taught;
    try t.expectEqualSlices(f32, &.{ 1, 1, 0, 0, 0 }, @as([*]const f32, @ptrFromInt(ptr))[0..5]);
    fail_copy = copies + 1;
    try t.expectError(error.NotReady, extra.stageScales(&e, &parts, 5));
    try t.expectEqual(ptr, e.pockets.rowsBuffer());
    fail_copy = 0;
    try extra.stageScales(&e, &parts, 5);
    try t.expectEqualSlices(f32, &.{ 1, 1, 0, 0, 0 }, @as([*]const f32, @ptrFromInt(e.taught))[0..5]);
}
