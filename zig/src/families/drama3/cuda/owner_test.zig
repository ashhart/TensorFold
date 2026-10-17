//! Actual Tts/Codec ownership, virtual device allocations and mocked completion, no CUDA context.
const std = @import("std");
const cuda = @import("cuda");
const Tts = @import("tts.zig").Tts;
const Codec = @import("codec.zig").Codec;
const Engine = @import("engine.zig").Engine;
const a = std.testing.allocator;
const Mock = struct {
    var next: usize = 0;
    var fail_next: bool = false;
    var live: [256]bool = @splat(false);
    var root: u64 = 0;
    var root_frees: usize = 0;
    var unloads: usize = 0;
    var fail_stream: cuda.abi.Stream = null;
    var pending: bool = false;
    var unsafe_frees: usize = 0;
    var waits: usize = 0;
    fn alloc(p: *u64, _: usize) callconv(.c) c_int {
        if (fail_next) {
            fail_next = false;
            return cuda.abi.error_not_ready;
        }
        next += 1;
        p.* = next;
        live[next] = true;
        return 0;
    }
    fn free(p: u64) callconv(.c) c_int {
        if (pending) unsafe_frees += 1;
        live[p] = false;
        if (p == root) root_frees += 1;
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) c_int {
        unloads += 1;
        return 0;
    }
    fn stream(p: *cuda.abi.Stream, _: c_uint) callconv(.c) c_int {
        p.* = @ptrFromInt(0x40);
        return 0;
    }
    fn close(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn wait(s: cuda.abi.Stream) callconv(.c) c_int {
        waits += 1;
        if (s == fail_stream) return cuda.abi.error_not_ready;
        if (@intFromPtr(s) == 3 or @intFromPtr(s) == 0x40) pending = false;
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn zeroAsync(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        pending = true;
        return 0;
    }
    fn init(e: *Engine) void {
        next = 0;
        fail_next = false;
        live = @splat(false);
        root_frees = 0;
        unloads = 0;
        fail_stream = null;
        pending = false;
        unsafe_frees = 0;
        waits = 0;
        e.* = undefined;
        e.gpa = a;
        e.quarantined = false;
        e.owner = null;
        e.driver.api.cuMemAlloc_v2 = alloc;
        e.driver.api.cuMemFree_v2 = free;
        e.driver.api.cuModuleUnload = unload;
        e.driver.api.cuStreamCreate = stream;
        e.driver.api.cuStreamDestroy_v2 = close;
        e.driver.api.cuStreamSynchronize = wait;
        e.driver.api.cuMemsetD8_v2 = zero;
        e.driver.api.cuMemsetD8Async = zeroAsync;
        e.stream = .{ .d = &e.driver, .handle = @ptrFromInt(1) };
        e.stream2 = .{ .d = &e.driver, .handle = @ptrFromInt(2) };
    }
    fn base(e: *Engine) !*Tts {
        const c = try a.create(Codec);
        c.* = undefined;
        c.gpa = a;
        c.owns_weights = true;
        c.root = null;
        c.refs = .init(1);
        c.d = &e.driver;
        c.k = &e.k;
        c.mod = .{ .d = &e.driver, .handle = @ptrFromInt(8) };
        c.bufs = .empty;
        c.host_layers = .empty;
        c.o = .{ .max_chunk = 1, .ring = 1 };
        const b = try cuda.DeviceBuffer.alloc(&e.driver, 256);
        root = b.ptr;
        try c.bufs.append(a, b);
        const layers = try a.alloc(std.meta.Child(@TypeOf(c.post.layers)), 1);
        try c.host_layers.append(a, layers);
        for (&c.dec) |*v| {
            v.cin = 1;
            v.cout = 1;
            v.s = 1;
        }
        const empty = @as(@import("codec.zig").Stage, .{ .buf = 0, .margin = 0, .cols = 0 });
        c.st_up = @splat(empty);
        c.st_conv0 = empty;
        c.st_final = empty;
        c.st_tconv = @splat(empty);
        c.st_ru = @splat(@splat(empty));
        const t = try a.create(Tts);
        t.* = .{ .gpa = a, .e = e, .c = c, .cs = .{ .d = &e.driver, .handle = @ptrFromInt(3) }, .audio_dev = try cuda.DeviceBuffer.alloc(&e.driver, 256), .audio_host = try a.alloc(f32, 1), .max_chunk = 1 };
        return t;
    }
    fn count() usize {
        var n: usize = 0;
        for (live) |v| n += @intFromBool(v);
        return n;
    }
    fn partial(_: void, t: *Tts) !void {
        try t.c.reset(t.cs);
        t.warm_zero = try cuda.DeviceBuffer.alloc(&t.e.driver, 256);
        try t.warm_zero.?.fill8(0, t.cs.handle);
        return error.InjectedWarmFailure;
    }
};
test "drama owner: failed fork teardown pins shared codec buffers descriptors and module through base teardown" {
    var e: Engine = undefined;
    Mock.init(&e);
    const base = try Mock.base(&e);
    var f = e;
    f.owner = &e;
    const second = try Tts.fork(base, &f);
    Mock.fail_stream = second.cs.handle;
    second.destroy();
    base.destroy();
    const freed = Mock.root_frees;
    const unloaded = Mock.unloads;
    try std.testing.expect(e.quarantined and f.quarantined);
    Mock.fail_stream = null;
    second.destroy();
    try std.testing.expectEqual(@as(usize, 0), freed);
    try std.testing.expectEqual(@as(usize, 0), unloaded);
    try std.testing.expectEqual(@as(usize, 1), Mock.root_frees);
    try std.testing.expectEqual(@as(usize, 1), Mock.unloads);
    try std.testing.expectEqual(@as(usize, 0), Mock.count());
}
test "drama owner: partial warmup failure retains real Tts codec and zero buffer until drain succeeds" {
    var e: Engine = undefined;
    Mock.init(&e);
    const t = try Mock.base(&e);
    Mock.fail_stream = t.cs.handle;
    try std.testing.expectError(error.InjectedWarmFailure, Tts.warmWith(t, {}, Mock.partial));
    const freed = Mock.root_frees;
    const unsafe_count = Mock.unsafe_frees;
    if (freed == 0) {
        Mock.fail_stream = null;
        t.destroy();
    }
    try std.testing.expect(e.quarantined);
    try std.testing.expectEqual(@as(usize, 0), freed);
    try std.testing.expectEqual(@as(usize, 0), unsafe_count);
    try std.testing.expectEqual(@as(usize, 0), Mock.count());
}
test "drama owner: partial warmup failure releases all owners after a successful drain" {
    var e: Engine = undefined;
    Mock.init(&e);
    const t = try Mock.base(&e);
    try std.testing.expectError(error.InjectedWarmFailure, Tts.warmWith(t, {}, Mock.partial));
    try std.testing.expect(!e.quarantined);
    try std.testing.expectEqual(@as(usize, 0), Mock.unsafe_frees);
    try std.testing.expectEqual(@as(usize, 0), Mock.count());
    try std.testing.expectEqual(@as(usize, 1), Mock.unloads);
}

test "drama owner: normal nested forks retain root after base release and unwind failed fork allocation" {
    var e: Engine = undefined;
    Mock.init(&e);
    const base = try Mock.base(&e);
    Mock.fail_next = true;
    try std.testing.expectError(error.NotReady, Tts.fork(base, &e));
    const second = try Tts.fork(base, &e);
    const third = try Tts.fork(second, &e);
    base.destroy();
    try std.testing.expectEqual(@as(usize, 0), Mock.root_frees);
    second.destroy();
    try std.testing.expectEqual(@as(usize, 0), Mock.unloads);
    third.destroy();
    try std.testing.expectEqual(@as(usize, 1), Mock.root_frees);
    try std.testing.expectEqual(@as(usize, 1), Mock.unloads);
    try std.testing.expectEqual(@as(usize, 0), Mock.count());
}
