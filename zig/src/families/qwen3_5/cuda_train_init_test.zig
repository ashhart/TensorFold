//! Real dense allocation and constructor unwind, with accepted work and poisoned-list sentinels.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const shape = @import("cuda_learn_shape.zig");
const a = std.testing.allocator;
const F = struct {
    var shared_calls: usize = 0;
    var shared_bytes: c_int = 0;
    var pending = false;
    var early: usize = 0;
    var allocated: usize = 0;
    var freed: usize = 0;
    var modules: usize = 0;
    var want: usize = 1;
    var fail_wait = false;
    var fail_context = false;
    var list: usize = 0;
    var owned_start: usize = 0;
    var owned_freed: usize = 0;
    var asynchronous = true;
    var list_dead = false;
    var fatal = false;
    fn check() void {
        if (pending) {
            early += 1;
            if (fatal) std.c._exit(71);
        }
    }
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocated += 1;
        out.* = allocated * 256;
        return 0;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        check();
        if (ptr / 256 > owned_start) {
            if (list_dead) std.c._exit(73);
            owned_freed += 1;
        }
        freed += 1;
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn fillAsync(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = true;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_wait and pending) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        if (fail_context) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn module(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        modules += 1;
        out.* = @ptrFromInt(modules);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        check();
        modules -= 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, _: [*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn attribute(_: cuda.abi.Function, attr: cuda.abi.FunctionAttribute, value: c_int) callconv(.c) cuda.abi.Result {
        std.debug.assert(attr == .max_dynamic_shared_size_bytes and value > 0);
        shared_calls += 1;
        shared_bytes = value;
        return 0;
    }
    fn driver() cuda.Driver {
        shared_calls = 0;
        shared_bytes = 0;
        pending = false;
        early = 0;
        allocated = 0;
        freed = 0;
        modules = 0;
        list = 0;
        list_dead = false;
        owned_start = 0;
        owned_freed = 0;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = fill;
        d.api.cuMemsetD8Async = fillAsync;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        d.api.cuModuleLoadData = module;
        d.api.cuModuleUnload = unload;
        d.api.cuModuleGetFunction = function;
        d.api.cuFuncSetAttribute = attribute;
        return d;
    }
    fn allocator() std.mem.Allocator {
        return .{ .ptr = undefined, .vtable = &.{ .alloc = cpuAlloc, .resize = cpuResize, .remap = cpuRemap, .free = cpuFree } };
    }
    fn cpuAlloc(_: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const p = a.rawAlloc(n, al, ra) orelse return null;
        if (n == 8 * @sizeOf(cuda.DeviceBuffer)) list = @intFromPtr(p);
        return p;
    }
    fn cpuResize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn cpuRemap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn cpuFree(_: *anyopaque, bytes: []u8, al: std.mem.Alignment, ra: usize) void {
        if (@intFromPtr(bytes.ptr) == list) {
            if (owned_freed < want) std.c._exit(73);
            list_dead = true;
        }
        a.rawFree(bytes, al, ra);
    }
    fn weights(t: anytype, _: std.Io) !void {
        owned_start = allocated;
        for (0..want) |_| {
            if (asynchronous) _ = try t.device(16) else _ = try t.devBytes(&.{ 1, 2, 3, 4 });
        }
        var fail = std.testing.FailingAllocator.init(t.gpa, .{ .fail_index = 0 });
        const b = try fail.allocator().alloc(u8, 1);
        fail.allocator().free(b);
    }
};
fn model(d: *cuda.Driver) engine.Engine {
    var e: engine.Engine = undefined;
    e.kit = null;
    e.d = d;
    e.stream = .{ .d = d, .handle = @ptrFromInt(1) };
    e.n_taps = 0;
    const g = shape.the_27b;
    e.g = .{ .hidden = g.hidden, .intermediate = g.intermediate, .layers = g.layers, .vocab = shape.vocab, .k_heads = g.linear_k_heads, .v_heads = g.linear_v_heads, .q_heads = g.query_heads, .kv_heads = g.kv_heads, .eps = shape.eps, .context = 1024, .tied = false, .linear = @splat(false) };
    for (e.g.linear[0..g.layers], 0..) |*v, i| v.* = shape.linear(i);
    return e;
}
fn registration(comptime T: type) !void {
    var d = F.driver();
    var e = model(&d);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var t: T = .{ .gpa = failing.allocator(), .e = &e, .g = undefined, .k = undefined, .gk = undefined };
    try std.testing.expectError(error.OutOfMemory, t.device(16));
    try std.testing.expectEqual(@as(usize, 0), F.early);
    try std.testing.expect(!F.pending);
    try std.testing.expectEqual(F.allocated, F.freed);
}
fn constructor(comptime init: anytype) !void {
    var d = F.driver();
    var e = model(&d);
    try std.testing.expectError(error.OutOfMemory, init(F.allocator(), std.testing.io, &e, F.weights));
    try std.testing.expectEqual(@as(usize, 0), F.early);
    try std.testing.expectEqual(F.allocated, F.freed);
    try std.testing.expectEqual(@as(usize, 0), F.modules);
}
const Child = struct {
    extern "c" fn fork() c_int;
};
fn child(comptime init: anytype, n: usize, accepted: bool, recover: bool) !void {
    const pid = Child.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        F.want = n;
        F.asynchronous = accepted;
        F.fail_wait = accepted;
        F.fail_context = !recover;
        F.fatal = true;
        constructor(init) catch std.c._exit(74);
        std.c._exit(0);
    }
    var status: c_int = 0;
    _ = std.c.waitpid(pid, &status, 0);
    try std.testing.expectEqual(@as(c_int, 0), status & 0x7f);
    try std.testing.expectEqual(@as(c_int, if (recover or !accepted) 0 else 1), (status >> 8) & 0xff);
}
pub fn registrationFailure(comptime T: type) !void {
    try registration(T);
}
pub fn listFailure(comptime init: anytype) !void {
    for ([_]usize{ 7, 1 }) |n| try child(init, n, false, true);
}
pub fn pendingFailure(comptime init: anytype) !void {
    for ([_]usize{ 7, 1 }) |n| {
        try child(init, n, true, true);
        try child(init, n, true, false);
    }
}

test "dense constructor: fake driver records shared memory setup" {
    var d = F.driver();
    const f: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(1) };
    try f.allowDynamicShared(48 * 1024);
    try std.testing.expectEqual(@as(usize, 1), F.shared_calls);
    try std.testing.expectEqual(@as(c_int, 48 * 1024), F.shared_bytes);
}
