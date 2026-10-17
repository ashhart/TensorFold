//! Native GPU parents outlive a failed family load until destination writes have drained in their context.
const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
pub const Gpu = struct { driver: cuda.Driver, ctx: cuda.Context, gpa: std.mem.Allocator, next: ?*Gpu = null };
var head: ?*Gpu = null;
var mutex: std.atomic.Mutex = .unlocked;
fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}
pub fn keep(g: *Gpu) void {
    lock();
    defer mutex.unlock();
    g.next = head;
    head = g;
}
fn close(g: *Gpu) !void {
    try g.ctx.makeCurrent();
    if (nemotron.engine.load_owner.recover(&g.ctx) != 0) return error.LoadRetained;
    try g.driver.synchronize();
    g.ctx.deinit();
    g.driver.close();
    g.gpa.destroy(g);
}
var recovery: std.atomic.Mutex = .unlocked;
pub fn recover() usize {
    // A detached transaction remains a pending owner at every parent/admission boundary.
    if (!recovery.tryLock()) return 1;
    defer recovery.unlock();
    lock();
    var at = head;
    head = null;
    mutex.unlock();
    if (comptime @import("builtin").is_test) if (Testing.detached) |hook| {
        Testing.detached = null;
        hook();
    };
    while (at) |g| {
        at = g.next;
        g.next = null;
        close(g) catch {
            keep(g);
        };
    }
    lock();
    defer mutex.unlock();
    var n: usize = 0;
    at = head;
    while (at) |g| {
        n += 1;
        at = g.next;
    }
    return n;
}

const t = std.testing;
const Fake = struct {
    var pending = false;
    var fail_wait = true;
    var fail_drain = true;
    var unsafe: usize = 0;
    var frees: usize = 0;
    var allocs: usize = 0;
    var modules: usize = 0;
    var releases: usize = 0;
    var released: [2]usize = @splat(0);
    var destinations: [64]struct { ptr: u64, owner: usize, live: bool } = undefined;
    var next: u64 = 256;
    var alloc_fail = false;
    var host: [4096]u8 align(16) = undefined;
    fn hostAlloc(out: *?*anyopaque, _: usize, _: c_uint) callconv(.c) c_int {
        out.* = &host;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) c_int {
        if (pending) unsafe += 1;
        return 0;
    }
    fn asyncCopy(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return copy(dst, src, n);
    }
    fn eventCreate(_: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        return cuda.abi.error_not_ready;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        if (alloc_fail) return cuda.abi.error_not_ready;
        destinations[allocs] = .{ .ptr = next, .owner = if (bound_context == @as(cuda.abi.Context, @ptrFromInt(7))) 1 else 0, .live = true };
        allocs += 1;
        out.* = next;
        next += n + 256;
        return 0;
    }
    fn free(ptr: u64) callconv(.c) c_int {
        if (pending) unsafe += 1;
        for (destinations[0..allocs]) |*dst| if (dst.ptr == ptr) {
            if (!dst.live) unsafe += 1;
            dst.live = false;
        };
        frees += 1;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        pending = true;
        return 0;
    }
    fn stream(_: cuda.abi.Stream) callconv(.c) c_int {
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn sync() callconv(.c) c_int {
        if (fail_drain) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    var bound_context: cuda.abi.Context = null;
    fn current(ctx: cuda.abi.Context) callconv(.c) c_int {
        bound_context = ctx;
        return 0;
    }
    fn event(_: cuda.abi.Event) callconv(.c) c_int {
        if (pending) unsafe += 1;
        return 0;
    }
    fn streamFree(_: cuda.abi.Stream) callconv(.c) c_int {
        if (pending) unsafe += 1;
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) c_int {
        if (pending) unsafe += 1;
        modules += 1;
        return 0;
    }
    fn release(device: cuda.abi.Device) callconv(.c) c_int {
        if (pending) unsafe += 1;
        const i: usize = @intCast(device);
        released[i] += 1;
        for (destinations[0..allocs]) |dst| if (dst.live and dst.owner == i) {
            unsafe += 1;
        };
        releases += 1;
        return 0;
    }
    fn gpu() !*Gpu {
        pending = false;
        fail_wait = true;
        fail_drain = true;
        unsafe = 0;
        frees = 0;
        allocs = 0;
        modules = 0;
        releases = 0;
        released = @splat(0);
        next = 256;
        alloc_fail = false;
        const g = try t.allocator.create(Gpu);
        g.gpa = t.allocator;
        g.next = null;
        g.driver.lib = try std.DynLib.open(if (@import("builtin").os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
        g.driver.api.cuMemAlloc_v2 = alloc;
        g.driver.api.cuMemFree_v2 = free;
        g.driver.api.cuMemcpyHtoD_v2 = copy;
        g.driver.api.cuMemHostAlloc = hostAlloc;
        g.driver.api.cuMemFreeHost = hostFree;
        g.driver.api.cuMemcpyHtoDAsync_v2 = asyncCopy;
        g.driver.api.cuEventCreate = eventCreate;
        g.driver.api.cuCtxSetCurrent = current;
        g.driver.api.cuCtxSynchronize = sync;
        g.driver.api.cuStreamSynchronize = stream;
        g.driver.api.cuStreamDestroy_v2 = streamFree;
        g.driver.api.cuEventDestroy_v2 = event;
        g.driver.api.cuModuleUnload = unload;
        g.driver.api.cuDevicePrimaryCtxRelease_v2 = release;
        g.ctx = .{ .d = &g.driver, .device = 0, .handle = @ptrFromInt(1) };
        try g.ctx.makeCurrent();
        return g;
    }
};
test "native Bolt load failure keeps destinations and kernel GPU parents through recovery" {
    const g = try Fake.gpu();
    const d = &g.driver;
    const e = try t.allocator.create(nemotron.Engine);
    e.gpa = t.allocator;
    e.ctx = &g.ctx;
    e.load_next = null;
    e.load_has_weights = false;
    e.stream = .{ .d = d, .handle = @ptrFromInt(2) };
    e.side = .{ .s = .{ .d = d, .handle = @ptrFromInt(3) }, .fork = .{ .d = d, .handle = @ptrFromInt(4) }, .join = .{ .d = d, .handle = @ptrFromInt(5) } };
    e.k.d = d;
    e.k.modelopt = null;
    e.k.ggml = null;
    e.k.triton = null;
    e.k.affine.split = null;
    const module: cuda.Module = .{ .d = d, .handle = @ptrFromInt(6) };
    e.k.emb.mod = module;
    e.k.norms.mod = module;
    e.k.train.module = module;
    for (&e.k.mods) |*m| m.* = module;
    const p = try t.allocator.create(nemotron.weights.PendingLoad);
    p.* = .{ .gpa = t.allocator, .ops = e.ops(), .w = .{ .gpa = t.allocator, .config = undefined } };
    const dst = try cuda.DeviceBuffer.alloc(d, 16);
    try p.w.buffers.append(t.allocator, dst);
    p.loader = .{ .gpa = t.allocator, .ops = e.ops(), .w = &p.w, .src = undefined, .scratch = try cuda.DeviceBuffer.alloc(d, 16) };
    try t.expectError(error.NotReady, p.ops.upload(dst.ptr, &.{ 1, 2, 3, 4 }));
    e.load_pending = null;
    try t.expectError(error.LoadRetained, p.fail(&e.load_pending, error.NotReady));
    nemotron.engine.load_owner.keep(e);
    keep(g);
    try t.expectEqual(@as(usize, 1), recover());
    try t.expect(Fake.pending);
    try t.expectEqual(@as(usize, 0), Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.modules);
    try t.expectEqual(@as(usize, 0), Fake.releases);
    try t.expectEqual(@as(usize, 0), Fake.unsafe);
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    try t.expectEqual(@as(usize, 0), recover());
    try t.expectEqual(@as(usize, 2), Fake.frees);
    try t.expectEqual(@as(usize, 25), Fake.modules);
    try t.expectEqual(@as(usize, 1), Fake.releases);
    try t.expectEqual(@as(usize, 0), Fake.unsafe);
}

test "Bolt loader scratch allocation failure leaves the old owner intact" {
    const g = try Fake.gpu();
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    defer {
        g.ctx.deinit();
        g.driver.close();
        g.gpa.destroy(g);
    }
    var k: nemotron.kernels.Kernels = undefined;
    k.d = &g.driver;
    const ops: nemotron.kernels.Ops = .{ .k = &k, .s = .{ .d = &g.driver, .handle = null } };
    var src: @import("nemotron").weights.Source = .{ .gpa = t.allocator, .ops = ops };
    var weights: nemotron.weights.Weights = .{ .gpa = t.allocator, .config = undefined };
    var L: nemotron.weights.Loader = .{ .gpa = t.allocator, .ops = ops, .w = &weights, .src = &src, .scratch = try cuda.DeviceBuffer.alloc(&g.driver, 16) };
    var old = L.scratch;
    defer if (Fake.frees == 0) old.free();
    Fake.alloc_fail = true;
    try t.expectError(error.NotReady, L.tmp(32));
    try t.expectEqual(old.ptr, L.scratch.ptr);
    try t.expectEqual(@as(usize, 0), Fake.frees);
}

fn metadataFailure(a: std.mem.Allocator) !void {
    const g = try Fake.gpu();
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    defer {
        g.ctx.deinit();
        g.driver.close();
        g.gpa.destroy(g);
    }
    var k: nemotron.kernels.Kernels = undefined;
    k.d = &g.driver;
    const ops: nemotron.kernels.Ops = .{ .k = &k, .s = .{ .d = &g.driver, .handle = null } };
    var weights: nemotron.weights.Weights = .{ .gpa = a, .config = undefined };
    var L: nemotron.weights.Loader = .{ .gpa = a, .ops = ops, .w = &weights, .src = undefined, .scratch = undefined };
    const result = L.alloc("fixture", 16);
    weights.deinit();
    try t.expectEqual(Fake.allocs, Fake.frees);
    _ = try result;
}
test "Bolt device allocation cannot escape a failed metadata publication" {
    try t.checkAllAllocationFailures(t.allocator, metadataFailure, .{});
}

pub const Testing = if (@import("builtin").is_test) struct {
    pub var detached: ?*const fn () void = null;
} else struct {};

const Barrier = @import("load_barrier_test.zig");
fn secondGpu(first: *Gpu) !*Gpu {
    const g = try t.allocator.create(Gpu);
    g.* = first.*;
    g.driver.lib = try std.DynLib.open(if (@import("builtin").os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
    g.ctx = .{ .d = &g.driver, .device = 1, .handle = @ptrFromInt(7) };
    g.next = null;
    return g;
}
fn partialEngine(g: *Gpu) !*nemotron.Engine {
    try g.ctx.makeCurrent();
    const e = try t.allocator.create(nemotron.Engine);
    e.gpa = t.allocator;
    e.ctx = &g.ctx;
    e.load_next = null;
    e.load_has_weights = false;
    e.load_pending = null;
    const d = &g.driver;
    e.stream = .{ .d = d, .handle = @ptrFromInt(2) };
    e.side = .{ .s = .{ .d = d, .handle = @ptrFromInt(3) }, .fork = .{ .d = d, .handle = @ptrFromInt(4) }, .join = .{ .d = d, .handle = @ptrFromInt(5) } };
    e.k.d = d;
    e.k.modelopt = null;
    e.k.ggml = null;
    e.k.triton = null;
    e.k.affine.split = null;
    const module: cuda.Module = .{ .d = d, .handle = @ptrFromInt(6) };
    e.k.emb.mod = module;
    e.k.norms.mod = module;
    e.k.train.module = module;
    for (&e.k.mods) |*m| m.* = module;
    const p = try t.allocator.create(nemotron.weights.PendingLoad);
    p.* = .{ .gpa = t.allocator, .ops = e.ops(), .w = .{ .gpa = t.allocator, .config = undefined } };
    try p.w.buffers.append(t.allocator, try cuda.DeviceBuffer.alloc(d, 4));
    e.load_pending = p;
    nemotron.engine.load_owner.keep(e);
    return e;
}
const Worker = struct {
    fn engine(ctx: *const cuda.Context) void {
        _ = nemotron.engine.load_owner.recover(ctx);
    }
    fn radio(d: *const cuda.Driver) void {
        _ = nemotron.engine.load_owner.Testing.radioRecover(d);
    }
};
fn interleaving(radio: bool) !void {
    const a = try Fake.gpu();
    const b = try secondGpu(a);
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    _ = try partialEngine(a);
    _ = try partialEngine(b);
    const owner = nemotron.engine.load_owner;
    if (radio) {
        try a.ctx.makeCurrent();
        try owner.Testing.radio.tiny(t.allocator, &a.driver);
        try b.ctx.makeCurrent();
        try owner.Testing.radio.tiny(t.allocator, &b.driver);
    }
    Barrier.reset();
    if (radio) owner.Testing.radio.detached = Barrier.pause else owner.Testing.detached = Barrier.pause;
    defer {
        owner.Testing.detached = null;
        owner.Testing.radio.detached = null;
    }
    const thread = if (radio) try std.Thread.spawn(.{}, Worker.radio, .{&a.driver}) else try std.Thread.spawn(.{}, Worker.engine, .{&a.ctx});
    var joined = false;
    defer if (!joined) {
        Barrier.release();
        thread.join();
    };
    Barrier.wait();
    keep(b);
    const left = recover();
    try t.expectEqual(@as(usize, 0), Fake.releases);
    try t.expectEqual(@as(usize, 0), Fake.modules);
    try t.expectEqual(@as(usize, 0), Fake.frees);
    try t.expect(left != 0);
    Barrier.release();
    thread.join();
    joined = true;
    owner.Testing.detached = null;
    owner.Testing.radio.detached = null;
    keep(a);
    try t.expectEqual(@as(usize, 0), recover());
    try t.expectEqual(@as(usize, 2), Fake.releases);
    try t.expectEqual(@as(usize, 50), Fake.modules);
    try t.expectEqualSlices(usize, &.{ 1, 1 }, &Fake.released);
    try t.expectEqual(@as(usize, if (radio) 4 else 2), Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.unsafe);
    try t.expect(nemotron.engine.load_owner.Testing.empty());
    try t.expect(nemotron.engine.load_owner.Testing.radio.empty());
    try t.expectEqual(@as(usize, 0), recover());
}
test "Engine/Gpu overlapping recovery keeps detached children and native parents" {
    try interleaving(false);
}
test "RADIO/Engine overlapping recovery keeps detached tower and both parents" {
    try interleaving(true);
}

pub fn closeEntry(ctx: ?*const cuda.Context) !void {
    _ = recover();
    if (ctx) |c| try c.makeCurrent();
}
test "native close entry retries old loads and restores the closing context" {
    const a = try Fake.gpu();
    const b = try secondGpu(a);
    _ = try partialEngine(a);
    keep(a);
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    try closeEntry(&b.ctx);
    const releases = Fake.releases;
    const modules = Fake.modules;
    const frees = Fake.frees;
    const bound = Fake.bound_context;
    _ = recover();
    const expected = b.ctx.handle;
    b.ctx.deinit();
    b.driver.close();
    t.allocator.destroy(b);
    try t.expectEqual(@as(usize, 1), releases);
    try t.expectEqual(@as(usize, 25), modules);
    try t.expectEqual(@as(usize, 1), frees);
    try t.expectEqual(expected, bound);
}
test "engine close entry retries other retained contexts and restores its own" {
    const a = try Fake.gpu();
    const b = try secondGpu(a);
    _ = try partialEngine(a);
    keep(a);
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    try nemotron.engine.load_owner.closeEntry(&b.ctx);
    const modules = Fake.modules;
    const bound = Fake.bound_context;
    _ = recover();
    const expected = b.ctx.handle;
    b.ctx.deinit();
    b.driver.close();
    t.allocator.destroy(b);
    try t.expectEqual(@as(usize, 25), modules);
    try t.expectEqual(expected, bound);
}
fn nativeOverlap(new_keep: bool) !void {
    const a = try Fake.gpu();
    const b = try secondGpu(a);
    _ = try partialEngine(a);
    _ = try partialEngine(b);
    Fake.fail_wait = false;
    Fake.fail_drain = false;
    keep(a);
    if (!new_keep) keep(b);
    Barrier.reset();
    Testing.detached = Barrier.pause;
    defer Testing.detached = null;
    const W = struct {
        fn run() void {
            _ = recover();
        }
    };
    const thread = try std.Thread.spawn(.{}, W.run, .{});
    var joined = false;
    defer if (!joined) {
        Barrier.release();
        thread.join();
    };
    Barrier.wait();
    if (new_keep) keep(b);
    const pending = recover();
    try t.expect(pending != 0);
    try t.expectEqual(@as(usize, 0), Fake.releases);
    Barrier.release();
    thread.join();
    joined = true;
    try t.expectEqual(@as(usize, if (new_keep) 1 else 2), Fake.releases);
    try t.expectEqual(@as(usize, 0), recover());
    try t.expectEqual(@as(usize, 2), Fake.releases);
    try t.expectEqual(@as(usize, 50), Fake.modules);
    try t.expectEqualSlices(usize, &.{ 1, 1 }, &Fake.released);
    try t.expectEqual(@as(usize, 2), Fake.frees);
    try t.expectEqual(@as(usize, 0), recover());
}

test "native registry overlap cannot admit through a detached GPU snapshot" {
    try nativeOverlap(false);
}
test "new native owner stays pending during another recovery snapshot" {
    try nativeOverlap(true);
}
