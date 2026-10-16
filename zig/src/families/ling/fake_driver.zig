//! A host stand-in for the CUDA driver in checks: device memory in host memory, no-op kernels, a failure on cue.
const std = @import("std");
const cuda = @import("cuda");

const abi = cuda.abi;
const R = abi.Result;
const ok = abi.success;
const backing = std.heap.page_allocator;

/// What the entry points share (they are C functions, so process-wide): allocations, handles, the countdown.
var held: std.AutoHashMapUnmanaged(u64, usize) = .empty; // device and pinned host allocations by address
var handles: isize = 0; // modules, streams, events, graphs and execs made and not yet released
var calls: u64 = 0; // fallible calls since reset
var fail_at: ?u64 = null; // the call that fails
var code: R = 2; // what it returns (2: out of memory; NOT_READY reaches the caller without a log line)
var bad: usize = 0; // frees of something not held, releases of nothing
var lazy = false; // async copies, memsets and launches stay pending until a stream or context drains
var sticky = false; // every call from the failing one on fails too, drains included
var pending: usize = 0; // async work queued since the last drain (lazy only)
var early: usize = 0; // frees while work was pending
var stuck: usize = 0; // drains that failed
var hit = false; // the failing call came with work pending
var next: usize = 0x1000;
var globals: [1 << 16]u8 align(256) = undefined;

/// From now on, fallible call number `at` (1-based) returns `with`; null fails none. Bad releases start at 0.
pub fn reset(at: ?u64, with: R) void {
    calls = 0;
    bad = 0;
    pending = 0;
    early = 0;
    stuck = 0;
    hit = false;
    fail_at = at;
    code = with;
}

/// From now on async work stays pending until a successful drain (`queue`); failures persist (`stick`).
pub fn mode(queue: bool, stick: bool) void {
    lazy = queue;
    sticky = stick;
}

/// Frees of device or pinned memory while async work was pending, since reset; work pending now; failed drains.
pub fn freedEarly() usize {
    return early;
}

pub fn queued() usize {
    return pending;
}

pub fn failedDrains() usize {
    return stuck;
}

pub fn failedPending() bool {
    return hit;
}

/// Everything still held freed (what a failed drain leaked on purpose); returns how many.
pub fn sweep() usize {
    const n = held.count();
    var it = held.iterator();
    while (it.next()) |kv| backing.free(@as([*]align(256) u8, @ptrFromInt(kv.key_ptr.*))[0..kv.value_ptr.*]);
    held.clearRetainingCapacity();
    return n;
}

/// Fallible calls since reset, and whether the failure was handed out.
pub fn count() u64 {
    return calls;
}

pub fn fired() bool {
    return if (fail_at) |n| calls >= n else false;
}

/// Allocations and handles held now; frees and releases of things never made.
pub fn live() usize {
    return held.count() + @as(usize, @intCast(@max(handles, 0)));
}

pub fn badReleases() usize {
    return bad + @as(usize, @intCast(@max(-handles, 0)));
}

/// A Driver whose entry points are these (its library is never opened or closed).
pub fn driver() cuda.Driver {
    var api: abi.Api = undefined;
    inline for (@typeInfo(abi.Api).@"struct".field_names) |name| @field(api, name) = @ptrCast(&passes);
    inline for (comptime std.meta.declarations(table)) |name| @field(api, name) = &@field(table, name);
    return .{ .lib = undefined, .api = api };
}

/// A context on `d` (device 0) that nothing retains.
pub fn context(d: *const cuda.Driver) cuda.Context {
    return .{ .d = d, .device = 0, .handle = @ptrFromInt(0x10) };
}

/// The countdown: each fallible call takes one step, and the chosen one fails.
fn step() R {
    calls += 1;
    const n = fail_at orelse return ok;
    if (calls == n) hit = pending > 0;
    return if (calls == n or (sticky and calls > n)) code else ok;
}

/// An async call's result: queued work pending until a drain when lazy.
fn later(r: R) R {
    if (r == ok and lazy) pending += 1;
    return r;
}

fn drain() R {
    const r = step();
    if (r == ok) pending = 0 else stuck += 1;
    return r;
}

/// Every entry point without out-parameters: a step, nothing else (the C ABI lets a caller pass arguments it ignores).
fn passes() callconv(.c) R {
    return step();
}

fn handle() ?*anyopaque {
    next += 16;
    handles += 1;
    return @ptrFromInt(next);
}

fn release() R {
    handles -= 1;
    return ok;
}

fn take(n: usize) ?u64 {
    const m = backing.alignedAlloc(u8, .fromByteUnits(256), @max(n, 1)) catch return null;
    held.put(backing, @intFromPtr(m.ptr), @max(n, 1)) catch {
        backing.free(m);
        return null;
    };
    return @intFromPtr(m.ptr);
}

fn give(p: u64) R {
    if (pending > 0) early += 1;
    const kv = held.fetchRemove(p) orelse {
        bad += 1;
        return 1;
    };
    backing.free(@as([*]align(256) u8, @ptrFromInt(p))[0..kv.value]);
    return ok;
}

fn bytes(p: u64, n: usize) []u8 {
    return @as([*]u8, @ptrFromInt(p))[0..n];
}

fn cbytes(p: ?*const anyopaque, n: usize) []const u8 {
    return @as([*]const u8, @ptrCast(p.?))[0..n];
}

const table = struct {
    pub fn cuDriverGetVersion(v: *c_int) callconv(.c) R {
        v.* = 13000;
        return ok;
    }
    pub fn cuDeviceGet(d: *abi.Device, _: c_int) callconv(.c) R {
        d.* = 0;
        return ok;
    }
    pub fn cuDeviceGetCount(n: *c_int) callconv(.c) R {
        n.* = 1;
        return ok;
    }
    pub fn cuDeviceGetName(out: [*]u8, len: c_int, _: abi.Device) callconv(.c) R {
        const name = "fake";
        @memcpy(out[0..@min(name.len, @as(usize, @intCast(len)))], name[0..@min(name.len, @as(usize, @intCast(len)))]);
        if (len > name.len) out[name.len] = 0;
        return ok;
    }
    pub fn cuDeviceGetAttribute(v: *c_int, a: abi.DeviceAttribute, _: abi.Device) callconv(.c) R {
        v.* = switch (a) {
            .max_threads_per_block => 1024,
            .warp_size => 32,
            .clock_rate => 1_000_000,
            .multiprocessor_count => 48,
            .integrated => 1,
            .l2_cache_size => 24 << 20,
            .compute_capability_major => 12,
            .compute_capability_minor => 1,
            .max_shared_memory_per_multiprocessor, .max_shared_memory_per_block_optin => 101376,
        };
        return step();
    }
    pub fn cuDeviceTotalMem_v2(n: *usize, _: abi.Device) callconv(.c) R {
        n.* = 128 << 30;
        return step();
    }
    pub fn cuDevicePrimaryCtxRetain(c: *abi.Context, _: abi.Device) callconv(.c) R {
        c.* = @ptrFromInt(0x10);
        return step();
    }
    pub fn cuDevicePrimaryCtxRelease_v2(_: abi.Device) callconv(.c) R {
        return ok;
    }
    pub fn cuCtxGetCurrent(c: *abi.Context) callconv(.c) R {
        c.* = @ptrFromInt(0x10);
        return ok;
    }
    pub fn cuMemGetInfo_v2(free: *usize, total: *usize) callconv(.c) R {
        free.* = 100 << 30;
        total.* = 128 << 30;
        return step();
    }
    pub fn cuMemAlloc_v2(p: *abi.DevicePtr, n: usize) callconv(.c) R {
        if (step() != ok) return code;
        p.* = take(n) orelse return 2;
        return ok;
    }
    pub fn cuMemFree_v2(p: abi.DevicePtr) callconv(.c) R {
        return give(p);
    }
    pub fn cuMemHostAlloc(p: *?*anyopaque, n: usize, _: c_uint) callconv(.c) R {
        if (step() != ok) return code;
        p.* = @ptrFromInt(take(n) orelse return 2);
        return ok;
    }
    pub fn cuMemHostGetDevicePointer_v2(p: *abi.DevicePtr, host: ?*anyopaque, _: c_uint) callconv(.c) R {
        p.* = @intFromPtr(host);
        return step();
    }
    pub fn cuMemFreeHost(p: ?*anyopaque) callconv(.c) R {
        return give(@intFromPtr(p));
    }
    pub fn cuMemHostUnregister(_: ?*anyopaque) callconv(.c) R {
        return ok;
    }
    pub fn cuMemcpyHtoD_v2(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize) callconv(.c) R {
        if (step() != ok) return code;
        @memcpy(bytes(dst, n), cbytes(src, n));
        return ok;
    }
    pub fn cuMemcpyDtoH_v2(dst: ?*anyopaque, src: abi.DevicePtr, n: usize) callconv(.c) R {
        if (step() != ok) return code;
        @memcpy(bytes(@intFromPtr(dst), n), bytes(src, n));
        return ok;
    }
    pub fn cuMemcpyDtoD_v2(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize) callconv(.c) R {
        if (step() != ok) return code;
        @memmove(bytes(dst, n), bytes(src, n));
        return ok;
    }
    pub fn cuMemcpyHtoDAsync_v2(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) R {
        return later(cuMemcpyHtoD_v2(dst, src, n));
    }
    pub fn cuMemcpyDtoHAsync_v2(dst: ?*anyopaque, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) R {
        return later(cuMemcpyDtoH_v2(dst, src, n));
    }
    pub fn cuMemcpyDtoDAsync_v2(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) R {
        return later(cuMemcpyDtoD_v2(dst, src, n));
    }
    pub fn cuMemcpy2DAsync_v2(m: *const abi.Memcpy2D, _: abi.Stream) callconv(.c) R {
        if (step() != ok) return code;
        for (0..m.height) |r| @memmove(bytes(m.dst + r * m.dst_pitch, m.width), bytes(m.src + r * m.src_pitch, m.width));
        return later(ok);
    }
    pub fn cuMemsetD8_v2(p: abi.DevicePtr, v: u8, n: usize) callconv(.c) R {
        if (step() != ok) return code;
        @memset(bytes(p, n), v);
        return ok;
    }
    pub fn cuMemsetD32_v2(p: abi.DevicePtr, v: c_uint, n: usize) callconv(.c) R {
        if (step() != ok) return code;
        @memset(@as([*]align(1) u32, @ptrFromInt(p))[0..n], v);
        return ok;
    }
    pub fn cuMemsetD8Async(p: abi.DevicePtr, v: u8, n: usize, _: abi.Stream) callconv(.c) R {
        return later(cuMemsetD8_v2(p, v, n));
    }
    pub fn cuMemsetD32Async(p: abi.DevicePtr, v: c_uint, n: usize, _: abi.Stream) callconv(.c) R {
        return later(cuMemsetD32_v2(p, v, n));
    }
    pub fn cuLaunchKernel(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) R {
        return later(step());
    }
    pub fn cuLaunchKernelEx(_: *const abi.LaunchConfig, _: abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) R {
        return later(step());
    }
    pub fn cuStreamSynchronize(_: abi.Stream) callconv(.c) R {
        return drain();
    }
    pub fn cuCtxSynchronize() callconv(.c) R {
        return drain();
    }
    pub fn cuStreamCreate(s: *abi.Stream, _: c_uint) callconv(.c) R {
        if (step() != ok) return code;
        s.* = @ptrCast(handle());
        return ok;
    }
    pub fn cuStreamDestroy_v2(_: abi.Stream) callconv(.c) R {
        return release();
    }
    pub fn cuStreamIsCapturing(_: abi.Stream, s: *abi.CaptureStatus) callconv(.c) R {
        s.* = .none;
        return step();
    }
    pub fn cuEventCreate(e: *abi.Event, _: c_uint) callconv(.c) R {
        if (step() != ok) return code;
        e.* = @ptrCast(handle());
        return ok;
    }
    pub fn cuEventDestroy_v2(_: abi.Event) callconv(.c) R {
        return release();
    }
    pub fn cuEventElapsedTime(ms: *f32, _: abi.Event, _: abi.Event) callconv(.c) R {
        ms.* = 0;
        return step();
    }
    pub fn cuModuleLoadData(m: *abi.Module, _: ?*const anyopaque) callconv(.c) R {
        if (step() != ok) return code;
        m.* = @ptrCast(handle());
        return ok;
    }
    pub fn cuModuleLoadDataEx(m: *abi.Module, image: ?*const anyopaque, _: c_uint, _: ?[*]c_int, _: ?[*]?*anyopaque) callconv(.c) R {
        return cuModuleLoadData(m, image);
    }
    pub fn cuModuleUnload(_: abi.Module) callconv(.c) R {
        return release();
    }
    pub fn cuModuleGetFunction(f: *abi.Function, _: abi.Module, _: [*:0]const u8) callconv(.c) R {
        f.* = @ptrFromInt(0x20);
        return step();
    }
    pub fn cuModuleGetGlobal_v2(p: *abi.DevicePtr, n: *usize, _: abi.Module, _: [*:0]const u8) callconv(.c) R {
        p.* = @intFromPtr(&globals);
        n.* = globals.len;
        return step();
    }
    pub fn cuFuncGetAttribute(v: *c_int, a: abi.FunctionAttribute, _: abi.Function) callconv(.c) R {
        v.* = switch (a) {
            .max_threads_per_block => 1024,
            .num_regs => 32,
            .ptx_version, .binary_version => 121,
            else => 0,
        };
        return step();
    }
    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(n: *c_int, _: abi.Function, _: c_int, _: usize) callconv(.c) R {
        n.* = 4;
        return step();
    }
    pub fn cuStreamEndCapture(_: abi.Stream, g: *abi.Graph) callconv(.c) R {
        if (step() != ok) return code;
        g.* = @ptrCast(handle());
        return ok;
    }
    pub fn cuGraphCreate(g: *abi.Graph, _: c_uint) callconv(.c) R {
        if (step() != ok) return code;
        g.* = @ptrCast(handle());
        return ok;
    }
    pub fn cuGraphDestroy(_: abi.Graph) callconv(.c) R {
        return release();
    }
    pub fn cuGraphAddKernelNode_v2(n: *abi.GraphNode, _: abi.Graph, _: ?[*]const abi.GraphNode, _: usize, _: *const abi.KernelNodeParams) callconv(.c) R {
        n.* = @ptrFromInt(0x30);
        return step();
    }
    pub fn cuGraphGetNodes(_: abi.Graph, _: ?[*]abi.GraphNode, n: *usize) callconv(.c) R {
        n.* = 0;
        return step();
    }
    pub fn cuGraphInstantiateWithFlags(x: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) R {
        if (step() != ok) return code;
        x.* = @ptrCast(handle());
        return ok;
    }
    pub fn cuGraphExecDestroy(_: abi.GraphExec) callconv(.c) R {
        return release();
    }
    pub fn cuGetErrorName(_: R, s: *?[*:0]const u8) callconv(.c) R {
        s.* = "CUDA_ERROR_INJECTED";
        return ok;
    }
    pub fn cuGetErrorString(_: R, s: *?[*:0]const u8) callconv(.c) R {
        s.* = "a failure the fake driver was told to give";
        return ok;
    }
};

test "the fake driver: buffers live in host memory, the chosen call fails, frees of strangers are counted" {
    const d = driver();
    reset(null, ok);
    var b = try cuda.DeviceBuffer.fromHost(&d, "abcdefgh");
    var back: [8]u8 = undefined;
    try b.download(0, &back);
    try std.testing.expectEqualStrings("abcdefgh", &back);
    try std.testing.expectEqual(@as(usize, 1), live());
    b.free();
    try std.testing.expectEqual(@as(usize, 0), live());
    reset(2, abi.error_not_ready);
    var s = try cuda.Stream.init(&d, false);
    try std.testing.expectError(error.NotReady, cuda.DeviceBuffer.alloc(&d, 64));
    try std.testing.expect(fired());
    s.deinit();
    try std.testing.expectEqual(@as(usize, 0), live());
    _ = d.api.cuMemFree_v2(0x1234);
    try std.testing.expectEqual(@as(usize, 1), badReleases());
    reset(null, ok);
}
