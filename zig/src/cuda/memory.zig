//! Device memory and pinned host memory, each owned by one value; copies and fills, blocking or on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// Bytes this process holds in DeviceBuffers and HostBuffers, and device/pinned peaks since the last reset.
pub const Usage = struct { device: u64, host: u64, peak: u64, host_peak: u64 = 0, allocs: u64 = 0 };

var device_bytes: std.atomic.Value(u64) = .init(0);
var host_bytes: std.atomic.Value(u64) = .init(0);
var host_peak_bytes: std.atomic.Value(u64) = .init(0);
var peak_bytes: std.atomic.Value(u64) = .init(0);
var driver_allocs: std.atomic.Value(u64) = .init(0); // live DeviceBuffers from cuMemAlloc itself, not from a hook
var counts_mutex: std.atomic.Mutex = .unlocked;
var allocation_limit: ?u64 = null;
var reserved_bytes: u64 = 0;

/// Caps DeviceBuffer plus pinned HostBuffer bytes (driver-private ones excluded); returns the previous limit.
pub fn setAllocationLimit(limit: ?u64) Error!?u64 {
    lockCounts();
    defer counts_mutex.unlock();
    const used = device_bytes.load(.monotonic) +| host_bytes.load(.monotonic) +| reserved_bytes;
    if (limit) |cap| if (used > cap) return error.OutOfDeviceMemory;
    const previous = allocation_limit;
    allocation_limit = limit;
    return previous;
}

fn reserveAllocation(n: usize) Error!void {
    lockCounts();
    defer counts_mutex.unlock();
    const used = device_bytes.load(.monotonic) +| host_bytes.load(.monotonic) +| reserved_bytes;
    if (allocation_limit) |cap| if (n > cap -| used) return error.OutOfDeviceMemory;
    reserved_bytes = std.math.add(u64, reserved_bytes, n) catch return error.OutOfDeviceMemory;
}

/// A failed driver call refunds its reservation without changing usage or the measured allocation peak.
fn finishAllocation(counter: ?*std.atomic.Value(u64), n: usize) void {
    lockCounts();
    defer counts_mutex.unlock();
    reserved_bytes -= n;
    if (counter) |c| recordAllocation(c, n);
}

/// The counts now; `reset_peak` starts a new peak at the current device and pinned bytes. Safe from any thread.
pub fn usage(reset_peak: bool) Usage {
    const NoHook = struct {
        fn afterRead(_: @This()) void {}
    };
    return usageWithHook(reset_peak, NoHook{});
}

/// Captures the completed interval and resets both peaks atomically to live bytes.
pub fn takeUsage() Usage {
    const NoHook = struct {
        fn afterRead(_: @This()) void {}
    };
    return sample(true, NoHook{}, true);
}

fn usageWithHook(reset_peak: bool, hook: anytype) Usage {
    return sample(reset_peak, hook, false);
}

fn sample(reset_peak: bool, hook: anytype, pre_reset: bool) Usage {
    lockCounts();
    defer counts_mutex.unlock();
    const now = device_bytes.load(.monotonic);
    hook.afterRead();
    const host = host_bytes.load(.monotonic);
    const peak = @max(now, peak_bytes.load(.monotonic));
    const host_peak = @max(host, host_peak_bytes.load(.monotonic));
    if (reset_peak) {
        peak_bytes.store(now, .monotonic);
        host_peak_bytes.store(host, .monotonic);
    }
    return .{ .device = now, .host = host, .allocs = driver_allocs.load(.monotonic), .peak = if (pre_reset) peak else @max(now, peak_bytes.load(.monotonic)), .host_peak = if (pre_reset) host_peak else @max(host, host_peak_bytes.load(.monotonic)) };
}

fn recordAllocation(counter: *std.atomic.Value(u64), n: usize) void {
    const now = counter.fetchAdd(n, .monotonic) + n;
    if (counter == &device_bytes) _ = peak_bytes.fetchMax(now, .monotonic);
    if (counter == &host_bytes) _ = host_peak_bytes.fetchMax(now, .monotonic);
}

fn held(counter: *std.atomic.Value(u64), n: usize) void {
    lockCounts();
    defer counts_mutex.unlock();
    recordAllocation(counter, n);
}

fn freed(counter: *std.atomic.Value(u64), n: usize) void {
    lockCounts();
    defer counts_mutex.unlock();
    _ = counter.fetchSub(n, .monotonic);
}

fn lockCounts() void {
    while (!counts_mutex.tryLock()) std.Thread.yield() catch {};
}

/// A sub-allocator under DeviceBuffer (arena.zig): `alloc` gives an address or null for cuMemAlloc, `free` owns it.
pub const Hook = struct {
    ctx: *anyopaque,
    alloc: *const fn (ctx: *anyopaque, len: usize) ?abi.DevicePtr,
    free: *const fn (ctx: *anyopaque, p: abi.DevicePtr) bool, // true when the address was the hook's
};

var alloc_hook: ?Hook = null;

/// Installs, or with null removes, the sub-allocator: once at boot, before the buffers it should hold.
pub fn setHook(h: ?Hook) void {
    alloc_hook = h;
}

pub fn currentHook() ?Hook {
    return alloc_hook;
}

pub const DeviceBuffer = struct {
    d: *const Driver,
    ptr: abi.DevicePtr,
    len: usize,
    owner: ?Hook = null, // captured at allocation; changing the installed hook cannot transfer ownership
    borrowed: bool = false, // a span of memory another value owns (a carveout's): free leaves it alone

    /// `len` bytes, 256-byte aligned (512 from an installed arena); zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        try reserveAllocation(len);
        errdefer finishAllocation(null, len);
        var p: abi.DevicePtr = 0;
        var owner: ?Hook = null;
        if (len > 0) {
            if (alloc_hook) |h| if (h.alloc(h.ctx, len)) |ptr| {
                p = ptr;
                owner = h;
            };
            if (p == 0) {
                try d.check(d.api.cuMemAlloc_v2(&p, len), "cuMemAlloc");
                _ = driver_allocs.fetchAdd(1, .monotonic);
            }
        }
        finishAllocation(&device_bytes, len);
        return .{ .d = d, .ptr = p, .len = len, .owner = owner };
    }

    /// Allocates and fills from host bytes in one call.
    pub fn fromHost(d: *const Driver, bytes: []const u8) Error!DeviceBuffer {
        var b = try alloc(d, bytes.len);
        b.upload(0, bytes) catch |e| { // the copy may still write `b`: it is freed once completion is proved
            @import("completion.zig").drain(d, .{ .d = d, .handle = null }, "an upload's default stream and context did not drain");
            b.free();
            return e;
        };
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (!self.borrowed) {
            if (self.ptr != 0) {
                if (self.owner) |h| {
                    std.debug.assert(h.free(h.ctx, self.ptr));
                } else {
                    _ = self.d.api.cuMemFree_v2(self.ptr);
                    _ = driver_allocs.fetchSub(1, .monotonic);
                }
            }
            freed(&device_bytes, self.len);
        }
        self.* = undefined;
    }

    /// The device address `offset` bytes in; out of range is refused.
    pub fn at(self: DeviceBuffer, offset: usize) Error!abi.DevicePtr {
        if (offset > self.len) return error.Invalid;
        return self.ptr + offset;
    }

    fn span(self: DeviceBuffer, offset: usize, n: usize) Error!abi.DevicePtr {
        if (offset > self.len or n > self.len - offset) return error.Invalid;
        return self.ptr + offset;
    }

    /// Returns once the bytes have landed: a pageable copy can return early, and non-blocking streams won't wait.
    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try @import("copy_source.zig").synchronousCopy(self.d, dst, bytes, std.heap.page_allocator);
    }

    pub fn download(self: DeviceBuffer, offset: usize, out: []u8) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "cuMemcpyDtoH");
    }

    /// Owns a pinned source copy until completion. Captured uploads must use copy_source.Stage or Buffer.
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try @import("copy_source.zig").enqueue(self.d, dst, bytes, .{ .d = self.d, .handle = stream });
    }

    pub fn downloadAsync(self: DeviceBuffer, offset: usize, out: []u8, stream: abi.Stream) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyDtoHAsync_v2(out.ptr, src, out.len, stream), "cuMemcpyDtoHAsync");
    }

    pub fn copyFrom(self: DeviceBuffer, offset: usize, src: abi.DevicePtr, n: usize, stream: ?abi.Stream) Error!void {
        const dst = try self.span(offset, n);
        if (n == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, s), "cuMemcpyDtoDAsync");
        } else {
            try self.d.check(self.d.api.cuMemcpyDtoD_v2(dst, src, n), "cuMemcpyDtoD");
        }
    }

    pub fn fill8(self: DeviceBuffer, value: u8, stream: ?abi.Stream) Error!void {
        if (self.len == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.cuMemsetD8Async(self.ptr, value, self.len, s), "cuMemsetD8Async");
        } else {
            try self.d.check(self.d.api.cuMemsetD8_v2(self.ptr, value, self.len), "cuMemsetD8");
        }
    }

    /// Fills whole 32-bit words; the length must be a multiple of four.
    pub fn fill32(self: DeviceBuffer, value: u32, stream: ?abi.Stream) Error!void {
        if (self.len % 4 != 0) return error.Invalid;
        if (self.len == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.cuMemsetD32Async(self.ptr, value, self.len / 4, s), "cuMemsetD32Async");
        } else {
            try self.d.check(self.d.api.cuMemsetD32_v2(self.ptr, value, self.len / 4), "cuMemsetD32");
        }
    }
};

/// Page-locked host memory: the only host memory an asynchronous copy can read or write without staging.
pub const HostBuffer = struct {
    d: *const Driver,
    bytes: []align(16) u8,

    pub fn alloc(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_alloc_portable);
    }

    /// Pinned memory kernels write directly (mapped into the device's address space): no copy reads it back.
    pub fn allocMapped(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_alloc_portable | abi.host_alloc_devicemap);
    }

    /// The device address of a mapped buffer.
    pub fn device(self: HostBuffer) Error!abi.DevicePtr {
        var p: abi.DevicePtr = 0;
        try self.d.check(self.d.api.cuMemHostGetDevicePointer_v2(&p, self.bytes.ptr, 0), "cuMemHostGetDevicePointer");
        return p;
    }

    fn allocFlags(d: *const Driver, len: usize, flags: c_uint) Error!HostBuffer {
        if (len == 0) return error.Invalid;
        try reserveAllocation(len);
        errdefer finishAllocation(null, len);
        var p: ?*anyopaque = null;
        try d.check(d.api.cuMemHostAlloc(&p, len, flags), "cuMemHostAlloc");
        finishAllocation(&host_bytes, len);
        const base: [*]align(16) u8 = @ptrCast(@alignCast(p.?));
        return .{ .d = d, .bytes = base[0..len] };
    }

    pub fn free(self: *HostBuffer) void {
        _ = self.d.api.cuMemFreeHost(self.bytes.ptr);
        freed(&host_bytes, self.bytes.len);
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};

test "usage counts held bytes and the device peak" {
    const before = usage(true);
    held(&device_bytes, 1000);
    held(&host_bytes, 24);
    freed(&device_bytes, 1000);
    const after = usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host + 24, after.host);
    try std.testing.expectEqual(before.device + 1000, after.peak);
    try std.testing.expectEqual(before.device, usage(true).peak);
    freed(&host_bytes, 24);
}

test "freeing a borrowed span leaves the driver and the counts alone" {
    const before = usage(false);
    var b: DeviceBuffer = .{ .d = undefined, .ptr = 0x1000, .len = 4096, .borrowed = true };
    b.free();
    try std.testing.expectEqual(before.device, usage(false).device);
}

test "a buffer the hook owns is freed by the hook, never by the driver" {
    const Owner = struct {
        var frees: u32 = 0;
        fn alloc(_: *anyopaque, _: usize) ?abi.DevicePtr {
            return 0x2000;
        }
        fn free(_: *anyopaque, p: abi.DevicePtr) bool {
            frees += 1;
            return p == 0x2000;
        }
    };
    var ctx: u8 = 0;
    setHook(.{ .ctx = &ctx, .alloc = Owner.alloc, .free = Owner.free });
    defer setHook(null);
    var b = try DeviceBuffer.alloc(undefined, 4096);
    b.free();
    var c: DeviceBuffer = .{ .d = undefined, .ptr = 0x3000, .len = 64, .borrowed = true };
    c.free();
    try std.testing.expectEqual(@as(u32, 1), Owner.frees);
}

test "the driver allocation count follows cuMemAlloc and cuMemFree, not a hook's or a borrowed span's" {
    const Fake = struct {
        var next: abi.DevicePtr = 0x10_0000;
        fn alloc(p: *abi.DevicePtr, len: usize) callconv(.c) abi.Result {
            p.* = next;
            next += len;
            return abi.success;
        }
        fn free(_: abi.DevicePtr) callconv(.c) abi.Result {
            return abi.success;
        }
        fn hookAlloc(_: *anyopaque, len: usize) ?abi.DevicePtr {
            return if (len == 4096) 0x9000 else null; // the hook serves one size, the rest go to the driver
        }
        fn hookFree(_: *anyopaque, p: abi.DevicePtr) bool {
            return p == 0x9000;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    const before = usage(false).allocs;
    var a = try DeviceBuffer.alloc(&d, 1000);
    var b = try DeviceBuffer.alloc(&d, 2000);
    try std.testing.expectEqual(before + 2, usage(false).allocs);
    var ctx: u8 = 0;
    setHook(.{ .ctx = &ctx, .alloc = Fake.hookAlloc, .free = Fake.hookFree });
    var h = try DeviceBuffer.alloc(&d, 4096);
    var c = try DeviceBuffer.alloc(&d, 64);
    try std.testing.expectEqual(@as(abi.DevicePtr, 0x9000), h.ptr);
    try std.testing.expectEqual(before + 3, usage(false).allocs);
    h.free();
    c.free();
    setHook(null);
    var s: DeviceBuffer = .{ .d = &d, .ptr = a.ptr + 256, .len = 256, .borrowed = true };
    s.free();
    try std.testing.expectEqual(before + 2, usage(false).allocs);
    a.free();
    b.free();
    try std.testing.expectEqual(before, usage(false).allocs);
}

test "peak reset preserves an allocation between its read and store" {
    const before = usage(true);
    held(&device_bytes, 100);
    defer freed(&device_bytes, 100);
    const Probe = struct {
        read: std.atomic.Value(bool) = .init(false),
        attempted: std.atomic.Value(bool) = .init(false),
        blocked: std.atomic.Value(bool) = .init(false),
        allocated: std.atomic.Value(bool) = .init(false),
        finish: std.atomic.Value(bool) = .init(false),

        fn wait(flag: *std.atomic.Value(bool)) void {
            while (!flag.load(.acquire)) std.Thread.yield() catch {};
        }

        fn afterRead(probe: *@This()) void {
            probe.read.store(true, .release);
            wait(&probe.attempted);
            if (!probe.blocked.load(.acquire)) wait(&probe.allocated);
        }

        fn allocate(probe: *@This()) void {
            wait(&probe.read);
            const acquired = counts_mutex.tryLock();
            if (acquired) counts_mutex.unlock();
            probe.blocked.store(!acquired, .release);
            probe.attempted.store(true, .release);
            held(&device_bytes, 300);
            probe.allocated.store(true, .release);
            wait(&probe.finish);
            freed(&device_bytes, 300);
        }
    };
    var probe: Probe = .{};
    const thread = try std.Thread.spawn(.{}, Probe.allocate, .{&probe});
    _ = usageWithHook(true, &probe);
    probe.finish.store(true, .release);
    thread.join();
    const after = usage(false);
    try std.testing.expectEqual(before.device + 100, after.device);
    try std.testing.expectEqual(before.device + 400, after.peak);
}

test "allocation cap includes pinned and in-flight bytes and refunds failed driver calls" {
    const before = usage(true);
    const base = before.device + before.host;
    const previous = try setAllocationLimit(base + 16);
    defer _ = setAllocationLimit(previous) catch unreachable;
    var pending: usize = 0;
    defer if (pending > 0) finishAllocation(null, pending);

    try reserveAllocation(12);
    pending = 12;
    try std.testing.expectError(error.OutOfDeviceMemory, reserveAllocation(5));
    try std.testing.expectEqual(before.device, usage(false).device);
    finishAllocation(null, 12);
    pending = 0;
    try std.testing.expectEqual(before.peak, usage(false).peak);

    try reserveAllocation(12);
    finishAllocation(&host_bytes, 12);
    defer freed(&host_bytes, 12);
    try reserveAllocation(4);
    finishAllocation(&device_bytes, 4);
    defer freed(&device_bytes, 4);
    try std.testing.expectError(error.OutOfDeviceMemory, reserveAllocation(1));
    try std.testing.expectError(error.OutOfDeviceMemory, setAllocationLimit(base + 15));
    try std.testing.expectEqual(before.host + 12, usage(false).host);
    try std.testing.expectEqual(before.device + 4, usage(false).device);
}

test "public allocations refused by the cap skip the driver, and failed driver calls are refunded" {
    const Fake = struct {
        var calls: usize = 0;
        // NOT_READY fails the call without the driver's error log, which would fail the test
        fn memAlloc(_: *abi.DevicePtr, _: usize) callconv(.c) abi.Result {
            calls += 1;
            return abi.error_not_ready;
        }
        fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) abi.Result {
            calls += 1;
            return abi.error_not_ready;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = Fake.memAlloc;
    d.api.cuMemHostAlloc = Fake.hostAlloc;
    const before = usage(true);
    const previous = try setAllocationLimit(before.device + before.host + 4096);
    defer _ = setAllocationLimit(previous) catch unreachable;

    // the second call fits only if the first refunded its reservation
    try std.testing.expectError(error.NotReady, DeviceBuffer.alloc(&d, 4096));
    try std.testing.expectError(error.NotReady, HostBuffer.alloc(&d, 4096));
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    try std.testing.expectEqual(before, usage(false));

    try std.testing.expectError(error.OutOfDeviceMemory, DeviceBuffer.alloc(&d, 4097));
    try std.testing.expectError(error.OutOfDeviceMemory, HostBuffer.alloc(&d, 4097));
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    try std.testing.expectEqual(before, usage(false));
}

test "pinned peak retains freed transients and resets to live bytes" {
    const before = usage(true);
    held(&host_bytes, 40);
    held(&host_bytes, 160);
    freed(&host_bytes, 160);
    try std.testing.expectEqual(before.host + 40, usage(false).host);
    try std.testing.expectEqual(before.host + 200, usage(false).host_peak);
    try std.testing.expectEqual(before.host + 40, usage(true).host_peak);
    held(&host_bytes, 10);
    freed(&host_bytes, 50);
    try std.testing.expectEqual(before.host, usage(false).host);
    try std.testing.expectEqual(before.host + 50, usage(false).host_peak);
    _ = usage(true);
}

test "pinned peak reset preserves an allocation between its read and store" {
    const before = usage(true);
    held(&host_bytes, 100);
    defer freed(&host_bytes, 100);
    const Probe = struct {
        read: std.atomic.Value(bool) = .init(false),
        attempted: std.atomic.Value(bool) = .init(false),
        blocked: std.atomic.Value(bool) = .init(false),
        allocated: std.atomic.Value(bool) = .init(false),
        finish: std.atomic.Value(bool) = .init(false),

        fn wait(flag: *std.atomic.Value(bool)) void {
            while (!flag.load(.acquire)) std.Thread.yield() catch {};
        }

        fn afterRead(probe: *@This()) void {
            probe.read.store(true, .release);
            wait(&probe.attempted);
            if (!probe.blocked.load(.acquire)) wait(&probe.allocated);
        }

        fn allocate(probe: *@This()) void {
            wait(&probe.read);
            const acquired = counts_mutex.tryLock();
            if (acquired) counts_mutex.unlock();
            probe.blocked.store(!acquired, .release);
            probe.attempted.store(true, .release);
            held(&host_bytes, 300);
            probe.allocated.store(true, .release);
            wait(&probe.finish);
            freed(&host_bytes, 300);
        }
    };
    var probe: Probe = .{};
    const thread = try std.Thread.spawn(.{}, Probe.allocate, .{&probe});
    _ = usageWithHook(true, &probe);
    probe.finish.store(true, .release);
    thread.join();
    const after = usage(false);
    try std.testing.expectEqual(before.host + 100, after.host);
    try std.testing.expectEqual(before.host + 400, after.host_peak);
}

test "taking an interval preserves transient peaks before resetting them" {
    const before = usage(true);
    held(&host_bytes, 48);
    held(&device_bytes, 96);
    freed(&host_bytes, 48);
    freed(&device_bytes, 96);
    const interval = takeUsage();
    try std.testing.expectEqual(before.host + 48, interval.host_peak);
    try std.testing.expectEqual(before.device + 96, interval.peak);
    try std.testing.expectEqual(before.host, usage(false).host_peak);
    try std.testing.expectEqual(before.device, usage(false).peak);
}

pub fn releaseRetainedHost(bytes: usize) void {
    freed(&host_bytes, bytes);
}

test "Failure boundaries: a buffer keeps its allocation hook when the installed hook changes" {
    const Owner = struct {
        freed: usize = 0,
        fn alloc(_: *anyopaque, _: usize) ?abi.DevicePtr {
            return 0x2000;
        }
        fn free(ctx: *anyopaque, _: abi.DevicePtr) bool {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.freed += 1;
            return true;
        }
        fn hook(o: *@This()) Hook {
            return .{ .ctx = o, .alloc = alloc, .free = free };
        }
    };
    var first: Owner = .{};
    var second: Owner = .{};
    setHook(first.hook());
    defer setHook(null);
    var b = try DeviceBuffer.alloc(undefined, 64);
    setHook(second.hook());
    b.free();
    try std.testing.expectEqual(@as(usize, 1), first.freed);
    try std.testing.expectEqual(@as(usize, 0), second.freed);
}
