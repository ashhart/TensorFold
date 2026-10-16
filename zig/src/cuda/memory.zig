//! Device memory and pinned host memory, each owned by one value; copies and fills, blocking or on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// Bytes this process holds in DeviceBuffers and HostBuffers, and the device peak since the last reset.
pub const Usage = struct { device: u64, host: u64, peak: u64 };

var device_bytes: std.atomic.Value(u64) = .init(0);
var host_bytes: std.atomic.Value(u64) = .init(0);
var peak_bytes: std.atomic.Value(u64) = .init(0);
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

/// The counts now; `reset_peak` starts a new peak at the current device bytes. Safe from any thread.
pub fn usage(reset_peak: bool) Usage {
    const NoHook = struct {
        fn afterRead(_: @This()) void {}
    };
    return usageWithHook(reset_peak, NoHook{});
}

fn usageWithHook(reset_peak: bool, hook: anytype) Usage {
    lockCounts();
    defer counts_mutex.unlock();
    const now = device_bytes.load(.monotonic);
    hook.afterRead();
    if (reset_peak) peak_bytes.store(now, .monotonic);
    return .{ .device = now, .host = host_bytes.load(.monotonic), .peak = @max(now, peak_bytes.load(.monotonic)) };
}

fn recordAllocation(counter: *std.atomic.Value(u64), n: usize) void {
    const now = counter.fetchAdd(n, .monotonic) + n;
    if (counter == &device_bytes) _ = peak_bytes.fetchMax(now, .monotonic);
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

pub const DeviceBuffer = struct {
    d: *const Driver,
    ptr: abi.DevicePtr,
    len: usize,
    borrowed: bool = false, // a span of memory another value owns (a carveout's): free leaves it alone

    /// `len` bytes, 256-byte aligned by the driver; zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        try reserveAllocation(len);
        errdefer finishAllocation(null, len);
        var p: abi.DevicePtr = 0;
        if (len > 0) try d.check(d.api.cuMemAlloc_v2(&p, len), "cuMemAlloc");
        finishAllocation(&device_bytes, len);
        return .{ .d = d, .ptr = p, .len = len };
    }

    /// Allocates and fills from host bytes in one call.
    pub fn fromHost(d: *const Driver, bytes: []const u8) Error!DeviceBuffer {
        var b = try alloc(d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (!self.borrowed) {
            if (self.ptr != 0) _ = self.d.api.cuMemFree_v2(self.ptr);
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
        try self.d.check(self.d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
        try self.d.check(self.d.api.cuStreamSynchronize(null), "cuStreamSynchronize");
    }

    pub fn download(self: DeviceBuffer, offset: usize, out: []u8) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "cuMemcpyDtoH");
    }

    /// Asynchronous only when `bytes` is pinned (HostBuffer); pageable memory makes the driver stage it.
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, stream), "cuMemcpyHtoDAsync");
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
