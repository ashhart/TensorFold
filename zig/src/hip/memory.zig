//! Owned HIP device bytes; sync copies and fills check ranges and use the null stream, so synchronize before mixing in non-blocking streams.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const std = @import("std");

/// Bytes this process holds in DeviceBuffers and HostBuffers, and the device peak since the last reset.
pub const Usage = struct { device: u64, host: u64, peak: u64 };

var counts_lock: std.atomic.Mutex = .unlocked;
var device_bytes: u64 = 0;
var host_bytes: u64 = 0;
var peak_bytes: u64 = 0;

/// The counts now; `reset_peak` starts a new peak at the current device bytes. Safe from any thread.
pub fn usage(reset_peak: bool) Usage {
    lock();
    defer counts_lock.unlock();
    if (reset_peak) peak_bytes = device_bytes;
    return .{ .device = device_bytes, .host = host_bytes, .peak = peak_bytes };
}

fn count(device: bool, held: usize, freed: usize) void {
    lock();
    defer counts_lock.unlock();
    if (device) {
        device_bytes = device_bytes + held - freed;
        peak_bytes = @max(peak_bytes, device_bytes);
    } else host_bytes = host_bytes + held - freed;
}

fn lock() void {
    while (!counts_lock.tryLock()) std.Thread.yield() catch {};
}

pub const DeviceBuffer = struct {
    r: *const runtime.Runtime,
    ptr: abi.DevicePtr,
    len: usize,

    pub fn alloc(r: *const runtime.Runtime, len: usize) runtime.Error!DeviceBuffer {
        var ptr: abi.DevicePtr = null;
        if (len != 0) {
            try runtime.check(r.api.hipMalloc(&ptr, len));
            if (ptr == null) return error.Invalid;
        }
        count(true, len, 0);
        return .{ .r = r, .ptr = ptr, .len = len };
    }

    /// Allocates and fills from host bytes in one call.
    pub fn fromHost(r: *const runtime.Runtime, bytes: []const u8) runtime.Error!DeviceBuffer {
        var b = try alloc(r, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (self.ptr != null) _ = self.r.api.hipFree(self.ptr);
        count(true, 0, self.len);
        self.* = undefined;
    }

    /// The device address `offset` bytes in, as kernel arguments and device copies take it; past the end is refused.
    pub fn address(self: DeviceBuffer, offset: usize) runtime.Error!u64 {
        if (offset > self.len) return error.Invalid;
        return @intFromPtr(self.ptr) + offset;
    }

    fn span(self: DeviceBuffer, offset: usize, len: usize) runtime.Error!abi.DevicePtr {
        if (offset > self.len or len > self.len - offset) return error.Invalid;
        if (self.ptr) |p| return @ptrFromInt(@intFromPtr(p) + offset);
        return null;
    }

    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) runtime.Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len != 0) try runtime.check(self.r.api.hipMemcpyHtoD(dst, bytes.ptr, bytes.len));
    }

    pub fn download(self: DeviceBuffer, offset: usize, bytes: []u8) runtime.Error!void {
        const src = try self.span(offset, bytes.len);
        if (bytes.len != 0) try runtime.check(self.r.api.hipMemcpyDtoH(bytes.ptr, src, bytes.len));
    }

    /// Blocks until the null-stream fill completes; later streams can read it.
    pub fn fill8(self: DeviceBuffer, value: u8) runtime.Error!void {
        if (self.len != 0) {
            try runtime.check(self.r.api.hipMemset(self.ptr, value, self.len));
            try runtime.check(self.r.api.hipStreamSynchronize(null));
        }
    }

    /// The buffer must remain alive until the supplied stream completes.
    pub fn fill8Async(self: DeviceBuffer, value: u8, stream: @import("stream.zig").Stream) runtime.Error!void {
        if (self.r != stream.r) return error.Invalid;
        if (self.len != 0) try runtime.check(self.r.api.hipMemsetAsync(self.ptr, value, self.len, stream.handle));
    }

    /// Host storage must stay alive and unmodified until the stream completes.
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, host: HostBuffer, stream: @import("stream.zig").Stream) runtime.Error!void {
        if (self.r != host.r or self.r != stream.r) return error.Invalid;
        const dst = try self.span(offset, host.bytes.len);
        if (host.bytes.len != 0) try runtime.check(self.r.api.hipMemcpyHtoDAsync(dst, host.bytes.ptr, host.bytes.len, stream.handle));
    }

    pub fn downloadAsync(self: DeviceBuffer, offset: usize, host: HostBuffer, stream: @import("stream.zig").Stream) runtime.Error!void {
        if (self.r != host.r or self.r != stream.r) return error.Invalid;
        const src = try self.span(offset, host.bytes.len);
        if (host.bytes.len != 0) try runtime.check(self.r.api.hipMemcpyDtoHAsync(host.bytes.ptr, src, host.bytes.len, stream.handle));
    }

    /// Blocks until the device-to-device copy from `src` (a device address) completes.
    pub fn copyFrom(self: DeviceBuffer, offset: usize, src: u64, n: usize) runtime.Error!void {
        const dst = try self.span(offset, n);
        if (n == 0) return;
        try runtime.check(self.r.api.hipMemcpyDtoD(dst, @ptrFromInt(src), n));
        try runtime.check(self.r.api.hipStreamSynchronize(null));
    }

    /// `src` must stay allocated until the stream completes.
    pub fn copyFromAsync(self: DeviceBuffer, offset: usize, src: u64, n: usize, stream: @import("stream.zig").Stream) runtime.Error!void {
        if (self.r != stream.r) return error.Invalid;
        const dst = try self.span(offset, n);
        if (n != 0) try runtime.check(self.r.api.hipMemcpyDtoDAsync(dst, @ptrFromInt(src), n, stream.handle));
    }

    /// Fills whole 32-bit words and blocks until done; the length must be a multiple of four.
    pub fn fill32(self: DeviceBuffer, value: u32) runtime.Error!void {
        if (self.len % 4 != 0) return error.Invalid;
        if (self.len != 0) {
            try runtime.check(self.r.api.hipMemsetD32(self.ptr, @bitCast(value), self.len / 4));
            try runtime.check(self.r.api.hipStreamSynchronize(null));
        }
    }

    pub fn fill32Async(self: DeviceBuffer, value: u32, stream: @import("stream.zig").Stream) runtime.Error!void {
        if (self.r != stream.r or self.len % 4 != 0) return error.Invalid;
        if (self.len != 0) try runtime.check(self.r.api.hipMemsetD32Async(self.ptr, @bitCast(value), self.len / 4, stream.handle));
    }
};

pub const HostBuffer = struct {
    r: *const runtime.Runtime,
    bytes: []u8,

    pub fn alloc(r: *const runtime.Runtime, len: usize) runtime.Error!HostBuffer {
        return allocFlags(r, len, 0);
    }

    /// Pinned bytes mapped into the device's address space: kernels read and write them with no copy.
    pub fn allocMapped(r: *const runtime.Runtime, len: usize) runtime.Error!HostBuffer {
        return allocFlags(r, len, abi.host_malloc_portable | abi.host_malloc_mapped);
    }

    fn allocFlags(r: *const runtime.Runtime, len: usize, flags: c_uint) runtime.Error!HostBuffer {
        if (len == 0) return error.Invalid;
        var ptr: abi.DevicePtr = null;
        try runtime.check(r.api.hipHostMalloc(&ptr, len, flags));
        if (ptr == null) return error.Invalid;
        const bytes: [*]u8 = @ptrCast(ptr.?);
        count(false, len, 0);
        return .{ .r = r, .bytes = bytes[0..len] };
    }

    /// The device address of a mapped buffer.
    pub fn device(self: HostBuffer) runtime.Error!u64 {
        var ptr: abi.DevicePtr = null;
        try runtime.check(self.r.api.hipHostGetDevicePointer(&ptr, self.bytes.ptr, 0));
        if (ptr == null) return error.Invalid;
        return @intFromPtr(ptr);
    }

    /// All streams using these bytes must have completed before release.
    pub fn free(self: *HostBuffer) void {
        _ = self.r.api.hipHostFree(self.bytes.ptr);
        count(false, 0, self.bytes.len);
        self.* = undefined;
    }
};

test "fills synchronize only the synchronous path and propagate errors" {
    const Mock = struct {
        var calls: u32 = 0;
        var fail_fill: bool = false;
        var fail_sync: bool = false;
        var seen_stream: abi.Stream = null;
        fn fill(_: abi.DevicePtr, _: c_int, _: usize) callconv(.c) abi.Result {
            calls = calls * 10 + 1;
            return if (fail_fill) 1 else 0;
        }
        fn sync(s: abi.Stream) callconv(.c) abi.Result {
            seen_stream = s;
            calls = calls * 10 + 2;
            return if (fail_sync) 1 else 0;
        }
        fn asyncFill(_: abi.DevicePtr, _: c_int, _: usize, s: abi.Stream) callconv(.c) abi.Result {
            seen_stream = s;
            calls = calls * 10 + 3;
            return if (fail_fill) 1 else 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipMemset = Mock.fill;
    r.api.hipStreamSynchronize = Mock.sync;
    r.api.hipMemsetAsync = Mock.asyncFill;
    const b: DeviceBuffer = .{ .r = &r, .ptr = @ptrFromInt(16), .len = 8 };
    Mock.calls = 0;
    Mock.fail_fill = false;
    Mock.fail_sync = false;
    try b.fill8(7);
    try std.testing.expectEqual(@as(u32, 12), Mock.calls);
    try std.testing.expect(Mock.seen_stream == null);
    Mock.calls = 0;
    const s: abi.Stream = @ptrFromInt(32);
    try b.fill8Async(7, .{ .r = &r, .handle = s });
    try std.testing.expectEqual(@as(u32, 3), Mock.calls);
    try std.testing.expectEqual(s, Mock.seen_stream);
    Mock.calls = 0;
    Mock.fail_fill = true;
    try std.testing.expectError(error.HipFailed, b.fill8(7));
    try std.testing.expectEqual(@as(u32, 1), Mock.calls);
    try std.testing.expectError(error.HipFailed, b.fill8Async(7, .{ .r = &r, .handle = s }));
    Mock.fail_fill = false;
    Mock.fail_sync = true;
    try std.testing.expectError(error.HipFailed, b.fill8(7));
}

test "empty buffers and rejected spans do not call HIP" {
    const r: runtime.Runtime = undefined;
    var b = try DeviceBuffer.alloc(&r, 0);
    defer b.free();
    try b.upload(0, &.{});
    var empty: [0]u8 = .{};
    try b.download(0, &empty);
    try b.fill8(7);
    try b.fill8Async(7, .{ .r = &r, .handle = null });
    try std.testing.expectError(error.Invalid, b.upload(1, &.{}));
    try std.testing.expectError(error.Invalid, b.upload(0, &.{1}));
    try std.testing.expectError(error.Invalid, HostBuffer.alloc(&r, 0));
}

test "async copies reject foreign owners and out-of-range bytes before HIP" {
    const r: runtime.Runtime = undefined;
    var bytes = [_]u8{1};
    const b = DeviceBuffer{ .r = &r, .ptr = @ptrFromInt(16), .len = 1 };
    const host = HostBuffer{ .r = &r, .bytes = &bytes };
    const foreign = HostBuffer{ .r = @ptrFromInt(32), .bytes = &bytes };
    const stream = @import("stream.zig").Stream{ .r = &r, .handle = null };
    try std.testing.expectError(error.Invalid, b.fill8Async(7, .{ .r = foreign.r, .handle = null }));
    try std.testing.expectError(error.Invalid, b.uploadAsync(0, foreign, stream));
    try std.testing.expectError(error.Invalid, b.downloadAsync(0, foreign, stream));
    try std.testing.expectError(error.Invalid, b.uploadAsync(1, host, stream));
    try std.testing.expectError(error.Invalid, b.downloadAsync(1, host, stream));
}

test "word fills refuse partial words and synchronize only the blocking path" {
    const Mock = struct {
        var calls: u32 = 0;
        var words: usize = 0;
        fn fill(_: abi.DevicePtr, value: c_int, n: usize) callconv(.c) abi.Result {
            calls = calls * 10 + 1;
            words = n;
            return if (value == -559038737) 0 else 1;
        }
        fn asyncFill(_: abi.DevicePtr, _: c_int, n: usize, _: abi.Stream) callconv(.c) abi.Result {
            calls = calls * 10 + 3;
            words = n;
            return 0;
        }
        fn sync(_: abi.Stream) callconv(.c) abi.Result {
            calls = calls * 10 + 2;
            return 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipMemsetD32 = Mock.fill;
    r.api.hipMemsetD32Async = Mock.asyncFill;
    r.api.hipStreamSynchronize = Mock.sync;
    const b: DeviceBuffer = .{ .r = &r, .ptr = @ptrFromInt(16), .len = 32 };
    try b.fill32(0xdeadbeef);
    try std.testing.expectEqual(@as(u32, 12), Mock.calls);
    try std.testing.expectEqual(@as(usize, 8), Mock.words);
    Mock.calls = 0;
    try b.fill32Async(0, .{ .r = &r, .handle = @ptrFromInt(32) });
    try std.testing.expectEqual(@as(u32, 3), Mock.calls);
    Mock.calls = 0;
    const odd: DeviceBuffer = .{ .r = &r, .ptr = @ptrFromInt(16), .len = 30 };
    try std.testing.expectError(error.Invalid, odd.fill32(0));
    try std.testing.expectError(error.Invalid, odd.fill32Async(0, .{ .r = &r, .handle = null }));
    try std.testing.expectEqual(@as(u32, 0), Mock.calls);
}

test "addresses and device copies stay inside the buffer" {
    const Mock = struct {
        var calls: u32 = 0;
        fn copy(_: abi.DevicePtr, _: abi.DevicePtr, _: usize) callconv(.c) abi.Result {
            calls += 1;
            return 0;
        }
        fn sync(_: abi.Stream) callconv(.c) abi.Result {
            return 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipMemcpyDtoD = Mock.copy;
    r.api.hipStreamSynchronize = Mock.sync;
    const b: DeviceBuffer = .{ .r = &r, .ptr = @ptrFromInt(4096), .len = 64 };
    try std.testing.expectEqual(@as(u64, 4096 + 64), try b.address(64));
    try std.testing.expectError(error.Invalid, b.address(65));
    try std.testing.expectError(error.Invalid, b.copyFrom(32, 8192, 33));
    try std.testing.expectEqual(@as(u32, 0), Mock.calls);
    try b.copyFrom(32, 8192, 32);
    try std.testing.expectEqual(@as(u32, 1), Mock.calls);
}

test "usage counts held bytes and the device peak" {
    const before = usage(true);
    count(true, 100, 0);
    count(false, 7, 0);
    count(true, 0, 60);
    const now = usage(false);
    try std.testing.expectEqual(before.device + 40, now.device);
    try std.testing.expectEqual(before.host + 7, now.host);
    try std.testing.expectEqual(before.device + 100, now.peak);
    count(true, 0, 40);
    count(false, 0, 7);
    try std.testing.expectEqual(before.device, usage(true).peak);
}
