//! Owned HIP device bytes; synchronous copies validate ranges before calling the driver.
//! Sync copies/fill use the null stream; explicitly synchronize before mixing with non-blocking streams.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const std = @import("std");

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
        return .{ .r = r, .ptr = ptr, .len = len };
    }

    pub fn free(self: *DeviceBuffer) void {
        if (self.ptr != null) _ = self.r.api.hipFree(self.ptr);
        self.* = undefined;
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
};

pub const HostBuffer = struct {
    r: *const runtime.Runtime,
    bytes: []u8,

    pub fn alloc(r: *const runtime.Runtime, len: usize) runtime.Error!HostBuffer {
        if (len == 0) return error.Invalid;
        var ptr: abi.DevicePtr = null;
        try runtime.check(r.api.hipHostMalloc(&ptr, len, 0));
        if (ptr == null) return error.Invalid;
        const bytes: [*]u8 = @ptrCast(ptr.?);
        return .{ .r = r, .bytes = bytes[0..len] };
    }

    /// All streams using these bytes must have completed before release.
    pub fn free(self: *HostBuffer) void {
        _ = self.r.api.hipHostFree(self.bytes.ptr);
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
