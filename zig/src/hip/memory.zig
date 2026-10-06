//! Owned HIP device bytes; synchronous copies validate ranges before calling the driver.
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

    pub fn fill8(self: DeviceBuffer, value: u8) runtime.Error!void {
        if (self.len != 0) try runtime.check(self.r.api.hipMemset(self.ptr, value, self.len));
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

test "empty buffers and rejected spans do not call HIP" {
    const r: runtime.Runtime = undefined;
    var b = try DeviceBuffer.alloc(&r, 0);
    defer b.free();
    try b.upload(0, &.{});
    var empty: [0]u8 = .{};
    try b.download(0, &empty);
    try b.fill8(7);
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
    try std.testing.expectError(error.Invalid, b.uploadAsync(0, foreign, stream));
    try std.testing.expectError(error.Invalid, b.downloadAsync(0, foreign, stream));
    try std.testing.expectError(error.Invalid, b.uploadAsync(1, host, stream));
    try std.testing.expectError(error.Invalid, b.downloadAsync(1, host, stream));
}
