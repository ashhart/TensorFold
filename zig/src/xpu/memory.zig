//! Device memory and pinned host memory, each owned by one value; copies and fills queue on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const Error = @import("driver.zig").Error;

pub const DeviceBuffer = struct {
    ctx: *const Context,
    ptr: ?*anyopaque,
    len: usize,

    /// `len` bytes, 64-byte aligned; zero bytes allocate nothing.
    pub fn alloc(ctx: *const Context, len: usize) Error!DeviceBuffer {
        var p: ?*anyopaque = null;
        if (len > 0) try ctx.d.check(ctx.d.api.zeMemAllocDevice(ctx.handle, &.{}, len, 64, ctx.entry.device, &p), "zeMemAllocDevice");
        return .{ .ctx = ctx, .ptr = p, .len = len };
    }

    /// Allocates and fills from host bytes in one call (waits for the copy).
    pub fn fromHost(ctx: *const Context, s: Stream, bytes: []const u8) Error!DeviceBuffer {
        var b = try alloc(ctx, bytes.len);
        errdefer b.free();
        try b.upload(s, 0, bytes);
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (self.ptr != null) _ = self.ctx.d.api.zeMemFree(self.ctx.handle, self.ptr);
        self.* = undefined;
    }

    /// The device address `offset` bytes in; out of range is refused.
    pub fn at(self: DeviceBuffer, offset: usize) Error!?*anyopaque {
        if (offset > self.len) return error.Invalid;
        return if (self.ptr) |p| @ptrFromInt(@intFromPtr(p) + offset) else null;
    }

    fn span(self: DeviceBuffer, offset: usize, n: usize) Error!?*anyopaque {
        if (offset > self.len or n > self.len - offset) return error.Invalid;
        return self.at(offset);
    }

    /// Queues the copy; `bytes` must stay valid until the stream is synchronized.
    pub fn uploadAsync(self: DeviceBuffer, s: Stream, offset: usize, bytes: []const u8) Error!void {
        try s.copy(try self.span(offset, bytes.len), bytes.ptr, bytes.len);
    }

    pub fn downloadAsync(self: DeviceBuffer, s: Stream, offset: usize, out: []u8) Error!void {
        try s.copy(out.ptr, try self.span(offset, out.len), out.len);
    }

    pub fn upload(self: DeviceBuffer, s: Stream, offset: usize, bytes: []const u8) Error!void {
        try self.uploadAsync(s, offset, bytes);
        try s.synchronize();
    }

    pub fn download(self: DeviceBuffer, s: Stream, offset: usize, out: []u8) Error!void {
        try self.downloadAsync(s, offset, out);
        try s.synchronize();
    }

    /// Queues a fill of the whole buffer with one byte value.
    pub fn fill8(self: DeviceBuffer, s: Stream, value: u8) Error!void {
        try s.fill8(self.ptr, value, self.len);
    }
};

/// Page-locked host memory the device copies from and to without staging.
pub const HostBuffer = struct {
    ctx: *const Context,
    bytes: []align(16) u8,

    pub fn alloc(ctx: *const Context, len: usize) Error!HostBuffer {
        if (len == 0) return error.Invalid;
        var p: ?*anyopaque = null;
        try ctx.d.check(ctx.d.api.zeMemAllocHost(ctx.handle, &.{}, len, 64, &p), "zeMemAllocHost");
        const base: [*]align(16) u8 = @ptrCast(@alignCast(p.?));
        return .{ .ctx = ctx, .bytes = base[0..len] };
    }

    pub fn free(self: *HostBuffer) void {
        _ = self.ctx.d.api.zeMemFree(self.ctx.handle, self.bytes.ptr);
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};
