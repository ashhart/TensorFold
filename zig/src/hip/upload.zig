//! The HIP backend's side of a format's `load`: host tensors onto the device, kept for the model to free.

const std = @import("std");
const quant = @import("core").quant;
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");

/// Device memory of one model: every buffer a format uploads is kept in `buffers`, which the model frees.
pub const Upload = struct {
    r: *const runtime.Runtime,
    buffers: *std.ArrayList(memory.DeviceBuffer),
    gpa: std.mem.Allocator,

    /// What a format's `upload` takes.
    pub fn uploader(u: *const Upload) quant.Uploader {
        return .{ .ctx = @ptrCast(@constCast(u)), .put = put };
    }

    fn put(ctx: *anyopaque, t: quant.Tensor) anyerror!quant.Buf {
        const u: *const Upload = @ptrCast(@alignCast(ctx));
        var b = try memory.DeviceBuffer.fromHost(u.r, t.bytes);
        errdefer b.free();
        try u.buffers.append(u.gpa, b);
        return .{ .ptr = @intFromPtr(b.ptr), .len = b.len, .dtype = t.dtype, .rank = t.rank, .shape = t.shape };
    }
};
