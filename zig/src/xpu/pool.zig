//! Device allocations of one owner: every buffer made through it is freed together, with a running byte total.

const std = @import("std");
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const HostBuffer = @import("memory.zig").HostBuffer;
const Error = @import("driver.zig").Error;

/// Host bytes a weight passes through on its way to the device.
pub const chunk_bytes: usize = 64 << 20;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    ctx: *const Context,
    stream: Stream,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    total: u64 = 0,
    stage: ?HostBuffer = null,

    pub fn deinit(p: *Pool) void {
        p.releaseStage();
        for (p.buffers.items) |*b| b.free();
        p.buffers.deinit(p.gpa);
    }

    /// Frees the pinned staging chunk once loading is done.
    pub fn releaseStage(p: *Pool) void {
        if (p.stage) |*s| s.free();
        p.stage = null;
    }

    pub fn empty(p: *Pool, bytes: usize) Error!DeviceBuffer {
        var b = try DeviceBuffer.alloc(p.ctx, bytes);
        errdefer b.free();
        p.buffers.append(p.gpa, b) catch return error.OutOfMemory;
        p.total += bytes;
        return b;
    }

    /// Zeroed device memory; the fill is queued, in order before any later launch.
    pub fn zeros(p: *Pool, bytes: usize) Error!DeviceBuffer {
        const b = try p.empty(bytes);
        try b.fill8(p.stream, 0);
        return b;
    }

    /// A new device buffer holding `bytes`, streamed through the staging chunk.
    pub fn fromBytes(p: *Pool, bytes: []const u8) Error!DeviceBuffer {
        const b = try p.empty(bytes.len);
        if (p.stage == null) p.stage = try HostBuffer.alloc(p.ctx, chunk_bytes);
        const stage = p.stage.?;
        var off: usize = 0;
        while (off < bytes.len) {
            const n = @min(bytes.len - off, chunk_bytes);
            @memcpy(stage.bytes[0..n], bytes[off..][0..n]);
            try b.uploadAsync(p.stream, off, stage.bytes[0..n]);
            try p.stream.synchronize();
            off += n;
        }
        return b;
    }

    /// A small bf16 tensor widened to fp32 on the device (exact).
    pub fn fromBf16AsF32(p: *Pool, bytes: []const u8) Error!DeviceBuffer {
        const n = bytes.len / 2;
        const wide = p.gpa.alloc(f32, n) catch return error.OutOfMemory;
        defer p.gpa.free(wide);
        for (wide, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16);
        const b = try p.empty(n * 4);
        try b.upload(p.stream, 0, std.mem.sliceAsBytes(wide));
        return b;
    }
};
