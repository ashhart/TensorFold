//! The CUDA weight store weights.zig's loaders fill: device buffers for the engine's life, one staging buffer.
const std = @import("std");
const cuda = @import("cuda");

pub const Store = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,
    d: *const cuda.Driver,
    staging: ?cuda.DeviceBuffer = null,

    const staging_bytes: usize = 24576 * 4096 * 2;

    pub fn alloc(s: *Store, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(s.d, @max(bytes, 256));
        try s.bufs.append(s.gpa, b);
        s.bytes += bytes;
        return b.ptr;
    }

    pub fn upload(s: *Store, bytes: []const u8) !u64 {
        const p = try s.alloc(bytes.len);
        var b = s.bufs.items[s.bufs.items.len - 1];
        try b.upload(0, bytes);
        return p;
    }

    /// `bytes` in the staging buffer (made on first use), for the quantizer to read.
    pub fn stage(s: *Store, bytes: []const u8) !u64 {
        if (bytes.len > staging_bytes) return error.StagingTooSmall;
        if (s.staging == null) s.staging = try cuda.DeviceBuffer.alloc(s.d, staging_bytes);
        try s.staging.?.upload(0, bytes);
        return s.staging.?.ptr;
    }

    pub fn unstage(s: *Store) void {
        if (s.staging) |*b| b.free();
        s.staging = null;
    }

    pub fn deinit(s: *Store) void {
        s.unstage();
        for (s.bufs.items) |*b| b.free();
        s.bufs.deinit(s.gpa);
    }
};
