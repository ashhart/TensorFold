//! File-backed tensors read into one owned pinned stage; a failed completion quarantines that stage.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const source = @import("copy_source.zig");
const capture = @import("copy_source_capture.zig");
const chunk_bytes = 64 << 20;

/// The reader supplies aligned file reads; CUDA owns every byte the queued upload may still read.
pub fn Source(comptime Direct: type) type {
    return struct {
        const Self = @This();
        const Map = struct { base: usize, len: usize, file: Direct.File };
        gpa: std.mem.Allocator,
        d: *const Driver,
        maps: std.ArrayList(Map) = .empty,
        stage: source.Stage = .{},

        /// Disabled sources touch no checkpoint or driver; discrete cards retain their original uploader.
        pub fn open(gpa: std.mem.Allocator, d: *const Driver, ck: anytype, enabled: bool) !Self {
            var s: Self = .{ .gpa = gpa, .d = d };
            if (!enabled) return s;
            errdefer s.deinit();
            for (ck.files.items) |f| {
                var file = try Direct.File.open(f.path);
                errdefer file.close();
                try s.maps.append(gpa, .{ .base = @intFromPtr(f.map.memory.ptr), .len = f.map.memory.len, .file = file });
            }
            _ = try s.stage.take(d, Direct.span(0, chunk_bytes) + Direct.alignment);
            return s;
        }
        pub fn deinit(s: *Self) void {
            s.stage.deinit(); // on failure, the shared owner keeps the allocation after these file handles close
            for (s.maps.items) |*m| m.file.close();
            s.maps.deinit(s.gpa);
            s.* = undefined;
        }
        /// Constructed vectors are not shard spans: false leaves them with the caller's ordinary copy custody.
        pub fn upload(s: *Self, destination: memory.DeviceBuffer, offset: usize, bytes: []const u8) !bool {
            const at = @intFromPtr(bytes.ptr);
            for (s.maps.items) |m| {
                if (at < m.base or at - m.base > m.len or bytes.len > m.len - (at - m.base)) continue;
                if (destination.d != s.d) return error.Invalid;
                if (capture.active(s.d, source.current())) return error.UnownedCaptureSource;
                if (offset > destination.len or bytes.len > destination.len - offset) return error.Invalid;
                var done: usize = 0;
                while (done < bytes.len) {
                    const n = @min(bytes.len - done, Direct.fits(chunk_bytes));
                    const file_at = try std.math.add(u64, at - m.base, done);
                    try s.stage.wait();
                    const into = s.stage.buf.?.bytes;
                    if (@intFromPtr(into.ptr) % Direct.alignment != 0) return error.UnalignedPinnedSlot;
                    const got = try m.file.read(@alignCast(into), file_at, n);
                    if (got.len != n) return error.EndOfStream;
                    const start = @intFromPtr(got.ptr);
                    const base = @intFromPtr(into.ptr);
                    if (start < base or start - base > into.len or n > into.len - (start - base)) return error.Invalid;
                    const from = start - base;
                    try s.stage.copyRangeTo(destination, offset + done, from, n, .{ .d = s.d, .handle = null });
                    try s.stage.wait();
                    done += n;
                }
                return true;
            }
            return false;
        }
    };
}
