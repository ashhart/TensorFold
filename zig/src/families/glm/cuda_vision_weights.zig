//! The GLM vision tower's tensors: where they live (device or mapped pinned host memory) and how they are found.
const std = @import("std");
const cuda = @import("cuda");

/// TF_GLM_VISION_HOST=1: the tower's weights live in pinned host memory, staged onto the device a unit at a time.
pub fn onHost() bool {
    const v = std.c.getenv("TF_GLM_VISION_HOST") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The tower's tensors: a device buffer, or pinned host memory mapped into the device's address space.
pub const Weights = union(enum) {
    device: cuda.DeviceBuffer,
    host: struct { buf: cuda.HostBuffer, ptr: u64 },

    pub fn alloc(d: *const cuda.Driver, len: usize, host: bool) !Weights {
        if (!host) return .{ .device = try cuda.DeviceBuffer.alloc(d, len) };
        var buf = try cuda.HostBuffer.allocMapped(d, len);
        errdefer buf.free();
        return .{ .host = .{ .buf = buf, .ptr = try buf.device() } };
    }

    pub fn ptr(w: Weights) u64 {
        return switch (w) {
            .device => |b| b.ptr,
            .host => |h| h.ptr,
        };
    }

    /// Device bytes the weights hold: none when they live on the host.
    pub fn deviceBytes(w: Weights) u64 {
        return switch (w) {
            .device => |b| b.len,
            .host => 0,
        };
    }

    pub fn free(w: *Weights) void {
        switch (w.*) {
            .device => |*b| b.free(),
            .host => |*h| h.buf.free(),
        }
    }
};

/// A projection's weight: the checkpoint's bf16 for cuBLASLt, or MXFP8 or NVFP4 as stored for qlinear's prompt rows.
pub const Lin = union(enum) { bf16: u64, q: cuda.qlinear.Weight };

/// The address of the tensor named by `fmt` and `args` (after the tower's prefix).
pub fn tensor(m: *const std.StringHashMap(u64), comptime fmt: []const u8, args: anytype) !u64 {
    var nb: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, fmt, args);
    return m.get(name) orelse {
        std.log.err("glm vision: no tensor {s}", .{name});
        return error.MissingVisionTensor;
    };
}

/// The projection named by `fmt` and `args`: packed for qlinear when its recipe quantizes it, else its bf16 weight.
pub fn linear(m: *const std.StringHashMap(u64), qm: *const std.StringHashMap(cuda.qlinear.Weight), comptime fmt: []const u8, args: anytype) !Lin {
    var nb: [128]u8 = undefined;
    const module = try std.fmt.bufPrint(&nb, fmt, args);
    if (qm.get(module)) |q| return .{ .q = q };
    var wb: [136]u8 = undefined;
    const name = try std.fmt.bufPrint(&wb, "{s}.weight", .{module});
    return .{ .bf16 = m.get(name) orelse {
        std.log.err("glm vision: no tensor {s}", .{name});
        return error.MissingVisionTensor;
    } };
}
