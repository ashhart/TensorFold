//! The joint schema head's weights on the device: every BF16 tensor in one buffer, uploaded once at open.
const std = @import("std");
const js = @import("core").joint_schema;
const Driver = @import("cuda").Driver;
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;

/// The head's tensors as device addresses, in the host loader's structure.
pub const Dev = js.tensors.WeightsOf(u64);

pub const Weights = struct {
    buf: DeviceBuffer,
    arena: std.heap.ArenaAllocator,
    w: Dev,
    s: js.device.Scalars,

    /// Uploads the mapped head `file` (checked by the host loader) and keeps the scalars.
    pub fn upload(gpa: std.mem.Allocator, d: *const Driver, file: anytype, host: *const js.tensors.Weights) !Weights {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const total = try js.device.bytes(arena.allocator(), host);
        const staging = try gpa.alloc(u8, total);
        defer gpa.free(staging);
        @memset(staging, 0);
        var buf = try DeviceBuffer.alloc(d, total);
        errdefer buf.free();
        const w = try js.device.stage(arena.allocator(), host, file, staging, buf.ptr);
        try put(buf, 0, staging);
        return .{ .buf = buf, .arena = arena, .w = w, .s = js.device.scalars(host) };
    }

    pub fn deinit(w: *Weights) void {
        w.buf.free();
        w.arena.deinit();
        w.* = undefined;
    }
};

/// `bytes` into `buf` at `off`; a failed copy may still write `buf`, so it drains before the error returns.
pub fn put(buf: DeviceBuffer, off: usize, bytes: []const u8) !void {
    buf.upload(off, bytes) catch |e| {
        cuda.completion.drain(buf.d, .{ .d = buf.d, .handle = null }, "a joint schema upload did not drain");
        return e;
    };
}
