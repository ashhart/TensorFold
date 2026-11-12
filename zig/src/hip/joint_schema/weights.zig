//! The joint schema head's weights on the device: every BF16 tensor in one buffer, uploaded once at open.
const std = @import("std");
const js = @import("core").joint_schema;
const runtime = @import("../runtime.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

/// The head's tensors as device addresses, in the host loader's structure.
pub const Dev = js.tensors.WeightsOf(u64);

pub const Scalars = js.device.Scalars;
pub const scalars = js.device.scalars;
pub const bytes = js.device.bytes;

pub const Weights = struct {
    buf: DeviceBuffer,
    arena: std.heap.ArenaAllocator,
    w: Dev,
    s: Scalars,

    /// Uploads the mapped head `file` (checked by the host loader) and keeps the scalars.
    pub fn upload(gpa: std.mem.Allocator, r: *const runtime.Runtime, file: anytype, host: *const js.tensors.Weights) !Weights {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const total = try bytes(arena.allocator(), host);
        const staging = try gpa.alloc(u8, total);
        defer gpa.free(staging);
        @memset(staging, 0);
        var buf = try DeviceBuffer.alloc(r, total);
        errdefer buf.free();
        const w = try js.device.stage(arena.allocator(), host, file, staging, buf.base());
        try buf.upload(0, staging);
        return .{ .buf = buf, .arena = arena, .w = w, .s = scalars(host) };
    }

    pub fn deinit(w: *Weights) void {
        w.buf.free();
        w.arena.deinit();
        w.* = undefined;
    }
};
