//! Kernel launches: every argument set in order, then the grid queued on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Kernel = @import("module.zig").Kernel;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const Error = @import("driver.zig").Error;

/// How a launch argument is passed to the kernel parameter at its position.
pub const ArgKind = enum { buffer, pointer, float, integer };

pub fn argKind(comptime T: type) ArgKind {
    if (T == DeviceBuffer) return .buffer;
    if (T == ?*anyopaque) return .pointer;
    if (T == f32 or T == comptime_float) return .float;
    return switch (@typeInfo(T)) {
        .int, .comptime_int => .integer,
        else => @compileError("kernel argument of type " ++ @typeName(T)),
    };
}

/// Queues `k` over `groups` work-groups; the group size was fixed when the kernel was made.
pub fn launch(k: Kernel, s: Stream, groups: [3]u32) Error!void {
    const g: abi.GroupCount = .{ .x = groups[0], .y = groups[1], .z = groups[2] };
    if (g.x == 0 or g.y == 0 or g.z == 0) return error.Invalid;
    try k.ctx.d.check(k.ctx.d.api.zeCommandListAppendLaunchKernel(s.list, k.handle, &g, null, 0, null), "zeCommandListAppendLaunchKernel");
}

/// Sets every argument of the tuple in order (buffer, device pointer, f32 or u32) and queues the kernel.
pub fn dispatch(k: Kernel, s: Stream, groups: [3]u32, args: anytype) Error!void {
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
        switch (comptime argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
    try launch(k, s, groups);
}

test "argument kinds follow the Zig type" {
    try std.testing.expectEqual(ArgKind.buffer, argKind(DeviceBuffer));
    try std.testing.expectEqual(ArgKind.pointer, argKind(?*anyopaque));
    try std.testing.expectEqual(ArgKind.float, argKind(f32));
    try std.testing.expectEqual(ArgKind.integer, argKind(u32));
    try std.testing.expectEqual(ArgKind.integer, argKind(comptime_int));
    try std.testing.expectEqual(ArgKind.integer, argKind(usize));
}
