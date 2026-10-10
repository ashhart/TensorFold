//! The launchers' shared shorthand: argument types, device addresses and grid arithmetic.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const launch = @import("../launch.zig");
const Function = @import("../module.zig").Function;

pub const Error = runtime.Error;

/// A stream handle; `counting` launches nothing, so a dry run measures the scratch a forward takes.
pub const S = abi.Stream;
pub const counting: S = @ptrFromInt(0x1);
pub const P = ?*anyopaque;
pub const C = ?*const anyopaque;
pub const F = ?[*]f32;
pub const CF = ?[*]const f32;
pub const I = ?[*]i32;
pub const CI = ?[*]const i32;

pub const Dim3 = launch.Dim3;
pub const Args = launch.Args;

/// A device address as the kernel argument it is.
pub fn ad(p: anytype) u64 {
    return @intFromPtr(p);
}

pub fn dim(x: anytype, y: anytype, z: anytype) Dim3 {
    return .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
}

/// ceil(n / by) as a grid extent.
pub fn cdiv(n: anytype, by: anytype) usize {
    return (@as(usize, @intCast(n)) + by - 1) / by;
}

/// The kernel of a triple for an activation kind: fp16 and bf16 by number, anything else the first.
pub fn tri(kind: c_int) usize {
    return if (kind == 1 or kind == 2) @intCast(kind) else 0;
}

/// By kind (0 fp32, 1 fp16, 2 bf16) or by cache kind (0 fp16, 1 bf16, 2 fp32), the kernel one instantiation.
pub const Triple = [3]Function;

pub fn invalid(what: []const u8) Error {
    std.log.err("{s}: shape", .{what});
    return error.Invalid;
}
