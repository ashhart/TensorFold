//! Inkling 2 Mini's loop kernels on HIP (zig/kernels/hip/inkling2/loop.hip): their shapes, checks and launches.
const std = @import("std");
const hip = @import("hip");

/// Threads a math probe block (tf_ink2_loop_math).
pub const math_threads = 256;
/// The probe's functions: the baselines' f32 tanh, exp and 1 / sqrt as the kernels run them, then the device's own.
pub const Math = enum(u8) { tanh, exp, inv_sqrt, device_tanh, device_exp, device_rsqrt };

/// The inkling2_loop image's launches (hip.kernels.Group.inkling2_loop).
pub const Kernels = struct {
    mod: hip.Module,
    math: hip.Function,

    pub fn load(r: *const hip.Runtime, image: []align(8) const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        return .{ .mod = mod, .math = try mod.function("tf_ink2_loop_math") };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

fn int(x: anytype) c_int {
    return @intCast(x);
}

fn go(f: hip.Function, s: hip.Stream, grid: hip.launch.Dim3, threads: u32, a: *hip.launch.Args) !void {
    try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// `count` f32 inputs through one of the probe's functions.
pub fn math(k: *const Kernels, s: hip.Stream, in: u64, out: u64, count: u32, op: Math) !void {
    if (count == 0 or count > std.math.maxInt(c_int)) return error.Inkling2LoopShape;
    var a: hip.launch.Args = .{};
    for ([_]u64{ in, out }) |v| try a.add(v);
    try a.add(int(count));
    try a.add(int(@backingInt(op)));
    try go(k.math, s, .{ .x = std.math.divCeil(u32, count, math_threads) catch unreachable }, math_threads, &a);
}

test {
    _ = @import("hip_loop_test.zig");
}
