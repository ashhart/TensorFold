//! Inkling 2 Mini's loop kernels on HIP (zig/kernels/hip/inkling2/loop.hip): their shapes, checks and launches.
const std = @import("std");
const hip = @import("hip");

/// Mini's loop: hidden 2,304, four message roles (q, k, v, mlp) a core layer, 12 core layers.
pub const hidden = 2304;
pub const roles = 4;
pub const core_layers = 12;
/// Threads a message block, which holds one row of one message: wave 0 reduces B_12, wave 1 P_18.
pub const message_threads = 64;
/// Most rows a call takes: the CUDA engine's 2,048-row prompt chunk.
pub const max_rows = 2048;
/// Threads a math probe block (tf_ink2_loop_math).
pub const math_threads = 256;
/// rho's eps: float64 1e-6 narrowed to f32, as main's Inkling 2 rms narrows it.
pub const eps: f32 = @floatCast(@as(f64, 1e-6));

/// binary32(1 / sqrt(hidden)), Mini's 1/48, rounded once from the float64 quotient.
pub fn queryScale() f32 {
    return @floatCast(1 / @sqrt(@as(f64, @floatFromInt(hidden))));
}

/// The probe's functions: the baselines' f32 tanh, exp and 1 / sqrt as the kernels run them, then the device's own.
pub const Math = enum(u8) { tanh, exp, inv_sqrt, device_tanh, device_exp, device_rsqrt };

/// A message launch: the served kernel, or a contract control (rsqrt, a fused share, no +0 start, serial rho, reversed butterfly).
pub const Messages = enum { served, rsqrt, fma, signed_zero, serial_rho, butterfly_desc };

/// The inkling2_loop image's launches (hip.kernels.Group.inkling2_loop).
pub const Kernels = struct {
    mod: hip.Module,
    math: hip.Function,
    messages: [6]hip.Function,

    pub fn load(r: *const hip.Runtime, image: []align(8) const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        const names = [_][:0]const u8{ "tf_ink2_loop_messages", "tf_ink2_loop_ctl_rsqrt", "tf_ink2_loop_ctl_fma", "tf_ink2_loop_ctl_signed_zero", "tf_ink2_loop_ctl_serial_rho", "tf_ink2_loop_ctl_butterfly_desc" };
        var k: Kernels = .{ .mod = mod, .math = try mod.function("tf_ink2_loop_math"), .messages = undefined };
        for (&k.messages, names) |*f, name| f.* = try mod.function(name);
        return k;
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

/// One call's messages: `rows` rows, `count` core layers from the first whose query rows qm and kv point at.
pub const MessageCall = struct {
    bank0: u64, // B_12, bf16 [rows][hidden]
    bank1: u64, // P_18 of the previous pass, bf16 [rows][hidden]
    qm: u64, // middle_cycle_cross_pass_queries, f32, from row 2 * first: [2 * count][hidden]
    kv: u64, // middle_cycle_cross_pass_kv_queries, likewise
    out: u64, // f32 [count][4][rows][hidden]: each role's rows one contiguous plane
    rows: u32,
    count: u32,
};

/// The cross-pass messages of `c.count` core layers under HIP-RMS-W32 and HIP-RETRIEVE-W32.
pub fn messages(k: *const Kernels, s: hip.Stream, c: MessageCall) !void {
    return messagesAs(k, s, .served, c);
}

/// `messages` through a contract control's kernel (or the served one).
pub fn messagesAs(k: *const Kernels, s: hip.Stream, which: Messages, c: MessageCall) !void {
    if (c.rows == 0 or c.rows > max_rows or c.count == 0 or c.count > core_layers) return error.Inkling2LoopShape;
    for ([_]u64{ c.bank0, c.bank1, c.qm, c.kv, c.out }) |p| if (p % 16 != 0) return error.Inkling2LoopAlign;
    var a: hip.launch.Args = .{};
    for ([_]u64{ c.bank0, c.bank1, c.qm, c.kv, c.out }) |v| try a.add(v);
    try a.add(int(c.rows));
    try a.add(queryScale());
    try a.add(eps);
    const grid: hip.launch.Dim3 = .{ .x = c.rows, .y = c.count * roles };
    try go(k.messages[@backingInt(which)], s, grid, message_threads, &a);
}

test {
    _ = @import("hip_loop_test.zig");
}
