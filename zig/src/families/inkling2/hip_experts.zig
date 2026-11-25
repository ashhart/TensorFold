//! Inkling 2 Mini's routed experts on HIP (zig/kernels/hip/inkling2/experts.hip): their shapes, checks and launches.
const std = @import("std");
const hip = @import("hip");

/// Mini's MoE: 128 routed experts, 8 a token, hidden 2,304, expert width 1,152 (w13 holds 2,304 interleaved rows).
pub const experts = 128;
pub const topk = 8;
pub const hidden = 2304;
pub const width = 1152;
/// Every bf16 bit pattern: the inputs the activation factors' proof covers.
pub const factor_inputs = 65536;
/// Threads a factor block (tf_ink2_act_factors).
pub const factor_threads = 256;

/// The inkling2_experts image's launches (hip.kernels.Group.inkling2_experts).
pub const Kernels = struct {
    mod: hip.Module,
    act_factors: hip.Function,

    pub fn load(r: *const hip.Runtime, image: []align(8) const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .act_factors = try mod.function("tf_ink2_act_factors"),
        };
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

/// The activation's fp32 factors at `count` bf16 inputs; `native` uses the device's tanhf and expf (a control only).
pub fn actFactors(k: *const Kernels, s: hip.Stream, in: u64, gate: u64, up: u64, count: u32, native: bool) !void {
    if (count == 0) return error.Inkling2ExpertsShape;
    var a: hip.launch.Args = .{};
    for ([_]u64{ in, gate, up }) |v| try a.add(v);
    try a.add(int(count));
    try a.add(int(@intFromBool(native)));
    try go(k.act_factors, s, .{ .x = std.math.divCeil(u32, count, factor_threads) catch unreachable }, factor_threads, &a);
}

test {
    _ = @import("hip_experts_test.zig");
}
