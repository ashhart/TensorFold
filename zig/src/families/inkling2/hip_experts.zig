//! Inkling 2 Mini's routed experts on HIP (zig/kernels/hip/inkling2/experts.hip): their shapes, checks and launches.
const std = @import("std");
const hip = @import("hip");

/// Mini's MoE: 128 routed experts, 8 a token, hidden 2,304, expert width 1,152 (w13 holds 2,304 interleaved rows).
pub const experts = 128;
pub const topk = 8;
pub const hidden = 2304;
pub const width = 1152;
/// Outputs a block (experts.hpp kThreads), the longest reduction a block stages (kMaxK), slots the combine orders.
pub const block_outputs = 64;
pub const max_k = 2304;
pub const max_topk = 16;
/// Most routes (or tokens) a launch's second grid dimension holds.
pub const max_rows = 65535;
/// Every bf16 bit pattern: the inputs the activation factors' proof covers.
pub const factor_inputs = 65536;
/// Threads a factor block (tf_ink2_act_factors).
pub const factor_threads = 256;

/// The checkpoint's stored formats: NVFP4 (E2M1 codes, an E4M3 scale per 16, amax and a denominator) or MXFP8.
pub const Format = enum { nvfp4, mxfp8 };

/// One projection's experts on the device, [experts][rows][k] codes and their scales; amax and den are NVFP4's.
pub const Weights = struct { format: Format, codes: u64, scales: u64, amax: u64 = 0, den: u64 = 0 };

/// A call: `rows` tokens (topk routes each), `n` outputs a route, reductions of `k`; ids past `experts` give NaN.
pub const Call = struct { rows: u32, n: u32, k: u32 = 0, topk: u32 = topk, experts: u32 = experts };

/// The inkling2_experts image's launches (hip.kernels.Group.inkling2_experts).
pub const Kernels = struct {
    mod: hip.Module,
    act_factors: hip.Function,
    nvfp4_down: hip.Function,
    mxfp8_down: hip.Function,

    pub fn load(r: *const hip.Runtime, image: []align(8) const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .act_factors = try mod.function("tf_ink2_act_factors"),
            .nvfp4_down = try mod.function("tf_ink2_nvfp4_down"),
            .mxfp8_down = try mod.function("tf_ink2_mxfp8_down"),
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

/// What a call must hold before any launch: whole scale groups, at most 2,304 inputs, rows for one grid, codes aligned.
fn check(w: ?Weights, c: Call, routes: u64) !void {
    if (c.rows == 0 or c.n == 0 or c.topk == 0 or c.topk > max_topk or routes > max_rows) return error.Inkling2ExpertsShape;
    const x = w orelse return;
    const per: u32 = if (x.format == .nvfp4) 16 else 32;
    if (c.k == 0 or c.k > max_k or c.k % per != 0) return error.Inkling2ExpertsShape;
    if (x.format == .nvfp4 and (x.amax == 0 or x.den == 0)) return error.Inkling2ExpertsShape;
    if (x.codes % @as(u64, if (x.format == .nvfp4) 8 else 16) != 0) return error.Inkling2ExpertsAlign;
}

fn weightArgs(a: *hip.launch.Args, w: Weights, x: u64) !void {
    for ([_]u64{ x, w.codes, w.scales }) |v| try a.add(v);
    if (w.format == .nvfp4) for ([_]u64{ w.amax, w.den }) |v| try a.add(v);
}

fn blocks(n: u32) u32 {
    return std.math.divCeil(u32, n, block_outputs) catch unreachable;
}

/// Down: x [rows * topk][k] bf16 (each route's activations, k = width), w2 [experts][n][k]; y [rows * topk][n].
pub fn down(k: *const Kernels, s: hip.Stream, w: Weights, x: u64, ids: u64, y: u64, c: Call) !void {
    const routes = @as(u64, c.rows) * c.topk;
    try check(w, c, routes);
    var a: hip.launch.Args = .{};
    try weightArgs(&a, w, x);
    try a.add(ids);
    try a.add(y);
    try addAll(&a, .{ int(c.k), int(c.n), int(c.experts) });
    try go(if (w.format == .nvfp4) k.nvfp4_down else k.mxfp8_down, s, .{ .x = blocks(c.n), .y = @intCast(routes) }, block_outputs, &a);
}

fn addAll(a: *hip.launch.Args, values: anytype) !void {
    inline for (values) |v| try a.add(v);
}

test {
    _ = @import("hip_experts_test.zig");
}
