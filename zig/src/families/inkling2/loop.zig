//! Mini's loop launches (loop.cu): message mixes and rank-r gates; a C2 stand-in (#4337's arithmetic), contract C2's.
const std = @import("std");
const cuda = @import("cuda");

pub const max_blocks = 8;
pub const max_queries = 4;
pub const max_gates = 3;
pub const max_rank = 128;
const threads = 256;

/// loop.cu's LoopMixArgs, field for field.
pub const MixArgs = extern struct {
    blocks: [max_blocks]u64 = @splat(0),
    partial: u64,
    query: [max_queries]u64 = @splat(0),
    out: u64,
    out_stride: i64,
    n_blocks: i32,
    partial_f32: i32,
    nq: i32,
    rows: i32,
    h: i32,
    eps: f32,
    scale: f32,
};

/// loop.cu's LoopGateArgs, field for field.
pub const GateArgs = extern struct {
    h: [max_gates]u64 = @splat(0),
    msg: u64,
    msg_stride: i64,
    slot: [max_gates]i32 = @splat(0),
    down_w: [max_gates]u64 = @splat(0),
    forget: [max_gates]u64 = @splat(0),
    gain: [max_gates]u64 = @splat(0),
    norm: [max_gates]u64 = @splat(0),
    down: u64,
    out: u64,
    out_stride: i64,
    rows: i32,
    h_dim: i32,
    rank: i32,
    eps_unit: f32,
    eps: f32,
};

pub const Kernels = struct {
    mod: cuda.Module,
    mix: cuda.Function,
    gate_down: cuda.Function,
    gate_mix: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .mix = try mod.function("tf_ink2_loop_mix"),
            .gate_down = try mod.function("tf_ink2_loop_gate_down"),
            .gate_mix = try mod.function("tf_ink2_loop_gate_mix"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// The shapes the kernels hold: blocks, queries, gates and rank in their arrays; a refusal names the bound.
pub fn mixFits(a: MixArgs) error{Inkling2LoopShape}!void {
    if (a.rows <= 0 or a.h <= 0 or a.nq <= 0 or a.nq > max_queries or a.n_blocks < 0 or a.n_blocks > max_blocks)
        return error.Inkling2LoopShape;
}

pub fn gateFits(a: GateArgs, count: usize) error{Inkling2LoopShape}!void {
    if (a.rows <= 0 or a.h_dim <= 0 or a.rank <= 0 or a.rank > max_rank or count == 0 or count > max_gates)
        return error.Inkling2LoopShape;
}

/// fp32 mixes of `nq` queries over a row's blocks and partial (no norm): a block a row.
pub fn mix(k: *const Kernels, s: cuda.Stream, a: MixArgs) !void {
    try mixFits(a);
    var args: cuda.Args = .{};
    args.add(a);
    try cuda.launch.launch(k.mix, .{ .grid = .{ .x = @intCast(a.rows) }, .block = .{ .x = threads } }, s, &args);
}

/// The gates' rank-r down rows, then the gated, normed inputs: a block a (row, gate) each.
pub fn gates(k: *const Kernels, s: cuda.Stream, a: GateArgs, n: usize) !void {
    try gateFits(a, n);
    var args: cuda.Args = .{};
    args.add(a);
    const grid: cuda.Dim3 = .{ .x = @intCast(a.rows), .y = @intCast(n) };
    const h: u32 = @intCast(a.h_dim);
    try cuda.launch.launch(k.gate_down, .{ .grid = grid, .block = .{ .x = threads }, .shared = 2 * h * 4 }, s, &args);
    try cuda.launch.launch(k.gate_mix, .{ .grid = grid, .block = .{ .x = threads }, .shared = h * 4 }, s, &args);
}

test "loop: the argument structs keep loop.cu's layouts" {
    try std.testing.expectEqual(@as(usize, 8 * 8 + 8 + 4 * 8 + 8 + 8 + 5 * 4 + 2 * 4 + 4), @sizeOf(MixArgs));
    try std.testing.expectEqual(@as(usize, 3 * 8 + 8 + 8 + 3 * 4 + 4 + 4 * 3 * 8 + 3 * 8 + 3 * 4 + 2 * 4 + 4), @sizeOf(GateArgs));
}

test "loop: shapes past the kernels' arrays are refused by name" {
    const ok: MixArgs = .{ .partial = 1, .out = 1, .out_stride = 0, .n_blocks = 1, .partial_f32 = 0, .nq = 4, .rows = 1, .h = 2304, .eps = 1e-6, .scale = 1 };
    try mixFits(ok);
    var bad = ok;
    bad.nq = 5;
    try std.testing.expectError(error.Inkling2LoopShape, mixFits(bad));
    bad = ok;
    bad.n_blocks = 9;
    try std.testing.expectError(error.Inkling2LoopShape, mixFits(bad));
    const g: GateArgs = .{ .msg = 1, .msg_stride = 0, .down = 1, .out = 1, .out_stride = 0, .rows = 1, .h_dim = 2304, .rank = 64, .eps_unit = 1e-6, .eps = 1e-6 };
    try gateFits(g, 3);
    try std.testing.expectError(error.Inkling2LoopShape, gateFits(g, 4));
    var wide = g;
    wide.rank = 129;
    try std.testing.expectError(error.Inkling2LoopShape, gateFits(wide, 1));
}
