//! The Mamba-2 mixer the HIP forward calls: this file is its interface; the kernels behind it plug in here.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");

const Tensor = hip.ops.Tensor;

pub const Error = hip.ops.Error;

/// Whether `mixer` runs: the native server refuses Nemotron until it does.
pub const ready = true;

/// One Mamba layer's state of one stream on this rank: device addresses the stream's buffers hold.
pub const State = struct {
    /// The conv tail: the last conv_kernel - 1 rows of x, B and C (convDim channels), in the activation dtype.
    conv: u64,
    /// The SSM state: (mamba_heads, mamba_head_dim, state) fp32.
    ssm: u64,
};

/// Bytes of a State's conv tail on this rank.
pub fn convBytes(d: model.Dims, act: model.Kind) usize {
    return (d.conv_kernel - 1) * d.convDim() * act.size();
}

/// Bytes of a State's SSM state on this rank.
pub fn ssmBytes(d: model.Dims) usize {
    return d.mamba_heads * d.mamba_head_dim * d.state * 4;
}

/// mixer: x (rows, hidden) act normed rows of one stream at its next positions -> y (rows, hidden); s moves past them.
pub fn mixer(
    o: hip.ops.Ops,
    d: model.Dims,
    act: model.Kind,
    w: model.Mamba,
    s: State,
    x: Tensor,
    rows: usize,
) Error!Tensor {
    // in_proj's rows [z | x B C | dt] in act; the conv, dt, scan and gated norm run over them and move the state
    const proj = try o.project(x, w.in_proj, rows, false);
    const xc = try o.arena.take(rows * d.convDim() * act.size());
    const dt = try o.arena.of(f32, rows * d.mamba_heads);
    const y = try o.arena.take(rows * d.inner() * act.size());
    const normed: Tensor = .{ .ptr = try o.arena.take(rows * d.inner() * act.size()), .kind = act };
    try o.mambaMixer(
        proj,
        .{ .conv = s.conv, .ssm = s.ssm },
        .{ .xc = xc, .dt = dt, .y = y },
        normed.ptr,
        .{ .conv_w = w.conv_w, .conv_b = w.conv_b, .dt_bias = w.dt_bias, .a_log = w.a_log, .d = w.d, .gnorm = w.gnorm },
        rows,
        .{
            .heads = d.mamba_heads,
            .dh = d.mamba_head_dim,
            .groups = d.groups,
            .state = d.state,
            .conv_kernel = d.conv_kernel,
        },
        .{ .dt_min = d.dt_min, .dt_max = d.dt_max, .eps = d.eps },
    );
    // y in act, or fp32 when w.out_proj.partial (under tp: this rank's share, which the caller sums)
    return o.project(normed, w.out_proj, rows, w.out_proj.partial);
}

test "a State's bytes are the plan's" {
    const config = @import("../config.zig");
    const plan = @import("plan.zig");
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"), &why);
    for ([_]u32{ 1, 2, 4 }) |world| {
        const d = try model.Dims.of(&c, world, 0);
        const p = try plan.MambaState.of(&c, world, 2);
        try std.testing.expectEqual(p.conv, convBytes(d, .bf16));
        try std.testing.expectEqual(p.ssm, ssmBytes(d));
    }
}
