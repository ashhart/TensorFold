//! Dense projections and routed-expert compute dispatches for Mini's forward visits.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const dense_mma = @import("dense.zig");
const affine4 = @import("affine4.zig");
const experts = @import("experts.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;

const Ref = state.Ref;

fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn row(buf: mtl.Buffer, i: usize, row_bytes: usize) Ref {
    return at(buf, i * row_bytes);
}
fn x(x_: Ref) experts.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
pub const DenseStep = struct { variant: usize, rows: usize };

/// Ceiling-power-of-two matvec variant for `left` rows, reading weights once per <=32 rows.
pub fn denseStep(left: usize) DenseStep {
    std.debug.assert(left > 0);
    const take = @min(left, 32);
    const variant: usize = if (take <= 1) 0 else if (take <= 2) 1 else if (take <= 4) 2 else if (take <= 8) 3 else if (take <= 16) 4 else 5;
    return .{ .variant = variant, .rows = take };
}

/// Every projection uses the shared BF16 matvec's FP32 reduction and one BF16 store.
pub fn dense(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    encodeDense(mtl.ComputeEncoder, m, e, weight, input, out, rows);
}

/// Dispatch helper parameterized on encoder so host tests record exact launch counts and arguments.
pub fn encodeDense(comptime Encoder: type, m: anytype, e: Encoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const k = weight.shape[weight.rank - 1];
    const n = weight.shape[weight.rank - 2];
    var base: usize = 0;
    while (base < rows) {
        const step = denseStep(rows - base);
        e.setPipeline(m.dense[step.variant]);
        e.setBuffer(input.buf, input.off + base * k * 2, 0);
        e.setBuffer(weight.buffer, weight.offset, 1);
        e.setValue(DenseArgs{ .rows = @intCast(step.rows), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(out.buf, out.off + base * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        base += step.rows;
    }
}

/// Dispatch BF16 projections through dense or tensor-unit kernels and MLX 4-bit weights through affine4.
pub fn project(m: *const Model, e: mtl.ComputeEncoder, weight: model_file.DenseWeight, input: Ref, out: Ref, rows: usize) void {
    switch (weight) {
        .bf16 => |w| {
            const n: u32 = @intCast(w.shape[w.rank - 2]);
            const call = dense_mma.Call{ .rows = @intCast(rows), .n = n, .k = @intCast(w.shape[w.rank - 1]), .ldy = n };
            if (m.dense_mma) |mma| {
                if (dense_mma.Mma.serves(call)) return mma.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = w.buffer, .off = w.offset }, .{ .buf = out.buf, .off = out.off }, call);
            } else if (m.dense_tiled) |tiled| {
                if (dense_mma.Tiled.serves(call)) return tiled.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = w.buffer, .off = w.offset }, .{ .buf = out.buf, .off = out.off }, call);
            }
            dense(m, e, w, input, out, rows);
        },
        .affine4 => |q| affine4.dense(&m.affine4.?, e, input, q, out, @intCast(rows)),
    }
}

pub fn routedExperts(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer: usize, rows: usize) !void {
    const w = &m.weights.layers[layer];
    const lp = m.weight_plan.at(@intCast(layer)).precision;
    const down_call = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = if (lp == .mxfp8) .mxfp8 else .nvfp4 };
    if (lp == .tfq3) {
        const pipes = if (m.tfq3_pipes) |*p| p else return error.Tfq3NotLoaded;
        try pipes.dispatch(e, w.tfq3 orelse return error.Tfq3NotLoaded, at(s.mlp_normed, 0), at(s.route_ids, 0), at(s.gu, 0), at(s.down, 0), rows);
        return experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
    }
    const precision: experts.Format = if (lp == .nvfp4) .nvfp4 else if (lp == .mxfp8) .mxfp8 else return error.UnsupportedInkling2ExpertFormat;
    const gu_call = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision };
    if (m.experts.mma != null and rows >= experts.mma_rows) {
        // A row's sums are the same on either path, so a chunk or window takes the one that fits its width.
        const nv = precision == .nvfp4;
        const gu_globals: ?experts.Globals = if (nv) .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) } else null;
        const down_globals: ?experts.Globals = if (nv) .{ .amax = x(r(w.down.amax.?)), .denominator = x(r(w.down.denominator.?)) } else null;
        try experts.group(&m.experts, e, x(at(s.route_ids, 0)), gu_call);
        try experts.mma(&m.experts, e, false, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), gu_globals, x(at(s.gu, 0)), gu_call);
        try experts.mma(&m.experts, e, true, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), down_globals, x(at(s.down, 0)), down_call);
    } else if (rows >= 2) {
        var row0: usize = 0;
        while (row0 < rows) {
            const take: usize = @min(rows - row0, experts.max_grouped_rows);
            const chunk_gu = experts.Call{ .rows = @intCast(take), .k = state.hidden, .n = state.intermediate, .format = precision };
            const chunk_down = experts.Call{ .rows = @intCast(take), .k = state.intermediate, .n = state.hidden, .format = precision };
            const x_in = x(row(s.mlp_normed, row0, state.hidden * 2));
            const ids_in = x(row(s.route_ids, row0, state.experts_per_token * 4));
            const gu_out = x(row(s.gu, row0, state.experts_per_token * state.intermediate * 2));
            const down_out = x(row(s.down, row0, state.experts_per_token * state.hidden * 2));
            try experts.groupBuckets(&m.experts, e, ids_in, x(at(s.expert_group, 0)), chunk_gu);
            if (precision == .nvfp4) {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                try experts.nvfp4DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.expert_group, 0)), down_out, chunk_down);
            } else {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                try experts.mxfp8DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.expert_group, 0)), down_out, chunk_down);
            }
            row0 += take;
        }
    } else {
        if (precision == .nvfp4) {
            try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        } else {
            try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        }
        switch (precision) {
            .nvfp4 => try experts.nvfp4Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
            .mxfp8 => try experts.mxfp8Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
        }
    }
    try experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
}
