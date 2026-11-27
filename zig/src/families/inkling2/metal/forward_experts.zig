//! Mini's routed experts for one visit: the experts' launches in turn, a barrier after each step a later launch reads.
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const experts = @import("experts.zig");
const state = @import("state.zig");
const Model = @import("model.zig").Model;

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

/// Ends with a barrier: the combine's output is written.
pub fn routed(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer: usize, rows: usize) !void {
    const w = &m.weights.layers[layer];
    const precision: experts.Format = switch (m.weight_plan.at(@intCast(layer)).precision) {
        .nvfp4 => .nvfp4,
        .mxfp8 => .mxfp8,
        .exl3 => return error.UnsupportedInkling2ExpertFormat,
    };
    const gu_call = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision };
    const down_call = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision };
    if (m.experts.mma != null and rows >= experts.mma_rows) {
        // A row's sums are the same on either path, so a chunk or window takes the one that fits its width.
        const nv = precision == .nvfp4;
        const gu_globals: ?experts.Globals = if (nv) .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) } else null;
        const down_globals: ?experts.Globals = if (nv) .{ .amax = x(r(w.down.amax.?)), .denominator = x(r(w.down.denominator.?)) } else null;
        try experts.group(&m.experts, e, x(at(s.route_ids, 0)), gu_call);
        e.barrier();
        try experts.mma(&m.experts, e, false, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), gu_globals, x(at(s.gu, 0)), gu_call);
        e.barrier();
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
            // The chunks share the bucket table: each chunk's steps run in turn.
            if (row0 > 0) e.barrier();
            try experts.groupBuckets(&m.experts, e, ids_in, x(at(s.expert_group, 0)), chunk_gu);
            e.barrier();
            if (precision == .nvfp4) {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                e.barrier();
                try experts.nvfp4DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.expert_group, 0)), down_out, chunk_down);
            } else {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                e.barrier();
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
        e.barrier();
        switch (precision) {
            .nvfp4 => try experts.nvfp4Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
            .mxfp8 => try experts.mxfp8Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
        }
    }
    e.barrier();
    try experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
    e.barrier();
}
