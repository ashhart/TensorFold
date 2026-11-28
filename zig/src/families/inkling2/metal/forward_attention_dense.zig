//! Inkling 2 Mini attention projection routing and exact dense6 dispatch.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const glue = @import("glue.zig");
const dense_mma = @import("dense.zig");
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

fn dense(m: *const Model, e: mtl.ComputeEncoder, w: *const model_file.LayerWeights, s: *state.Scratch, knorm: Ref, vnorm: Ref, rows: usize) void {
    const weights = [6]ck.Tensor{ w.attn.wq, w.attn.wr, w.attn.sink, w.attn.gate, w.attn.wk, w.attn.wv };
    var w_refs: [6]Ref = undefined;
    var n_dims: [6]u32 = undefined;
    for (weights, 0..) |t, i| {
        w_refs[i] = r(t);
        n_dims[i] = @intCast(t.shape[t.rank - 2]);
        std.debug.assert(t.shape[t.rank - 1] == state.hidden);
    }
    glue.dense6(&m.glue, e, at(s.q_norm, 0), knorm, vnorm, w_refs, n_dims, .{
        at(s.q_raw, 0), at(s.rel_raw, 0), at(s.sink, 0), at(s.gate, 0), at(s.k_raw, 0), at(s.v_raw, 0),
    }, @intCast(rows), state.hidden);
}

/// Uses the existing matrix prompt route or one exact dense6 dispatch for all six projections.
pub fn project(m: *const Model, e: mtl.ComputeEncoder, w: *const model_file.LayerWeights, s: *state.Scratch, knorm: Ref, vnorm: Ref, rows: usize, comptime projectOne: anytype) void {
    if (m.dense_mma != null and rows >= dense_mma.min_rows) {
        projectOne(m, e, w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
        projectOne(m, e, w.attn.wk, knorm, at(s.k_raw, 0), rows);
        projectOne(m, e, w.attn.wv, vnorm, at(s.v_raw, 0), rows);
        projectOne(m, e, w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
        projectOne(m, e, w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
        projectOne(m, e, w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
    } else {
        dense(m, e, w, s, knorm, vnorm, rows);
    }
}
