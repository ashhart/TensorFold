//! Attention projections, preprocessing and merge for one verified layer.
const mtl = @import("metal");
const Frame = @import("gpu_frame.zig").Frame;
const fwd = @import("forward.zig");
const wts = @import("gpu_weights.zig");
const attn = @import("attention.zig");

pub fn forward(r: anytype, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
    const plan = (r.current_attention orelse return error.RoundNotReady).*;
    const layer = r.attention_index[block];
    var views: [attn.planning.max_streams]attn.Cache = undefined;
    const caches = r.layerCaches(layer, r.current_round orelse return error.RoundNotReady, &views);
    try r.bindings[layer].upload(plan, caches);
    const attn_loop = if (comptime @hasDecl(@import("core"), "sensenova_loop")) blk: {
        if (!r.model.config.looped) break :blk false;
        try @import("loop.zig").encodeAttnProj(&r.model.kernels, r.model.glue, e, weights, f, rows, @intCast(r.model.config.qDim()), @intCast(r.model.config.kvDim()));
        break :blk true;
    } else false;
    // the tail's other rows' queries reach only rows its o projection skips: their q rows stay unwritten
    const t0: u32 = if (block + 1 == r.model.config.layers) r.tail_from orelse 0 else 0;
    if (!attn_loop) try @import("attention_input.zig").project(&r.model.kernels, e, weights, f, fwd.sums(f, .sums, rows), rows, t0);
    try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .mixed = f.get(.mixed), .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = r.current_placed.h_end, .w_end = r.current_placed.w_end, .plane = f.capacity, .geometry = r.geometry, .shift = r.model.config.shift_norms, .rope = r.model.config.rope, .key = if (attn_loop) f.get(.key) else null, .value = if (attn_loop) f.get(.value) else null });
    const sums = if (fwd.outSums(r.model, r.reduce)) fwd.sums(f, .sums, rows) else null;
    if (attn_loop) {
        try r.attention.encodeContiguousSums(e, &r.scratch, &r.bindings[layer], plan, f.get(.query), f.get(.rotated_key), f.get(.value), f.get(.mixer_out), caches, f.get(.q_gate), r.model.kernels.prompt, r.reduce != null, sums);
    } else {
        try r.attention.encodeSums(e, &r.scratch, &r.bindings[layer], plan, f.get(.query), f.get(.rotated_key), f.get(.mixed), f.get(.mixer_out), caches, f.get(.q_gate), r.model.kernels.prompt, r.reduce != null, sums);
    }
}
