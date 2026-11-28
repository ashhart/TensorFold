//! Attention projections, preprocessing and merge for one verified layer.
const mtl = @import("metal");
const Frame = @import("gpu_frame.zig").Frame;
const fwd = @import("forward.zig");
const Ref = @import("projection.zig").Ref;
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
    if (!attn_loop) {
        // the tail's other rows' queries reach only rows its o projection skips: their q rows stay unwritten
        const t0: u32 = if (block + 1 == r.model.config.layers) r.tail_from orelse 0 else 0;
        var q_sums = fwd.sums(f, .sums, rows);
        q_sums.ref.offset += @as(usize, t0) * 4;
        const q_in: Ref = .{ .buffer = f.get(.input).buffer, .offset = f.get(.input).offset + @as(usize, t0) * weights.q.k * 2 };
        const q_out: Ref = .{ .buffer = f.get(.q_gate).buffer, .offset = f.get(.q_gate).offset + @as(usize, t0) * weights.q.n * 2 };
        try r.model.kernels.quant(e, weights.q, q_in, q_sums, f.get(.dims), q_out, rows - t0);
        try r.model.kernels.quant(e, weights.kv, f.get(.input), fwd.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
        e.barrier();
        const kv: u32 = @intCast(r.model.config.kvDim());
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.key), .{ .rows = rows, .width = kv, .stride = 2 * kv, .offset = 0 });
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.value), .{ .rows = rows, .width = kv, .stride = 2 * kv, .offset = kv });
        e.barrier();
    }
    try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .key = f.get(.key), .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = r.current_placed.h_end, .w_end = r.current_placed.w_end, .plane = f.capacity, .geometry = r.geometry, .shift = r.model.config.shift_norms, .rope = r.model.config.rope });
    try r.attention.encodeSums(e, &r.scratch, &r.bindings[layer], plan, f.get(.query), f.get(.rotated_key), f.get(.value), f.get(.mixer_out), caches, f.get(.q_gate), r.model.kernels.prompt, r.reduce != null, if (fwd.outSums(r.model, r.reduce)) fwd.sums(f, .sums, rows) else null);
}
