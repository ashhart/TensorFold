//! The Forward's prompt-chunk path (embed, block pre, mixer, post, finish), re-exported from cuda_forward.zig.

const std = @import("std");
const kern = @import("cuda_kernels.zig");
const latent = @import("cuda_latent.zig");
const kept = @import("cuda_marks.zig");
const adapt = @import("cuda_lesson_trace.zig").adapt;
const fwd = @import("cuda_forward.zig");
const Forward = fwd.Forward;
const Walk = fwd.Walk;

/// Engine.prefill_chunk: `rows` tokens in p_ids at positions pos..; commits them and samples the next token.
pub fn chunk(f: *const Forward, rows: usize, pos: usize) !void {
    var w: Walk = .{ .rows = rows, .pos = pos };
    try f.chunkBegin(&w);
    for (0..f.w.blocks.len) |i| {
        try f.chunkPre(&w, i);
        try f.chunkMixer(&w, i);
        if (f.unread and i == lastState(f.w.blocks)) return;
        try f.chunkPost(&w, i);
    }
    try f.chunkFinish(&w);
}

/// The last block that writes state: every block past it reads and writes its rows alone.
fn lastState(blocks: anytype) usize {
    var i = blocks.len - 1;
    while (i > 0 and blocks[i].kind == .moe) i -= 1;
    return i;
}

/// A chunk's embedding, from p_ids.
pub fn chunkBegin(f: *const Forward, w: *Walk) !void {
    const b = f.b;
    try f.embed(b.p_ids, b.emb, w.rows);
    if (f.vision) |v| try v.splice(f.ops.s, w.pos, w.pos + w.rows, b.emb);
    if (f.audio) |a| try a.splice(f.ops.s, w.pos, w.pos + w.rows, b.emb);
    w.x = b.emb;
}

/// Block i up to its mixer: the norm, then the Mamba input projection or attention's q, k and v.
pub fn chunkPre(f: *const Forward, w: *Walk, i: usize) !void {
    const blk = f.w.blocks[i];
    const b = f.b;
    try f.norm(&w.x, w.delta, blk.norm, w.rows, false);
    switch (blk.kind) {
        .mamba => try f.ops.prefillDense(b.y, blk.mamba.in_proj, b.proj, w.rows),
        .attention => {
            try f.ops.prefillDense(b.y, blk.attn.qkv, b.qkv, w.rows);
            try f.ops.fill32(b.p_meta, @intCast(w.pos), 4);
        },
        .moe => {},
    }
}

/// A block mixer reads its Mamba state or attention cache from earlier rows.
pub fn chunkMixer(f: *const Forward, w: *Walk, i: usize) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    switch (f.w.blocks[i].kind) {
        .mamba => {
            const m = f.w.blocks[i].mamba;
            const layer: u64 = c.mamba_heads * c.mamba_head_dim * c.state * 4;
            const base = b.conv_base + @as(u64, w.mj) * 3 * c.convDim() * 2;
            var snaps: kern.Snaps = .{}; // the scan stores its state at each mark too, the bits a chunk ending there stores
            for (f.captures, 0..) |cap, k| {
                try f.keepTaps(cap, w, base); // before the commit below moves BASE on
                snaps.p[k] = cap.ssm + w.mj * layer;
                snaps.r[k] = @intCast(cap.at - w.pos);
            }
            try f.glue.convRows(b.proj, base, b.p_xc, m.conv_w, m.conv_b, w.rows, f.mshape());
            try o.scanRows(b.proj, b.p_xc, b.ssm + w.mj * layer, m.a, m.d, m.dt_bias, b.sy, w.rows, c.projDim(), c.mamba_heads, c.mamba_head_dim, c.convDim(), c.groups, c.dt_min, c.dt_max, snaps);
        },
        .attention => {
            const kc = f.cache(b, b.k_cache, w.aj);
            const vc = f.cache(b, b.v_cache, w.aj);
            const qd = c.heads * c.head_dim;
            try f.glue.kvWrite(b.qkv, kc, vc, b.p_meta, w.rows, f.ashape());
            try o.torch().copyRows(b.qkv, c.qkvDim() * 2, b.q, qd * 2, qd * 2, w.rows);
            const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
            try o.prefillAttention(b.q, kc, vc, b.att, w.pos, w.rows, c.heads, c.kv_heads, scale, b.table, f.page_elems);
        },
        .moe => {},
    }
}

/// A mark's conv taps in this layer: the raw inputs of the three rows before it, BASE's last rows then the chunk's.
pub fn keepTaps(f: *const Forward, cap: kept.Mark, w: *const Walk, base: u64) !void {
    const cd: u64 = f.c.convDim();
    const dst = cap.conv + w.mj * 3 * cd * 2;
    const old = 3 -| (cap.at - w.pos); // taps from before the chunk; a mark inside it has at least one row
    try f.ops.copy(dst, base + (3 - old) * cd * 2, old * cd * 2);
    const first = f.b.proj + (cap.at - w.pos - (3 - old)) * f.c.projDim() * 2 + f.c.inner() * 2;
    try f.ops.copyRows(dst + old * cd * 2, cd * 2, first, f.c.projDim() * 2, cd * 2, 3 - old);
}

/// Block i after its mixer: Mamba gate norm and out projection, attention's out projection, or the experts.
pub fn chunkPost(f: *const Forward, w: *Walk, i: usize) !void {
    const c = f.c;
    const b = f.b;
    const blk = f.w.blocks[i];
    switch (blk.kind) {
        .mamba => {
            try f.glue.groupRmsnorm(b.sy, blk.mamba.gnorm, b.g, b.gxs, w.rows, c.inner(), c.groups, c.eps);
            if (f.dump) |d| try d.input(f.ops, "g", b.g, w.rows * c.inner() * 2);
            try f.ops.prefillDense(b.g, blk.mamba.out_proj, b.delta, w.rows);
            if (blk.mamba.out_rest != 0) try f.ops.restRows(b.g, c.inner(), blk.mamba.out_rest, b.delta, c.hidden, false, w.rows, 1, 0, c.hidden, c.inner());
            if (blk.mamba.adapter) |ad| try adapt(f.ops, ad, b.g, c.inner(), b.delta, c.hidden, false, w.rows, &f.route);
            w.delta = .dense;
            w.mj += 1;
        },
        .attention => {
            if (f.dump) |d| try d.input(f.ops, "att", b.att, w.rows * c.heads * c.head_dim * 2);
            try f.ops.prefillDense(b.att, blk.attn.o, b.delta, w.rows);
            const qd = c.heads * c.head_dim;
            if (blk.attn.o_rest != 0) try f.ops.restRows(b.att, qd, blk.attn.o_rest, b.delta, c.hidden, false, w.rows, 1, 0, c.hidden, qd);
            if (blk.attn.adapter) |ad| try adapt(f.ops, ad, b.att, qd, b.delta, c.hidden, false, w.rows, &f.route);
            w.delta = .dense;
            w.aj += 1;
        },
        .moe => if (blk.moe.latent != null) {
            try latent.run(f, blk.moe, w.rows);
            w.delta = .dense;
        } else {
            try f.experts(blk.moe, w.rows, true);
            w.delta = if (blk.moe.bolt != null) .dense else .moe;
        },
    }
    // the drafter's taps: layer i's residual (x + this block's delta) into its ring slot.

    // setTaps admits a tap only where the delta is the bf16 plane (dense blocks; Bolt's MoE).

    // Write taps in the prompt path (chunk) like in windowRows (cuda_forward.zig ~276-280)
    if (f.n_draft_taps > 0 and w.delta == .dense) for (f.draft_taps[0..f.n_draft_taps], 0..) |tp, ti| if (tp == i) {
        // Create a small device buffer to hold the absolute start position for this chunk

        // This is needed because the chunk path doesn't have the standard meta buffer from windowRows
        const meta_buffer = b.p_meta; // Using the existing p_meta buffer as a scratch buffer
        // Write the absolute start position of this chunk into the meta buffer
        try f.ops.fill32(meta_buffer, @intCast(w.pos), 1);

        // Lazy TF_TAP_DEBUG gate check
        const tap_debug = if (fwd.tap_debug_env) |val| val else blk: {
            const val = (@import("sys").environment.read(bool, "TF_TAP_DEBUG", false) catch false);
            fwd.tap_debug_env = val;
            break :blk val;
        };

        // Print tap-write event if debug is enabled
        if (tap_debug) {
            std.debug.print("[tap-write] layer={d} tap_idx={d} pos={d} rows={d}\n", .{ i, ti, w.pos, w.rows });
        }

        // Launch the tap write operation

        // The tap writes the residual (w.x + w.delta) which is equivalent to the block's output residual

        // The buffer mapping:   H = w.x (the block's output residual from the norm)

        //   R = b.delta (the delta from this block's processing)

        //   META[0] = w.pos (absolute start position of this chunk)   row0 = 0 (chunk-local row offset)

        //   rows = w.rows (total rows in chunk)

        //   keep_rows = w.rows (KEEP ALL prompt rows - the drafter needs the full prompt)
        try f.ops.draftTap(w.x, b.delta, f.tap_ring, meta_buffer, f.tap_rows, f.n_draft_taps, ti, 0, w.rows, w.rows, c.hidden);
    };
}

/// The chunk's last norm alone: w.x then holds the residual it read, b.y every row's normed hidden state.
pub fn finalNorm(f: *const Forward, w: *Walk) !void {
    try f.norm(&w.x, w.delta, f.w.norm_f, w.rows, false);
}

/// The chunk's last norm, its rows' hidden states for the MTP head, and the token after its last row.
pub fn chunkFinish(f: *const Forward, w: *Walk) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    const rows = w.rows;
    try f.norm(&w.x, w.delta, f.w.norm_f, rows, false);
    try o.copy(b.p_hidden, b.y, @as(usize, rows) * c.hidden * 2);
    try o.prefillDense(b.y + @as(u64, rows - 1) * c.hidden * 2, f.w.head, b.p_logits, 1);
    try o.fill32(b.p_meta, @intCast(w.pos + rows - 1), 4); // sample_last: the chunk's last row is at pos + rows - 1
    try f.sample(b.p_logits, 1, b.p_meta, b.p_sampled, f.sampled, b.rule);
    if (f.dump) |d| try d.tail(f.ops, b.p_logits, b.p_sampled, 1, c.vocab);
}
