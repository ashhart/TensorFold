//! The LiLiCorr LIVE ROUND: forward() and hold() with the score ladder and the TF_LP_DUMP/hold-profiling state.
const std = @import("std");
const cuda = @import("cuda");
const cuda_glue = @import("cuda_glue.zig");
const head = @import("cuda_lilicorr_head.zig");
const walk = @import("cuda_lilicorr_walk.zig");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");
const lc = @import("cuda_lilicorr.zig");

const E = lc.E;
const Hold = lc.Hold;
const run = lc.run;
const f32acc = lc.f32acc;
const stageOn = lc.stageOn;
const stageDump = lc.stageDump;
const max_streams = lc.max_streams;
const segments_at = lc.segments_at;
const words_at = lc.words_at;
const anchor_at = lc.anchor_at;
const token_emb_at = lc.token_emb_at;
const pass_at = lc.pass_at;
const scores_at = lc.scores_at;
const Segment = lc.Segment;
const Ctx = lc.Ctx;
const Draft = lc.Draft;

pub var lp_dumped = false; // TF_LP_DUMP: the first hold's log_probs + lse (the head-features band check)
/// TF_HOLD_PROFILE: the hold's own phases in host-wall ms (the 16e round-budget drill):
/// fwd (the drafter's forward), body (the stream's launches), sync (its words DtoH + synchronize),
/// emb (the candidate-embed gather + upload), score (deviceScore/head), walk (walk.chain).
pub var hold_prof_env: ?bool = null;

pub var hp_fwd: f64 = 0;
pub var hp_body: f64 = 0;
pub var hp_sync: f64 = 0;
pub var hp_emb: f64 = 0;
pub var hp_score: f64 = 0;
pub var hp_walk: f64 = 0;
pub var hp_launch: f64 = 0;
pub var hp_sync2: f64 = 0;
pub var hp_d1: f64 = 0;
pub var hp_d2: f64 = 0;
pub fn holdProfOn() bool {
    if (hold_prof_env == null) hold_prof_env = std.c.getenv("TF_HOLD_PROFILE") != null;
    return hold_prof_env.?;
}

pub fn holdMs() f64 {
    const c = @import("sys").c;
    hold_clock_reads += 1;
    var t: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e3 + @as(f64, @floatFromInt(t.nsec)) / 1e6;
}

/// The zero-clock contract's own counter (the clock_reads pattern, owner A8): every hold-path
/// clock read ticks here, so a pass of the hold path with TF_HOLD_PROFILE latched off must
/// leave this at zero — starts open through holdStamp (which reads only under the gate) and
/// every end site re-checks the same latched gate.
pub var hold_clock_reads: u64 = 0;

/// A timed hold section's start: the clock is read ONLY when TF_HOLD_PROFILE is set, so a
/// disabled profile does zero clock reads — the site's cost is then an unclocked straight
/// flow through. The gate latches after its first read, so 0 can never mix with a real start.
pub fn holdStamp() f64 {
    return if (holdProfOn()) holdMs() else 0;
}

/// The frame forward: its same-engine source mapping is zig/src/families/qwen3_5/cuda_draft_frame.zig —
/// the qwen3_5 family's in-repo drafter frame this nemotron path ports (owner review round 2, item 8:
/// the earlier 'ref_qwen_frame.ziref' citation named no real file). The calibration ground truth is the
/// oracle's frame dump: forward_out.pt from checkpoint 0db47f8, read verbatim for the frame conditioning
/// below and pinned stage-by-stage by the body-fixture gate (deviceBodyFixtureAB) and the hold-path dumps.
pub fn forward(d: *Draft, items: []const Hold, contexts: []const *Ctx) !void {
    const c = d.c;
    const e = d.e;
    const B = items.len * c.block;

    // Stage + ids: ids[i*block] = pending, the rest mask; one Segment per stream
    const ids = d.staged.bytes[0 .. c.block * items.len * 4];
    const segments: []Segment = @alignCast(std.mem.bytesAsSlice(Segment, d.staged.bytes[segments_at..words_at]));

    // Fill the segments array with values
    for (items, contexts, 0..) |it, ctx, i| {
        // THE FRAME CONDITIONING (the oracle forward_out.pt verbatim): row 0 = the frame's
        // configured start token, rows 1..block-1 = the mask token (c.mask = 990). Row 0's
        // identity is the owner-A5 config rule (row0Token): a config that declares
        // parallel_drafting_token_id wins; the LiLiCorr checkpoint is silent, so the pending
        // token rides row 0 — the leg-9 measured-best face (23/39 vs the collapsed 1/59).
        const id_slice = @as([*]u32, @ptrCast(ids));
        id_slice[i * c.block] = lc.row0Token(&c, it.pending);
        @memset(id_slice[i * c.block + 1 ..][0 .. c.block - 1], c.mask);

        // Set segment info
        segments[i] = .{ .ring = ctx.ring.ptr, .begin = @intCast(ctx.begin), .position = @intCast(ctx.end) };
        if (stageOn() and lc.stage_fwd_count < 3) std.debug.print("[seg] i={d} begin={d} position={d} block={d} (ref frame positions=[1..8])\n", .{ i, ctx.begin, ctx.end, c.block });
    }

    // Copy ids and segments to device
    const hp_t_launch = holdStamp();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyHtoDAsync_v2(d.ids, d.staged.bytes.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyHtoDAsync_v2(d.segments, d.staged.bytes.ptr + segments_at, items.len * @sizeOf(Segment), e.stream.handle), "cuMemcpyHtoDAsync"); // THE SEGMENTS UPLOAD READ THE IDS-REGION OFFSET (block*items*4 = 32B) INSTEAD OF segments_at (512B) — the kernel received mask-id bytes as its DraftSegment (garbage ring pointer, garbage begin/position), the ring range computed empty, and the frame's attention never saw the ring (the 0%-acceptance root cause; upstream qwen3_5 uploads from st.ptr + segments_at)

    // THE SPAN GATE, DERIVED (owner A5): the reference drafter's attention contributes EXACTLY
    // ZERO for the first two decode rounds (frames 0-1 attnout all-zero at every layer, opens
    // large at frame 2) — so the span opens once >=2 accepted rows exist beyond the context's
    // own prompt+absorb phase (spanOpen/Ctx.span_base: the 11/24 numbered-frame oracle's rule,
    // history-derived per context; the pre-cure 38-row-absolute literal held only for the
    // 35-row prompt it was measured on). Every context must be open — the batch's span is.
    var span_open = true;
    for (contexts) |cx| span_open = span_open and lc.spanOpen(cx);
    const stag = stageOn() and B == 8 and span_open and lc.stage_fwd_count < 3;

    // Embedding step
    // The nemotron engine has NO embed method. The embedding path lives in cuda_forward.zig:
    // pub fn embed(f: *const Forward, ids, out, rows) — bf16 table: f.glue.embed16(ids, e.w, out, rows, hidden); 4-bit: f.glue.embed(ids, e.w, e.s, e.b, out, rows, hidden).
    // READ that function and mirror it: the driver needs the engine's embedding table handle and glue fns — find how cuda_forward.zig's Forward obtains `w`/`glue` (the engine's weights field and e.k.glue) and write the same two-branch call in the driver's forward.
    // The 4-bit branch needs the table's scale/bias pointers exactly as forward.zig passes them.
    if (c.embed_scale != 1.0) return error.UnsupportedEmbedScale;
    // the target's embedding table through the shared glue (cuda_forward.zig's embed, bf16 branch)
    const g = cuda_glue.Glue{ .set = if (e.k.triton) |*t| t else null, .f = &e.k.glue, .s = e.stream, .emb = &e.k.emb, .norms = &e.k.norms, .moe = &e.k.moe };
    if (e.w.embed.bf16) try g.embed16(d.ids, e.w.embed.w, d.h, B, c.hidden) else try g.embed(d.ids, e.w.embed.w, e.w.embed.s, e.w.embed.b, d.h, B, c.hidden);
    if (stag) try stageDump(d, d.h, B * c.hidden * 2, "ours_embed");

    // Process each layer
    for (d.w.layers, 0..) |l, i| {
        try norm(d, d.h, l.input_norm, d.x, B);

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.x, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_postnorm_in", .{i}));
        }
        try projDense(d, l.attn_conv, d.x, d.dyn, B);

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.dyn, B * c.dynamic() * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_dyn", .{i}));
        }
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.x, d.dyn, l.attn_base, @as(u64, 0), d.xin, @as(c_int, @intCast(c.hidden)), @as(c_int, 16), @as(c_int, 0), @as(c_int, @intCast(c.block)), f32acc() });
        try projDense(d, l.q, d.xin, d.qraw, B);
        try projDense(d, l.k, d.xin, d.kraw, B);
        try projDense(d, l.v, d.xin, d.vb, B);

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.xin, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_convprep", .{i}));
        }

        // Norm rope batch for q (heads) and k (kv_heads)
        try run(d.e, d.fns.norm_rope_batch, .{ B * c.heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.qraw, l.q_norm, d.q, @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.heads * c.head_dim)), d.segments, @as(c_int, @intCast(c.block)), c.eps, c.theta, f32acc() });
        try run(d.e, d.fns.norm_rope_batch, .{ B * c.kv_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.kraw, l.k_norm, d.kb, @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.kv_heads * c.head_dim)), d.segments, @as(c_int, @intCast(c.block)), c.eps, c.theta, f32acc() });

        // Attention: attn_partial then attn_merge (span-gated: zero contribution while closed)
        if (span_open) {
            try run(d.e, d.fns.attn_partial, .{ B, c.heads, c.chunks() }, .{ 128, 1, 1 }, .{ d.q, d.kb, d.vb, d.segments, @as(c_int, @intCast(c.block)), @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.ringRows())), @as(c_int, @intCast(c.window)), @as(c_int, @intCast(i)), @as(c_int, @intCast(c.chunks())), @as(c_int, 0), d.pm, d.pl, d.po });
            try run(d.e, d.fns.attn_merge, .{ B, c.heads, 1 }, .{ 32, 1, 1 }, .{ d.pm, d.pl, d.po, @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.chunks())), l.sink, d.att });
        } else {
            try d.e.ctx.d.check(d.e.ctx.d.api.cuMemsetD8Async(d.att, 0, B * c.qWidth() * 2, d.e.stream.handle), "cuMemsetD8Async span-closed att");
        }

        try projDense(d, l.o, d.att, d.o, B);

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.o, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_attnout", .{i}));
        }
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.o, d.dyn, l.attn_base, d.h, d.h2, @as(c_int, @intCast(c.hidden)), @as(c_int, 16), @as(c_int, 1), @as(c_int, @intCast(c.block)), f32acc() });
        try norm(d, d.h2, l.post_norm, d.x, B);

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.h2, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_convfin", .{i}));
            try stageDump(d, d.x, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_postnorm_out", .{i}));
        }
        try projDense(d, l.mlp_conv, d.x, d.dyn, B);
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.x, d.dyn, l.mlp_base, @as(u64, 0), d.xin, @as(c_int, @intCast(c.hidden)), @as(c_int, 16), @as(c_int, 0), @as(c_int, @intCast(c.block)), f32acc() });
        try projDense(d, l.gate, d.xin, d.gate, B);
        try projDense(d, l.up, d.xin, d.up, B);
        try run(d.e, d.fns.mlp_act, .{ (B * c.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), @as(c_int, @intCast(c.intermediate)), d.xs });
        try projDense(d, l.down, d.act, d.o, B);

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.o, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_mlpout", .{i}));
        }
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.o, d.dyn, l.mlp_base, d.h2, d.h, @as(c_int, @intCast(c.hidden)), @as(c_int, 16), @as(c_int, 1), @as(c_int, @intCast(c.block)), f32acc() });

        if (stag) {
            var nb: [40]u8 = undefined;
            try stageDump(d, d.h, B * c.hidden * 2, try std.fmt.bufPrint(&nb, "ours_L{d}_res_out", .{i}));
        }
    }

    // Final normalization and head
    const rows = items.len * (c.block - 1);
    var masked = d.h + c.hidden * 2;
    if (items.len > 1) {
        try run(d.e, d.fns.mask_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.h, d.h2, @as(c_int, @intCast(c.hidden)), @as(c_int, @intCast(c.block)) });
        masked = d.h2;
    }
    try norm(d, masked, d.w.final_norm, d.x, rows);

    if (stag) try stageDump(d, masked, rows * c.hidden * 2, "ours_masked");
    if (stag) try stageDump(d, d.x, rows * c.hidden * 2, "ours_finalnorm");

    // The target's lm_head over the block rows
    try e.ops().dense(d.x, d.xs, e.w.head, d.logits, rows);
    try e.k.draws.logprobRows(e.stream, d.logits, c.vocab, d.draft_picks, @intCast(c.topk), false, d.words, rows);

    // DtoH the words into staged buffer
    if (holdProfOn()) hp_body = holdMs() - hp_t_launch;
    const hp_t_sync = holdStamp();
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + words_at, d.words, rows * (3 + 2 * 20) * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    if (holdProfOn()) hp_sync = holdMs() - hp_t_sync;

    if (stag) lc.stage_fwd_count += 1;
}

pub fn widenBuf(bytes: []const u8, dst: []f32) void {
    for (dst, 0..) |*v, i| v.* = head.widenBits(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
}

/// S5b: the five candidate features + LayerNorm — the host score()'s exact code (ln_eps, the
/// biased variance, the same loop order). Stays host-side: the log_probs are host-derived from
/// the target's words buffer. Output: [56,5] bf16 bits, uploaded by the caller (1.1KB HtoD).
pub fn featuresBits(gpa: std.mem.Allocator, w: *const head.Weights, log_probs: []const f32, dst: []u16) !void {
    const feats = try gpa.alloc(f32, head.lattice * 5);
    defer gpa.free(feats);
    var mx: [head.slots]f32 = undefined;
    for (0..head.slots) |s| {
        var m: f32 = -std.math.floatMax(f32);
        for (0..head.topk) |k| m = @max(m, log_probs[s * head.topk + k]);
        mx[s] = m;
    }
    for (0..head.slots) |s| for (0..head.topk) |k| {
        const lp = log_probs[s * head.topk + k];
        feats[(s * head.topk + k) * 5 + 0] = lp;
        feats[(s * head.topk + k) * 5 + 1] = @exp(lp);
        feats[(s * head.topk + k) * 5 + 2] = lp - mx[s];
        feats[(s * head.topk + k) * 5 + 3] = @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(head.topk - 1));
        feats[(s * head.topk + k) * 5 + 4] = if (k == 0) 1.0 else 0.0;
    };
    for (0..head.lattice) |r| {
        var mean: f32 = 0;
        for (0..5) |k| mean += feats[r * 5 + k];
        mean /= @as(f32, @floatFromInt(5));
        var varr: f32 = 0;
        for (0..5) |k| {
            const dv = feats[r * 5 + k] - mean;
            varr += dv * dv;
        }
        varr /= @as(f32, @floatFromInt(5));
        const inv = 1.0 / @sqrt(varr + head.ln_eps);
        for (0..5) |k| feats[r * 5 + k] = (feats[r * 5 + k] - mean) * inv * w.feature_norm_w[k] + w.feature_norm_b[k];
    }
    for (feats, 0..) |v, i| dst[i] = @truncate(@as(u32, @bitCast(nvfp4.bf16(v))) >> 16);
}

pub fn runP(e: *E, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype, comptime name: []const u8) !void {
    const t0 = holdStamp();
    try run(e, f, grid, block, args);
    if (holdProfOn()) {
        const ms = holdMs() - t0;
        hp_n += 1;
        hp_sum += ms;
        if (ms > hp_max) {
            hp_max = ms;
            hp_max_name = name;
        }
    }
}

pub fn deviceScore(d: *Draft, cand_emb: u64, x: u64, anchor: u64, anchor_valid: bool, feat_bits: u64, start_scores: []f32, pair_scores: []f32) !void {
    const hp_t_l = holdStamp();
    defer if (holdProfOn()) {
        hp_launch = holdMs() - hp_t_l;
    };
    const rows = head.lattice;
    const hd = head.hd;
    const factor = head.factor;
    const topk = head.topk;
    const slots = head.slots;
    const dw = &d.dhw.?;
    const ds = &d.ds;

    // a. token projection: cand_emb bf16 [56,4096] -> token_states f32 [56,1024]
    try runP(d.e, d.fns.token_proj, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ cand_emb, dw.token_proj.w, dw.token_proj.b, d.token_states, @as(c_int, @intCast(head.model_hidden)), @as(c_int, @intCast(hd)), @as(c_int, @intCast(rows)) }, "token_proj");

    // b. pass projection: the pass row (d.x row 0) -> pass_states f32 [1024]
    try runP(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ x, dw.pass_proj.w, dw.pass_proj.b, ds.pass_states, @as(c_int, @intCast(head.model_hidden)), @as(c_int, @intCast(hd)), @as(c_int, 1) }, "edge_linear");

    // c. anchor projection, validity gate (memset), its f32 rms norm (the anchor norm stays f32)
    try runP(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ anchor, dw.ctx_proj.w, dw.ctx_proj.b, ds.anchor_state, @as(c_int, @intCast(head.model_hidden)), @as(c_int, @intCast(hd)), @as(c_int, 1) }, "edge_linear");
    if (!anchor_valid) {
        try d.e.ctx.d.check(d.e.ctx.d.api.cuMemsetD8Async(ds.anchor_state, 0, hd * 4, d.e.stream.handle), "cuMemsetD8Async anchor zero");
    }
    try runP(d.e, d.fns.rms_f32, .{ 1, 1, 1 }, .{ hd, 1, 1 }, .{ ds.anchor_state, dw.anchor_norm, ds.anchor_normed, @as(c_int, @intCast(hd)), @as(f32, head.rms_eps) }, "rms_f32");

    // d/e. the feature mlp: feat_bits [56,5] -> fm_up f32 -> silu bf16 -> fm f32
    try runP(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ feat_bits, dw.feature_up.w, dw.feature_up.b, ds.fm_up, @as(c_int, 5), @as(c_int, @intCast(hd)), @as(c_int, @intCast(rows)) }, "edge_linear");
    try runP(d.e, d.fns.silu_bf16, .{ (rows * hd + 1023) / 1024, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.fm_up, ds.fm_up_bits, @as(c_int, @intCast(rows * hd)) }, "silu_bf16");
    try runP(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.fm_up_bits, dw.feature_down.w, dw.feature_down.b, ds.fm, @as(c_int, @intCast(hd)), @as(c_int, @intCast(hd)), @as(c_int, @intCast(rows)) }, "edge_linear");

    // f. lattice assembly: TS/FM/PS/SLOT/RANK (f32) -> one bf16 store
    try runP(d.e, d.fns.lattice, .{ rows, 1, 1 }, .{ hd, 1, 1 }, .{ d.token_states, ds.pass_states, ds.fm, dw.slot_emb, dw.rank_emb, ds.lattice }, "lattice");

    // g. the two layers — the S5a chain verbatim (add's double-cast: bf16(proj) before the add)
    for (0..2) |layer_i| {
        try runP(d.e, d.fns.rms, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.lattice, dw.layers[layer_i].attn_norm, ds.ain, @as(c_int, @intCast(hd)), @as(f32, head.rms_eps) }, "rms");
        try runP(d.e, d.fns.edge_linear, .{ (3 * hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.ain, dw.layers[layer_i].in_proj, dw.layers[layer_i].in_bias, ds.qkv, @as(c_int, @intCast(hd)), @as(c_int, @intCast(3 * hd)), @as(c_int, @intCast(rows)) }, "edge_linear");
        try runP(d.e, d.fns.lat_attn, .{ rows * head.heads, 1, 1 }, .{ 128, 1, 1 }, .{ ds.qkv, ds.bias, ds.att, @as(c_int, @intCast(rows)), @as(c_int, @intCast(hd)), @as(c_int, @intCast(head.head_dim)) }, "lat_attn");
        try runP(d.e, d.fns.lin_f32, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.att, dw.layers[layer_i].out_proj, dw.layers[layer_i].out_bias, ds.proj, @as(c_int, @intCast(hd)), @as(c_int, @intCast(hd)), @as(c_int, @intCast(rows)) }, "lin_f32");
        try runP(d.e, d.fns.add_bf16, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.lattice, ds.proj, ds.lattice2, @as(c_int, @intCast(rows * hd)) }, "add_bf16");
        try runP(d.e, d.fns.rms, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.lattice2, dw.layers[layer_i].mlp_norm, ds.ain, @as(c_int, @intCast(hd)), @as(f32, head.rms_eps) }, "rms");
        try runP(d.e, d.fns.edge_linear, .{ (2 * hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.ain, dw.layers[layer_i].up, dw.layers[layer_i].up_bias, ds.up, @as(c_int, @intCast(hd)), @as(c_int, @intCast(2 * hd)), @as(c_int, @intCast(rows)) }, "edge_linear");
        try runP(d.e, d.fns.silu_bf16, .{ (rows * 2 * hd + 1023) / 1024, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.up, ds.up_bf16, @as(c_int, @intCast(rows * 2 * hd)) }, "silu_bf16");
        try runP(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.up_bf16, dw.layers[layer_i].down, dw.layers[layer_i].down_bias, ds.proj, @as(c_int, @intCast(2 * hd)), @as(c_int, @intCast(hd)), @as(c_int, @intCast(rows)) }, "edge_linear");
        try runP(d.e, d.fns.add_bf16, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.lattice2, ds.proj, ds.lattice, @as(c_int, @intCast(rows * hd)) }, "add_bf16");
    }

    // out norm (bf16, the S5a shape) then widen to f32 — the factor kernel's A_IN is f32
    try runP(d.e, d.fns.rms, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.lattice, dw.out_norm, ds.ain, @as(c_int, @intCast(hd)), @as(f32, head.rms_eps) }, "rms");
    try runP(d.e, d.fns.widen, .{ rows, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.ain, ds.ain_f32, @as(c_int, @intCast(rows * hd)) }, "widen");

    // h. the S4 chain verbatim: factor -> fused edge head + anchor head -> l2norm grid(57) -> dots
    try runP(d.e, d.fns.factor, .{ (hd + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, .{ ds.ain_f32, ds.anchor_normed, dw.factor.w, dw.factor.b, ds.pre_bits, @as(c_int, @intCast(rows)), @as(c_int, @intCast(hd)) }, "factor");
    try runP(d.e, d.fns.edge_linear, .{ (2 * factor + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.pre_bits, ds.fused_w, ds.fused_b, ds.edges, @as(c_int, @intCast(hd)), @as(c_int, @intCast(2 * factor)), @as(c_int, @intCast(rows)) }, "edge_linear");
    try runP(d.e, d.fns.narrow, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.anchor_normed, ds.anchor_bits, @as(c_int, @intCast(hd)) }, "narrow");
    try runP(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ ds.anchor_bits, dw.anchor_out_head.w, dw.anchor_out_head.b, ds.edges + ds.anchor_tail_at, @as(c_int, @intCast(hd)), @as(c_int, @intCast(hd)), @as(c_int, 1) }, "edge_linear");
    try runP(d.e, d.fns.l2norm, .{ rows + 1, 1, 1 }, .{ 1024, 1, 1 }, .{ ds.edges, @as(c_int, @intCast(rows)), @as(c_int, @intCast(2 * factor)), @as(c_int, @intCast(hd)), @as(f32, head.vector_eps) }, "l2norm");
    try runP(d.e, d.fns.dots, .{ 8 + (slots - 1) * topk * topk, 1, 1 }, .{ 256, 1, 1 }, .{ ds.edges + ds.anchor_tail_at, ds.edges, ds.edges + @as(u64, factor) * 4, ds.start, ds.pair, @as(c_int, @intCast(2 * factor)) }, "dots");

    // i. DtoH: start [8] + pair [384] only, into the PINNED staged buffer (the pageable-stack
    // DtoH was the 16e round-budget finding: a staged pageable copy serializes the drain);
    // sync before the caller's host walk, then a plain host copy out of the pinned region.
    const hp_t_d1 = holdStamp();
    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + scores_at, ds.start, topk * 4, d.e.stream.handle), "cuMemcpyDtoHAsync start scores");
    if (holdProfOn()) hp_d1 = holdMs() - hp_t_d1;
    const hp_t_d2 = holdStamp();
    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + scores_at + topk * 4, ds.pair, (slots - 1) * topk * topk * 4, d.e.stream.handle), "cuMemcpyDtoHAsync pair scores");
    if (holdProfOn()) hp_d2 = holdMs() - hp_t_d2;
    const hp_t_s = holdStamp();
    try d.e.stream.synchronize();
    if (holdProfOn()) hp_sync2 = holdMs() - hp_t_s;
    @memcpy(start_scores, std.mem.bytesAsSlice(f32, d.staged.bytes[scores_at..][0 .. topk * 4]));
    @memcpy(pair_scores, std.mem.bytesAsSlice(f32, d.staged.bytes[scores_at + topk * 4 ..][0 .. (slots - 1) * topk * topk * 4]));
}

pub fn hold(d: *Draft, items: []const Hold) !void {
    const lp: usize = 3 + 2 * 20; // sample.cu's LP_WORDS
    if (items.len == 0) return;
    if (items.len > max_streams) return error.BadDraftFrame;
    var contexts: [max_streams]*Ctx = undefined;
    for (items, contexts[0..items.len]) |it, *ctx| {
        ctx.* = d.streams.get(it.stream) orelse return error.UnknownStream;
        if (it.lanes != null) return error.BadLanes;
        if (it.position == 0 or it.depth > d.c.block - 1) return error.BadDraftFrame;
        if (ctx.*.end != it.position - 1) return error.DraftContextBehind;
        ctx.*.held_n = 0;
    }

    // The ring-poison experiment: TF_RING_POISON=1 fills the drafter's ring with bf16 1.0
    // before the frame's forward — the frame's outputs MUST move if its attention reads the ring.
    if (std.c.getenv("TF_RING_POISON") != null) {
        const bytes = d.c.ringBytes();
        const pat = try d.gpa.alloc(u8, bytes);
        defer d.gpa.free(pat);
        var pi: usize = 0;
        while (pi + 2 <= pat.len) : (pi += 2) {
            pat[pi] = 0x80;
            pat[pi + 1] = 0x3F; // bf16 1.0
        }
        try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyHtoDAsync_v2(contexts[0].ring.ptr, pat.ptr, @intCast(bytes), d.e.stream.handle), "ring poison HtoD");
        try d.e.stream.synchronize();
        std.debug.print("[poison] ring filled bytes={d}\n", .{bytes});
    }
    // Load head weights if not already loaded
    // Perform full LiLiCorr head call + chain walk instead of top-1 fill
    const hp_t0 = holdStamp();
    hp_body = 0;
    hp_sync = 0;
    hp_emb = 0;
    hp_score = 0;
    hp_walk = 0;
    hp_n = 0;
    hp_sum = 0;
    hp_max = 0;
    hp_d1 = 0;
    hp_d2 = 0;
    try forward(d, items, contexts[0..items.len]);
    if (holdProfOn()) hp_fwd = holdMs() - hp_t0;

    // Build the head's inputs from the pinned words (pinned layout, logprob_rows.cuh):
    // words[0] = f32 bits of the log normalizer; words[2 + k] = the k-th best id;
    // words[2 + 20 + k] = f32 bits of the k-th best logit. log-softmax[k] = logit[k] - norm.
    const st = d.staged.bytes;
    const topk = d.c.topk;
    // S5b: the device path flips per-process via TF_LILICORR_DEVICE; unset keeps the host path
    // byte-for-byte (the oracle and the L6 fallback). No other hold() behavior changes.
    const device_path = std.c.getenv("TF_LILICORR_DEVICE") != null;
    // zero-filled: the embed gathers ALL 56 candidate ids every hold — the rows<slots tail
    // must be real ids (0) or the gather runs wild (the S3/S4-era undefined-tail crash class).
    // The walk consumes only the first `rows` slots, so zero tails are inert everywhere else.
    var cand: [head.slots * head.topk]u32 = @splat(0);
    var log_probs: [head.slots * head.topk]f32 = @splat(0);
    for (items, contexts[0..items.len], 0..) |it, ctx, i| {
        const rows: usize = @min(it.depth, head.slots);
        for (0..rows) |r| {
            const row_base = words_at + (i * (d.c.block - 1) + r) * lp * 4;
            const words = std.mem.bytesAsSlice(u32, st[row_base..][0 .. lp * 4]);
            const lse: f32 = @bitCast(words[0]);
            for (0..topk) |k| {
                cand[r * topk + k] = words[2 + k];
                const logit: f32 = @bitCast(words[2 + 20 + k]);
                log_probs[r * topk + k] = logit - lse;
            }
        }
        // token embeddings: the TARGET's embedding rows of the 56 candidate ids, gathered
        // device-side then downloaded (v1 host-oracle path)
        if (std.c.getenv("TF_WORDS_DUMP") != null and !lp_dumped and rows > 0) {
            lp_dumped = true;
            const wbase = words_at + i * (d.c.block - 1) * lp * 4;
            const f = try std.Io.Dir.cwd().createFile(d.e.io, "/tmp/our_words.bin", .{});
            defer f.close(d.e.io);
            try std.Io.File.writePositionalAll(f, d.e.io, std.mem.sliceAsBytes(st[wbase..][0 .. rows * lp * 4]), 0);
            for (0..rows) |r| {
                const wrow = std.mem.bytesAsSlice(u32, st[wbase + r * lp * 4 ..][0 .. lp * 4]);
                const w0: f32 = @bitCast(wrow[0]);
                var mx: f32 = -std.math.floatMax(f32);
                for (0..topk) |k| mx = @max(mx, @as(f32, @bitCast(wrow[2 + 20 + k])));
                std.debug.print("[words] r={d} w0(lse?)={d:.4} w1={d} w2(id)={d} maxlogit={d:.4} lse_ge_max={any} row32..39={any}\n", .{ r, w0, wrow[1], wrow[2], mx, w0 >= mx, wrow[32..40].* });
            }
        }
        var emb_ids: [head.slots * head.topk]i32 = undefined;
        for (cand, 0..) |id, k| emb_ids[k] = @intCast(id);
        const hp_t_emb = holdStamp();
        try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyHtoDAsync_v2(d.cand_ids, &emb_ids, emb_ids.len * 4, d.e.stream.handle), "cuMemcpyHtoDAsync cand ids");
        const hg = cuda_glue.Glue{ .set = if (d.e.k.triton) |*t| t else null, .f = &d.e.k.glue, .s = d.e.stream, .emb = &d.e.k.emb, .norms = &d.e.k.norms, .moe = &d.e.k.moe };
        if (d.e.w.embed.bf16) try hg.embed16(d.cand_ids, d.e.w.embed.w, d.cand_emb, emb_ids.len, d.c.hidden) else try hg.embed(d.cand_ids, d.e.w.embed.w, d.e.w.embed.s, d.e.w.embed.b, d.cand_emb, emb_ids.len, d.c.hidden);
        if (holdProfOn()) hp_emb += holdMs() - hp_t_emb;
        if (device_path) {
            // S5b device path: the features stay host-side (the log_probs are host-derived from
            // the target's words buffer), then the full device chain; DtoH start+pair only; the
            // walk stays host-side on the device scores. Scratch (ds.*) was carved at open.
            var fb: [head.lattice * 5]u16 = undefined;
            try featuresBits(d.gpa, &d.hw, &log_probs, &fb);
            try d.ds.buf.upload(@intCast(d.ds.feat_bits - d.ds.buf.ptr), std.mem.sliceAsBytes(&fb));
            var start_f32: [head.topk]f32 = undefined;
            var pair_f32: [(head.slots - 1) * head.topk * head.topk]f32 = undefined;
            const hp_t_score = holdStamp();
            try deviceScore(d, d.cand_emb, d.x, d.anchor_buf.ptr, ctx.anchor_valid, d.ds.feat_bits, &start_f32, &pair_f32);
            if (holdProfOn()) hp_score += holdMs() - hp_t_score;
            const hp_t_walk = holdStamp();
            const ch = try walk.chain(&start_f32, &pair_f32, &cand);
            if (holdProfOn()) hp_walk += holdMs() - hp_t_walk;

            // Debug print for draft proposals
            const debug_env = std.c.getenv("TF_DRAFT_DEBUG") != null;
            if (debug_env) {
                const max_print: usize = @min(4, rows);
                std.debug.print("[draft-debug-d] rows={d} cand[0..{d}]=", .{ rows, max_print });
                for (0..max_print) |ci| std.debug.print("{d} ", .{cand[ci]});
                std.debug.print("chosen=", .{});
                for (0..max_print) |ri| std.debug.print("{d} ", .{cand[ri * topk + ch.index[ri]]});
                std.debug.print("start={d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4} chain0={d}\n", .{ start_f32[0], start_f32[1], start_f32[2], start_f32[3], start_f32[4], start_f32[5], start_f32[6], start_f32[7], ch.index[0] });
                // per-candidate feature distinctness: the 8 row-0 candidates' token_proj outputs + the anchor
                {
                    const hc = head.model_hidden;
                    const ce = try d.gpa.alloc(u8, 8 * hc * 2);
                    defer d.gpa.free(ce);
                    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(ce.ptr, d.cand_emb, 8 * hc * 2, d.e.stream.handle), "ce dump");
                    try d.e.stream.synchronize();
                    var ph: [8]u64 = @splat(0xcbf29ce484222325);
                    for (0..8) |ci| for (0..hc * 2) |bi| {
                        ph[ci] = (ph[ci] ^ @as(u64, ce[ci * hc * 2 + bi])) *% 0x100000001b3;
                    };
                    const an = try d.gpa.alloc(u8, d.c.hidden * 2);
                    defer d.gpa.free(an);
                    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(an.ptr, d.anchor, d.c.hidden * 2, d.e.stream.handle), "an dump");
                    try d.e.stream.synchronize();
                    var ah: u64 = 0xcbf29ce484222325;
                    for (an) |byte| ah = (ah ^ @as(u64, byte)) *% 0x100000001b3;
                    std.debug.print("[head-inp] candemb={x:0>8},{x:0>8},{x:0>8},{x:0>8},{x:0>8},{x:0>8},{x:0>8},{x:0>8} anchor={x:0>16}\n", .{ ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6], ph[7], ah });
                }

                // Extended debug dump for all staged lattice rows: direct prints (no
                // allocation — std.ArrayList(u8).init is gone in Zig 0.17's unmanaged std)
                std.debug.print("[draft-debug-rows] i={d} r0=", .{i});
                const pending_row_base = words_at + (i * (d.c.block - 1) + 0) * lp * 4;
                const pending_words = std.mem.bytesAsSlice(u32, st[pending_row_base..][0 .. lp * 4]);
                for (0..8) |k| {
                    std.debug.print("{d} ", .{pending_words[2 + k]});
                }
                for (1..(d.c.block - 1)) |r| {
                    const row_base = words_at + (i * (d.c.block - 1) + r) * lp * 4;
                    const words = std.mem.bytesAsSlice(u32, st[row_base..][0 .. lp * 4]);
                    std.debug.print("r{d}=", .{r - 1});
                    for (0..8) |k| {
                        std.debug.print("{d} ", .{words[2 + k]});
                    }
                }
                std.debug.print("\n", .{});

                // FNV-1a 64-bit hash of the normed mask rows
                // Calculate n_rows = items.len * (c.block - 1)
                const n_rows = items.len * (d.c.block - 1);
                const bytes_len = n_rows * d.c.hidden * 2; // bf16 bytes
                if (bytes_len > 0) {
                    // In debug branch ONLY: allocate host buffer, copy from device, then hash
                    const host_buffer = try d.gpa.alloc(u8, bytes_len);
                    defer d.gpa.free(host_buffer);

                    // Issue cuMemcpyDtoHAsync_v2 to copy bytes_len bytes FROM d.x INTO the host buffer
                    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(host_buffer.ptr, d.x, bytes_len, d.e.stream.handle), "cuMemcpyDtoHAsync dxhash");
                    // Synchronize the stream to ensure copy completes
                    try d.e.stream.synchronize();

                    // Create a hash of the bytes
                    var hash: u64 = 0xcbf29ce484222325;
                    for (host_buffer) |byte| {
                        hash = hash ^ @as(u64, byte);
                        hash = hash * 0x100000001b3;
                    }
                    std.debug.print("[draft-debug-x] dxhash={x:0>16} rows={d}\n", .{ hash, n_rows });
                }
            }

            for (0..rows) |r| ctx.held[r] = cand[r * topk + ch.index[r]];
            ctx.held_n = rows;
            continue;
        }
        try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + token_emb_at, d.cand_emb, emb_ids.len * d.c.hidden * 2, d.e.stream.handle), "cuMemcpyDtoHAsync token emb");
        // pass hidden: the block rows' post-final-norm hidden (d.x), one DtoH
        try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + pass_at, d.x, rows * d.c.hidden * 2, d.e.stream.handle), "cuMemcpyDtoHAsync pass");
        try d.e.stream.synchronize();
        widenBuf(st[token_emb_at..][0 .. head.lattice * head.model_hidden * 2], &d.token_emb_f32);
        widenBuf(st[pass_at..][0 .. head.lattice * head.model_hidden * 2], &d.pass_f32);
        widenBuf(st[anchor_at..][0 .. head.model_hidden * 2], &d.anchor_f32);

        const hp_t_score = holdStamp();
        const got = try head.score(d.gpa, &d.hw, &d.token_emb_f32, &log_probs, &d.pass_f32, &d.anchor_f32, ctx.anchor_valid);
        defer got.deinit(d.gpa); // the call's two survivors; every intermediate died with the call
        if (holdProfOn()) hp_score += holdMs() - hp_t_score;
        const hp_t_walk = holdStamp();
        const ch = try walk.chain(got.start, got.pair, &cand);
        if (holdProfOn()) hp_walk += holdMs() - hp_t_walk;
        for (0..rows) |r| ctx.held[r] = cand[r * topk + ch.index[r]];
        ctx.held_n = rows;
    }
    if (holdProfOn()) std.debug.print("[hold-profile] fwd={d:.2} body={d:.2} sync={d:.2} emb={d:.2} score={d:.2} walk={d:.2} launch={d:.2} n={d} sum={d:.2} max={d:.2} d1={d:.2} d2={d:.2} dssync={d:.2}\n", .{ hp_fwd, hp_body, hp_sync, hp_emb, hp_score, hp_walk, hp_launch, hp_n, hp_sum, hp_max, hp_d1, hp_d2, hp_sync2 });
}

pub fn norm(d: *Draft, x: u64, w: u64, out: u64, rows: usize) !void {
    return d.e.k.norms.addRmsTail(d.e.stream, x, null, w, d.c.hidden, out, null, rows, d.c.hidden, d.c.eps, .{ .round = true });
}

/// 16c: a checkpoint-bf16 projection through the engine's dense decode — TRUE bf16, no
/// affine-4 requant (the body fixture priced the requant at ~9%/projection; LEG 16b).
pub fn projDense(d: *Draft, q: cuda.qlinear.Weight, x: u64, out: u64, rows: usize) !void {
    const n = q.outputs();
    try run(d.e, d.fns.sums, .{ rows, (n / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(n)), @as(u64, 0) });
    return d.e.ops().dense(x, d.xs, q, out, rows);
}

/// The projection helpers (the absorb + forward backbone)
/// Ported from qwen3_5's cuda_draft.zig
pub fn projAgain(d: *Draft, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !void {
    return projSums(d, l, x, out, rows);
}

pub fn projMany(d: *Draft, ls: []const cuda.qlinear.Affine4, x: u64, outs: []const u64, rows: usize) !void {
    for (ls, outs) |l, out| try projSums(d, l, x, out, rows);
}

pub fn projSums(d: *Draft, l: cuda.qlinear.Affine4, x: u64, out: u64, rows: usize) !void {
    // the group sums first: the gemv face and the verify tiles both read them (our norm emits none)
    try run(d.e, d.fns.sums, .{ rows, (l.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(l.k)), @as(u64, 0) });
    // prompt ingest takes the GEMV face; live rounds tile the verify path (qwen precision.affine's shape)
    if (d.projection_role == .prompt and rows > 16) return d.affine.gemv(d.e.stream, x, d.xs, l, out, rows, 0);
    const tile = d.affine.verifyRows();
    var at: usize = 0;
    while (at < rows) : (at += tile) {
        try d.affine.verify(d.e.stream, x + at * l.k * 2, d.xs + at * (l.k / 64) * 4, l, out + at * l.n * 2, @min(tile, rows - at));
    }
}

/// S5b: the full device chain — the host score()'s stage order, the S5a layer chain and the S4
/// factor/edge/norm/dots chain verbatim. Inputs: cand_emb bf16 [56,4096], x bf16 [4096] (the
/// pass row), anchor bf16 [4096], feat_bits bf16 [56,5] (host features). DtoH: start [8] +
/// pair [384] ONLY — the walk stays host-side on the device scores. Scratch: ds.* (carved at
/// open, L11); hold()-time launches only. The anchor norm stays f32 (the reference's dtype).
pub var hp_n: u32 = 0;
pub var hp_sum: f64 = 0;
pub var hp_max: f64 = 0;
pub var hp_max_name: []const u8 = "";
