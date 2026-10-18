//! Drama 3's forward on the engine's buffers: backbone and tower layers, the MoE, and the frame step's launches.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const eng = @import("engine.zig");

const Engine = eng.Engine;
const Sampling = eng.Sampling;
const Probe = eng.Probe;
const H = cfg.hidden;
const W = cfg.frame_width;
const max_split_rows = eng.max_split_rows;

/// The scratch and stream one chain of launches uses: the backbone and the tower run concurrently in a frame.
pub const Lane = struct { s: cuda.Stream, xn: u64, tmp: u64, proj: u64, y: u64, q: u64, pos: u64, pml: u64, pacc: u64 };

const lay_bb: kern.Kernels.QkLayout = .{ .q_off = 0, .q_head_stride = 2 * cfg.head_dim, .k_off = 2 * cfg.heads * cfg.head_dim, .v_off = 2 * cfg.heads * cfg.head_dim + cfg.kv_heads * cfg.head_dim, .gate_off = cfg.head_dim, .gate_head_stride = 2 * cfg.head_dim };
const lay_tw: kern.Kernels.QkLayout = .{ .q_off = 0, .q_head_stride = cfg.head_dim, .k_off = 2 * cfg.heads * cfg.head_dim, .v_off = 2 * cfg.heads * cfg.head_dim + cfg.kv_heads * cfg.head_dim, .gate_off = cfg.heads * cfg.head_dim, .gate_head_stride = cfg.head_dim };

/// Gated attention of t rows from the lane's projection rows (layout `lay`) and caches kc / vc into the lane's y.
fn attention(e: *Engine, ln: Lane, t: usize, start: usize, lay: kern.Kernels.QkLayout, qn: u64, kn: u64, kc: u64, vc: u64, interleaved: bool) !void {
    const k = &e.k;
    const s = ln.s;
    const ps = cfg.attn_proj;
    try k.qkRope(s, ln.proj, ps, lay, qn, kn, ln.pos, ln.q, kc, vc, t, interleaved, cfg.rope_theta, cfg.eps);
    if (t == 1) {
        if (t <= max_split_rows) return k.attnSplit(s, ln.q, kc, vc, ln.pos, ln.pml, ln.pacc, ln.proj, ps, lay, ln.y, t);
        return k.attnDecode(s, ln.q, kc, vc, ln.pos, ln.proj, ps, lay, ln.y, t);
    }
    std.debug.assert(ln.pos == e.pos and ln.s.handle == e.stream.handle); // prompt rows: the main stream
    // prompt rows a chunk of queries at a time: fp32 scores q K^T / 16, causal softmax, then P V
    const ac = @min(e.o.attn_chunk, e.o.max_rows);
    var c0: usize = 0;
    while (c0 < t) : (c0 += ac) {
        const tc = @min(ac, t - c0);
        const L = start + c0 + tc;
        const ld = (L + 7) / 8 * 8;
        if (c0 > 0) try e.setPos(start + c0);
        const qrow = ln.q + c0 * 4096 * 2;
        try k.qGroup(s, qrow, e.qg, tc);
        const m8 = 8 * tc;
        try k.gemm(s, .t, .n, L, m8, 256, 0.0625, kc, 512, 256, e.qg, 256, m8 * 256, e.scores, ld, m8 * ld, 2, .bf16, .f32);
        try k.causalSoftmax(s, e.scores, tc, ld, e.pos);
        try k.bf16F32(s, vc, e.vf, L * 512);
        try k.gemm(s, .n, .n, 256, m8, L, 1.0, e.vf, 512, 256, e.scores, ld, m8 * ld, e.og, 256, m8 * 256, 2, .f32, .f32);
        try k.attnGate(s, e.og, ln.proj + c0 * ps * 2, ps, lay, ln.y + c0 * 4096 * 2, tc);
    }
    if (t > ac) try e.setPos(start);
}

/// The backbone's lane: the main stream and its scratch.
pub fn laneBB(e: *Engine, pos: u64) Lane {
    return .{ .s = e.stream, .xn = e.xn, .tmp = e.tmp, .proj = e.proj, .y = e.y, .q = e.q, .pos = pos, .pml = e.pml[0], .pacc = e.pacc[0] };
}

/// The tower's lane on stream `s`, with its own scratch.
pub fn laneTW(e: *Engine, s: cuda.Stream) Lane {
    return .{ .s = s, .xn = e.t_xn, .tmp = e.t_tmp, .proj = e.t_proj, .y = e.t_y, .q = e.t_q, .pos = e.pos, .pml = e.pml[1], .pacc = e.pacc[1] };
}

/// One backbone layer over t rows, e.x (the residual stream) in and out.
pub fn backboneLayer(e: *Engine, ln: Lane, li: usize, t: usize, start: usize) !void {
    const k = &e.k;
    const s = ln.s;
    const L = &e.w.layers[li];
    try k.rmsnorm(s, e.x, H, L.in_norm, e.xn, H, t, H, cfg.eps, true);
    if (L.full) {
        try k.linear(s, L.attn.qkv, e.xn, H, H, cfg.attn_proj, e.proj, cfg.attn_proj, t);
        try attention(e, ln, t, start, lay_bb, L.attn.q_norm, L.attn.k_norm, e.kv_bb[li][0], e.kv_bb[li][1], false);
        try k.linear(s, L.attn.o, e.y, 4096, 4096, H, e.tmp, H, t);
    } else {
        const ps = cfg.gdn_proj;
        try k.linear(s, L.gdn.proj, e.xn, H, H, ps, e.proj, ps, t);
        try k.convSilu(s, e.proj, ps, t, L.gdn.conv, e.conv[li], e.conv_out, cfg.conv_channels);
        try k.gdn(s, e.conv_out, e.proj, ps, t, L.gdn.a_log, L.gdn.dt_bias, e.gstate[li], e.o32);
        try k.gatedNorm(s, e.o32, e.proj, ps, L.gdn.norm, e.y, t, cfg.eps);
        try k.linear(s, L.gdn.out, e.y, 4096, 4096, H, e.tmp, H, t);
    }
    try k.addRmsnorm(s, e.x, e.tmp, H, L.post_norm, e.xn, t, H, cfg.eps);
    try moe(e, s, L, t);
}

/// The MoE block: router, gated shared expert and the routed FP8 experts, added to the residual stream.
fn moe(e: *Engine, s: cuda.Stream, L: *const wts.Layer, t: usize) !void {
    const k = &e.k;
    const mp = cfg.moe_proj;
    try k.linear(s, L.moe_in, e.xn, H, H, mp, e.sp, mp, t);
    try k.router(s, e.sp, mp, 2 * cfg.shared_width, e.picks, e.rw, t);
    try k.swiglu(s, e.sp, mp, 0, cfg.shared_width, cfg.shared_width, e.sact, t);
    try k.linear(s, L.shared_down, e.sact, cfg.shared_width, cfg.shared_width, H, e.shout, H, t);
    const pairs = t * cfg.top_k;
    const prompt = t > 2;
    const tile: usize = if (prompt) cuda.experts.promptTile(.fp8g) else 16;
    try k.plan.route(s, e.picks, pairs, cfg.experts, tile, e.plan);
    const items = cuda.grouped.maxItems(pairs, cfg.experts, tile);
    // activations to FP8 per row and 128 inputs before each expert projection, as the reference serves them
    try k.actQuant(s, e.xn, H, e.xq, H, e.xsc, t, H);
    try k.experts(s, true, prompt, e.xq, H, cfg.top_k, e.xsc, L.experts, e.plan, e.gu, items);
    try k.actQuant(s, e.gu, cfg.expert_width, e.gq, cfg.expert_width, e.gsc, pairs, cfg.expert_width);
    try k.experts(s, false, prompt, e.gq, cfg.expert_width, 0, e.gsc, L.experts, e.plan, e.down, items);
    try k.moeCombine(s, e.x, e.down, e.rw, e.shout, e.sp, mp, 2 * cfg.shared_width + cfg.experts, H, t);
}

/// Tower layer ti over t rows: hidden hin -> hout; label frames lf (rows lfs ints apart), input frames inf.
pub fn towerLayer(e: *Engine, ln: Lane, ti: usize, t: usize, start: usize, hin: u64, hout: u64, lf: u64, lfs: usize, inf: u64) !void {
    const k = &e.k;
    const s = ln.s;
    const T = &e.w.tower[ti];
    const code_col: usize = if (ti < cfg.shared_layers) 1 else 1 + (ti - cfg.shared_layers);
    try k.towerMix(s, lf, lfs, code_col, inf, ti == 0, T.table, e.w.cb_emb, hin, T.enorm, T.hnorm, e.mix, H, e.c.ids.sem_begin, e.c.ids.sem_end, cfg.codebook_size, cfg.eps, t);
    try k.linear(s, T.eh, e.mix, 2 * H, 2 * H, H, e.mixed, H, t);
    try k.rmsnorm(s, e.mixed, H, T.attn_norm, ln.xn, H, t, H, cfg.eps, true);
    try k.linear(s, T.qkv, ln.xn, H, H, cfg.attn_proj, ln.proj, cfg.attn_proj, t);
    try attention(e, ln, t, start, lay_tw, T.q_norm, T.k_norm, e.kv_tw[ti][0], e.kv_tw[ti][1], true);
    try k.linear(s, T.wo, ln.y, 4096, 4096, H, ln.tmp, H, t);
    try k.addRmsnorm(s, e.mixed, ln.tmp, H, T.ffn_norm, ln.xn, t, H, cfg.eps); // resid = o + mixed
    try k.linear(s, T.w13, ln.xn, H, H, 2 * cfg.tower_ffn, ln.proj, 2 * cfg.tower_ffn, t);
    try k.swiglu(s, ln.proj, 2 * cfg.tower_ffn, 0, cfg.tower_ffn, cfg.tower_ffn, e.act, t);
    try k.linear(s, T.w2, e.act, cfg.tower_ffn, cfg.tower_ffn, H, ln.tmp, H, t);
    try k.addRmsnorm(s, e.mixed, ln.tmp, H, T.final_norm, hout, t, H, cfg.eps);
}

fn spOf(s: Sampling) kern.Sampling {
    return .{ .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .seed = s.seed };
}

/// A frame step's launches, one graph: sample, the tower, record, then the backbone row at the next position.
pub fn enqueueFrame(e: *Engine, smp: Sampling, probe: ?Probe, force: ?Engine.Force) !void {
    const k = &e.k;
    const s = e.stream;
    const sem_begin = e.c.ids.sem_begin;
    const greedy = smp.greedy();
    const sp = spOf(smp);
    // the tail of the tower runs on the second stream, except under a probe whose reads would race
    const split = e.o.split and probe == null;
    try k.gemv(s, e.w.cand_head, e.hlast, H, e.w.n_cand, 0, e.cand_logits);
    if (probe) |p| try p.f(p.ctx, "tower_decode0.candidate_logits", e.cand_logits, 1, e.w.n_cand, true);
    if (force) |fc| {
        try k.argmax(s, e.cand_logits, e.w.n_cand, e.w.cand_ids, fc.picks, sem_begin, cfg.codebook_size);
        try k.copyI32(s, fc.frame, e.frame, 2);
    } else if (greedy) {
        try k.argmax(s, e.cand_logits, e.w.n_cand, e.w.cand_ids, e.frame, sem_begin, cfg.codebook_size);
    } else {
        try k.sampleMain(s, e.cand_logits, e.w.n_cand, e.w.cand_ids, e.window, sp, e.step, e.frame, sem_begin, cfg.codebook_size);
    }
    var hin = e.hlast;
    var ts = s;
    for (0..cfg.tower_layers) |ti| {
        const hout = e.th[ti % 2];
        try towerLayer(e, laneTW(e, ts), ti, 1, 0, hin, hout, e.frame, W, e.cur_input);
        if (probe) |p| {
            var nb: [64]u8 = undefined;
            try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_decode0.layer{d:0>2}", .{ti}), hout, 1, H, false);
        }
        hin = hout;
        if (ti >= cfg.shared_layers) {
            const hi = ti - cfg.shared_layers;
            try k.gemv(ts, e.w.heads[hi], hout, H, cfg.codebook_size, 0, e.code_logits);
            if (probe) |p| {
                var nb: [64]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_decode0.head{d}_logits", .{hi}), e.code_logits, 1, cfg.codebook_size, true);
            }
            const out = e.frame + (2 + hi) * 4;
            if (force) |fc| {
                try k.argmax(ts, e.code_logits, cfg.codebook_size, 0, fc.picks + (2 + hi) * 4, 0, 0);
                try k.copyI32(ts, fc.frame + (2 + hi) * 4, out, 1);
            } else if (greedy) {
                try k.argmax(ts, e.code_logits, cfg.codebook_size, 0, out, 0, 0);
            } else {
                try k.sampleCode(ts, e.code_logits, cfg.codebook_size, sp, e.step, @intCast(2 + hi), out);
            }
            if (hi == 0) {
                // main token, code 0 and code 1 are drawn: the next backbone row can start
                try k.nextInput(s, e.frame, e.bb_input, e.pos, e.pos_bb);
                if (split) {
                    try e.ev_fork.record(s);
                    try e.stream2.wait(e.ev_fork);
                    ts = e.stream2;
                }
            }
        }
    }
    if (split) try e.ev_join.record(e.stream2);
    // the backbone row of the new frame at pos + 1
    const bb = laneBB(e, e.pos_bb);
    try k.embedBackbone(s, e.bb_input, e.w.embed, e.w.cb_emb, e.x, 1, H, sem_begin, e.c.ids.sem_end, cfg.codebook_size);
    for (0..cfg.layers) |li| try backboneLayer(e, bb, li, 1, 0);
    try k.rmsnorm(s, e.x, H, e.w.final_norm, e.hlast, H, 1, H, cfg.eps, true);
    if (split) try s.wait(e.ev_join);
    try k.finishFrame(s, e.frame, e.frames_out, e.cur_input, e.window, e.pos, e.step, e.count);
}
