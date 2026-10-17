//! Drama 3's forward on Metal: backbone and tower layers, the MoE, and one frame step's dispatches.
const std = @import("std");
const cfg = @import("../config.zig");
const common = @import("../common.zig");
const wts = @import("weights.zig");
const k = @import("kernels.zig");
const eng = @import("engine.zig");

const Engine = eng.Engine;
const Sampling = common.Sampling;
const Probe = common.Probe;
const H = cfg.hidden;
const W = cfg.frame_width;

/// The scratch one chain of dispatches uses (the backbone's, or the tower's own).
pub const Lane = struct { xn: u64, tmp: u64, proj: u64, y: u64, q: u64, pos: u64 };

const lay_bb: k.QkLayout = .{ .q_off = 0, .q_head_stride = 2 * cfg.head_dim, .k_off = 2 * cfg.heads * cfg.head_dim, .v_off = 2 * cfg.heads * cfg.head_dim + cfg.kv_heads * cfg.head_dim, .gate_off = cfg.head_dim, .gate_head_stride = 2 * cfg.head_dim };
const lay_tw: k.QkLayout = .{ .q_off = 0, .q_head_stride = cfg.head_dim, .k_off = 2 * cfg.heads * cfg.head_dim, .v_off = 2 * cfg.heads * cfg.head_dim + cfg.kv_heads * cfg.head_dim, .gate_off = cfg.heads * cfg.head_dim, .gate_head_stride = cfg.head_dim };

/// Gated attention of t rows from the lane's projection rows into the lane's y; K / V go to cache rows pos...
fn attention(e: *Engine, ln: Lane, t: usize, lay: k.QkLayout, qn: u64, kn: u64, kc: u64, vc: u64, interleaved: bool) void {
    const ps = cfg.attn_proj;
    k.qkRope(e.g, ln.proj, ps, lay, qn, kn, ln.pos, ln.q, kc, vc, t, interleaved, cfg.rope_theta, cfg.eps);
    k.attn(e.g, ln.q, kc, vc, ln.pos, ln.proj, ps, lay, ln.y, t);
}

pub fn laneBB(e: *Engine, pos: u64) Lane {
    return .{ .xn = e.xn, .tmp = e.tmp, .proj = e.proj, .y = e.y, .q = e.q, .pos = pos };
}

pub fn laneTW(e: *Engine) Lane {
    return .{ .xn = e.t_xn, .tmp = e.t_tmp, .proj = e.t_proj, .y = e.t_y, .q = e.t_q, .pos = e.pos };
}

/// One backbone layer over t rows, e.x (the residual stream) in and out.
pub fn backboneLayer(e: *Engine, ln: Lane, li: usize, t: usize) void {
    const g = e.g;
    const L = &e.w.layers[li];
    k.rmsnorm(g, e.x, H, L.in_norm, e.xn, H, t, H, cfg.eps, true);
    if (L.full) {
        k.linear(g, L.attn.qkv, e.xn, H, H, cfg.attn_proj, e.proj, cfg.attn_proj, t, false);
        attention(e, ln, t, lay_bb, L.attn.q_norm, L.attn.k_norm, e.kv_bb[li][0], e.kv_bb[li][1], false);
        k.linear(g, L.attn.o, e.y, 4096, 4096, H, e.tmp, H, t, false);
    } else {
        const ps = cfg.gdn_proj;
        k.linear(g, L.gdn.proj, e.xn, H, H, ps, e.proj, ps, t, false);
        k.convSilu(g, e.proj, ps, t, L.gdn.conv, e.conv[li], e.conv_out, cfg.conv_channels);
        k.gdn(g, e.conv_out, e.proj, ps, t, L.gdn.a_log, L.gdn.dt_bias, e.gstate[li], e.o32);
        k.gatedNorm(g, e.o32, e.proj, ps, L.gdn.norm, e.y, t, cfg.eps);
        k.linear(g, L.gdn.out, e.y, 4096, 4096, H, e.tmp, H, t, false);
    }
    k.addRmsnorm(g, e.x, e.tmp, H, L.post_norm, e.xn, t, H, cfg.eps);
    moe(e, L, t);
}

/// The MoE block: router, gated shared expert and the routed FP8 experts, added to the residual stream.
fn moe(e: *Engine, L: *const wts.Layer, t: usize) void {
    const g = e.g;
    const mp = cfg.moe_proj;
    k.linear(g, L.moe_in, e.xn, H, H, mp, e.sp, mp, t, false);
    k.router(g, e.sp, mp, 2 * cfg.shared_width, e.picks, e.rw, t);
    k.swiglu(g, e.sp, mp, 0, cfg.shared_width, cfg.shared_width, e.sact, t);
    k.linear(g, L.shared_down, e.sact, cfg.shared_width, cfg.shared_width, H, e.shout, H, t, false);
    const pairs = t * cfg.top_k;
    const prompt = t > 1;
    if (prompt) k.plan(g, e.picks, pairs, e.counts, e.members, e.o.max_rows * cfg.top_k);
    const r: k.Route = .{ .table = L.experts, .picks = e.picks, .counts = e.counts, .members = e.members, .cap = e.o.max_rows * cfg.top_k, .pairs = pairs, .prompt = prompt };
    // activations to FP8 per row and 128 inputs before each expert projection, as the reference serves them
    k.actQuant(g, e.xn, H, e.xq, H, e.xsc, t, H);
    k.experts(g, true, e.xq, e.xsc, r, H, cfg.expert_width, e.gu);
    k.actQuant(g, e.gu, cfg.expert_width, e.gq, cfg.expert_width, e.gsc, pairs, cfg.expert_width);
    k.experts(g, false, e.gq, e.gsc, r, cfg.expert_width, H, e.down);
    k.moeCombine(g, e.x, e.down, e.rw, e.shout, e.sp, mp, 2 * cfg.shared_width + cfg.experts, H, t);
}

/// Tower layer ti over t rows: hidden hin -> hout; label frames lf (rows lfs ints apart), input frames inf.
pub fn towerLayer(e: *Engine, ln: Lane, ti: usize, t: usize, hin: u64, hout: u64, lf: u64, lfs: usize, inf: u64) void {
    const g = e.g;
    const T = &e.w.tower[ti];
    const code_col: usize = if (ti < cfg.shared_layers) 1 else 1 + (ti - cfg.shared_layers);
    k.towerMix(g, .{ .lf = lf, .lfs = lfs, .code_col = code_col, .inf = inf, .with_shared = ti == 0, .table = T.table, .cb_emb = e.w.cb_emb, .h = hin, .enorm = T.enorm, .hnorm = T.hnorm, .out = e.mix }, H, e.c.ids.sem_begin, e.c.ids.sem_end, cfg.codebook_size, cfg.eps, t);
    k.linear(g, T.eh, e.mix, 2 * H, 2 * H, H, e.mixed, H, t, false);
    k.rmsnorm(g, e.mixed, H, T.attn_norm, ln.xn, H, t, H, cfg.eps, true);
    k.linear(g, T.qkv, ln.xn, H, H, cfg.attn_proj, ln.proj, cfg.attn_proj, t, false);
    attention(e, ln, t, lay_tw, T.q_norm, T.k_norm, e.kv_tw[ti][0], e.kv_tw[ti][1], true);
    k.linear(g, T.wo, ln.y, 4096, 4096, H, ln.tmp, H, t, false);
    k.addRmsnorm(g, e.mixed, ln.tmp, H, T.ffn_norm, ln.xn, t, H, cfg.eps); // resid = o + mixed
    k.linear(g, T.w13, ln.xn, H, H, 2 * cfg.tower_ffn, ln.proj, 2 * cfg.tower_ffn, t, false);
    k.swiglu(g, ln.proj, 2 * cfg.tower_ffn, 0, cfg.tower_ffn, cfg.tower_ffn, e.act, t);
    k.linear(g, T.w2, e.act, cfg.tower_ffn, cfg.tower_ffn, H, ln.tmp, H, t, false);
    k.addRmsnorm(g, e.mixed, ln.tmp, H, T.final_norm, hout, t, H, cfg.eps);
}

fn probeName(buf: []u8, comptime f: []const u8, i: usize) []const u8 {
    return std.fmt.bufPrint(buf, f, .{i}) catch unreachable;
}

/// A frame step's dispatches: sample, the tower, record, then the backbone row at the next position.
pub fn enqueueFrame(e: *Engine, smp: Sampling, probe: ?Probe, force: ?Engine.Force) !void {
    const g = e.g;
    const sem_begin = e.c.ids.sem_begin;
    const greedy = smp.greedy();
    const sp = k.sampling(smp);
    if (!greedy and (smp.top_k < 1 or smp.top_k > 64)) return error.TopKUnsupportedOnMetal;
    k.linear(g, e.w.cand_head, e.hlast, H, H, e.w.n_cand, e.cand_logits, e.w.n_cand, 1, true);
    if (probe) |p| try p.f(p.ctx, "tower_decode0.candidate_logits", e.cand_logits, 1, e.w.n_cand, true);
    if (force) |fc| {
        k.argmax(g, e.cand_logits, e.w.n_cand, e.w.cand_ids, fc.picks, sem_begin, cfg.codebook_size);
        k.copyI32(g, fc.frame, e.frame, 2);
    } else if (greedy) {
        k.argmax(g, e.cand_logits, e.w.n_cand, e.w.cand_ids, e.frame, sem_begin, cfg.codebook_size);
    } else {
        k.sampleMain(g, e.cand_logits, e.w.n_cand, e.w.cand_ids, e.window, sp, e.step, e.frame, sem_begin, cfg.codebook_size);
    }
    var hin = e.hlast;
    for (0..cfg.tower_layers) |ti| {
        const hout = e.th[ti % 2];
        towerLayer(e, laneTW(e), ti, 1, hin, hout, e.frame, W, e.cur_input);
        var nb: [64]u8 = undefined;
        if (probe) |p| try p.f(p.ctx, probeName(&nb, "tower_decode0.layer{d:0>2}", ti), hout, 1, H, false);
        hin = hout;
        if (ti < cfg.shared_layers) continue;
        const hi = ti - cfg.shared_layers;
        k.linear(g, e.w.heads[hi], hout, H, H, cfg.codebook_size, e.code_logits, cfg.codebook_size, 1, true);
        if (probe) |p| try p.f(p.ctx, probeName(&nb, "tower_decode0.head{d}_logits", hi), e.code_logits, 1, cfg.codebook_size, true);
        const out = e.frame + (2 + hi) * 4;
        if (force) |fc| {
            k.argmax(g, e.code_logits, cfg.codebook_size, 0, fc.picks + (2 + hi) * 4, 0, 0);
            k.copyI32(g, fc.frame + (2 + hi) * 4, out, 1);
        } else if (greedy) {
            k.argmax(g, e.code_logits, cfg.codebook_size, 0, out, 0, 0);
        } else {
            k.sampleCode(g, e.code_logits, cfg.codebook_size, sp, e.step, @intCast(2 + hi), out);
        }
        // main token, code 0 and code 1 are drawn: the next backbone row's input is known
        if (hi == 0) k.nextInput(g, e.frame, e.bb_input, e.pos, e.pos_bb);
    }
    // the backbone row of the new frame at pos + 1
    k.embedBackbone(g, e.bb_input, e.w.embed, e.w.cb_emb, e.x, 1, H, sem_begin, e.c.ids.sem_end, cfg.codebook_size);
    for (0..cfg.layers) |li| backboneLayer(e, laneBB(e, e.pos_bb), li, 1);
    k.rmsnorm(g, e.x, H, e.w.final_norm, e.hlast, H, 1, H, cfg.eps, true);
    k.finishFrame(g, e.frame, e.frames_out, e.cur_input, e.window, e.pos, e.step, e.count);
}
