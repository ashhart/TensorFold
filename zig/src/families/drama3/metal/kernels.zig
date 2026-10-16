//! Drama 3's Metal launches: each wrapper binds its buffers and argument struct as zig/kernels/metal/drama3 declares.
const std = @import("std");
const gpu = @import("gpu.zig");
const common = @import("../common.zig");

const Gpu = gpu.Gpu;

fn u(x: usize) u32 {
    return @intCast(x);
}

pub const QkLayout = struct { q_off: usize, q_head_stride: usize, k_off: usize, v_off: usize, gate_off: usize, gate_head_stride: usize };

pub const Sampling = extern struct { temperature: f32, top_k: i32, top_p: f32, pad: i32 = 0, seed: u64 };

pub fn sampling(s: common.Sampling) Sampling {
    return .{ .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .seed = s.seed };
}

/// Rows y [t, n] = x [t, k] W^T in bf16 (fp32 sums); `f32out` writes bf16-rounded fp32 (logits).
pub fn linear(g: *Gpu, w: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, t: usize, f32out: bool) void {
    const A = extern struct { K: u32, N: u32, rows: u32, xs: u32, ys: u32, f32out: u32 };
    const a: A = .{ .K = u(kdim), .N = u(n), .rows = u(t), .xs = u(xs), .ys = u(ys), .f32out = @intFromBool(f32out) };
    const r: usize = if (t == 1) 1 else 8;
    g.go(if (t == 1) .d3_rows_r1 else .d3_rows_r8, .{ (n + 15) / 16, (t + r - 1) / r, 1 }, 128, &.{ w, x, y }, a);
}

pub fn rmsnorm(g: *Gpu, x: u64, xs: usize, w: u64, y: u64, ys: usize, rows: usize, dim: usize, eps: f32, zero_centred: bool) void {
    const A = extern struct { xs: u32, ys: u32, D: u32, zc: u32, eps: f32 };
    g.go(.d3_rmsnorm, .{ rows, 1, 1 }, 256, &.{ x, w, y }, A{ .xs = u(xs), .ys = u(ys), .D = u(dim), .zc = @intFromBool(zero_centred), .eps = eps });
}

pub fn addRmsnorm(g: *Gpu, r: u64, add: u64, add_stride: usize, w: u64, y: u64, rows: usize, dim: usize, eps: f32) void {
    const A = extern struct { as: u32, D: u32, eps: f32 };
    g.go(.d3_add_rmsnorm, .{ rows, 1, 1 }, 256, &.{ r, add, w, y }, A{ .as = u(add_stride), .D = u(dim), .eps = eps });
}

pub fn embedBackbone(g: *Gpu, frames: u64, embed: u64, cb_emb: u64, out: u64, rows: usize, dim: usize, sem_begin: i32, sem_end: i32, cb: i32) void {
    const A = extern struct { D: u32, sb: i32, se: i32, cb: i32 };
    g.go(.d3_embed_backbone, .{ rows, 1, 1 }, 256, &.{ frames, embed, cb_emb, out }, A{ .D = u(dim), .sb = sem_begin, .se = sem_end, .cb = cb });
}

pub fn swiglu(g: *Gpu, x: u64, stride: usize, g_off: usize, u_off: usize, n: usize, y: u64, t: usize) void {
    const A = extern struct { stride: u32, g: u32, u: u32, N: u32 };
    g.go(.d3_swiglu, .{ (n + 255) / 256, t, 1 }, 256, &.{ x, y }, A{ .stride = u(stride), .g = u(g_off), .u = u(u_off), .N = u(n) });
}

pub fn moeCombine(g: *Gpu, r: u64, down: u64, w: u64, shared: u64, sp: u64, ss: usize, sg_off: usize, dim: usize, t: usize) void {
    const A = extern struct { ss: u32, sg: u32, D: u32 };
    g.go(.d3_moe_combine, .{ t, 1, 1 }, 256, &.{ r, down, w, shared, sp }, A{ .ss = u(ss), .sg = u(sg_off), .D = u(dim) });
}

pub const Mix = struct { lf: u64, lfs: usize, code_col: usize, inf: u64, with_shared: bool, table: u64, cb_emb: u64, h: u64, enorm: u64, hnorm: u64, out: u64 };

pub fn towerMix(g: *Gpu, m: Mix, dim: usize, sem_begin: i32, sem_end: i32, cb: i32, eps: f32, t: usize) void {
    const A = extern struct { lfs: u32, col: u32, ws: u32, D: u32, sb: i32, se: i32, cb: i32, eps: f32 };
    const a: A = .{ .lfs = u(m.lfs), .col = u(m.code_col), .ws = @intFromBool(m.with_shared), .D = u(dim), .sb = sem_begin, .se = sem_end, .cb = cb, .eps = eps };
    g.go(.d3_tower_mix, .{ t, 1, 1 }, 256, &.{ m.lf, m.inf, m.table, m.cb_emb, m.h, m.enorm, m.hnorm, m.out }, a);
}

pub fn convSilu(g: *Gpu, proj: u64, ps: usize, t: usize, w: u64, state: u64, out: u64, channels: usize) void {
    const A = extern struct { ps: u32, T: u32, C: u32 };
    g.go(.d3_conv_silu, .{ (channels + 127) / 128, 1, 1 }, 128, &.{ proj, w, state, out }, A{ .ps = u(ps), .T = u(t), .C = u(channels) });
}

pub fn gdn(g: *Gpu, conv: u64, proj: u64, ps: usize, t: usize, a_log: u64, dt_bias: u64, state: u64, o: u64) void {
    const A = extern struct { ps: u32, T: u32 };
    g.go(.d3_gdn, .{ 32, 4, 1 }, 128, &.{ conv, proj, a_log, dt_bias, state, o }, A{ .ps = u(ps), .T = u(t) });
}

pub fn gatedNorm(g: *Gpu, o: u64, proj: u64, ps: usize, w: u64, y: u64, t: usize, eps: f32) void {
    const A = extern struct { ps: u32, eps: f32 };
    g.go(.d3_gated_norm, .{ t, 4, 1 }, 256, &.{ o, proj, w, y }, A{ .ps = u(ps), .eps = eps });
}

pub fn qkRope(g: *Gpu, proj: u64, ps: usize, lay: QkLayout, qn: u64, kn: u64, pos0: u64, qout: u64, kc: u64, vc: u64, t: usize, interleaved: bool, theta: f32, eps: f32) void {
    const A = extern struct { ps: u32, qo: u32, qhs: u32, ko: u32, vo: u32, il: u32, theta: f32, eps: f32 };
    const a: A = .{ .ps = u(ps), .qo = u(lay.q_off), .qhs = u(lay.q_head_stride), .ko = u(lay.k_off), .vo = u(lay.v_off), .il = @intFromBool(interleaved), .theta = theta, .eps = eps };
    g.go(.d3_qk_rope, .{ t, 18, 1 }, 256, &.{ proj, qn, kn, pos0, qout, kc, vc }, a);
}

pub fn attn(g: *Gpu, q: u64, kc: u64, vc: u64, pos0: u64, proj: u64, ps: usize, lay: QkLayout, out: u64, t: usize) void {
    const A = extern struct { ps: u32, go: u32, ghs: u32 };
    g.go(.d3_attn, .{ t, 16, 1 }, 256, &.{ q, kc, vc, pos0, proj, out }, A{ .ps = u(ps), .go = u(lay.gate_off), .ghs = u(lay.gate_head_stride) });
}

pub fn router(g: *Gpu, sp: u64, ss: usize, r_off: usize, picks: u64, weights: u64, t: usize) void {
    const A = extern struct { ss: u32, off: u32 };
    g.go(.d3_router, .{ t, 1, 1 }, 256, &.{ sp, picks, weights }, A{ .ss = u(ss), .off = u(r_off) });
}

/// x rows to FP8 values (kept in bf16) and their scales [rows, kdim / 128].
pub fn actQuant(g: *Gpu, x: u64, xs: usize, q: u64, qs: usize, sc: u64, rows: usize, kdim: usize) void {
    const A = extern struct { xs: u32, qs: u32, groups: u32 };
    g.go(.d3_act_quant, .{ rows, kdim / 128, 1 }, 128, &.{ x, q, sc }, A{ .xs = u(xs), .qs = u(qs), .groups = u(kdim / 128) });
}

pub fn plan(g: *Gpu, picks: u64, pairs: usize, counts: u64, members: u64, cap: usize) void {
    const A = extern struct { pairs: u32, cap: u32 };
    g.go(.d3_plan, .{ 1, 1, 1 }, 1024, &.{ picks, counts, members }, A{ .pairs = u(pairs), .cap = u(cap) });
}

pub const Route = struct { table: u64, picks: u64, counts: u64, members: u64, cap: usize, pairs: usize, prompt: bool };

/// Routed experts: `up` maps token rows (x) to SwiGLU pair rows, else pair rows to fp32 down rows.
pub fn experts(g: *Gpu, up: bool, x: u64, xsc: u64, r: Route, kdim: usize, n: usize, out: u64) void {
    const A = extern struct { K: u32, N: u32, prompt: u32, cap: u32, slots: u32 };
    const a: A = .{ .K = u(kdim), .N = u(n), .prompt = @intFromBool(r.prompt), .cap = u(r.cap), .slots = 8 };
    const gy: usize = if (r.prompt) 256 else r.pairs;
    g.go(if (up) .d3_expert_up else .d3_expert_down, .{ n / @as(usize, if (up) 8 else 16), gy, 1 }, 128, &.{ x, xsc, r.table, r.picks, r.counts, r.members, out }, a);
}

/// Greedy: cand_ids non-zero picks a main token (frame[0..2]), zero a code (frame[0]).
pub fn argmax(g: *Gpu, logits: u64, count: usize, cand_ids: u64, frame: u64, sem_begin: i32, cb: i32) void {
    const A = extern struct { count: u32, ids: u32, sb: i32, cb: i32 };
    const ids = if (cand_ids != 0) cand_ids else logits;
    g.go(.d3_argmax, .{ 1, 1, 1 }, 1024, &.{ logits, ids, frame }, A{ .count = u(count), .ids = @intFromBool(cand_ids != 0), .sb = sem_begin, .cb = cb });
}

const SampleArgs = extern struct { sp: Sampling, count: u32, stream: i32, sb: i32, cb: i32 };

pub fn sampleMain(g: *Gpu, logits: u64, count: usize, cand_ids: u64, window: u64, sp: Sampling, step: u64, frame: u64, sem_begin: i32, cb: i32) void {
    g.go(.d3_sample_main, .{ 1, 1, 1 }, 256, &.{ logits, cand_ids, window, step, frame }, SampleArgs{ .sp = sp, .count = u(count), .stream = 0, .sb = sem_begin, .cb = cb });
}

pub fn sampleCode(g: *Gpu, logits: u64, count: usize, sp: Sampling, step: u64, stream: i32, out: u64) void {
    g.go(.d3_sample_code, .{ 1, 1, 1 }, 256, &.{ logits, step, out }, SampleArgs{ .sp = sp, .count = u(count), .stream = stream, .sb = 0, .cb = 0 });
}

pub fn finishFrame(g: *Gpu, frame: u64, frames_out: u64, next_input: u64, window: u64, pos: u64, step: u64, count: u64) void {
    g.go(.d3_finish_frame, .{ 1, 1, 1 }, 32, &.{ frame, frames_out, next_input, window, pos, step, count }, {});
}

pub fn nextInput(g: *Gpu, frame: u64, bb_input: u64, pos: u64, pos_bb: u64) void {
    g.go(.d3_next_input, .{ 1, 1, 1 }, 32, &.{ frame, bb_input, pos, pos_bb }, {});
}

pub fn copyI32(g: *Gpu, src: u64, dst: u64, n: usize) void {
    g.go(.d3_copy_i32, .{ 1, 1, 1 }, 64, &.{ src, dst }, @as(u32, u(n)));
}

test "argument structs match the kernels' layouts" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Sampling));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(SampleArgs));
}
