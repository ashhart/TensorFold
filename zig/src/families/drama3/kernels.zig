//! Drama 3's launches: drama3.cu by symbol, the FP8 experts (drama3_fp8.cu) with grouped.zig's plan, cuBLAS GEMMs.
const std = @import("std");
const cuda = @import("cuda");
const blas = @import("blas.zig");

pub const Sampling = extern struct {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    pad: i32 = 0,
    seed: u64,
};

const names = .{
    "d3_gemv",        "d3_rmsnorm",    "d3_add_rmsnorm", "d3_embed_backbone", "d3_conv_silu",
    "d3_gdn",         "d3_gated_norm", "d3_qk_rope",     "d3_attn_decode",    "d3_q_group",
    "d3_causal_softmax", "d3_attn_gate", "d3_router",   "d3_swiglu",         "d3_moe_combine",
    "d3_tower_mix",   "d3_sample_main", "d3_sample_code", "d3_argmax_main",   "d3_finish_frame",
    "d3_copy_i32",    "d3_act_quant",  "d3_bf16_f32",  "d3_next_input", "d3_attn_partial", "d3_attn_merge", "d3_sample_main_fast", "d3_sample_code_fast",
};

pub const F = enum(u8) {
    gemv,
    rmsnorm,
    add_rmsnorm,
    embed_backbone,
    conv_silu,
    gdn,
    gated_norm,
    qk_rope,
    attn_decode,
    q_group,
    causal_softmax,
    attn_gate,
    router,
    swiglu,
    moe_combine,
    tower_mix,
    sample_main,
    sample_code,
    argmax_main,
    finish_frame,
    copy_i32,
    act_quant,
    bf16_f32,
    next_input,
    attn_partial,
    attn_merge,
    sample_main_fast,
    sample_code_fast,
};

/// drama3_fp8.cu's instances: [decode, prompt][gate-up with the SwiGLU epilogue, down to fp32].
const fp8_symbols = [2][2][:0]const u8{
    .{ "_ZN9tf_d3_fp817fp8_expert_kernelILi2ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_PvifiS8_", "_ZN9tf_d3_fp817fp8_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_PvifiS8_" },
    .{ "_ZN9tf_d3_fp824fp8_expert_prompt_kernelILi2ELi4ELi2ELi2EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_PvifiS8_", "_ZN9tf_d3_fp824fp8_expert_prompt_kernelILi1ELi0ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_PvifiS8_" },
};

const n_fns = @typeInfo(F).@"enum".field_names.len;

pub const Kernels = struct {
    d: *const cuda.Driver,
    mod: cuda.Module,
    fns: [n_fns]cuda.Function,
    experts_mod: cuda.Module,
    plan: cuda.grouped.Router,
    fp8a_mod: cuda.Module,
    fp8a: [2][2]cuda.Function,
    fp8a_resident: [2][2]usize,
    blas: blas.Blas,
    ws: cuda.DeviceBuffer,
    sms: usize,
    pdl: bool = false, // GEMVs with programmatic dependent launch

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        k.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        k.mod = try cuda.Module.load(d, cuda.kernels.drama3);
        inline for (names, 0..) |name, j| k.fns[j] = try k.mod.function(name);
        try k.fns[@intFromEnum(F.sample_main)].allowDynamicShared(8192 * 8);
        try k.fns[@intFromEnum(F.gemv)].allowDynamicShared(6144 * 2);
        k.experts_mod = try cuda.Module.load(d, cuda.kernels.experts);
        k.plan = try cuda.grouped.Router.resolve(k.experts_mod);
        k.fp8a_mod = try cuda.Module.load(d, cuda.kernels.drama3_fp8);
        for (fp8_symbols, 0..) |set, pi| for (set, 0..) |name, ei| {
            k.fp8a[pi][ei] = try k.fp8a_mod.function(name);
            k.fp8a_resident[pi][ei] = @max(1, try k.fp8a[pi][ei].occupancy(128, 0)) * k.sms;
        };
        const ws_bytes: usize = 32 << 20;
        k.ws = try cuda.DeviceBuffer.alloc(d, ws_bytes);
        k.blas = try blas.Blas.open(k.ws.ptr, ws_bytes);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.blas.close();
        k.ws.free();
        k.fp8a_mod.unload();
        k.experts_mod.unload();
        k.mod.unload();
    }

    fn f(k: *const Kernels, which: F) cuda.Function {
        return k.fns[@intFromEnum(which)];
    }

    fn go(k: *const Kernels, which: F, s: cuda.Stream, gx: usize, gy: usize, block: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(k.f(which), .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) }, .shared = @intCast(shared) }, s, a);
    }

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    // dense linear y [T, N] = x [T, K] W^T in bf16: decode rows through the GEMV, prompt rows through cuBLAS

    /// One row: y (bf16, may be 0) and / or yf (fp32, may be 0).
    pub fn gemv(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, kdim: usize, n: usize, y: u64, yf: u64) !void {
        var a: cuda.Args = .{};
        a.add(w);
        a.add(x);
        a.add(i(kdim));
        a.add(i(n));
        a.add(y);
        a.add(yf);
        try cuda.launch.launch(k.f(.gemv), .{ .grid = .{ .x = @intCast((n + 7) / 8) }, .block = .{ .x = 256 }, .shared = @intCast(kdim * 2), .pdl = k.pdl }, s, &a);
    }

    /// Rows: x rows xs elements apart, y rows ys apart (bf16 out, fp32 accumulate).
    pub fn linear(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, t: usize) !void {
        if (t == 1 and xs >= kdim) return k.gemv(s, w, x, kdim, n, y, 0);
        try k.blas.gemm(s, .t, .n, n, t, kdim, 1.0, w, kdim, 0, x, xs, 0, 0.0, y, ys, 0, 1, .bf16, .bf16);
    }

    /// Column-major C = alpha op(A) op(B), batched, fp32 compute (the prompt attention's score and value GEMMs).
    pub fn gemm(k: *const Kernels, s: cuda.Stream, ta: blas.Op, tb: blas.Op, m: usize, n: usize, kk: usize, alpha: f32, a_: u64, lda: usize, sa: usize, b_: u64, ldb: usize, sb: usize, c: u64, ldc: usize, sc: usize, batch: usize, in_t: blas.DataType, out_t: blas.DataType) !void {
        return k.blas.gemm(s, ta, tb, m, n, kk, alpha, a_, lda, sa, b_, ldb, sb, 0.0, c, ldc, sc, batch, in_t, out_t);
    }

    /// fp32 Y [t, o] (rows ldy apart) = X [t, cin] (rows ldx apart) W [o, cin]^T + beta Y, no TF32: the codec's GEMMs.
    pub fn sgemm(k: *const Kernels, s: cuda.Stream, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32) !void {
        return k.blas.gemm(s, .t, .n, o, t, cin, 1.0, w, cin, 0, x, ldx, 0, beta, y, ldy, 0, 1, .f32, .f32);
    }

    // norms and elementwise

    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, rows: usize, dim: usize, eps: f32, zero_centred: bool) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(xs));
        a.add(w);
        a.add(y);
        a.add(i(ys));
        a.add(i(dim));
        a.add(eps);
        a.add(@as(i32, @intFromBool(zero_centred)));
        try k.go(.rmsnorm, s, rows, 1, 256, 0, &a);
    }

    pub fn addRmsnorm(k: *const Kernels, s: cuda.Stream, r: u64, add: u64, add_stride: usize, w: u64, y: u64, rows: usize, dim: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(r);
        a.add(add);
        a.add(i(add_stride));
        a.add(w);
        a.add(y);
        a.add(i(dim));
        a.add(eps);
        try k.go(.add_rmsnorm, s, rows, 1, 256, 0, &a);
    }

    pub fn embedBackbone(k: *const Kernels, s: cuda.Stream, frames: u64, embed: u64, cb_emb: u64, out: u64, rows: usize, dim: usize, sem_begin: i32, sem_end: i32, cb: i32) !void {
        var a: cuda.Args = .{};
        a.add(frames);
        a.add(embed);
        a.add(cb_emb);
        a.add(out);
        a.add(i(dim));
        a.add(sem_begin);
        a.add(sem_end);
        a.add(cb);
        try k.go(.embed_backbone, s, rows, 1, 256, 0, &a);
    }

    pub fn convSilu(k: *const Kernels, s: cuda.Stream, proj: u64, ps: usize, t: usize, w: u64, state: u64, out: u64, channels: usize) !void {
        var a: cuda.Args = .{};
        a.add(proj);
        a.add(i(ps));
        a.add(i(t));
        a.add(w);
        a.add(state);
        a.add(out);
        a.add(i(channels));
        try k.go(.conv_silu, s, (channels + 127) / 128, 1, 128, 0, &a);
    }

    pub fn gdn(k: *const Kernels, s: cuda.Stream, conv: u64, proj: u64, ps: usize, t: usize, a_log: u64, dt_bias: u64, state: u64, o: u64) !void {
        var a: cuda.Args = .{};
        a.add(conv);
        a.add(proj);
        a.add(i(ps));
        a.add(i(t));
        a.add(a_log);
        a.add(dt_bias);
        a.add(state);
        a.add(o);
        try k.go(.gdn, s, 32, 4, 32, 0, &a);
    }

    pub fn gatedNorm(k: *const Kernels, s: cuda.Stream, o: u64, proj: u64, ps: usize, w: u64, y: u64, t: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(o);
        a.add(proj);
        a.add(i(ps));
        a.add(w);
        a.add(y);
        a.add(eps);
        try k.go(.gated_norm, s, t, 4, 256, 0, &a);
    }

    pub const QkLayout = struct { q_off: usize, q_head_stride: usize, k_off: usize, v_off: usize, gate_off: usize, gate_head_stride: usize };

    pub fn qkRope(k: *const Kernels, s: cuda.Stream, proj: u64, ps: usize, lay: QkLayout, qn: u64, kn: u64, pos0: u64, qout: u64, kc: u64, vc: u64, t: usize, interleaved: bool, theta: f32, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(proj);
        a.add(i(ps));
        a.add(i(lay.q_off));
        a.add(i(lay.q_head_stride));
        a.add(i(lay.k_off));
        a.add(i(lay.v_off));
        a.add(qn);
        a.add(kn);
        a.add(pos0);
        a.add(qout);
        a.add(kc);
        a.add(vc);
        a.add(@as(i32, @intFromBool(interleaved)));
        a.add(theta);
        a.add(eps);
        try k.go(.qk_rope, s, t, 18, 256, 0, &a);
    }

    pub fn attnDecode(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos0: u64, proj: u64, ps: usize, lay: QkLayout, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kc);
        a.add(vc);
        a.add(pos0);
        a.add(proj);
        a.add(i(ps));
        a.add(i(lay.gate_off));
        a.add(i(lay.gate_head_stride));
        a.add(out);
        try k.go(.attn_decode, s, t, 16, 256, 0, &a);
    }

    pub const attn_splits = 16;

    /// Split-key decode attention of t rows: partials in pml [t, 16, S, 2] and pacc [t, 16, S, 256], then the merge.
    pub fn attnSplit(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos0: u64, pml: u64, pacc: u64, proj: u64, ps: usize, lay: QkLayout, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, pos0, pml, pacc }) |v| a.add(v);
        try cuda.launch.launch(k.f(.attn_partial), .{ .grid = .{ .x = @intCast(t), .y = 16, .z = attn_splits }, .block = .{ .x = 256 } }, s, &a);
        var b: cuda.Args = .{};
        b.add(pml);
        b.add(pacc);
        b.add(@as(i32, attn_splits));
        b.add(proj);
        b.add(i(ps));
        b.add(i(lay.gate_off));
        b.add(i(lay.gate_head_stride));
        b.add(out);
        try k.go(.attn_merge, s, t, 16, 256, 0, &b);
    }

    pub fn qGroup(k: *const Kernels, s: cuda.Stream, q: u64, qg: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(qg);
        a.add(i(t));
        try k.go(.q_group, s, t, 16, 256, 0, &a);
    }

    pub fn causalSoftmax(k: *const Kernels, s: cuda.Stream, sc: u64, t: usize, ld: usize, pos0: u64) !void {
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(i(t));
        a.add(i(ld));
        a.add(pos0);
        try k.go(.causal_softmax, s, 2 * 8 * t, 1, 256, 0, &a);
    }

    pub fn attnGate(k: *const Kernels, s: cuda.Stream, og: u64, proj: u64, ps: usize, lay: QkLayout, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(og);
        a.add(proj);
        a.add(i(ps));
        a.add(i(lay.gate_off));
        a.add(i(lay.gate_head_stride));
        a.add(out);
        a.add(i(t));
        try k.go(.attn_gate, s, t, 16, 256, 0, &a);
    }

    pub fn router(k: *const Kernels, s: cuda.Stream, sp: u64, ss: usize, r_off: usize, picks: u64, weights: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(sp);
        a.add(i(ss));
        a.add(i(r_off));
        a.add(picks);
        a.add(weights);
        try k.go(.router, s, t, 1, 256, 0, &a);
    }

    pub fn swiglu(k: *const Kernels, s: cuda.Stream, x: u64, stride: usize, g_off: usize, u_off: usize, n: usize, y: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(stride));
        a.add(i(g_off));
        a.add(i(u_off));
        a.add(i(n));
        a.add(y);
        try k.go(.swiglu, s, (n + 255) / 256, t, 256, 0, &a);
    }

    pub fn moeCombine(k: *const Kernels, s: cuda.Stream, r: u64, down: u64, w: u64, shared: u64, sp: u64, ss: usize, sg_off: usize, dim: usize, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(r);
        a.add(down);
        a.add(w);
        a.add(shared);
        a.add(sp);
        a.add(i(ss));
        a.add(i(sg_off));
        a.add(i(dim));
        try k.go(.moe_combine, s, t, 1, 256, 0, &a);
    }

    pub fn towerMix(k: *const Kernels, s: cuda.Stream, lf: u64, lfs: usize, code_col: usize, inf: u64, with_shared: bool, table: u64, cb_emb: u64, h: u64, enorm: u64, hnorm: u64, out: u64, dim: usize, sem_begin: i32, sem_end: i32, cb: i32, eps: f32, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(lf);
        a.add(i(lfs));
        a.add(i(code_col));
        a.add(inf);
        a.add(@as(i32, @intFromBool(with_shared)));
        a.add(table);
        a.add(cb_emb);
        a.add(h);
        a.add(enorm);
        a.add(hnorm);
        a.add(out);
        a.add(i(dim));
        a.add(sem_begin);
        a.add(sem_end);
        a.add(cb);
        a.add(eps);
        try k.go(.tower_mix, s, t, 1, 256, 0, &a);
    }

    pub fn sampleMain(k: *const Kernels, s: cuda.Stream, logits: u64, count: usize, cand_ids: u64, window: u64, sp: Sampling, step: u64, frame: u64, sem_begin: i32, cb: i32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(count));
        a.add(cand_ids);
        a.add(window);
        a.add(sp);
        a.add(step);
        a.add(frame);
        a.add(sem_begin);
        a.add(cb);
        if (sp.top_k >= 1 and sp.top_k <= 64) return k.go(.sample_main_fast, s, 1, 1, 1024, 0, &a);
        try k.go(.sample_main, s, 1, 1, 1024, 8192 * 8, &a);
    }

    pub fn sampleCode(k: *const Kernels, s: cuda.Stream, logits: u64, count: usize, sp: Sampling, step: u64, stream_id: i32, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(count));
        a.add(sp);
        a.add(step);
        a.add(stream_id);
        a.add(out);
        if (sp.top_k >= 1 and sp.top_k <= 64) return k.go(.sample_code_fast, s, 1, 1, 1024, 0, &a);
        try k.go(.sample_code, s, 1, 1, 1024, 4096 * 8, &a);
    }

    /// Greedy: cand_ids non-zero picks a main token (frame[0..2]), zero a code (frame[0]).
    pub fn argmax(k: *const Kernels, s: cuda.Stream, logits: u64, count: usize, cand_ids: u64, frame: u64, sem_begin: i32, cb: i32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(count));
        a.add(cand_ids);
        a.add(frame);
        a.add(sem_begin);
        a.add(cb);
        try k.go(.argmax_main, s, 1, 1, 1024, 0, &a);
    }

    pub fn finishFrame(k: *const Kernels, s: cuda.Stream, frame: u64, frames_out: u64, next_input: u64, window: u64, pos: u64, step: u64, count: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ frame, frames_out, next_input, window, pos, step, count }) |v| a.add(v);
        try k.go(.finish_frame, s, 1, 1, 32, 0, &a);
    }

    /// SGLang's per_token_group_quant_fp8: x rows to q (bf16 holding e4m3) and scales [rows, kdim / 128].
    pub fn actQuant(k: *const Kernels, s: cuda.Stream, x: u64, xs: usize, q: u64, qs: usize, sc: u64, rows: usize, kdim: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(xs));
        a.add(q);
        a.add(i(qs));
        a.add(sc);
        try k.go(.act_quant, s, rows, kdim / 128, 128, 0, &a);
    }

    /// Routed FP8 experts on FP8 rows: `up` maps token rows to SwiGLU pair rows, else pair rows to fp32 down rows.
    pub fn experts(k: *const Kernels, s: cuda.Stream, up: bool, prompt: bool, x: u64, stride: usize, slots: usize, xsc: u64, l: cuda.experts.Layer, p: cuda.grouped.Plan, out: u64, items: usize) !void {
        const kg = (if (up) l.dims else l.width) / 32;
        const nb = (if (up) l.width else l.dims) / 32;
        const n = if (up) l.width else l.dims;
        const pi: usize = @intFromBool(prompt);
        const ei: usize = if (up) 0 else 1;
        const grid: usize = if (prompt) @min(items * (nb / @as(usize, if (up) 2 else 4)), k.fp8a_resident[pi][ei]) else @min((items * nb + 3) / 4, k.fp8a_resident[pi][ei]);
        if (grid < 1) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(stride));
        a.add(i(if (up) slots else 0));
        a.add(if (up) l.up else l.down);
        a.add(if (up) l.up_scale else l.down_scale);
        a.add(i(kg));
        a.add(i(nb));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(i(n));
        a.add(@as(f32, 0.0));
        a.add(@as(i32, -1));
        a.add(xsc);
        try cuda.launch.launch(k.fp8a[pi][ei], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn bf16F32(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i(n));
        try k.go(.bf16_f32, s, @min((n + 255) / 256, 4096), 1, 256, 0, &a);
    }

    pub fn nextInput(k: *const Kernels, s: cuda.Stream, frame: u64, bb_input: u64, pos: u64, pos_bb: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ frame, bb_input, pos, pos_bb }) |v| a.add(v);
        try k.go(.next_input, s, 1, 1, 32, 0, &a);
    }

    pub fn copyI32(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(i(n));
        try k.go(.copy_i32, s, 1, 1, 64, 0, &a);
    }
};
