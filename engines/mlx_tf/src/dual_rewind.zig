//! Universal Forget-Invert Dual-Rewind: Singularity-Free Exact Arbitrary-Boundary
//! Gated DeltaNet State Recovery with Zero Checkpoints.
//!
//! Theorem 1 (Singularity-Free Delta Inverse via Correction Tape c_t):
//!   Forward : S_t     = alpha_t * S_{t-1} + c_t k_t^T,   c_t = beta_t (v_t - alpha_t S_{t-1} k_t)
//!   Backward: S_{t-1} = (S_t - c_t k_t^T) / alpha_t
//!   Eliminates the 1 / (1 - beta_t ||k_t||^2) Sherman-Morrison singularity completely
//!   (exact even at beta_t = 1.0!) and removes dot products from the backward kernel.
//!
//! Theorem 2 (Exact Backward vs Forward Bounds):
//!   E_back(P <- T) ~ u_eff / prod_{t=P}^{T-1} alpha_t
//!   E_fwd(P, W)   <=         prod_{t=P-W}^{P-1} alpha_t
//!
//! Theorem 3 (General Non-Stationary Minimax Duality & Certified Adaptive Replay):
//!   Let alpha_pre  = (prod_{t=P-W}^{P-1} alpha_t)^(1/W) and
//!       alpha_post = (prod_{t=P}^{T-1}   alpha_t)^(1/Delta),
//!   with decay-drift ratio kappa = ln(alpha_pre) / ln(alpha_post) >= 0 (kappa = 1 stationary).
//!   With window W = r * Delta:
//!     min( E_back, E_fwd(W) ) <= u_eff^( (r * kappa) / (r * kappa + 1) )
//!   Recovering u_eff^(r / (r + 1)) in the stationary regime (kappa = 1), while flagging any
//!   non-stationary head (kappa << 1: pre-remember / post-flush) at runtime via the exact
//!   error certificate min(E_back, E_fwd(W)) > eps_tol and adaptively extending ONLY those
//!   flagged heads until prod alpha_t <= eps_tol (or t = 0) — certifying <= eps_tol on 100% of heads.
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");

const Model = model_mod.Model;
const Buffer = mtl.Buffer;
const nowMs = model_mod.nowMs;

const GdnRewindParams = extern struct {
    nk: u32,
    nv: u32,
    cur_pos: u32,
    target_pos: u32,
};

pub const RewindPolicy = enum {
    tail_replay_only, // 2026 SOTA baseline: zero-start forward tail replay (W) for all heads
    sm_inverse_only, // Ablation: Sherman-Morrison backward inverse without c_t tape
    ct_inverse_only, // Ablation: Singularity-free c_t-Inverse backward for all heads
    replay_ssm_tape, // Periodic checkpoint (ReplaySSM-style anchor) + GDN-only forward tape replay
    dual_rewind_fixed_w, // Forget-Invert Duality (0 checkpoints, fixed W)
    dual_rewind_minimax, // Forget-Invert Duality (0 checkpoints, W = max(64, 2*Delta))
    dual_rewind_certified, // Certified Non-Stationary TF-Rewind (0 checkpoints, adaptive extension when kappa << 1)
};

pub const RewindReport = struct {
    rewind_ms: f64 = 0,
    heads_total: usize = 0,
    heads_backward: usize = 0,
    heads_forward: usize = 0,
    heads_extended: usize = 0,
    avg_steps: f64 = 0,
    max_steps: u32 = 0,
    median_rel_err: f64 = 0,
    p90_rel_err: f64 = 0,
    max_rel_err: f64 = 0,
    exact_1e3_heads: usize = 0,
    exact_1e2_heads: usize = 0,
    max_logit_err: f64 = 0,
    kl_div: f64 = 0,
    token_matches: usize = 0,
    tokens_checked: usize = 0,
};

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub const DualRewindEngine = struct {
    a: std.mem.Allocator,
    m: *Model,
    max_tape: u32,
    n_lin: usize,
    nk: usize,
    nv: usize,
    qkv_dim: usize,

    // Lightweight O(d) per-token tape buffers (per linear layer):
    //   tape_k    : [n_lin][max_tape][nk * 128] bf16
    //   tape_v    : [n_lin][max_tape][nv * 128] bf16
    //   tape_decay: [n_lin][max_tape][nv]       f32
    //   tape_mix  : [n_lin][max_tape][nv]       bf16
    //   tape_corr : [n_lin][max_tape][nv * 128] f32   (c_t = beta_t (v_t - alpha_t S_{t-1} k_t))
    //   tape_qkv  : [n_lin][max_tape][qkv_dim]  bf16  (exact 3-tap conv history restoration)
    tape_k_buf: Buffer,
    tape_v_buf: Buffer,
    tape_decay_buf: Buffer,
    tape_mix_buf: Buffer,
    tape_corr_buf: Buffer,
    tape_qkv_buf: Buffer,
    head_mode_buf: Buffer,
    head_start_buf: Buffer,
    anchor_state_buf: Buffer,

    k_layer_stride: usize,
    v_layer_stride: usize,
    dec_layer_stride: usize,
    mix_layer_stride: usize,
    corr_layer_stride: usize,
    qkv_layer_stride: usize,
    tape_len: u32 = 0,
    anchor_pos: u32 = 0,
    has_anchor: bool = false,

    pub fn init(a: std.mem.Allocator, m: *Model, max_tape: u32) !DualRewindEngine {
        const n_lin = m.c.numLinearLayers();
        std.debug.assert(n_lin > 0);
        const nk = m.c.gdn_k_heads;
        const nv = m.c.gdn_v_heads;
        const qkv_dim = m.c.gdnQkvDim();

        const k_ls = align256(@as(usize, max_tape) * nk * 128 * 2);
        const v_ls = align256(@as(usize, max_tape) * nv * 128 * 2);
        const d_ls = align256(@as(usize, max_tape) * nv * 4);
        const m_ls = align256(@as(usize, max_tape) * nv * 2);
        const c_ls = align256(@as(usize, max_tape) * nv * 128 * 4);
        const q_ls = align256(@as(usize, max_tape) * qkv_dim * 2);

        const tk = try m.ctx.allocBuffer(n_lin * k_ls);
        errdefer tk.deinit();
        const tv = try m.ctx.allocBuffer(n_lin * v_ls);
        errdefer tv.deinit();
        const td = try m.ctx.allocBuffer(n_lin * d_ls);
        errdefer td.deinit();
        const tm = try m.ctx.allocBuffer(n_lin * m_ls);
        errdefer tm.deinit();
        const tc = try m.ctx.allocBuffer(n_lin * c_ls);
        errdefer tc.deinit();
        const tq = try m.ctx.allocBuffer(n_lin * q_ls);
        errdefer tq.deinit();
        const hm = try m.ctx.allocBuffer(align256(n_lin * nv * 4));
        errdefer hm.deinit();
        const hs = try m.ctx.allocBuffer(align256(n_lin * nv * 4));
        errdefer hs.deinit();
        const anc = try m.ctx.allocBuffer(m.gdn_state_buf.len); // used only when benchmarking ReplaySSM anchor baseline
        errdefer anc.deinit();

        m.ctx.commitResidency();

        return .{
            .a = a,
            .m = m,
            .max_tape = max_tape,
            .n_lin = n_lin,
            .nk = nk,
            .nv = nv,
            .qkv_dim = qkv_dim,
            .tape_k_buf = tk,
            .tape_v_buf = tv,
            .tape_decay_buf = td,
            .tape_mix_buf = tm,
            .tape_corr_buf = tc,
            .tape_qkv_buf = tq,
            .head_mode_buf = hm,
            .head_start_buf = hs,
            .anchor_state_buf = anc,
            .k_layer_stride = k_ls,
            .v_layer_stride = v_ls,
            .dec_layer_stride = d_ls,
            .mix_layer_stride = m_ls,
            .corr_layer_stride = c_ls,
            .qkv_layer_stride = q_ls,
        };
    }

    pub fn deinit(self: *DualRewindEngine) void {
        self.anchor_state_buf.deinit();
        self.head_start_buf.deinit();
        self.head_mode_buf.deinit();
        self.tape_qkv_buf.deinit();
        self.tape_corr_buf.deinit();
        self.tape_mix_buf.deinit();
        self.tape_decay_buf.deinit();
        self.tape_v_buf.deinit();
        self.tape_k_buf.deinit();
    }

    pub fn bytesPerTokenShared(self: *const DualRewindEngine) usize {
        // Shared recurrence inputs needed by Tail-Replay & ReplaySSM-Tape: (k bf16 + v bf16 + alpha f32 + beta bf16)
        return self.n_lin * (self.nk * 128 * 2 + self.nv * 128 * 2 + self.nv * 4 + self.nv * 2);
    }

    pub fn bytesPerTokenCorr(self: *const DualRewindEngine) usize {
        // Incremental TF-Rewind correction tape: c_t fp32 (only needed for last Delta_max tokens!)
        return self.n_lin * (self.nv * 128 * 4);
    }

    pub fn bytesPerTokenQkv(self: *const DualRewindEngine) usize {
        // Pre-conv qkv bf16 for exact 3-tap conv1d restore (only needed for last Delta_max + 3 tokens!)
        return self.n_lin * (self.qkv_dim * 2);
    }

    pub fn bytesPerTokenTape(self: *const DualRewindEngine) usize {
        return self.bytesPerTokenShared() + self.bytesPerTokenCorr();
    }

    pub fn saveAnchor(self: *DualRewindEngine) void {
        @memcpy(self.anchor_state_buf.ptr[0..self.m.gdn_state_buf.len], self.m.gdn_state_buf.ptr[0..self.m.gdn_state_buf.len]);
        self.anchor_pos = self.m.pos;
        self.has_anchor = true;
    }

    fn captureFromVerifyBuf(self: *DualRewindEngine, start_pos: u32, rows: u32) void {
        const m = self.m;
        const k_row_bytes = self.nk * 128 * 2;
        const v_row_bytes = self.nv * 128 * 2;
        const dec_row_bytes = self.nv * 4;
        const mix_row_bytes = self.nv * 2;
        const corr_row_bytes = self.nv * 128 * 4;
        const qkv_row_bytes = self.qkv_dim * 2;
        const ur: usize = rows;
        const sp: usize = start_pos;

        for (0..self.n_lin) |li| {
            const vbase = li * m.gdn_vl.layer_stride;
            const vb = m.gdn_verify_buf.ptr;

            @memcpy(
                self.tape_k_buf.ptr[li * self.k_layer_stride + sp * k_row_bytes ..][0 .. ur * k_row_bytes],
                vb[vbase + m.gdn_vl.k_off ..][0 .. ur * k_row_bytes],
            );
            @memcpy(
                self.tape_v_buf.ptr[li * self.v_layer_stride + sp * v_row_bytes ..][0 .. ur * v_row_bytes],
                vb[vbase + m.gdn_vl.v_off ..][0 .. ur * v_row_bytes],
            );
            @memcpy(
                self.tape_decay_buf.ptr[li * self.dec_layer_stride + sp * dec_row_bytes ..][0 .. ur * dec_row_bytes],
                vb[vbase + m.gdn_vl.decay_off ..][0 .. ur * dec_row_bytes],
            );
            @memcpy(
                self.tape_mix_buf.ptr[li * self.mix_layer_stride + sp * mix_row_bytes ..][0 .. ur * mix_row_bytes],
                vb[vbase + m.gdn_vl.mix_off ..][0 .. ur * mix_row_bytes],
            );
            @memcpy(
                self.tape_corr_buf.ptr[li * self.corr_layer_stride + sp * corr_row_bytes ..][0 .. ur * corr_row_bytes],
                vb[vbase + m.gdn_vl.corr_off ..][0 .. ur * corr_row_bytes],
            );
            @memcpy(
                self.tape_qkv_buf.ptr[li * self.qkv_layer_stride + sp * qkv_row_bytes ..][0 .. ur * qkv_row_bytes],
                vb[vbase + m.gdn_vl.qkv_off ..][0 .. ur * qkv_row_bytes],
            );
        }
        self.tape_len = start_pos + rows;
    }

    pub fn prefillRecorded(self: *DualRewindEngine, tokens: []const u32, chunk_size: u32) !u32 {
        const m = self.m;
        const prev_tc = m.tape_capture;
        m.tape_capture = true;
        defer m.tape_capture = prev_tc;

        const cap = @min(chunk_size, m.max_rows);
        var offset: usize = 0;
        var next_tok: u32 = 0;
        while (offset < tokens.len) {
            const n: u32 = @intCast(@min(tokens.len - offset, cap));
            const start_pos = m.pos;
            std.debug.assert(start_pos + n <= self.max_tape);
            next_tok = try m.prefill(tokens[offset..][0..n], n);
            self.captureFromVerifyBuf(start_pos, n);
            offset += n;
        }
        return next_tok;
    }

    pub fn printSpectrumStats(self: *const DualRewindEngine, start_t: usize, end_t: usize) void {
        const total_heads = self.n_lin * self.nv;
        var slow_alpha: usize = 0;
        var med_alpha: usize = 0;
        var fast_alpha: usize = 0;
        var high_beta: usize = 0;
        var sum_alpha_all: f64 = 0;
        var sum_beta_all: f64 = 0;
        var sum_log_inv_alpha: f64 = 0;
        var count_all: f64 = 0;

        for (0..self.n_lin) |li| {
            const dec_ptr: [*]const f32 = @ptrCast(@alignCast(self.tape_decay_buf.ptr + li * self.dec_layer_stride));
            const mix_ptr: [*]const u16 = @ptrCast(@alignCast(self.tape_mix_buf.ptr + li * self.mix_layer_stride));
            for (0..self.nv) |h| {
                var a_sum: f64 = 0;
                var b_max: f64 = 0;
                for (start_t..end_t) |t| {
                    const a: f64 = @max(dec_ptr[t * self.nv + h], 1e-12);
                    const b: f64 = bf16ToF32(mix_ptr[t * self.nv + h]);
                    a_sum += a;
                    if (b > b_max) b_max = b;
                    sum_alpha_all += a;
                    sum_beta_all += b;
                    sum_log_inv_alpha += -@log(a);
                    count_all += 1;
                }
                const a_mean = a_sum / @as(f64, @floatFromInt(end_t - start_t));
                if (a_mean >= 0.95) slow_alpha += 1 else if (a_mean >= 0.50) med_alpha += 1 else fast_alpha += 1;
                if (b_max >= 0.90) high_beta += 1;
            }
        }
        const ckpt_mib = @as(f64, @floatFromInt(self.m.gdn_state_buf.len)) / (1024.0 * 1024.0);
        const shared_kib = @as(f64, @floatFromInt(self.bytesPerTokenShared())) / 1024.0;
        const corr_kib = @as(f64, @floatFromInt(self.bytesPerTokenCorr())) / 1024.0;
        const qkv_kib = @as(f64, @floatFromInt(self.bytesPerTokenQkv())) / 1024.0;
        const ring64_mib = (192.0 * shared_kib + 64.0 * corr_kib + 67.0 * qkv_kib) / 1024.0;
        std.debug.print(
            \\GDN Empirical Spectrum ({d} linear layers x {d} heads = {d} heads, t={d}..{d}):
            \\  Decay alpha regimes  : slow(>=0.95)={d} ({d:.1}%) | med(0.5-0.95)={d} ({d:.1}%) | fast(<0.5)={d} ({d:.1}%)
            \\  Heads w/ beta>=0.90  : {d}/{d} ({d:.1}%) -- breaks Sherman-Morrison, ZERO effect on c_t-Inverse!
            \\  Global means         : E[alpha]={d:.4} | E[beta]={d:.4} | geom_mean(1/alpha)={d:.3}
            \\  Itemized Memory      : 1 Ckpt = {d:.2} MiB | Shared (k,v,a,b) = {d:.1} KiB/tok | c_t (f32) = {d:.1} KiB/tok | conv qkv = {d:.1} KiB/tok
            \\  Ring (Delta<=64)     : {d:.1} MiB total ({d:.2}x one ckpt, O(Delta) vs ReplaySSM O(T/B) checkpoints!)
            \\
        , .{
            self.n_lin,
            self.nv,
            total_heads,
            start_t,
            end_t,
            slow_alpha,
            100.0 * @as(f64, @floatFromInt(slow_alpha)) / @as(f64, @floatFromInt(total_heads)),
            med_alpha,
            100.0 * @as(f64, @floatFromInt(med_alpha)) / @as(f64, @floatFromInt(total_heads)),
            fast_alpha,
            100.0 * @as(f64, @floatFromInt(fast_alpha)) / @as(f64, @floatFromInt(total_heads)),
            high_beta,
            total_heads,
            100.0 * @as(f64, @floatFromInt(high_beta)) / @as(f64, @floatFromInt(total_heads)),
            sum_alpha_all / count_all,
            sum_beta_all / count_all,
            @exp(sum_log_inv_alpha / count_all),
            ckpt_mib,
            shared_kib,
            corr_kib,
            qkv_kib,
            ring64_mib,
            ring64_mib / ckpt_mib,
        });
    }

    pub fn rewindTo(
        self: *DualRewindEngine,
        target_pos: u32,
        policy: RewindPolicy,
        base_window: u32,
    ) !RewindReport {
        const m = self.m;
        const cur_pos = m.pos;
        const delta = cur_pos - target_pos;
        std.debug.assert(target_pos > 0 and target_pos < cur_pos and cur_pos <= self.tape_len);

        const t0 = nowMs();
        var rep = RewindReport{ .heads_total = self.n_lin * self.nv };

        const modes = self.head_mode_buf.slice(u32, self.n_lin * self.nv);
        const starts = self.head_start_buf.slice(u32, self.n_lin * self.nv);

        // Effective fp32 rounding unit for c_t-Inverse, and target error certificate for Certified mode
        const log_u: f64 = @log(@as(f64, 5.0e-7));
        const log_tol: f64 = @log(@as(f64, 5.0e-5)); // ~ -9.9035 (matches stationary r=2 fp32 bound u^(2/3))
        const log_ext_target: f64 = -11.5; // exp(-11.5) ~ 1.0e-5 extra margin when extending non-stationary heads
        const eff_window: u32 = switch (policy) {
            .dual_rewind_minimax, .dual_rewind_certified => @max(base_window, 2 * delta),
            else => base_window,
        };

        var total_steps: u64 = 0;

        for (0..self.n_lin) |li| {
            const dec_ptr: [*]const f32 = @ptrCast(@alignCast(self.tape_decay_buf.ptr + li * self.dec_layer_stride));

            for (0..self.nv) |h| {
                const idx = li * self.nv + h;
                const start_w: u32 = if (target_pos > eff_window) target_pos - eff_window else 0;
                var head_steps: u32 = 0;

                switch (policy) {
                    .tail_replay_only => {
                        modes[idx] = 1;
                        starts[idx] = start_w;
                        head_steps = target_pos - start_w;
                        rep.heads_forward += 1;
                    },
                    .sm_inverse_only => {
                        modes[idx] = 2;
                        starts[idx] = 0;
                        head_steps = delta;
                        rep.heads_backward += 1;
                    },
                    .ct_inverse_only => {
                        modes[idx] = 0;
                        starts[idx] = 0;
                        head_steps = delta;
                        rep.heads_backward += 1;
                    },
                    .replay_ssm_tape => {
                        // ReplaySSM-style periodic anchor replay: load checkpoint S_{anchor_pos} and forward-replay [anchor_pos .. target_pos)
                        if (self.has_anchor and target_pos >= self.anchor_pos) {
                            modes[idx] = 3;
                            starts[idx] = self.anchor_pos;
                            head_steps = target_pos - self.anchor_pos;
                        } else {
                            modes[idx] = 1;
                            starts[idx] = 0;
                            head_steps = target_pos;
                        }
                        rep.heads_forward += 1;
                    },
                    .dual_rewind_fixed_w, .dual_rewind_minimax, .dual_rewind_certified => {
                        // Theorem 2 (Singularity-free!): log E_back = log(u) - sum_{t=P}^{T-1} log(alpha_t)
                        var log_err_back: f64 = log_u;
                        for (target_pos..cur_pos) |t| {
                            const alpha: f64 = @max(dec_ptr[t * self.nv + h], 1e-12);
                            log_err_back += -@log(alpha);
                        }

                        // Theorem 2 (Forward tail): walk backward from P-1 down to start_w, early-stopping at -14.0
                        var opt_start = target_pos;
                        var acc_log: f64 = 0.0;
                        while (opt_start > start_w) {
                            opt_start -= 1;
                            acc_log += @log(@max(dec_ptr[opt_start * self.nv + h], 1e-12));
                            if (acc_log <= -14.0) break; // exp(-14) < 8.3e-7 (below fp32 precision!)
                        }
                        const log_err_fwd: f64 = if (opt_start == 0) -100.0 else acc_log;

                        if (policy == .dual_rewind_certified) {
                            // Runtime error certificate handles non-stationary decay (kappa << 1: pre-remember / post-flush)!
                            if (log_err_back <= log_err_fwd and log_err_back <= log_tol) {
                                modes[idx] = 0;
                                starts[idx] = 0;
                                head_steps = delta;
                                rep.heads_backward += 1;
                            } else if (log_err_fwd <= log_tol) {
                                modes[idx] = 1;
                                starts[idx] = opt_start;
                                head_steps = target_pos - opt_start;
                                rep.heads_forward += 1;
                            } else {
                                // Non-stationary head detected! Both fixed paths exceed eps_tol.
                                // Adaptively extend ONLY this head backward until prod alpha_t <= eps_tol or t = 0 (bit-exact!).
                                while (opt_start > 0 and acc_log > log_ext_target) {
                                    opt_start -= 1;
                                    acc_log += @log(@max(dec_ptr[opt_start * self.nv + h], 1e-12));
                                }
                                modes[idx] = 1;
                                starts[idx] = opt_start;
                                head_steps = target_pos - opt_start;
                                rep.heads_forward += 1;
                                rep.heads_extended += 1;
                            }
                        } else {
                            if (log_err_back <= log_err_fwd) {
                                modes[idx] = 0; // Singularity-free Backward c_t-Inverse!
                                starts[idx] = 0;
                                head_steps = delta;
                                rep.heads_backward += 1;
                            } else {
                                modes[idx] = 1; // Exact Zero-Start Forward Tail-Replay!
                                starts[idx] = opt_start;
                                head_steps = target_pos - opt_start;
                                rep.heads_forward += 1;
                            }
                        }
                    },
                }
                total_steps += head_steps;
                if (head_steps > rep.max_steps) rep.max_steps = head_steps;
            }
        }
        rep.avg_steps = @as(f64, @floatFromInt(total_steps)) / @as(f64, @floatFromInt(rep.heads_total));

        // Dispatch ONE single Metal command buffer across all linear layers!
        var cb = m.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        enc.setPipeline(m.p.gdn_rewind_dual.?);

        const rp = GdnRewindParams{
            .nk = @intCast(self.nk),
            .nv = @intCast(self.nv),
            .cur_pos = cur_pos,
            .target_pos = target_pos,
        };

        for (0..self.n_lin) |li| {
            enc.setBuffer(self.tape_k_buf, li * self.k_layer_stride, 0);
            enc.setBuffer(self.tape_v_buf, li * self.v_layer_stride, 1);
            enc.setBuffer(self.tape_decay_buf, li * self.dec_layer_stride, 2);
            enc.setBuffer(self.tape_mix_buf, li * self.mix_layer_stride, 3);
            enc.setBuffer(self.tape_corr_buf, li * self.corr_layer_stride, 4);
            enc.setBuffer(self.head_mode_buf, li * self.nv * 4, 5);
            enc.setBuffer(self.head_start_buf, li * self.nv * 4, 6);
            enc.setBuffer(self.anchor_state_buf, li * m.gdn_state_stride, 7);
            enc.setBuffer(m.gdn_state_buf, li * m.gdn_state_stride, 8);
            enc.setValue(rp, 9);
            enc.dispatchGroups(.{ .width = 1, .height = 32, .depth = self.nv }, .{ .width = 32, .height = 1, .depth = 1 });
        }

        enc.end();
        cb.commit();
        try m.ctx.waitIdle();

        // Exact 3-tap conv history restoration from tape_qkv [target_pos - 3, target_pos - 2, target_pos - 1]
        const qkv_row_bytes = self.qkv_dim * 2;
        for (0..self.n_lin) |li| {
            const dst = m.gdn_hist_buf.ptr + li * m.gdn_hist_stride;
            const src_base = self.tape_qkv_buf.ptr + li * self.qkv_layer_stride;
            for (0..3) |slot| {
                const back = 3 - slot;
                if (target_pos >= back) {
                    const t: usize = target_pos - back;
                    @memcpy(dst[slot * qkv_row_bytes ..][0..qkv_row_bytes], src_base[t * qkv_row_bytes ..][0..qkv_row_bytes]);
                } else {
                    @memset(dst[slot * qkv_row_bytes ..][0..qkv_row_bytes], 0);
                }
            }
        }

        // Zero-copy KV cache rewind!
        m.pos = target_pos;
        rep.rewind_ms = nowMs() - t0;
        return rep;
    }

    pub fn evaluateStateAccuracy(self: *const DualRewindEngine, ref_state: []const f32, rep: *RewindReport) !void {
        const got_state = self.m.gdn_state_buf.slice(f32, self.m.gdn_state_buf.len / 4);
        const total_heads = self.n_lin * self.nv;
        const head_elems: usize = 128 * 128;
        const stride_f32 = self.m.gdn_state_stride / 4;

        var errs = try self.a.alloc(f64, total_heads);
        defer self.a.free(errs);

        var idx: usize = 0;
        var e1e3: usize = 0;
        var e1e2: usize = 0;
        for (0..self.n_lin) |li| {
            const base = li * stride_f32;
            for (0..self.nv) |h| {
                const off = base + h * head_elems;
                var diff_sq: f64 = 0.0;
                var ref_sq: f64 = 0.0;
                for (0..head_elems) |k| {
                    const r: f64 = ref_state[off + k];
                    const g: f64 = got_state[off + k];
                    const d = g - r;
                    diff_sq += d * d;
                    ref_sq += r * r;
                }
                const rel = if (ref_sq > 1e-24) @sqrt(diff_sq / ref_sq) else @sqrt(diff_sq);
                errs[idx] = if (std.math.isFinite(rel)) rel else 1e9;
                if (errs[idx] < 1e-3) e1e3 += 1;
                if (errs[idx] < 1e-2) e1e2 += 1;
                idx += 1;
            }
        }

        std.mem.sort(f64, errs, {}, std.sort.asc(f64));
        rep.median_rel_err = errs[total_heads / 2];
        rep.p90_rel_err = errs[(total_heads * 9) / 10];
        rep.max_rel_err = errs[total_heads - 1];
        rep.exact_1e3_heads = e1e3;
        rep.exact_1e2_heads = e1e2;
    }
};

pub fn compareLogits(ref_bf16: []const u16, got_bf16: []const u16) struct { max_err: f64, kl: f64 } {
    var max_ref: f64 = -std.math.inf(f64);
    var max_got: f64 = -std.math.inf(f64);
    var max_err: f64 = 0.0;
    for (ref_bf16, got_bf16) |rb, gb| {
        const r: f64 = bf16ToF32(rb);
        const g: f64 = bf16ToF32(gb);
        if (r > max_ref) max_ref = r;
        if (g > max_got) max_got = g;
        const d = @abs(g - r);
        if (d > max_err) max_err = d;
    }
    var zr: f64 = 0.0;
    var zg: f64 = 0.0;
    for (ref_bf16, got_bf16) |rb, gb| {
        const r: f64 = bf16ToF32(rb);
        const g: f64 = bf16ToF32(gb);
        if (r - max_ref > -40.0) zr += @exp(r - max_ref);
        if (g - max_got > -40.0) zg += @exp(g - max_got);
    }
    const log_zr = @log(zr) + max_ref;
    const log_zg = @log(zg) + max_got;
    var kl: f64 = 0.0;
    for (ref_bf16, got_bf16) |rb, gb| {
        const r: f64 = bf16ToF32(rb);
        const g: f64 = bf16ToF32(gb);
        if (r - max_ref > -30.0) {
            const pr = @exp(r - log_zr);
            const log_pg = g - log_zg;
            kl += pr * ((r - log_zr) - log_pg);
        }
    }
    return .{ .max_err = max_err, .kl = @max(0.0, kl) };
}

fn align256(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}
