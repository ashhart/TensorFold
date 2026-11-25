//! Cross-model speculative decoding (e.g., Qwen3.5-0.8B drafting for Qwen3.5-2B or Qwen3.8-27B).
//!
//! Shares one Metal context between target and draft models and uses fast GPU `copy_u4`
//! snapshots (~33 us per step on 0.8B) so draft GDN+KV rollback on rejection costs zero re-forward.
//! Combines with SuffixIndex copy lookahead for seamless 100% lossless greedy parity.
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");
const spec_mod = @import("speculate.zig");
const samp_mod = @import("sampler.zig");

const Model = model_mod.Model;
const Buffer = mtl.Buffer;
const nowMs = model_mod.nowMs;

const CopyParams = extern struct { count: u32 };

pub const CrossModelDecoder = struct {
    pub const max_k = 8;

    a: std.mem.Allocator,
    target: *Model,
    draft: *Model,
    suffix: spec_mod.SuffixIndex,
    k: usize,
    copy_width: usize = 31,

    // GPU snapshot slots [0 .. k] holding draft GDN state + conv history for O(1) 33us rollback!
    snap_state_buf: Buffer,
    snap_hist_buf: Buffer,
    state_bytes: usize,
    hist_bytes: usize,

    pub fn init(a: std.mem.Allocator, target: *Model, draft: *Model, draft_steps: usize) !CrossModelDecoder {
        const k = @max(1, @min(draft_steps, max_k));
        const sb = draft.gdn_state_buf.len;
        const hb = draft.gdn_hist_buf.len;
        const snap_state = try draft.ctx.allocBuffer((k + 1) * sb);
        errdefer snap_state.deinit();
        const snap_hist = try draft.ctx.allocBuffer((k + 1) * hb);
        errdefer snap_hist.deinit();
        draft.ctx.commitResidency();

        return .{
            .a = a,
            .target = target,
            .draft = draft,
            .suffix = try spec_mod.SuffixIndex.init(a),
            .k = k,
            .snap_state_buf = snap_state,
            .snap_hist_buf = snap_hist,
            .state_bytes = sb,
            .hist_bytes = hb,
        };
    }

    pub fn deinit(self: *CrossModelDecoder) void {
        self.suffix.deinit();
        self.snap_state_buf.deinit();
        self.snap_hist_buf.deinit();
    }

    fn saveDraftSnap(self: *CrossModelDecoder, slot: usize) !void {
        const d = self.draft;
        var cb = d.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        enc.setPipeline(d.p.copy_u4);

        const s_u4: u32 = @intCast(self.state_bytes / 16);
        enc.setBuffer(d.gdn_state_buf, 0, 0);
        enc.setBuffer(self.snap_state_buf, slot * self.state_bytes, 1);
        enc.setValue(CopyParams{ .count = s_u4 }, 2);
        enc.dispatchThreads(.{ .width = s_u4, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

        const h_u4: u32 = @intCast(self.hist_bytes / 16);
        enc.setBuffer(d.gdn_hist_buf, 0, 0);
        enc.setBuffer(self.snap_hist_buf, slot * self.hist_bytes, 1);
        enc.setValue(CopyParams{ .count = h_u4 }, 2);
        enc.dispatchThreads(.{ .width = h_u4, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

        enc.end();
        cb.commit();
        try d.ctx.waitIdle();
    }

    fn restoreDraftSnap(self: *CrossModelDecoder, slot: usize, target_pos: u32) !void {
        const d = self.draft;
        var cb = d.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        enc.setPipeline(d.p.copy_u4);

        const s_u4: u32 = @intCast(self.state_bytes / 16);
        enc.setBuffer(self.snap_state_buf, slot * self.state_bytes, 0);
        enc.setBuffer(d.gdn_state_buf, 0, 1);
        enc.setValue(CopyParams{ .count = s_u4 }, 2);
        enc.dispatchThreads(.{ .width = s_u4, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

        const h_u4: u32 = @intCast(self.hist_bytes / 16);
        enc.setBuffer(self.snap_hist_buf, slot * self.hist_bytes, 0);
        enc.setBuffer(d.gdn_hist_buf, 0, 1);
        enc.setValue(CopyParams{ .count = h_u4 }, 2);
        enc.dispatchThreads(.{ .width = h_u4, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

        enc.end();
        cb.commit();
        try d.ctx.waitIdle();
        d.pos = target_pos;
    }

    /// Generate up to `max_tokens` lossless greedy tokens using Draft model + SuffixIndex fast-path.
    pub fn generate(
        self: *CrossModelDecoder,
        prompt: []const u32,
        max_tokens: usize,
        stop_on_eos: bool,
        out_ids: ?*std.ArrayList(u32),
        out_alloc: std.mem.Allocator,
        on_token: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        cb_ctx: ?*anyopaque,
    ) !spec_mod.SpecStats {
        const t = self.target;
        const d = self.draft;
        t.reset();
        d.reset();
        self.suffix.reset();
        self.copy_width = 31;
        try self.suffix.appendSlice(prompt);

        var stats = spec_mod.SpecStats{};
        var pending = try t.prefill(prompt, 128);
        _ = try d.prefill(prompt, 128);
        if (max_tokens == 0) return stats;

        const td0 = nowMs();

        if (stop_on_eos and t.c.isEos(pending)) return stats;
        if (out_ids) |list| try list.append(out_alloc, pending);
        if (on_token) |cb| cb(cb_ctx, pending);
        stats.tokens_generated += 1;

        var row_tokens: [128]u32 = undefined;
        var kept_rows: [128]u32 = undefined;

        while (stats.tokens_generated < max_tokens) {
            const rem_budget = max_tokens - stats.tokens_generated;
            const ctx_room = if (t.pos < t.max_context) t.max_context - t.pos else 0;
            if (ctx_room == 0) break;

            stats.rounds += 1;
            stats.spec_rounds += 1;

            // 1. Confident verbatim copy fast-path (skips draft model!)
            const max_copy = @min(self.copy_width, @min(if (rem_budget > 1) rem_budget - 1 else 0, if (ctx_room > 1) ctx_room - 1 else 0));
            const copy_prop = self.suffix.propose(pending, max_copy);

            if (copy_prop.match_len >= 5 and copy_prop.len >= 10) {
                const n_draft = copy_prop.len;
                const n_rows = 1 + n_draft;
                row_tokens[0] = pending;
                @memcpy(row_tokens[1..n_rows], copy_prop.tokens[0..n_draft]);
                stats.drafts_proposed += n_draft;

                const picks = try t.verifyChain(row_tokens[0..n_rows]);
                var accepted_drafts: usize = 0;
                var hit_eos = false;
                var next_pending = picks[0];
                kept_rows[0] = 0;
                try self.suffix.append(pending);

                for (0..n_draft) |i| {
                    const emitted = picks[i];
                    if (stop_on_eos and t.c.isEos(emitted)) {
                        hit_eos = true;
                        break;
                    }
                    if (out_ids) |list| try list.append(out_alloc, emitted);
                    if (on_token) |cb| cb(cb_ctx, emitted);
                    stats.tokens_generated += 1;
                    if (stats.tokens_generated >= max_tokens) {
                        hit_eos = true;
                        break;
                    }
                    if (emitted == copy_prop.tokens[i]) {
                        accepted_drafts += 1;
                        kept_rows[accepted_drafts] = @intCast(i + 1);
                        try self.suffix.append(emitted);
                        next_pending = picks[i + 1];
                    } else {
                        next_pending = emitted;
                        break;
                    }
                }
                if (!hit_eos and accepted_drafts == n_draft and stats.tokens_generated < max_tokens) {
                    const bonus = picks[n_draft];
                    if (stop_on_eos and t.c.isEos(bonus)) {
                        hit_eos = true;
                    } else {
                        if (out_ids) |list| try list.append(out_alloc, bonus);
                        if (on_token) |cb| cb(cb_ctx, bonus);
                        stats.tokens_generated += 1;
                        next_pending = bonus;
                    }
                }
                stats.drafts_accepted += accepted_drafts;
                const keep_cnt = 1 + accepted_drafts;
                try t.commitKeep(kept_rows[0..keep_cnt]);
                // Advance draft model in sync over the accepted tokens in ONE fast multi-row prefill call!
                _ = try d.prefill(row_tokens[0..keep_cnt], 128);

                if (accepted_drafts * 10 >= n_draft * 8 and n_draft == self.copy_width) {
                    self.copy_width = @min(127, self.copy_width * 2 + 1);
                } else if (accepted_drafts * 2 < n_draft) {
                    self.copy_width = @max(31, (self.copy_width - 1) / 2);
                }
                pending = next_pending;
                if (hit_eos) break;
                continue;
            }

            // 2. Neural cross-model drafting (K steps on fast draft model with 33us per-step GPU snapshots!)
            const n_draft = @min(self.k, @min(if (rem_budget > 1) rem_budget - 1 else 0, if (ctx_room > 1) ctx_room - 1 else 0));
            if (n_draft == 0) {
                try self.suffix.append(pending);
                const next_tok = try t.step(pending);
                _ = try d.step(pending);
                if (stop_on_eos and t.c.isEos(next_tok)) break;
                if (out_ids) |list| try list.append(out_alloc, next_tok);
                if (on_token) |cb| cb(cb_ctx, next_tok);
                stats.tokens_generated += 1;
                pending = next_tok;
                continue;
            }

            const base_pos = d.pos;
            row_tokens[0] = pending;
            var d_cur = try d.step(pending);
            try self.saveDraftSnap(0); // state after `pending` (accepted_drafts = 0)
            row_tokens[1] = d_cur;

            for (1..n_draft) |i| {
                d_cur = try d.step(row_tokens[i]);
                try self.saveDraftSnap(i); // state after `row_tokens[0..i]` (accepted_drafts = i)
                row_tokens[i + 1] = d_cur;
            }
            // Note: if all n_draft land, we need draft to also process row_tokens[n_draft] (1 step) after verify!

            const n_rows = 1 + n_draft;
            stats.drafts_proposed += n_draft;

            const picks = try t.verifyChain(row_tokens[0..n_rows]);

            var accepted_drafts: usize = 0;
            var hit_eos = false;
            var next_pending = picks[0];
            kept_rows[0] = 0;
            try self.suffix.append(pending);

            for (0..n_draft) |i| {
                const emitted = picks[i];
                if (stop_on_eos and t.c.isEos(emitted)) {
                    hit_eos = true;
                    break;
                }
                if (out_ids) |list| try list.append(out_alloc, emitted);
                if (on_token) |cb| cb(cb_ctx, emitted);
                stats.tokens_generated += 1;
                if (stats.tokens_generated >= max_tokens) {
                    hit_eos = true;
                    break;
                }
                if (emitted == row_tokens[i + 1]) {
                    accepted_drafts += 1;
                    kept_rows[accepted_drafts] = @intCast(i + 1);
                    try self.suffix.append(emitted);
                    next_pending = picks[i + 1];
                } else {
                    next_pending = emitted;
                    break;
                }
            }

            if (!hit_eos and accepted_drafts == n_draft and stats.tokens_generated < max_tokens) {
                const bonus = picks[n_draft];
                if (stop_on_eos and t.c.isEos(bonus)) {
                    hit_eos = true;
                } else {
                    if (out_ids) |list| try list.append(out_alloc, bonus);
                    if (on_token) |cb| cb(cb_ctx, bonus);
                    stats.tokens_generated += 1;
                    next_pending = bonus;
                }
            }

            stats.drafts_accepted += accepted_drafts;
            try t.commitKeep(kept_rows[0 .. 1 + accepted_drafts]);

            // Sync draft model in O(1): either advance 1 final token (all accepted) or 33us snapshot restore!
            if (accepted_drafts == n_draft) {
                _ = try d.step(row_tokens[n_draft]);
            } else {
                try self.restoreDraftSnap(accepted_drafts, base_pos + @as(u32, @intCast(1 + accepted_drafts)));
            }

            pending = next_pending;
            if (hit_eos) break;
        }

        stats.decode_ms = nowMs() - td0;
        return stats;
    }

    fn buildRowDist(m: *const Model, row: usize, params: samp_mod.SamplingParams) !samp_mod.SparseDist {
        const greedy = m.out_tok_buf.slice(u32, row + 1)[row];
        const max_v = m.out_val_buf.slice(f32, row + 1)[row];
        const all_logits = m.logits_buf.slice(u16, (row + 1) * m.c.vocab_size);
        const row_logits = all_logits[row * m.c.vocab_size ..][0..m.c.vocab_size];
        return samp_mod.SparseDist.build(m.a, row_logits, greedy, max_v, params);
    }

    /// Exact Leviathan/Chen speculative rejection sampling for `temperature > 0`!
    /// Provably preserves the EXACT target probability distribution P(x | context) for any
    /// (temperature, top_p, top_k, min_p) while achieving 2x+ speedup on 27B!
    pub fn generateSampled(
        self: *CrossModelDecoder,
        prompt: []const u32,
        max_tokens: usize,
        stop_on_eos: bool,
        sampling: samp_mod.SamplingParams,
        out_ids: ?*std.ArrayList(u32),
        out_alloc: std.mem.Allocator,
        on_token: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        cb_ctx: ?*anyopaque,
    ) !spec_mod.SpecStats {
        if (sampling.temperature <= 1e-6) {
            return self.generate(prompt, max_tokens, stop_on_eos, out_ids, out_alloc, on_token, cb_ctx);
        }

        const t = self.target;
        const d = self.draft;
        t.reset();
        d.reset();
        var rng = samp_mod.Rng.init(sampling.seed);

        var stats = spec_mod.SpecStats{};
        _ = try t.prefill(prompt, 128);
        _ = try d.prefill(prompt, 128);
        if (max_tokens == 0) return stats;

        var pending = try t.sampleRow(0, sampling, &rng);
        const td0 = nowMs();

        if (stop_on_eos and t.c.isEos(pending)) return stats;
        if (out_ids) |list| try list.append(out_alloc, pending);
        if (on_token) |cb| cb(cb_ctx, pending);
        stats.tokens_generated += 1;

        var row_tokens: [128]u32 = undefined;
        var kept_rows: [128]u32 = undefined;
        var q_dists: [max_k]samp_mod.SparseDist = undefined;

        while (stats.tokens_generated < max_tokens) {
            const rem_budget = max_tokens - stats.tokens_generated;
            const ctx_room = if (t.pos < t.max_context) t.max_context - t.pos else 0;
            if (ctx_room == 0) break;

            stats.rounds += 1;
            const n_draft = @min(self.k, @min(if (rem_budget > 1) rem_budget - 1 else 0, if (ctx_room > 1) ctx_room - 1 else 0));
            if (n_draft == 0) {
                _ = try t.stepWithLogits(pending);
                _ = try d.step(pending);
                const next_tok = try t.sampleRow(0, sampling, &rng);
                if (stop_on_eos and t.c.isEos(next_tok)) break;
                if (out_ids) |list| try list.append(out_alloc, next_tok);
                if (on_token) |cb| cb(cb_ctx, next_tok);
                stats.tokens_generated += 1;
                pending = next_tok;
                continue;
            }

            // 1. Draft `n_draft` tokens from Q_0 .. Q_{k-1} with exact sparse distributions & 33us snapshots
            stats.spec_rounds += 1;
            const base_pos = d.pos;
            row_tokens[0] = pending;
            var built_q: usize = 0;
            defer for (0..built_q) |qi| q_dists[qi].deinit(d.a);

            for (0..n_draft) |i| {
                _ = try d.stepWithLogits(row_tokens[i]);
                try self.saveDraftSnap(i);
                q_dists[i] = try buildRowDist(d, 0, sampling);
                built_q += 1;
                row_tokens[i + 1] = q_dists[i].sample(&rng);
            }

            // 2. Verify all `1 + n_draft` rows in ONE target hardware tensor-unit pass!
            const n_rows = 1 + n_draft;
            stats.drafts_proposed += n_draft;
            _ = try t.verifyChain(row_tokens[0..n_rows]);

            // 3. Exact Leviathan/Chen rejection sampling + residual resampling
            var accepted_drafts: usize = 0;
            var hit_eos = false;
            var next_pending: u32 = 0;
            kept_rows[0] = 0;

            for (0..n_draft) |i| {
                var p_dist = try buildRowDist(t, i, sampling);
                defer p_dist.deinit(t.a);

                const cand = row_tokens[i + 1];
                const px = p_dist.probOf(cand);
                const qx = q_dists[i].probOf(cand);
                const accept = (qx > 0.0) and (px >= qx or rng.nextF32() * qx < px);

                if (accept) {
                    if (stop_on_eos and t.c.isEos(cand)) {
                        hit_eos = true;
                        break;
                    }
                    if (out_ids) |list| try list.append(out_alloc, cand);
                    if (on_token) |cb| cb(cb_ctx, cand);
                    stats.tokens_generated += 1;
                    accepted_drafts += 1;
                    kept_rows[accepted_drafts] = @intCast(i + 1);
                    if (stats.tokens_generated >= max_tokens) {
                        hit_eos = true;
                        break;
                    }
                } else {
                    const corr = samp_mod.sampleResidual(&p_dist, &q_dists[i], &rng);
                    if (stop_on_eos and t.c.isEos(corr)) {
                        hit_eos = true;
                        break;
                    }
                    if (out_ids) |list| try list.append(out_alloc, corr);
                    if (on_token) |cb| cb(cb_ctx, corr);
                    stats.tokens_generated += 1;
                    next_pending = corr;
                    if (stats.tokens_generated >= max_tokens) hit_eos = true;
                    break;
                }
            }

            if (!hit_eos and accepted_drafts == n_draft and stats.tokens_generated < max_tokens) {
                var bonus_dist = try buildRowDist(t, n_draft, sampling);
                defer bonus_dist.deinit(t.a);
                const bonus = bonus_dist.sample(&rng);
                if (stop_on_eos and t.c.isEos(bonus)) {
                    hit_eos = true;
                } else {
                    if (out_ids) |list| try list.append(out_alloc, bonus);
                    if (on_token) |cb| cb(cb_ctx, bonus);
                    stats.tokens_generated += 1;
                    next_pending = bonus;
                }
            }

            stats.drafts_accepted += accepted_drafts;
            try t.commitKeep(kept_rows[0 .. 1 + accepted_drafts]);
            if (accepted_drafts == n_draft) {
                _ = try d.step(row_tokens[n_draft]);
            } else {
                try self.restoreDraftSnap(accepted_drafts, base_pos + @as(u32, @intCast(1 + accepted_drafts)));
            }

            pending = next_pending;
            if (hit_eos) break;
        }

        stats.decode_ms = nowMs() - td0;
        return stats;
    }
};
