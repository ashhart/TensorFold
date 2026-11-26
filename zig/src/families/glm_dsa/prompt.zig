//! A prompt window on the mesh: every layer's projections and experts for all of the window's rows at once.
const std = @import("std");
const mtl = @import("metal");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const ly = @import("layer.zig");
const lp = @import("layer_prompt.zig");
const wt = @import("weights.zig");
const Model = @import("model.zig").Model;
const Ref = kn.Ref;

const StageIdx = enum(usize) {
    x_g = 0,
    prep_idx,
    q_b,
    wq_b,
    idx_topk,
    absorb,
    attend,
    uv_g,
    o_proj,
    add_rms1,
    route,
    moe_jobs,
    moe_gu,
    moe_down,
    moe_comb,
    add_rms2,
    count,
};

fn runStep(queue: mtl.Queue, t_gpu: *f64, t_wall: *f64, comptime f: anytype, args: anytype) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = @import("sys").c.mach_absolute_time();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try @call(.auto, f, .{enc} ++ args);
    enc.end();
    cb.commit();
    cb.wait();
    const t1 = @import("sys").c.mach_absolute_time();
    if (cb.failure()) |msg| {
        std.log.err("glm_dsa profile: GPU failed: {s}", .{msg});
        return error.GpuFailure;
    }
    t_gpu.* += (cb.gpuEnd() - cb.gpuStart()) * 1000.0;
    t_wall.* += @as(f64, @floatFromInt(t1 - t0)) / 24_000.0;
}

/// Profiled backbone (when TF_GLM_PROMPT_PROFILE=1): times every kernel stage per layer over the window.
pub fn backboneProfiled(m: *Model, queue: mtl.Queue, s: *const gpu.Scratch, q: *const gpu.Prompt, rows: u32, pos0: u32) !void {
    const hid = m.c.hidden;
    const nh = m.c.localHeads();
    const v = &m.vocab;
    {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        if (m.c.isFirstStage()) m.embedTokens(enc, .{ .buf = q.ids }, .{ .buf = q.h }, rows) else m.pair.pipeRecv(enc, .{ .buf = q.h }, rows, hid);
        enc.barrier();
        const L0 = &m.layers[0];
        kn.addRms(&m.K.k, enc, .{ .buf = q.h }, .{ .buf = q.h }, L0.k.at(L0.w.in_norm), m.glue(hid, hid), .{ .buf = q.x }, rows);
        enc.end();
        cb.commit();
        cb.wait();
    }
    var gpu_ms: [@backingInt(StageIdx.count)]f64 = @splat(0);
    var wall_ms: [@backingInt(StageIdx.count)]f64 = @splat(0);
    var full_layers: u32 = 0;
    const p = m.pair;
    const t = Ref{ .buf = q.rows };
    const loc = m.c.localHalves() * 128;
    for (m.layers, 0..) |*L, i| {
        const a = L.w.attn;
        const ma: kn.MlaArgs = .{ .rows = rows, .heads = nh, .slot_keys = s.context, .proj_stride = wt.proj_width, .q_stride = nh * m.c.qHead(), .kvb = a.kvb_w, .kvb_scale = a.kvb_s };
        var mp = ma;
        mp.q_stride = m.c.q_lora;
        const ia: kn.IndexArgs = .{ .rows = rows, .slot_keys = s.context, .proj_stride = wt.proj_width, .k_col = wt.wk_col, .w_col = wt.wp_col, .q_stride = m.c.i_heads * m.c.i_dim, .score_stride = s.context, .pad0 = 1, .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.c.i_dim))), .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.c.i_heads))) };
        if (L.w.full) full_layers += 1;
        const next = if (i + 1 < m.layers.len) m.layers[i + 1].k.at(m.layers[i + 1].w.in_norm) else if (m.c.isLastStage()) v.k.at(v.norm) else L.k.at(L.w.in_norm);
        const S = struct {
            fn dense(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, g: gpu.Groups, x: Ref, y: Ref, r: u32, xs: usize, ys: usize) !void {
                if (L_.k.fast_dense) {
                    try ly.pp.encodeDenseFast(&K.pp, e, x, L_.k.dense64(g), L_.k.dense(g), y, r, xs, ys, true);
                } else {
                    try ly.pp.encodeDense(&K.pp, e, x, L_.k.dense(g), y, r, xs, ys, true);
                }
            }
            fn prepIdx(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, q_: *const gpu.Prompt, t_: Ref, mp_: kn.MlaArgs) !void {
                if (L_.w.full) {
                    kn.prepFull(&K.k, e, .{ .buf = q_.proj }, L_.k.at(L_.w.attn.q_norm), L_.k.at(L_.w.attn.kv_norm), .{ .buf = K.inv }, t_, mp_, .{ .buf = q_.q_resid }, .{ .buf = L_.cache }, L_.k.at(L_.w.attn.k_norm), L_.k.at(L_.w.attn.k_bias), .{ .buf = L_.index_cache.? }, null);
                } else {
                    kn.prep(&K.k, e, .{ .buf = q_.proj }, L_.k.at(L_.w.attn.q_norm), L_.k.at(L_.w.attn.kv_norm), .{ .buf = K.inv }, t_, mp_, .{ .buf = q_.q_resid }, .{ .buf = L_.cache });
                }
            }
            fn idxTop(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, q_: *const gpu.Prompt, t_: Ref, ia_: kn.IndexArgs, lp_: u32) !void {
                kn.indexTopk(&K.k, e, .{ .buf = q_.qidx }, .{ .buf = q_.proj }, .{ .buf = L_.index_cache.? }, t_, ia_, .{ .buf = q_.score }, .{ .buf = K.inv }, .{ .buf = q_.keys }, lp_);
            }
            fn absorbAll(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, q_: *const gpu.Prompt, ma_: kn.MlaArgs) !void {
                kn.absorbPrompt(&K.k, e, .{ .buf = q_.qq }, L_.k.bytes, .{ .buf = q_.rows }, ma_, .{ .buf = q_.qlat }, .{ .buf = q_.qrope }, .{ .buf = K.inv });
            }
            fn attendAll(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, q_: *const gpu.Prompt, ma_: kn.MlaArgs, nh_: u32, kl_: u32, rp_: u32, r_: u32) !void {
                if (r_ >= 32 and nh_ >= 8 and (nh_ % 8 == 0)) {
                    kn.attendFlash(&K.k, e, .{ .buf = q_.qlat }, .{ .buf = q_.qrope }, .{ .buf = q_.rows }, .{ .buf = q_.keys }, .{ .buf = L_.cache }, ma_, .{ .buf = q_.olat });
                    return;
                }
                const AR = gpu.prompt_attend_rows;
                var k: u32 = 0;
                while (k < r_) : (k += AR) {
                    var mk = ma_;
                    mk.rows = @min(AR, r_ - k);
                    const tk: Ref = .{ .buf = q_.rows, .off = @as(usize, k) * @sizeOf(kn.Row) };
                    kn.attend(&K.k, e, .{ .buf = q_.qlat, .off = @as(usize, k) * nh_ * kl_ * 4 }, .{ .buf = q_.qrope, .off = @as(usize, k) * nh_ * rp_ * 4 }, tk, .{ .buf = q_.keys }, .{ .buf = L_.cache }, mk, .{ .buf = q_.partial }, .{ .buf = q_.olat, .off = @as(usize, k) * nh_ * kl_ * 4 });
                    e.barrier();
                }
            }
            fn addRmsOp(e: mtl.ComputeEncoder, p_: *@import("../../core/pair.zig").Pair, q_: *const gpu.Prompt, w_: Ref, r_: u32, h_: u32, eps_: f32) !void {
                p_.addRms(e, .{ .buf = q_.h }, w_, .{ .buf = q_.x }, r_, h_, eps_);
            }
            fn routeOp(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, q_: *const gpu.Prompt, c_: *const @import("config.zig").Config, r_: u32) !void {
                try lp.route(K, L_, s_, q_, e, c_, r_, .{ .buf = q_.x });
            }
            fn moeJobs(e: mtl.ComputeEncoder, K: *const ly.Kernels, q_: *const gpu.Prompt, fa: kn.Fp8MoeArgs) !void {
                e.setPipeline(K.k.fp8_moe_prompt_jobs);
                e.setBuffer(q_.pick, 0, 0);
                e.setBuffer(q_.wts8, 0, 1);
                e.setValue(fa, 2);
                e.setBuffer(q_.jobs, 0, 3);
                e.setBuffer(q_.members, 0, 4);
                e.setBuffer(q_.wts, 0, 5);
                e.setBuffer(q_.dispatch, 0, 6);
                e.dispatchGroups(.{ .width = 1 }, .{ .width = 1024 });
            }
            fn moeGu(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, q_: *const gpu.Prompt, fa: kn.Fp8MoeArgs) !void {
                const fm = L_.fp8_moe.?;
                e.setPipeline(K.k.fp8_moe_prompt_gu);
                e.setBuffer(q_.x, 0, 0);
                e.setBuffer(fm.gate_w, 0, 1);
                e.setBuffer(fm.gate_s, 0, 2);
                e.setBuffer(fm.up_w, 0, 3);
                e.setBuffer(fm.up_s, 0, 4);
                e.setBuffer(q_.jobs, 0, 5);
                e.setBuffer(q_.members, 0, 6);
                e.setValue(fa, 7);
                e.setBuffer(q_.act, 0, 8);
                e.dispatchIndirect(q_.dispatch, 0, .{ .width = 128 });
            }
            fn moeDown(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, q_: *const gpu.Prompt, fa: kn.Fp8MoeArgs, o_: Ref) !void {
                const fm = L_.fp8_moe.?;
                e.setPipeline(K.k.fp8_moe_prompt_down);
                e.setBuffer(q_.act, 0, 0);
                e.setBuffer(fm.down_w, 0, 1);
                e.setBuffer(fm.down_s, 0, 2);
                e.setBuffer(q_.jobs, 0, 3);
                e.setBuffer(q_.members, 0, 4);
                e.setValue(fa, 5);
                e.setBuffer(q_.z, 0, 6);
                e.dispatchIndirect(q_.dispatch, 16, .{ .width = 128 });
                e.barrier();
                e.setPipeline(K.k.fp8_moe_combine);
                e.setBuffer(q_.z, 0, 0);
                e.setBuffer(q_.wts, 0, 1);
                e.setValue(fa, 2);
                e.setBuffer(o_.buf, o_.off, 3);
                e.dispatchGroups(.{ .width = (fa.hidden / 4 + 63) / 64, .height = fa.rows }, .{ .width = 64 });
            }
        };
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.x_g)], &wall_ms[@backingInt(StageIdx.x_g)], S.dense, .{ &m.K, L, L.groups[ly.Layer.x_g], Ref{ .buf = q.x }, Ref{ .buf = q.proj }, rows, hid, wt.proj_width });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.prep_idx)], &wall_ms[@backingInt(StageIdx.prep_idx)], S.prepIdx, .{ &m.K, L, q, t, mp });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.q_b)], &wall_ms[@backingInt(StageIdx.q_b)], S.dense, .{ &m.K, L, L.groups[ly.Layer.q_b_g], Ref{ .buf = q.q_resid }, Ref{ .buf = q.qq }, rows, m.c.q_lora, nh * m.c.qHead() });
        if (L.w.full and (pos0 + rows - 1 >= m.c.i_topk)) {
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.wq_b)], &wall_ms[@backingInt(StageIdx.wq_b)], S.dense, .{ &m.K, L, L.groups[ly.Layer.wq_b_g], Ref{ .buf = q.q_resid }, Ref{ .buf = q.qidx }, rows, m.c.q_lora, m.c.i_heads * m.c.i_dim });
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.idx_topk)], &wall_ms[@backingInt(StageIdx.idx_topk)], S.idxTop, .{ &m.K, L, q, t, ia, pos0 + rows - 1 });
        }
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.absorb)], &wall_ms[@backingInt(StageIdx.absorb)], S.absorbAll, .{ &m.K, L, q, ma });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.attend)], &wall_ms[@backingInt(StageIdx.attend)], S.attendAll, .{ &m.K, L, q, ma, nh, m.c.kv_lora, m.c.rope, rows });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.uv_g)], &wall_ms[@backingInt(StageIdx.uv_g)], S.dense, .{ &m.K, L, L.groups[ly.Layer.uv_g], Ref{ .buf = q.olat }, Ref{ .buf = q.o }, rows, nh * m.c.kv_lora, nh * m.c.v_dim });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.o_proj)], &wall_ms[@backingInt(StageIdx.o_proj)], S.dense, .{ &m.K, L, L.groups[ly.Layer.o_g], Ref{ .buf = q.o }, p.part(), rows, nh * m.c.v_dim, hid });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.add_rms1)], &wall_ms[@backingInt(StageIdx.add_rms1)], S.addRmsOp, .{ p, q, L.k.at(L.w.post_norm), rows, hid, m.c.eps });
        const o = p.part();
        if (L.w.mlp == .moe and L.fp8_moe != null) {
            const fa: kn.Fp8MoeArgs = .{ .rows = rows, .loc = loc, .hidden = hid, .topk = m.c.topk, .slots = gpu.slots, .experts = m.c.experts, .loc_blocks = m.c.localHalves(), .hid_blocks = hid / 128 };
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.route)], &wall_ms[@backingInt(StageIdx.route)], S.routeOp, .{ &m.K, L, s, q, &m.c, rows });
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_jobs)], &wall_ms[@backingInt(StageIdx.moe_jobs)], S.moeJobs, .{ &m.K, q, fa });
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_gu)], &wall_ms[@backingInt(StageIdx.moe_gu)], S.moeGu, .{ &m.K, L, q, fa });
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_down)], &wall_ms[@backingInt(StageIdx.moe_down)], S.moeDown, .{ &m.K, L, q, fa, o });
        }
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.add_rms2)], &wall_ms[@backingInt(StageIdx.add_rms2)], S.addRmsOp, .{ p, q, next, rows, hid, m.c.eps });
    }
    if (m.c.rank == 0) {
        const nl: f64 = @floatFromInt(m.layers.len);
        const names = [_][]const u8{ "x_g", "prep_idx", "q_b", "wq_b", "idx_topk", "absorb", "attend", "uv_g", "o_proj", "addRms1", "route", "moe_jobs", "moe_gu", "moe_down", "moe_comb", "addRms2" };
        var tot: f64 = 0;
        std.log.info("=== GLM PROMPT PROFILE (rows={d}, pos0={d}, layers={d}, full={d}, tp={d}) ===", .{ rows, pos0, m.layers.len, full_layers, m.c.tp });
        for (names, 0..) |nm, idx| {
            const per = gpu_ms[idx] / nl;
            const wper = wall_ms[idx] / nl;
            tot += per;
            std.log.info("  {s: <10}: gpu={d:6.2} ms/L  wall={d:6.2} ms/L  (total={d:7.1} ms)", .{ nm, per, wper, gpu_ms[idx] });
        }
        std.log.info("  TOTAL     : gpu={d:6.2} ms/L  ({d:7.1} ms for {d} layers)", .{ tot, tot * nl, m.layers.len });
    }
}

/// Profiled decode window (when TF_GLM_PROMPT_PROFILE=1): times every kernel stage per layer in us/layer.
pub fn windowProfiled(m: *Model, queue: mtl.Queue, s: *const gpu.Scratch, rows: u32, last_pos: u32, ids: Ref) !void {
    const dg = @import("../../core/exlr8_dense_gpu.zig");
    const hid = m.c.hidden;
    const nh = m.c.localHeads();
    const v = &m.vocab;
    {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        if (m.c.isFirstStage()) m.embedTokens(enc, ids, .{ .buf = s.h }, rows) else m.pair.pipeRecv(enc, .{ .buf = s.h }, rows, hid);
        enc.barrier();
        const L0 = &m.layers[0];
        var g0 = m.glue(hid, hid);
        if (rows >= 2 and rows <= 8) g0.pad0 = @intCast(gpu.max_rows * (hid / 4));
        kn.addRms(&m.K.k, enc, .{ .buf = s.h }, .{ .buf = s.y }, L0.k.at(L0.w.in_norm), g0, .{ .buf = s.x }, rows);
        enc.end();
        cb.commit();
        cb.wait();
    }
    var gpu_ms: [@backingInt(StageIdx.count)]f64 = @splat(0);
    var wall_ms: [@backingInt(StageIdx.count)]f64 = @splat(0);
    const p = m.pair;
    const t = Ref{ .buf = s.rows, .off = s.rows_off };
    const loc = m.c.localHalves() * 128;
    const use_xh = rows >= 2 and rows <= 8;
    for (m.layers, 0..) |*L, i| {
        const a = L.w.attn;
        const ma: kn.MlaArgs = .{ .rows = rows, .heads = nh, .slot_keys = s.context, .proj_stride = wt.proj_width, .q_stride = nh * m.c.qHead(), .pad1 = if (use_xh) 1.0 else 0.0, .kvb = a.kvb_w, .kvb_scale = a.kvb_s };
        var mp = ma;
        mp.q_stride = m.c.q_lora;
        mp.pad2 = if (use_xh) gpu.max_rows * m.c.q_lora else 0;
        const ia: kn.IndexArgs = .{ .rows = rows, .slot_keys = s.context, .proj_stride = wt.proj_width, .k_col = wt.wk_col, .w_col = wt.wp_col, .q_stride = m.c.i_heads * m.c.i_dim, .score_stride = s.context, .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.c.i_dim))), .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.c.i_heads))) };
        const next = if (i + 1 < m.layers.len) m.layers[i + 1].k.at(m.layers[i + 1].w.in_norm) else if (m.c.isLastStage()) v.k.at(v.norm) else L.k.at(L.w.in_norm);
        const fuse_uv = L.k.fast_dense and m.c.nope == 192 and m.c.v_dim == 256 and m.c.kv_lora == 512;
        const D = struct {
            fn xProj(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, r: u32, h: u32) !void {
                if (r <= 8 and L_.fp8_x_n >= 2) {
                    const m0 = L_.fp8_x[0].?;
                    const m1 = L_.fp8_x[1].?;
                    const m2 = if (L_.fp8_x_n >= 3) L_.fp8_x[2] else null;
                    const pt: u32 = if (r == 1 or (r >= 3 and r <= 5)) 8 else 16;
                    kn.fast.fp8Gemv3(&K.k.fast_p, e, if (r >= 2) s_.xhRef(h) else .{ .buf = s_.x }, L_.k.bytes, .{ .o0 = m0.outs, .o1 = m1.outs, .o2 = if (m2) |x| x.outs else 0, .ins = m0.ins, .rows = r, .x_stride = h, .y_stride = wt.proj_width, .sb_cols = m0.sb_cols, .yc0 = m0.y_col, .yc1 = m1.y_col, .yc2 = if (m2) |x| x.y_col else 0, .g0 = (m0.outs + pt - 1) / pt, .g1 = (m1.outs + pt - 1) / pt, .g2 = if (m2) |x| (x.outs + pt - 1) / pt else 0, .w0 = m0.w, .s0 = m0.s, .w1 = m1.w, .s1 = m1.s, .w2 = if (m2) |x| x.w else 0, .s2 = if (m2) |x| x.s else 0 }, .{ .buf = s_.proj }, null);
                } else try dg.encodeProject(&K.dg, e, .{ .buf = s_.x }, L_.k.denseFor(L_.groups[ly.Layer.x_g], r), .{ .buf = s_.proj }, r, h, wt.proj_width, true);
            }
            fn qB(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, r: u32, ql: u32, nh_: u32, qh: u32) !void {
                if (r <= 8 and L_.fp8_q_b != null) {
                    const fm = L_.fp8_q_b.?;
                    kn.fast.fp8Gemv(&K.k.fast_p, e, if (r >= 2) s_.qhRef(ql) else .{ .buf = s_.q_resid }, L_.k.bytes, .{ .outs = fm.outs, .ins = fm.ins, .rows = r, .x_stride = ql, .y_stride = nh_ * qh, .y_col = 0, .sb_cols = fm.sb_cols, .w_off = fm.w, .s_off = fm.s }, .{ .buf = s_.q }, null, 2);
                } else try dg.encodeProject(&K.dg, e, .{ .buf = s_.q_resid }, L_.k.denseFor(L_.groups[ly.Layer.q_b_g], r), .{ .buf = s_.q }, r, ql, nh_ * qh, true);
            }
            fn oProj(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, p_: *@import("../../core/pair.zig").Pair, r: u32, nv: u32, h: u32) !void {
                const pk = p_.packTarget();
                if (r <= 8 and L_.fp8_o != null) {
                    const fm = L_.fp8_o.?;
                    kn.fast.fp8OPack(&K.k.fast_p, e, if (r >= 2) s_.ohRef(nv / 256, 256) else .{ .buf = s_.o }, L_.k.bytes, .{ .outs = fm.outs, .ins = fm.ins, .rows = r, .x_stride = nv, .y_stride = h, .y_col = 0, .sb_cols = fm.sb_cols, .w_off = fm.w, .s_off = fm.s }, p_.part(), pk.info, pk.base, null);
                } else try dg.encodeProject(&K.dg, e, .{ .buf = s_.o }, L_.k.denseFor(L_.groups[ly.Layer.o_g], r), p_.part(), r, nv, h, true);
            }
            fn dense(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, g: gpu.Groups, x: Ref, y: Ref, r: u32, xs: usize, ys: usize) !void {
                try dg.encodeProject(&K.dg, e, x, L_.k.denseFor(g, r), y, r, xs, ys, true);
            }
            fn prepIdx(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, t_: Ref, mp_: kn.MlaArgs) !void {
                if (L_.w.full) {
                    kn.prepFull(&K.k, e, .{ .buf = s_.proj }, L_.k.at(L_.w.attn.q_norm), L_.k.at(L_.w.attn.kv_norm), .{ .buf = K.inv }, t_, mp_, .{ .buf = s_.q_resid }, .{ .buf = L_.cache }, L_.k.at(L_.w.attn.k_norm), L_.k.at(L_.w.attn.k_bias), .{ .buf = L_.index_cache.? }, null);
                } else {
                    kn.prep(&K.k, e, .{ .buf = s_.proj }, L_.k.at(L_.w.attn.q_norm), L_.k.at(L_.w.attn.kv_norm), .{ .buf = K.inv }, t_, mp_, .{ .buf = s_.q_resid }, .{ .buf = L_.cache });
                }
            }
            fn idxTop(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, t_: Ref, ia_: kn.IndexArgs, lp_: u32) !void {
                kn.indexTopk(&K.k, e, .{ .buf = s_.qidx }, .{ .buf = s_.proj }, .{ .buf = L_.index_cache.? }, t_, ia_, .{ .buf = s_.score }, .{ .buf = K.inv }, .{ .buf = s_.keys }, lp_);
            }
            fn absorbOp(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, t_: Ref, ma_: kn.MlaArgs) !void {
                kn.absorb(&K.k, e, .{ .buf = s_.q }, L_.k.bytes, t_, ma_, .{ .buf = s_.qlat }, .{ .buf = s_.qrope }, .{ .buf = K.inv });
            }
            fn attendOp(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, t_: Ref, ma_: kn.MlaArgs, lp_: u32, fuv: bool) !void {
                _ = kn.attendDecode(&K.k, e, .{ .buf = s_.qlat }, .{ .buf = s_.qrope }, t_, .{ .buf = s_.keys }, .{ .buf = L_.cache }, ma_, .{ .buf = s_.partial }, .{ .buf = s_.olat }, lp_, if (fuv) L_.k.bytes else null, if (fuv) Ref{ .buf = s_.o } else null, null);
            }
            fn addRmsPre(e: mtl.ComputeEncoder, p_: *@import("../../core/pair.zig").Pair, s_: *const gpu.Scratch, w_: Ref, r_: u32, h_: u32, eps_: f32) !void {
                const xo4: u32 = if (r_ >= 2 and r_ <= 8) @intCast(gpu.max_rows * (h_ / 4)) else 0;
                _ = p_.addRmsPrepackedXhGated(e, .{ .buf = s_.h }, w_, .{ .buf = s_.x }, r_, h_, eps_, xo4, null);
            }
            fn routeOp(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, c_: *const @import("config.zig").Config, r_: u32) !void {
                _ = try ly.route(K, L_, s_, e, c_, r_, .{ .buf = s_.x }, null, null);
            }
            fn moeGu1(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, fa: kn.Fp8MoeArgs) !void {
                const fm = L_.fp8_moe.?;
                e.setPipeline(K.k.fp8_moe_gu1_fast);
                e.setBuffer(s_.x, 0, 0);
                e.setBuffer(fm.gate_w, 0, 1);
                e.setBuffer(fm.gate_s, 0, 2);
                e.setBuffer(fm.up_w, 0, 3);
                e.setBuffer(fm.up_s, 0, 4);
                e.setBuffer(s_.route[0], 0, 5);
                e.setValue(fa, 6);
                e.setBuffer(s_.act, 0, 7);
                e.dispatchGroups(.{ .width = (fa.loc + 7) / 8, .height = fa.slots }, .{ .width = 128 });
            }
            fn moeDown1(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, fa: kn.Fp8MoeArgs, p_: *@import("../../core/pair.zig").Pair) !void {
                const fm = L_.fp8_moe.?;
                const pk = p_.packTarget();
                const o_ = p_.part();
                e.setPipeline(K.k.fp8_moe_down_combine1_fast);
                e.setBuffer(s_.act, 0, 0);
                e.setBuffer(fm.down_w, 0, 1);
                e.setBuffer(fm.down_s, 0, 2);
                e.setBuffer(s_.route[0], 0, 3);
                e.setBuffer(s_.route[1], 0, 4);
                e.setValue(fa, 5);
                e.setBuffer(o_.buf, o_.off, 6);
                e.setValue(pk.info, 7);
                e.setBuffer(pk.base.buf, pk.base.off, 8);
                e.dispatchGroups(.{ .width = (fa.hidden + 15) / 16, .height = 1 }, .{ .width = 128 });
            }
            fn moeGuN(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, fa: kn.Fp8MoeArgs) !void {
                const fm = L_.fp8_moe.?;
                const ci: usize = if (fa.rows <= 8) fa.rows - 2 else 7;
                e.setPipeline(K.k.fp8_moe_gu[ci]);
                const x_ref = if (fa.rows <= 8) s_.xhRef(fa.hidden) else Ref{ .buf = s_.x };
                e.setBuffer(x_ref.buf, x_ref.off, 0);
                inline for (.{ fm.gate_w, fm.gate_s, fm.up_w, fm.up_s }, 1..) |b, bi| e.setBuffer(b, 0, bi);
                e.setBuffer(s_.moe_jobs, 0, 5);
                e.setBuffer(s_.members, 0, 6);
                e.setValue(fa, 7);
                e.setBuffer(s_.act, 0, 8);
                e.dispatchIndirect(s_.dispatch, 0, .{ .width = if (fa.rows <= 5) 128 else 256 });
            }
            fn moeDnN(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, fa: kn.Fp8MoeArgs) !void {
                const fm = L_.fp8_moe.?;
                const ci: usize = if (fa.rows <= 8) fa.rows - 2 else 7;
                e.setPipeline(K.k.fp8_moe_down[ci]);
                e.setBuffer(s_.act, 0, 0);
                inline for (.{ fm.down_w, fm.down_s }, 1..) |b, bi| e.setBuffer(b, 0, bi);
                e.setBuffer(s_.moe_jobs, 0, 3);
                e.setBuffer(s_.members, 0, 4);
                e.setValue(fa, 5);
                e.setBuffer(s_.z, 0, 6);
                e.dispatchIndirect(s_.dispatch, 16, .{ .width = 256 });
            }
            fn moeCombN(e: mtl.ComputeEncoder, K: *const ly.Kernels, s_: *const gpu.Scratch, fa: kn.Fp8MoeArgs, p_: *@import("../../core/pair.zig").Pair) !void {
                const pk = p_.packTarget();
                const o_ = p_.part();
                e.setPipeline(K.k.fp8_moe_combine_pack_bf16);
                e.setBuffer(s_.z, 0, 0);
                e.setBuffer(s_.wts, 0, 1);
                e.setValue(fa, 2);
                e.setBuffer(o_.buf, o_.off, 3);
                e.setValue(pk.info, 4);
                e.setBuffer(pk.base.buf, pk.base.off, 5);
                e.dispatchGroups(.{ .width = (fa.hidden / 4 + 31) / 32, .height = fa.rows }, .{ .width = 32 });
            }
            fn denseMlpOp(e: mtl.ComputeEncoder, K: *const ly.Kernels, L_: *const ly.Layer, s_: *const gpu.Scratch, c_: *const @import("config.zig").Config, r_: u32, p_: *@import("../../core/pair.zig").Pair) !void {
                const pk = p_.packTarget();
                _ = try ly.denseMlpPack(K, L_, s_, e, c_, r_, .{ .buf = s_.x }, p_.part(), pk.info, pk.base);
            }
        };
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.x_g)], &wall_ms[@backingInt(StageIdx.x_g)], D.xProj, .{ &m.K, L, s, rows, hid });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.prep_idx)], &wall_ms[@backingInt(StageIdx.prep_idx)], D.prepIdx, .{ &m.K, L, s, t, mp });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.q_b)], &wall_ms[@backingInt(StageIdx.q_b)], D.qB, .{ &m.K, L, s, rows, m.c.q_lora, nh, m.c.qHead() });
        if (L.w.full and last_pos >= m.c.i_topk) {
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.wq_b)], &wall_ms[@backingInt(StageIdx.wq_b)], D.dense, .{ &m.K, L, L.groups[ly.Layer.wq_b_g], Ref{ .buf = s.q_resid }, Ref{ .buf = s.qidx }, rows, m.c.q_lora, m.c.i_heads * m.c.i_dim });
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.idx_topk)], &wall_ms[@backingInt(StageIdx.idx_topk)], D.idxTop, .{ &m.K, L, s, t, ia, last_pos });
        }
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.absorb)], &wall_ms[@backingInt(StageIdx.absorb)], D.absorbOp, .{ &m.K, L, s, t, ma });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.attend)], &wall_ms[@backingInt(StageIdx.attend)], D.attendOp, .{ &m.K, L, s, t, ma, last_pos, fuse_uv });
        if (!fuse_uv) try runStep(queue, &gpu_ms[@backingInt(StageIdx.uv_g)], &wall_ms[@backingInt(StageIdx.uv_g)], D.dense, .{ &m.K, L, L.groups[ly.Layer.uv_g], Ref{ .buf = s.olat }, Ref{ .buf = s.o }, rows, nh * m.c.kv_lora, nh * m.c.v_dim });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.o_proj)], &wall_ms[@backingInt(StageIdx.o_proj)], D.oProj, .{ &m.K, L, s, p, rows, nh * m.c.v_dim, hid });
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.add_rms1)], &wall_ms[@backingInt(StageIdx.add_rms1)], D.addRmsPre, .{ p, s, L.k.at(L.w.post_norm), rows, hid, m.c.eps });
        if (L.w.mlp == .moe and L.fp8_moe != null) {
            const fa: kn.Fp8MoeArgs = .{ .rows = rows, .loc = loc, .hidden = hid, .topk = m.c.topk, .slots = gpu.slots, .experts = m.c.experts, .loc_blocks = m.c.localHalves(), .hid_blocks = hid / 128 };
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.route)], &wall_ms[@backingInt(StageIdx.route)], D.routeOp, .{ &m.K, L, s, &m.c, rows });
            if (rows == 1) {
                try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_gu)], &wall_ms[@backingInt(StageIdx.moe_gu)], D.moeGu1, .{ &m.K, L, s, fa });
                try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_down)], &wall_ms[@backingInt(StageIdx.moe_down)], D.moeDown1, .{ &m.K, L, s, fa, p });
            } else {
                try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_gu)], &wall_ms[@backingInt(StageIdx.moe_gu)], D.moeGuN, .{ &m.K, L, s, fa });
                try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_down)], &wall_ms[@backingInt(StageIdx.moe_down)], D.moeDnN, .{ &m.K, L, s, fa });
                try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_comb)], &wall_ms[@backingInt(StageIdx.moe_comb)], D.moeCombN, .{ &m.K, s, fa, p });
            }
        } else {
            try runStep(queue, &gpu_ms[@backingInt(StageIdx.moe_gu)], &wall_ms[@backingInt(StageIdx.moe_gu)], D.denseMlpOp, .{ &m.K, L, s, &m.c, rows, p });
        }
        try runStep(queue, &gpu_ms[@backingInt(StageIdx.add_rms2)], &wall_ms[@backingInt(StageIdx.add_rms2)], D.addRmsPre, .{ p, s, next, rows, hid, m.c.eps });
    }
    if (m.c.rank == 0) {
        const nl: f64 = @floatFromInt(m.layers.len);
        const names = [_][]const u8{ "x_g", "prep_idx", "q_b", "wq_b", "idx_topk", "absorb", "attend", "uv_g", "o_proj", "addRms1", "route", "moe_jobs", "moe_gu", "moe_down", "moe_comb", "addRms2" };
        var tot: f64 = 0;
        std.log.info("=== GLM DECODE PROFILE (rows={d}, last_pos={d}, layers={d}, tp={d}) ===", .{ rows, last_pos, m.layers.len, m.c.tp });
        for (names, 0..) |nm, idx| {
            if (gpu_ms[idx] == 0 and wall_ms[idx] == 0) continue;
            const per = gpu_ms[idx] * 1000.0 / nl;
            const wper = wall_ms[idx] * 1000.0 / nl;
            tot += per;
            std.log.info("  {s: <10}: gpu={d:6.1} us/L  wall={d:6.1} us/L  (78L={d:5.2} ms)", .{ nm, per, wper, per * 78.0 / 1000.0 });
        }
        std.log.info("  TOTAL     : gpu={d:6.1} us/L  (78L={d:5.2} ms, ~{d:5.1} tok/s)", .{ tot, tot * 78.0 / 1000.0, 1000.0 * @as(f64, @floatFromInt(rows)) / (tot * 78.0 / 1000.0) });
    }
}

/// One decoder layer for the window's rows (consecutive positions from pos0): the TP exchanges cover them all.
fn layer(m: *Model, L: *const ly.Layer, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32, table: mtl.Buffer, next: Ref) !void {
    const p = m.pair;
    const hid = m.c.hidden;
    try lp.attention(&m.K, L, s, q, e, &m.c, rows, pos0, table, .{ .buf = q.x }, p.part());
    e.barrier();
    p.addRms(e, .{ .buf = q.h }, L.k.at(L.w.post_norm), .{ .buf = q.x }, rows, hid, m.c.eps);
    e.barrier();
    const o = p.part();
    if (L.w.mlp == .moe) {
        try lp.route(&m.K, L, s, q, e, &m.c, rows, .{ .buf = q.x });
        try lp.experts(&m.K, L, q, e, &m.c, rows, .{ .buf = q.x }, o);
    } else try lp.denseMlp(&m.K, L, q, e, &m.c, rows, .{ .buf = q.x }, o);
    e.barrier();
    p.addRms(e, .{ .buf = q.h }, next, .{ .buf = q.x }, rows, hid, m.c.eps);
    e.barrier();
}

/// The window's rows (q.ids at q.rows' positions from pos0) through this stage's layers (with PP recv/send).
pub fn backbone(m: *Model, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32) !void {
    var cb: mtl.CommandBuffer = undefined;
    var enc = e;
    _ = try backboneTap(m, s, q, undefined, &cb, &enc, .serial, rows, pos0, null);
}

/// Prompt backbone with optional DFlash2 residual tap capture and concurrent 32-core ANE `fc` launch after layer 75.
pub fn backboneTap(m: *Model, s: *const gpu.Scratch, q: *const gpu.Prompt, queue: mtl.Queue, cb: *mtl.CommandBuffer, enc: *mtl.ComputeEncoder, dispatch: mtl.DispatchType, rows: u32, pos0: u32, df2: anytype) !?u64 {
    const hid = m.c.hidden;
    const v = &m.vocab;
    if (m.c.isFirstStage()) {
        m.embedTokens(enc.*, .{ .buf = q.ids }, .{ .buf = q.h }, rows);
    } else {
        m.pair.pipeRecv(enc.*, .{ .buf = q.h }, rows, hid);
    }
    enc.barrier();
    const L0 = &m.layers[0];
    kn.addRms(&m.K.k, enc.*, .{ .buf = q.h }, .{ .buf = q.h }, L0.k.at(L0.w.in_norm), m.glue(hid, hid), .{ .buf = q.x }, rows);
    enc.barrier();
    var ane_wait: ?u64 = null;
    for (m.layers, 0..) |*L, i| {
        const next = if (i + 1 < m.layers.len)
            m.layers[i + 1].k.at(m.layers[i + 1].w.in_norm)
        else if (m.c.isLastStage())
            v.k.at(v.norm)
        else
            L.k.at(L.w.in_norm);
        try layer(m, L, s, q, enc.*, rows, pos0, q.rows, next);
        if (@TypeOf(df2) != @TypeOf(null)) if (df2) |d| {
            const lid = m.c.layer0 + @as(u32, @intCast(i));
            d.tap(enc.*, lid, .{ .buf = q.h }, rows);
            if (lid == d.c.taps[d.c.taps.len - 1]) ane_wait = d.startAneFc(queue, cb, enc, dispatch, rows);
        };
    }
    if (!m.c.isLastStage()) {
        m.pair.pipeSend(enc.*, .{ .buf = q.h }, rows, hid);
        enc.barrier();
    }
    return ane_wait;
}

/// The MTP layer over the window's rows: tokens q.ids_mtp, hidden rows m.head.prev, positions pos0 on in q.rows_mtp.
pub fn draft(m: *Model, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, rows: u32, pos0: u32) !void {
    const L = &(m.mtp orelse return error.NoMtpHead);
    const t = L.w.mtp.?;
    const hid = m.c.hidden;
    m.embedTokens(e, .{ .buf = q.ids_mtp }, .{ .buf = q.h }, rows);
    e.barrier();
    kn.addRms(&m.K.k, e, .{ .buf = q.h }, .{ .buf = q.h }, L.k.at(t.enorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat }, rows);
    kn.addRms(&m.K.k, e, .{ .buf = m.head.prev }, .{ .buf = q.h }, L.k.at(t.hnorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat, .off = hid * 4 }, rows);
    e.barrier();
    if (t.eh_sharded) {
        const count_u4: u32 = (rows * hid) / 4;
        e.setPipeline(m.K.k.zero_f32);
        e.setBuffer(q.h, 0, 0);
        e.setValue(count_u4, 1);
        e.dispatchGroups(.{ .width = (count_u4 + 255) / 256 }, .{ .width = 256 });
        if (L.k.fast_dense and t.eh_bf16_off != 0) {
            try ly.pp.encodeDenseBf16Fast(&m.K.pp, e, .{ .buf = m.head.cat }, L.k.dense64(L.groups[ly.Layer.eh_g]), L.k.dense(L.groups[ly.Layer.eh_g]), m.pair.part(), rows, 2 * hid, hid, true);
        } else {
            try lp.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], .{ .buf = m.head.cat }, 2 * hid, .{ .buf = q.rot }, m.pair.part(), hid, rows);
        }
        e.barrier();
        m.pair.addRms(e, .{ .buf = q.h }, L.k.at(L.w.in_norm), .{ .buf = q.x }, rows, hid, m.c.eps);
        e.barrier();
    } else {
        try lp.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], .{ .buf = m.head.cat }, 2 * hid, .{ .buf = q.rot }, .{ .buf = q.h }, hid, rows);
        e.barrier();
        kn.addRms(&m.K.k, e, .{ .buf = q.h }, .{ .buf = q.h }, L.k.at(L.w.in_norm), m.glue(hid, hid), .{ .buf = q.x }, rows);
        e.barrier();
    }
    try layer(m, L, s, q, e, rows, pos0, q.rows_mtp, L.k.at(t.head_norm));
}
