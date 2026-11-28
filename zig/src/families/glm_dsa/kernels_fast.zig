//! Fused decode Metal pipelines (router+jobs, wp, attend+UV, FP8 dense/fused3/pack, merge_h4+UV).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = @import("../../core/exlr8_projection.zig").Ref;

pub const RouterFusedArgs = extern struct {
    rows: u32,
    experts: u32,
    hidden: u32,
    topk: u32,
    loc: u32,
    slots: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    scale: f32,
    pad2: f32 = 0,
    pad3: f32 = 0,
    pad4: f32 = 0,
};

pub const WpBf16Args = extern struct {
    rows: u32,
    cols: u32,
    hidden: u32,
    proj_stride: u32,
    wp_col: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

pub const EhPackArgs = extern struct {
    outs: u32,
    ins: u32,
    x_col: u32,
    x_stride: u32,
    rank: u32,
    tp_size: u32,
    slot_u2: u32,
    pad0: u32 = 0,
};

pub const MtpCatArgs = extern struct {
    width: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    eps: f32 = 1e-5,
    pad3: f32 = 0,
    pad4: f32 = 0,
    pad5: f32 = 0,
};

pub const MtpAcceptArgs = extern struct {
    rows: u32,
    slot: u32,
    base_pos: u32,
    depth: u32,
    i_topk: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

pub const MtpGateArgs = extern struct {
    step: u32,
    n_grids: u32 = 15,
    thresh: f32 = 0.26,
    pad0: f32 = 0,
    grids: [15][4]u32,
};

pub const CompactKvArgs = extern struct {
    slot: u32,
    base_pos: u32,
    slot_keys: u32,
    kept: u32,
    has_idx: u32 = 0,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    path: [8]u32 = @splat(0),
};

pub const MtpTreeAcceptArgs = extern struct {
    rows: u32,
    slot: u32,
    base_pos: u32,
    depth: u32,
    i_topk: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    parents: [8]i32 = @splat(-1),
};

pub const Fp8MatArgs = extern struct {
    outs: u32,
    ins: u32,
    rows: u32,
    x_stride: u32,
    y_stride: u32,
    y_col: u32,
    sb_cols: u32,
    pad0: u32 = 0,
    w_off: u64,
    s_off: u64,
};

pub const Fp8Mat3Args = extern struct {
    o0: u32,
    o1: u32,
    o2: u32,
    ins: u32,
    rows: u32,
    x_stride: u32,
    y_stride: u32,
    sb_cols: u32,
    yc0: u32,
    yc1: u32,
    yc2: u32,
    g0: u32,
    g1: u32,
    g2: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    w0: u64,
    s0: u64,
    w1: u64,
    s1: u64,
    w2: u64,
    s2: u64,
};

pub const Pipelines = struct {
    router_gemv: [8]mtl.Pipeline, // rows = 1..8
    router_route_1: mtl.Pipeline,
    router_route_jobs: [7]mtl.Pipeline, // rows = 2..8
    wp_bf16_decode: mtl.Pipeline,
    attend_uv_4s: mtl.Pipeline,
    attend_uv_2s: mtl.Pipeline,
    attend_uv_1s: mtl.Pipeline,
    attend_short_8sg: mtl.Pipeline,
    eh_q8_pack_1: mtl.Pipeline,
    eh_q8_pack_n: mtl.Pipeline,
    compact_kv: mtl.Pipeline,
    compact_kv_indirect: mtl.Pipeline,
    mtp_cat_rms: mtl.Pipeline,
    mtp_accept: mtl.Pipeline,
    mtp_tree_accept: mtl.Pipeline,
    mtp_gate: mtl.Pipeline,
    lm_bf16_argmax_1: mtl.Pipeline,
    lm_bf16_argmax_n: mtl.Pipeline,
    lm_q4_argmax_1: mtl.Pipeline,
    lm_q4_argmax_2: mtl.Pipeline,
    lm_q8_argmax_1: mtl.Pipeline,
    lm_q8_argmax_2: mtl.Pipeline,
    lm_argmax_reduce: mtl.Pipeline,
    lm_argmax_prob_reduce: mtl.Pipeline,
    fp8_gemv: [8]mtl.Pipeline, // rows = 1..8
    fp8_o_pack: [8]mtl.Pipeline, // rows = 1..8
    fp8_gemv3: [8]mtl.Pipeline, // rows = 1..8
    mla_merge_h4: mtl.Pipeline,
    mla_uv: [8]mtl.Pipeline, // rows = 1..8
    mla_absorb_u2: [8]mtl.Pipeline, // rows = 1..8
    cast_xh: mtl.Pipeline,
    fp8_moe_shared_gu: [7]mtl.Pipeline, // rows = 2..8
    fp8_moe_shared_down: [7]mtl.Pipeline, // rows = 2..8

    pub fn load(device: mtl.Device, opts: mtl.CompileOptions, glue: mtl.Library, fp8l: mtl.Library) !Pipelines {
        const lib = try mtl.Library.fromSource(device, sources.glm_dsa_decode_fast, opts);
        defer lib.deinit();
        const apex = try mtl.Library.fromSource(device, sources.glm_dsa_apex_fast, opts);
        defer apex.deinit();
        var p: Pipelines = undefined;
        for (0..8) |i| {
            var name: [32]u8 = undefined;
            p.router_gemv[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name, "glm_router_gemv_{d}", .{i + 1}), false);
        }
        p.router_route_1 = try mtl.Pipeline.init(device, lib, "glm_router_route_1", false);
        for (0..7) |i| {
            var name: [32]u8 = undefined;
            p.router_route_jobs[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name, "glm_router_route_jobs_{d}", .{i + 2}), false);
        }
        p.wp_bf16_decode = try mtl.Pipeline.init(device, lib, "glm_wp_bf16_decode", false);
        p.attend_uv_4s = try mtl.Pipeline.init(device, lib, "glm_mla_attend_uv_4s", false);
        p.attend_uv_2s = try mtl.Pipeline.init(device, lib, "glm_mla_attend_uv_2s", false);
        p.attend_uv_1s = try mtl.Pipeline.init(device, lib, "glm_mla_attend_uv_1s", false);
        p.attend_short_8sg = try mtl.Pipeline.init(device, lib, "glm_mla_attend_short_8sg", false);
        p.eh_q8_pack_1 = try mtl.Pipeline.init(device, lib, "glm_eh_q8_pack_1", false);
        p.eh_q8_pack_n = try mtl.Pipeline.init(device, lib, "glm_eh_q8_pack_n", false);
        p.compact_kv = try mtl.Pipeline.init(device, glue, "glm_mla_compact_kv", false);
        p.compact_kv_indirect = try mtl.Pipeline.init(device, glue, "glm_mla_compact_kv_indirect", false);
        p.mtp_cat_rms = try mtl.Pipeline.init(device, lib, "glm_mtp_cat_rms", false);
        p.mtp_accept = try mtl.Pipeline.init(device, lib, "glm_mtp_accept", false);
        p.mtp_tree_accept = try mtl.Pipeline.init(device, glue, "glm_mtp_tree_accept", false);
        p.mtp_gate = try mtl.Pipeline.init(device, lib, "glm_mtp_gate", false);
        p.lm_bf16_argmax_1 = try mtl.Pipeline.init(device, glue, "glm_lm_bf16_argmax_1", false);
        p.lm_bf16_argmax_n = try mtl.Pipeline.init(device, glue, "glm_lm_bf16_argmax_n", false);
        p.lm_q4_argmax_1 = try mtl.Pipeline.init(device, glue, "glm_lm_q4_argmax_1", false);
        p.lm_q4_argmax_2 = try mtl.Pipeline.init(device, glue, "glm_lm_q4_argmax_2", false);
        p.lm_q8_argmax_1 = try mtl.Pipeline.init(device, apex, "glm_lm_q8_argmax_1", false);
        p.lm_q8_argmax_2 = try mtl.Pipeline.init(device, apex, "glm_lm_q8_argmax_2", false);
        p.lm_argmax_reduce = try mtl.Pipeline.init(device, glue, "glm_lm_argmax_reduce", false);
        p.lm_argmax_prob_reduce = try mtl.Pipeline.init(device, glue, "glm_lm_argmax_prob_reduce", false);
        for (0..8) |i| {
            var nb: [32]u8 = undefined;
            p.fp8_gemv[i] = try mtl.Pipeline.init(device, apex, try std.fmt.bufPrint(&nb, "glm_fp8_gemv_{d}", .{i + 1}), false);
            p.fp8_o_pack[i] = try mtl.Pipeline.init(device, apex, try std.fmt.bufPrint(&nb, "glm_fp8_o_pack_{d}", .{i + 1}), false);
            p.fp8_gemv3[i] = try mtl.Pipeline.init(device, apex, try std.fmt.bufPrint(&nb, "glm_fp8_gemv3_{d}", .{i + 1}), false);
            p.mla_uv[i] = try mtl.Pipeline.init(device, apex, try std.fmt.bufPrint(&nb, "glm_mla_uv_{d}", .{i + 1}), false);
            p.mla_absorb_u2[i] = try mtl.Pipeline.init(device, apex, try std.fmt.bufPrint(&nb, "glm_mla_absorb_u2_{d}", .{i + 1}), false);
        }
        p.mla_merge_h4 = try mtl.Pipeline.init(device, apex, "glm_mla_merge_h4", false);
        p.cast_xh = try mtl.Pipeline.init(device, fp8l, "glm_cast_xh", false);
        for (0..7) |i| {
            var nb: [40]u8 = undefined;
            p.fp8_moe_shared_gu[i] = try mtl.Pipeline.init(device, fp8l, try std.fmt.bufPrint(&nb, "glm_fp8_moe_shared_gu_{d}", .{i + 2}), false);
            p.fp8_moe_shared_down[i] = try mtl.Pipeline.init(device, fp8l, try std.fmt.bufPrint(&nb, "glm_fp8_moe_shared_down_{d}", .{i + 2}), false);
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.router_gemv) |q| q.deinit();
        for (p.router_route_jobs) |q| q.deinit();
        for (p.fp8_gemv) |q| q.deinit();
        for (p.fp8_o_pack) |q| q.deinit();
        for (p.fp8_gemv3) |q| q.deinit();
        for (p.mla_uv) |q| q.deinit();
        for (p.mla_absorb_u2) |q| q.deinit();
        for (p.fp8_moe_shared_gu) |q| q.deinit();
        for (p.fp8_moe_shared_down) |q| q.deinit();
        inline for (.{
            p.router_route_1,   p.wp_bf16_decode,        p.attend_uv_4s,     p.attend_uv_2s,
            p.attend_uv_1s,     p.attend_short_8sg,      p.eh_q8_pack_1,     p.eh_q8_pack_n,
            p.compact_kv,       p.compact_kv_indirect,   p.mtp_cat_rms,      p.mtp_accept,
            p.mtp_tree_accept,  p.mtp_gate,              p.lm_bf16_argmax_1, p.lm_bf16_argmax_n,
            p.lm_q4_argmax_1,   p.lm_q4_argmax_2,        p.lm_q8_argmax_1,   p.lm_q8_argmax_2,
            p.lm_argmax_reduce, p.lm_argmax_prob_reduce, p.mla_merge_h4,     p.cast_xh,
        }) |q| q.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, i: usize, r: Ref) void {
    e.setBuffer(r.buf, r.off, i);
}

pub fn dispOr(e: mtl.ComputeEncoder, gate: ?mtl.Buffer, slot: usize, grid: mtl.Size, tg: mtl.Size) void {
    if (gate) |g| e.dispatchIndirect(g, slot * 16, tg) else e.dispatchGroups(grid, tg);
}

pub fn mtpGate(p: *const Pipelines, e: mtl.ComputeEncoder, picks: Ref, meta: Ref, disp: Ref, a: MtpGateArgs) void {
    e.setPipeline(p.mtp_gate);
    inline for (.{ picks, meta, disp }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
}

pub fn routerRoute1(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, bias: Ref, a: RouterFusedArgs, logits: Ref, done: Ref, pick: Ref, wts: Ref, gate: ?mtl.Buffer) void {
    _ = done;
    e.setPipeline(p.router_gemv[0]);
    inline for (.{ x, w }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 2);
    bind(e, 3, logits);
    dispOr(e, gate, 6, .{ .width = 128 }, .{ .width = 256 });
    e.barrier();
    e.setPipeline(p.router_route_1);
    inline for (.{ logits, bias }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 2);
    inline for (.{ pick, wts }, 3..) |r, i| bind(e, i, r);
    dispOr(e, gate, 14, .{ .width = 1 }, .{ .width = 32 });
}

pub fn routerGemv(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, a: RouterFusedArgs, logits: Ref) void {
    e.setPipeline(p.router_gemv[a.rows - 1]);
    inline for (.{ x, w }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 2);
    bind(e, 3, logits);
    e.dispatchGroups(.{ .width = 128 }, .{ .width = 256 });
}

pub fn routerRouteJobsOnly(p: *const Pipelines, e: mtl.ComputeEncoder, logits: Ref, bias: Ref, a: RouterFusedArgs, pick: Ref, wts: Ref, groups: Ref, members: Ref, wts9: Ref, dispatch: Ref, keep_meta: ?mtl.Buffer) void {
    e.setPipeline(p.router_route_jobs[a.rows - 2]);
    inline for (.{ logits, bias }, 0..) |r, i| bind(e, i, r);
    var ra = a;
    ra.pad0 = if (keep_meta != null) 1 else 0;
    e.setValue(ra, 2);
    inline for (.{ pick, wts, groups, members, wts9, dispatch }, 3..) |r, i| bind(e, i, r);
    if (keep_meta) |km| e.setBuffer(km, 0, 9) else bind(e, 9, dispatch);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
}

pub fn routerRouteJobs(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, bias: Ref, a: RouterFusedArgs, logits: Ref, done: Ref, pick: Ref, wts: Ref, groups: Ref, members: Ref, wts9: Ref, dispatch: Ref, keep_meta: ?mtl.Buffer) void {
    _ = done;
    routerGemv(p, e, x, w, a, logits);
    e.barrier();
    var ra = a;
    if (a.rows <= 5) ra.pad1 |= 2;
    routerRouteJobsOnly(p, e, logits, bias, ra, pick, wts, groups, members, wts9, dispatch, keep_meta);
}

pub fn castXh(p: *const Pipelines, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n4: u32) void {
    e.setPipeline(p.cast_xh);
    bind(e, 0, src);
    bind(e, 1, dst);
    e.setValue(n4, 2);
    e.dispatchGroups(.{ .width = (n4 + 255) / 256 }, .{ .width = 256 });
}

pub fn fp8MoeSharedGu(p: *const Pipelines, e: mtl.ComputeEncoder, xh: Ref, gate_w: mtl.Buffer, gate_s: mtl.Buffer, up_w: mtl.Buffer, up_s: mtl.Buffer, a: anytype, act: Ref) void {
    e.setPipeline(p.fp8_moe_shared_gu[a.rows - 2]);
    bind(e, 0, xh);
    inline for (.{ gate_w, gate_s, up_w, up_s }, 1..) |b, i| e.setBuffer(b, 0, i);
    e.setValue(a, 5);
    bind(e, 6, act);
    e.dispatchGroups(.{ .width = (a.loc + 3) / 4 }, .{ .width = 128 });
}

pub fn fp8MoeSharedDown(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, down_w: mtl.Buffer, down_s: mtl.Buffer, a: anytype, z: Ref) void {
    e.setPipeline(p.fp8_moe_shared_down[a.rows - 2]);
    bind(e, 0, act);
    inline for (.{ down_w, down_s }, 1..) |b, i| e.setBuffer(b, 0, i);
    e.setValue(a, 3);
    bind(e, 4, z);
    e.dispatchGroups(.{ .width = (a.hidden + 15) / 16 }, .{ .width = 256 });
}

pub fn wpBf16Decode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, a: WpBf16Args, proj: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.wp_bf16_decode);
    bind(e, 0, x);
    bind(e, 1, w);
    e.setValue(a, 2);
    bind(e, 3, proj);
    dispOr(e, gate, 4, .{ .width = a.cols }, .{ .width = 256 });
}

fn attendUvNs(pipe: mtl.Pipeline, mult: u32, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, cache: Ref, weights: mtl.Buffer, a: anytype, o_out: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(pipe);
    inline for (.{ qlat, qrope, rows, cache }, 0..) |r, i| bind(e, i, r);
    e.setBuffer(weights, 0, 4);
    e.setValue(a, 5);
    bind(e, 6, o_out);
    dispOr(e, gate, 11, .{ .width = a.heads * mult, .height = a.rows }, .{ .width = 256 });
}
pub fn attendUv4s(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, cache: Ref, weights: mtl.Buffer, a: anytype, o_out: Ref, gate: ?mtl.Buffer) void {
    attendUvNs(p.attend_uv_4s, 4, e, qlat, qrope, rows, cache, weights, a, o_out, gate);
}
pub fn attendUv2s(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, cache: Ref, weights: mtl.Buffer, a: anytype, o_out: Ref, gate: ?mtl.Buffer) void {
    attendUvNs(p.attend_uv_2s, 2, e, qlat, qrope, rows, cache, weights, a, o_out, gate);
}
pub fn attendUv1s(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, cache: Ref, weights: mtl.Buffer, a: anytype, o_out: Ref, gate: ?mtl.Buffer) void {
    attendUvNs(p.attend_uv_1s, 1, e, qlat, qrope, rows, cache, weights, a, o_out, gate);
}

pub fn attendShort8sg(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, cache: Ref, a: anytype, olat_h: Ref) void {
    e.setPipeline(p.attend_short_8sg);
    inline for (.{ qlat, qrope, rows, cache }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 4);
    bind(e, 5, olat_h);
    e.dispatchGroups(.{ .width = a.heads, .height = a.rows }, .{ .width = 256 });
}

pub fn ehQ8PackN(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, a: EhPackArgs, y: Ref, send_base: Ref, keep_meta: ?mtl.Buffer) void {
    e.setPipeline(p.eh_q8_pack_n);
    inline for (.{ x, w, s }, 0..) |r, i| bind(e, i, r);
    var ea = a;
    if (keep_meta != null) ea.pad0 |= 16;
    e.setValue(ea, 3);
    bind(e, 4, y);
    bind(e, 5, send_base);
    if (keep_meta) |km| e.setBuffer(km, 0, 6) else bind(e, 6, y);
    e.dispatchGroups(.{ .width = (a.outs + 7) / 8 }, .{ .width = 128 });
}

pub fn compactKv(p: *const Pipelines, e: mtl.ComputeEncoder, cache: Ref, idx_cache: Ref, a: CompactKvArgs) void {
    e.setPipeline(p.compact_kv);
    bind(e, 0, cache);
    bind(e, 1, idx_cache);
    e.setValue(a, 2);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 128 });
}

pub fn compactKvIndirect(p: *const Pipelines, e: mtl.ComputeEncoder, cache: Ref, idx_cache: Ref, meta: Ref, a: CompactKvArgs, disp: mtl.Buffer) void {
    e.setPipeline(p.compact_kv_indirect);
    inline for (.{ cache, idx_cache, meta }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    e.dispatchIndirect(disp, 0, .{ .width = 128 });
}

pub fn ehQ8Pack1(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, a: EhPackArgs, y: Ref, send_base: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.eh_q8_pack_1);
    inline for (.{ x, w, s }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    bind(e, 4, y);
    bind(e, 5, send_base);
    dispOr(e, gate, 0, .{ .width = (a.outs + 15) / 16 }, .{ .width = 128 });
}

pub fn mtpCatRms(p: *const Pipelines, e: mtl.ComputeEncoder, emb: Ref, prev: Ref, w_e: Ref, w_h: Ref, a: MtpCatArgs, cat: Ref, rows: u32, gate: ?mtl.Buffer) void {
    e.setPipeline(p.mtp_cat_rms);
    inline for (.{ emb, prev, w_e, w_h }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 4);
    bind(e, 5, cat);
    dispOr(e, gate, 14, .{ .width = rows }, .{ .width = 1024 });
}

pub fn mtpAccept(p: *const Pipelines, e: mtl.ComputeEncoder, win_ids: Ref, win_picks: Ref, a: MtpAcceptArgs, meta: Ref, d1_ids: Ref, d2_idx: Ref, d_rows: Ref) void {
    e.setPipeline(p.mtp_accept);
    bind(e, 0, win_ids);
    bind(e, 1, win_picks);
    e.setValue(a, 2);
    inline for (.{ meta, d1_ids, d2_idx, d_rows }, 3..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
}

pub fn mtpTreeAccept(p: *const Pipelines, e: mtl.ComputeEncoder, win_ids: Ref, win_picks: Ref, a: MtpTreeAcceptArgs, meta: Ref, disp: Ref, d1_ids: Ref, d1_idx: Ref, d2_idx: Ref, d_rows: Ref) void {
    e.setPipeline(p.mtp_tree_accept);
    bind(e, 0, win_ids);
    bind(e, 1, win_picks);
    e.setValue(a, 2);
    inline for (.{ meta, disp, d1_ids, d1_idx, d2_idx, d_rows }, 3..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
}

pub fn lmBf16Argmax(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, dims: [4]u32, part: Ref, n_tgs: u32) void {
    e.setPipeline(if (dims[3] == 1) p.lm_bf16_argmax_1 else p.lm_bf16_argmax_n);
    inline for (.{ x, w }, 0..) |r, i| bind(e, i, r);
    e.setValue(dims, 2);
    bind(e, 3, part);
    const th: usize = if (dims[3] == 1) 128 else 256;
    e.dispatchGroups(.{ .width = n_tgs }, .{ .width = th });
}

pub fn lmDraftArgmax(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, dims: [4]u32, part: Ref, n_tgs: u32, is_q8: bool, gate: ?mtl.Buffer) void {
    e.setPipeline(if (is_q8) (if (dims[3] == 2) p.lm_q8_argmax_2 else p.lm_q8_argmax_1) else (if (dims[3] == 2) p.lm_q4_argmax_2 else p.lm_q4_argmax_1));
    inline for (.{ x, w, s }, 0..) |r, i| bind(e, i, r);
    e.setValue(dims, 3);
    bind(e, 4, part);
    dispOr(e, gate, 9, .{ .width = n_tgs }, .{ .width = 256 });
}

pub fn lmArgmaxReduce(p: *const Pipelines, e: mtl.ComputeEncoder, part: Ref, n_tgs: u32, out: Ref, rows: u32) void {
    e.setPipeline(p.lm_argmax_reduce);
    bind(e, 0, part);
    e.setValue(n_tgs, 1);
    bind(e, 2, out);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
}

pub fn lmArgmaxProbReduce(p: *const Pipelines, e: mtl.ComputeEncoder, part: Ref, n_tgs: u32, out: Ref, rows: u32, gate: ?mtl.Buffer) void {
    e.setPipeline(p.lm_argmax_prob_reduce);
    bind(e, 0, part);
    e.setValue(n_tgs, 1);
    bind(e, 2, out);
    dispOr(e, gate, 14, .{ .width = rows }, .{ .width = 256 });
}

pub fn fp8Gemv(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: mtl.Buffer, a: Fp8MatArgs, y: Ref, gate: ?mtl.Buffer, slot: usize) void {
    e.setPipeline(p.fp8_gemv[a.rows - 1]);
    bind(e, 0, x);
    e.setBuffer(w, 0, 1);
    e.setValue(a, 2);
    bind(e, 3, y);
    const gx: usize = if (a.rows == 2 or a.rows == 3 or a.rows == 5) (a.outs + 3) / 4 else if (a.rows == 1 or a.rows == 4) (a.outs + 7) / 8 else (a.outs + 15) / 16;
    const th: usize = if (a.rows == 2 or a.rows == 3 or a.rows == 5) 64 else if (a.rows == 1 or a.rows == 4) 128 else 256;
    dispOr(e, gate, slot, .{ .width = gx }, .{ .width = th });
}

pub fn fp8Gemv3(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: mtl.Buffer, a: Fp8Mat3Args, y: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.fp8_gemv3[a.rows - 1]);
    bind(e, 0, x);
    e.setBuffer(w, 0, 1);
    e.setValue(a, 2);
    bind(e, 3, y);
    const th: usize = if (a.rows >= 3 and a.rows <= 5) 64 else if (a.rows <= 2) 128 else 256;
    dispOr(e, gate, 1, .{ .width = a.g0 + a.g1 + a.g2 }, .{ .width = th });
}

pub fn fp8OPack(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: mtl.Buffer, a: Fp8MatArgs, y: Ref, pack_info: [4]u32, send_base: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.fp8_o_pack[a.rows - 1]);
    bind(e, 0, x);
    e.setBuffer(w, 0, 1);
    e.setValue(a, 2);
    bind(e, 3, y);
    e.setValue(pack_info, 4);
    bind(e, 5, send_base);
    const gx: usize = if (a.rows >= 3 and a.rows <= 5) (a.outs + 3) / 4 else (a.outs + 15) / 16;
    const th: usize = if (a.rows >= 3 and a.rows <= 5) 64 else if (a.rows <= 2) 128 else 256;
    dispOr(e, gate, 5, .{ .width = gx }, .{ .width = th });
}

pub fn mlaAbsorbU2(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, weights: mtl.Buffer, rows: Ref, a: anytype, qlat: Ref, qrope: Ref, inv: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.mla_absorb_u2[a.rows - 1]);
    bind(e, 0, q);
    e.setBuffer(weights, 0, 1);
    bind(e, 2, rows);
    e.setValue(a, 3);
    inline for (.{ qlat, qrope, inv }, 4..) |r, i| bind(e, i, r);
    const gx: usize = if (a.rows >= 2) 16 else 8;
    dispOr(e, gate, 13, .{ .width = gx, .height = a.heads }, .{ .width = 96 });
}

pub fn mlaMergeH4(p: *const Pipelines, e: mtl.ComputeEncoder, partial: Ref, a: anytype, olat_h: Ref) void {
    e.setPipeline(p.mla_merge_h4);
    bind(e, 0, partial);
    e.setValue(a, 1);
    bind(e, 2, olat_h);
    e.dispatchGroups(.{ .width = a.heads, .height = a.rows }, .{ .width = 128 });
}

pub fn mlaUv(p: *const Pipelines, e: mtl.ComputeEncoder, olat_h: Ref, weights: mtl.Buffer, a: anytype, o_out: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.mla_uv[a.rows - 1]);
    bind(e, 0, olat_h);
    e.setBuffer(weights, 0, 1);
    e.setValue(a, 2);
    bind(e, 3, o_out);
    dispOr(e, gate, 12, .{ .width = a.heads * 16 }, .{ .width = 256 });
}
