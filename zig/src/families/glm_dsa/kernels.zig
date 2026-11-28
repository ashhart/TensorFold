//! GLM-5.3's own Metal kernels (MLA, DSA indexer, fp32 glue, FP8 block-128 MoE) and core route, loaded once.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const route = @import("../../core/moe_route.zig");
pub const fast = @import("kernels_fast.zig");
pub const TreeArgs = @import("tree.zig").TreeArgs;

pub const Ref = @import("../../core/exlr8_projection.zig").Ref;

/// A window row: its cache slot, position, and its key list (offset, count) in the keys buffer.
pub const Row = extern struct { slot: u32, pos: u32, list: u32, nkeys: u32 = 0 };

pub const MlaArgs = extern struct {
    rows: u32,
    heads: u32,
    head0: u32 = 0,
    slot_keys: u32,
    proj_stride: u32,
    q_stride: u32,
    pad2: u32 = 0,
    kvb_in_blocks: u32 = 4,
    eps: f32 = 1e-6,
    scale: f32 = 0.0625,
    pad0: f32 = 0,
    pad1: f32 = 0,
    kvb: u64,
    kvb_scale: u64,
    tree: TreeArgs = .{},
};

pub const IndexArgs = extern struct {
    rows: u32,
    slot_keys: u32,
    proj_stride: u32,
    k_col: u32,
    w_col: u32,
    q_stride: u32,
    score_stride: u32,
    pad0: u32 = 0,
    eps: f32 = 1e-6,
    key_scale: f32,
    head_scale: f32,
    pad1: f32 = 0,
    tree: TreeArgs = .{},
};

pub const GlueArgs = extern struct {
    width: u32,
    x_stride: u32,
    y_stride: u32,
    parts: u32 = 0,
    part_stride: u32 = 0,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    eps: f32 = 1e-5,
    pad3: f32 = 0,
    pad4: f32 = 0,
    pad5: f32 = 0,
};

/// One expert's tables on this Mac for the EXLR8 jobs kernel (kernels/metal/glm_dsa/moe_jobs.metal).
pub const ExpertDesc = extern struct {
    gate_k: [16]u32 = @splat(0),
    gate_base: [16]u32 = @splat(0),
    gate_chan: [16]u32 = @splat(0),
    gate_block: [16]u32 = @splat(0),
    up_k: [16]u32 = @splat(0),
    up_base: [16]u32 = @splat(0),
    up_chan: [16]u32 = @splat(0),
    up_block: [16]u32 = @splat(0),
    down_k: u32 = 0,
    down_base: u32 = 0,
    down_suh: u32 = 0,
    down_block: u32 = 0,
};

pub const JobsArgs = extern struct { rows: u32, halves: u32, topk: u32, slots: u32, gu_cap: u32, pad0: u32 = 0, pad1: u32 = 0, pad2: u32 = 0 };
pub const PromptJobsArgs = extern struct { rows: u32, halves: u32, topk: u32, slots: u32, gu_cap: u32, experts: u32, pad0: u32 = 0, pad1: u32 = 0 };

pub const Fp8MoeArgs = extern struct {
    rows: u32,
    loc: u32,
    hidden: u32,
    topk: u32,
    slots: u32,
    experts: u32,
    loc_blocks: u32,
    hid_blocks: u32,
};

pub const PromptRouteArgs = extern struct { rows: u32, experts: u32, topk: u32, scale: f32 };
pub const RouterBf16Args = extern struct { rows: u32, experts: u32, hidden: u32, pad: u32 = 0 };

pub const splits = 16;
const capacities = [_]usize{ 1, 2, 4, 8, 16 };
const fp8_caps = [_]usize{ 2, 3, 4, 5, 6, 7, 8, 16 };

pub const Pipelines = struct {
    prep: mtl.Pipeline,
    prep_full: mtl.Pipeline,
    absorb: [capacities.len]mtl.Pipeline,
    absorb_prompt: mtl.Pipeline,
    attend: mtl.Pipeline,
    attend_decode: mtl.Pipeline,
    attend_flash: mtl.Pipeline,
    attend_flash_16h: mtl.Pipeline,
    attend_flash_decode: mtl.Pipeline,
    attend_flash_decode_16h: mtl.Pipeline,
    attend_flash_decode_32h: mtl.Pipeline,
    merge: mtl.Pipeline,
    merge_uv: mtl.Pipeline,
    index_key: mtl.Pipeline,
    index_scores_block: mtl.Pipeline,
    index_scores_dec_128: mtl.Pipeline,
    index_scores_prompt: mtl.Pipeline,
    topk: mtl.Pipeline,
    embed: mtl.Pipeline,
    embed_bf16: mtl.Pipeline,
    add_rms: mtl.Pipeline,
    silu_mul: mtl.Pipeline,
    gather: mtl.Pipeline,
    fast_p: fast.Pipelines,
    select: mtl.Pipeline,
    route_prompt: mtl.Pipeline,
    router_bf16_decode: mtl.Pipeline,
    zero_f32: mtl.Pipeline,
    moe_jobs: mtl.Pipeline,
    moe_jobs_prompt: mtl.Pipeline,
    fp8_moe_jobs_direct: mtl.Pipeline,
    fp8_moe_gu1_fast: mtl.Pipeline,
    fp8_moe_down_combine1_fast: mtl.Pipeline,
    fp8_moe_gu: [fp8_caps.len]mtl.Pipeline,
    fp8_moe_down: [fp8_caps.len]mtl.Pipeline,
    fp8_moe_combine: mtl.Pipeline,
    fp8_moe_combine_pack_bf16: mtl.Pipeline,
    fp8_moe_prompt_jobs: mtl.Pipeline,
    fp8_moe_prompt_gu: mtl.Pipeline,
    fp8_moe_prompt_down: mtl.Pipeline,
    fp8_moe_prompt_down_atomic: mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, experts: u32, hidden: u32, topk: u32) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const opts = mtl.CompileOptions.mlx();
        const mla = try mtl.Library.fromSource(device, sources.glm_dsa_mla, opts);
        defer mla.deinit();
        const mlap = try mtl.Library.fromSource(device, sources.glm_dsa_mla_prompt, opts);
        defer mlap.deinit();
        const idx = try mtl.Library.fromSource(device, sources.glm_dsa_index, opts);
        defer idx.deinit();
        const glue = try mtl.Library.fromSource(device, sources.glm_dsa_glue, opts);
        defer glue.deinit();
        const jl = try mtl.Library.fromSource(device, sources.glm_dsa_moe_jobs, opts);
        defer jl.deinit();
        const fp8l = try mtl.Library.fromSource(device, sources.glm_dsa_fp8_moe, opts);
        defer fp8l.deinit();
        const fp8p_raw = try frags.source(device, gpa, sources.glm_dsa_fp8_moe_prompt);
        defer gpa.free(fp8p_raw);
        const fp8p_src = if (!device.tensorUnits())
            try std.mem.concat(gpa, u8, &.{ "#define TF_GLM_M3_FAST 1\n", fp8p_raw })
        else
            try gpa.dupe(u8, fp8p_raw);
        defer gpa.free(fp8p_src);
        const fp8pl = try mtl.Library.fromSource(device, fp8p_src, opts);
        defer fp8pl.deinit();
        const src = try route.source(gpa, .{ .hidden = hidden, .experts = experts, .topk = topk });
        defer gpa.free(src);
        const rl = try mtl.Library.fromSource(device, src, opts);
        defer rl.deinit();
        var p: Pipelines = undefined;
        p.prep = try mtl.Pipeline.init(device, mla, "glm_mla_prep", false);
        p.prep_full = try mtl.Pipeline.init(device, mla, "glm_mla_prep_full", false);
        for (capacities, &p.absorb) |c, *q| {
            var name: [32]u8 = undefined;
            q.* = try mtl.Pipeline.init(device, mla, try std.fmt.bufPrint(&name, "glm_mla_absorb_{d}", .{c}), false);
        }
        p.absorb_prompt = try mtl.Pipeline.init(device, mlap, "glm_mla_absorb_prompt_mma", false);
        p.attend = try mtl.Pipeline.init(device, mla, "glm_mla_attend", false);
        p.attend_decode = try mtl.Pipeline.init(device, mla, "glm_mla_attend_decode", false);
        p.attend_flash = try mtl.Pipeline.init(device, mla, "glm_mla_attend_flash", false);
        p.attend_flash_16h = try mtl.Pipeline.init(device, mlap, "glm_mla_attend_flash_16h", false);
        p.attend_flash_decode = try mtl.Pipeline.init(device, mla, "glm_mla_attend_flash_decode", false);
        p.attend_flash_decode_16h = try mtl.Pipeline.init(device, mlap, "glm_mla_attend_flash_decode_16h", false);
        p.attend_flash_decode_32h = try mtl.Pipeline.init(device, mlap, "glm_mla_attend_flash_decode_32h", false);
        p.merge = try mtl.Pipeline.init(device, mla, "glm_mla_merge", false);
        p.merge_uv = try mtl.Pipeline.init(device, mlap, "glm_mla_merge_uv", false);
        p.index_key = try mtl.Pipeline.init(device, idx, "glm_index_key", false);
        p.index_scores_block = try mtl.Pipeline.init(device, idx, "glm_index_scores_block", false);
        p.index_scores_dec_128 = try mtl.Pipeline.init(device, idx, "glm_index_scores_dec_128", false);
        p.index_scores_prompt = try mtl.Pipeline.init(device, idx, "glm_index_scores_prompt", false);
        p.topk = try mtl.Pipeline.init(device, idx, "glm_index_topk", false);
        p.embed = try mtl.Pipeline.init(device, glue, "glm_embed", false);
        p.embed_bf16 = try mtl.Pipeline.init(device, glue, "glm_embed_bf16", false);
        p.add_rms = try mtl.Pipeline.init(device, glue, "glm_add_rms", false);
        p.silu_mul = try mtl.Pipeline.init(device, glue, "glm_silu_mul", false);
        p.gather = try mtl.Pipeline.init(device, glue, "glm_gather", false);
        p.fast_p = try fast.Pipelines.load(device, opts, glue, fp8l);
        p.select = try mtl.Pipeline.init(device, rl, route.names[1], false);
        p.route_prompt = try mtl.Pipeline.init(device, fp8pl, "glm_route_prompt", false);
        p.router_bf16_decode = try mtl.Pipeline.init(device, fp8pl, "glm_router_bf16_decode", false);
        p.zero_f32 = try mtl.Pipeline.init(device, fp8pl, "glm_zero_f32", false);
        p.moe_jobs = try mtl.Pipeline.init(device, jl, "glm_moe_jobs", false);
        p.moe_jobs_prompt = try mtl.Pipeline.init(device, jl, "glm_moe_jobs_prompt", false);
        p.fp8_moe_jobs_direct = try mtl.Pipeline.init(device, fp8l, "glm_fp8_moe_jobs_direct", false);
        p.fp8_moe_gu1_fast = try mtl.Pipeline.init(device, fp8l, "glm_fp8_moe_gu_silu_1_fast", false);
        p.fp8_moe_down_combine1_fast = try mtl.Pipeline.init(device, fp8l, "glm_fp8_moe_down_combine_1_fast", false);
        for (fp8_caps, 0..) |c, i| {
            var name: [48]u8 = undefined;
            p.fp8_moe_gu[i] = try mtl.Pipeline.init(device, fp8l, try std.fmt.bufPrint(&name, "glm_fp8_moe_gu_silu_{d}", .{c}), false);
            p.fp8_moe_down[i] = try mtl.Pipeline.init(device, fp8l, try std.fmt.bufPrint(&name, "glm_fp8_moe_down_{d}", .{c}), false);
        }
        p.fp8_moe_combine = try mtl.Pipeline.init(device, fp8l, "glm_fp8_moe_combine", false);
        p.fp8_moe_combine_pack_bf16 = try mtl.Pipeline.init(device, fp8l, "glm_fp8_moe_combine_pack_bf16", false);
        p.fp8_moe_prompt_jobs = try mtl.Pipeline.init(device, fp8pl, "glm_fp8_moe_prompt_jobs", false);
        p.fp8_moe_prompt_gu = try mtl.Pipeline.init(device, fp8pl, "glm_fp8_moe_prompt_gu_silu", false);
        p.fp8_moe_prompt_down = try mtl.Pipeline.init(device, fp8pl, "glm_fp8_moe_prompt_down", false);
        p.fp8_moe_prompt_down_atomic = try mtl.Pipeline.init(device, fp8pl, "glm_fp8_moe_prompt_down_atomic", false);
        if (p.topk.maxThreads() < 1024 or p.moe_jobs_prompt.maxThreads() < 1024) return error.UnsupportedGlmPipeline;
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.absorb) |q| q.deinit();
        for (p.fp8_moe_gu) |q| q.deinit();
        for (p.fp8_moe_down) |q| q.deinit();
        p.fast_p.deinit();
        inline for (.{
            p.prep,                       p.prep_full,           p.absorb_prompt,             p.attend,                  p.attend_decode,       p.attend_flash,
            p.attend_flash_16h,           p.attend_flash_decode, p.attend_flash_decode_16h,   p.attend_flash_decode_32h, p.merge,               p.merge_uv,
            p.index_key,                  p.index_scores_block,  p.index_scores_dec_128,      p.index_scores_prompt,     p.topk,                p.embed,
            p.embed_bf16,                 p.add_rms,             p.silu_mul,                  p.gather,                  p.select,              p.route_prompt,
            p.router_bf16_decode,         p.zero_f32,            p.moe_jobs,                  p.moe_jobs_prompt,         p.fp8_moe_jobs_direct, p.fp8_moe_gu1_fast,
            p.fp8_moe_down_combine1_fast, p.fp8_moe_combine,     p.fp8_moe_combine_pack_bf16, p.fp8_moe_prompt_jobs,     p.fp8_moe_prompt_gu,   p.fp8_moe_prompt_down,
            p.fp8_moe_prompt_down_atomic,
        }) |q| q.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, i: usize, r: Ref) void {
    e.setBuffer(r.buf, r.off, i);
}

fn capacity(rows: u32) usize {
    for (capacities, 0..) |c, i| if (rows <= c) return i;
    unreachable;
}

fn fp8Cap(rows: u32) usize {
    for (fp8_caps, 0..) |c, i| if (rows <= c) return i;
    return fp8_caps.len - 1;
}

pub fn prep(p: *const Pipelines, e: mtl.ComputeEncoder, proj: Ref, w_qa: Ref, w_kva: Ref, inv: Ref, rows: Ref, a: MlaArgs, q_resid: Ref, cache: Ref) void {
    e.setPipeline(p.prep);
    inline for (.{ proj, w_qa, w_kva, inv, rows }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, q_resid);
    bind(e, 7, cache);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 512 });
}

pub fn prepFull(p: *const Pipelines, e: mtl.ComputeEncoder, proj: Ref, w_qa: Ref, w_kva: Ref, inv: Ref, rows: Ref, a: MlaArgs, q_resid: Ref, cache: Ref, w_ik: Ref, b_ik: Ref, idx_cache: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.prep_full);
    inline for (.{ proj, w_qa, w_kva, inv, rows }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    inline for (.{ q_resid, cache, w_ik, b_ik, idx_cache }, 6..) |r, i| bind(e, i, r);
    fast.dispOr(e, gate, 14, .{ .width = a.rows }, .{ .width = 512 });
}

pub fn absorb(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, weights: mtl.Buffer, rows: Ref, a: MlaArgs, qlat: Ref, qrope: Ref, inv: Ref) void {
    return absorbGated(p, e, q, weights, rows, a, qlat, qrope, inv, null);
}

pub fn absorbGated(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, weights: mtl.Buffer, rows: Ref, a: MlaArgs, qlat: Ref, qrope: Ref, inv: Ref, gate: ?mtl.Buffer) void {
    if (a.rows >= 1 and a.rows <= 8) {
        fast.mlaAbsorbU2(&p.fast_p, e, q, weights, rows, a, qlat, qrope, inv, gate);
        return;
    }
    e.setPipeline(p.absorb[capacity(a.rows)]);
    bind(e, 0, q);
    e.setBuffer(weights, 0, 1);
    bind(e, 2, rows);
    e.setValue(a, 3);
    inline for (.{ qlat, qrope, inv }, 4..) |r, i| bind(e, i, r);
    fast.dispOr(e, gate, 13, .{ .width = 8, .height = a.heads }, .{ .width = 256 });
}

pub fn absorbPrompt(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, weights: mtl.Buffer, rows: Ref, a: MlaArgs, qlat: Ref, qrope: Ref, inv: Ref) void {
    e.setPipeline(p.absorb_prompt);
    bind(e, 0, q);
    e.setBuffer(weights, 0, 1);
    bind(e, 2, rows);
    e.setValue(a, 3);
    inline for (.{ qlat, qrope, inv }, 4..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 8, .height = a.heads, .depth = (a.rows + 31) / 32 }, .{ .width = 128 });
}

pub fn attendFlash(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, keys: Ref, cache: Ref, a: MlaArgs, olat: Ref) void {
    const use16 = a.heads >= 16 and (a.heads % 16 == 0);
    e.setPipeline(if (use16) p.attend_flash_16h else p.attend_flash);
    inline for (.{ qlat, qrope, rows, keys, cache }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, olat);
    e.dispatchGroups(.{ .width = if (use16) a.heads / 16 else a.heads / 8, .height = a.rows }, .{ .width = 256 });
}

pub fn routerBf16Decode(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, a: RouterBf16Args, logits: Ref) void {
    e.setPipeline(p.router_bf16_decode);
    bind(e, 0, x);
    bind(e, 1, w);
    e.setValue(a, 2);
    bind(e, 3, logits);
    e.dispatchGroups(.{ .width = a.experts }, .{ .width = 64 });
}

pub fn routePrompt(p: *const Pipelines, e: mtl.ComputeEncoder, logits: Ref, bias: Ref, a: PromptRouteArgs, pick: Ref, wts: Ref) void {
    e.setPipeline(p.route_prompt);
    bind(e, 0, logits);
    bind(e, 1, bias);
    e.setValue(a, 2);
    bind(e, 3, pick);
    bind(e, 4, wts);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 32 });
}

pub fn attendDecode(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, keys: Ref, cache: Ref, a: MlaArgs, partial: Ref, olat: Ref, last_pos: u32, weights_uv: ?mtl.Buffer, o_out: ?Ref, gate: ?mtl.Buffer) bool {
    if (a.rows >= 1 and a.rows <= 2 and last_pos < 256 and a.heads >= 8) if (weights_uv) |wuv| if (o_out) |oo| {
        fast.attendUv4s(&p.fast_p, e, qlat, qrope, rows, cache, wuv, a, oo, gate);
        return true;
    };
    if (a.rows >= 3 and a.rows <= 8 and last_pos < 256 and a.heads >= 8) if (weights_uv) |wuv| if (o_out) |oo| {
        fast.attendShort8sg(&p.fast_p, e, qlat, qrope, rows, cache, a, olat);
        e.barrier();
        fast.mlaUv(&p.fast_p, e, olat, wuv, a, oo, gate);
        return true;
    };
    const min_pos = if (a.tree.base == 0xFFFFFFFE) last_pos else if (a.tree.base != 0xFFFFFFFF) a.tree.base else (last_pos -| (a.rows - 1));
    const straddle_256 = a.rows >= 2 and a.rows <= 8 and last_pos >= 256 and min_pos < 256 and a.heads >= 8 and weights_uv != null and o_out != null;
    if (straddle_256) {
        var sa = a;
        sa.pad2 = 1;
        fast.attendShort8sg(&p.fast_p, e, qlat, qrope, rows, cache, sa, olat);
    }
    const use_flash = last_pos >= 32 and a.heads >= 8 and (a.heads % 8 == 0);
    const max_ns: u32 = @min(@as(u32, 64), (16 * splits) / @max(@as(u32, 1), a.rows));
    const ns: u32 = if (last_pos < 32) 1 else if (use_flash) @min(max_ns, if (last_pos < 128) @as(u32, 8) else @as(u32, 16)) else if (last_pos < 512) 8 else splits;
    var ma = a;
    ma.pad2 = ns;
    if (straddle_256) ma.pad0 = 1.0;
    e.setPipeline(if (use_flash) p.attend_flash_decode else p.attend_decode);
    inline for (.{ qlat, qrope, rows, keys, cache }, 0..) |r, i| bind(e, i, r);
    e.setValue(ma, 5);
    bind(e, 6, if (ns == 1) olat else partial);
    const gx = if (use_flash) a.heads / 8 else (a.heads + 7) / 8;
    fast.dispOr(e, gate, 11, .{ .width = gx, .height = ns, .depth = a.rows }, .{ .width = 256 });
    if (ns == 1) return false;
    e.barrier();
    if (a.rows >= 1 and a.rows <= 8) if (weights_uv) |wuv| if (o_out) |oo| {
        fast.mlaMergeH4(&p.fast_p, e, partial, ma, olat);
        e.barrier();
        fast.mlaUv(&p.fast_p, e, olat, wuv, ma, oo, gate);
        return true;
    };
    if (a.heads * a.rows >= 16) if (weights_uv) |wuv| if (o_out) |oo| {
        e.setPipeline(p.merge_uv);
        bind(e, 0, partial);
        e.setBuffer(wuv, 0, 1);
        e.setValue(ma, 2);
        bind(e, 3, oo);
        e.dispatchGroups(.{ .width = a.heads * 16, .height = a.rows }, .{ .width = 128 });
        return true;
    };
    e.setPipeline(p.merge);
    bind(e, 0, partial);
    bind(e, 1, rows);
    e.setValue(ma, 2);
    bind(e, 3, olat);
    e.dispatchGroups(.{ .width = a.heads, .height = a.rows }, .{ .width = 128 });
    return false;
}

pub fn attend(p: *const Pipelines, e: mtl.ComputeEncoder, qlat: Ref, qrope: Ref, rows: Ref, keys: Ref, cache: Ref, a: MlaArgs, partial: Ref, olat: Ref) void {
    e.setPipeline(p.attend);
    inline for (.{ qlat, qrope, rows, keys, cache }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, partial);
    e.dispatchGroups(.{ .width = (a.heads + 3) / 4, .height = splits, .depth = a.rows }, .{ .width = 128 });
    e.barrier();
    e.setPipeline(p.merge);
    bind(e, 0, partial);
    bind(e, 1, rows);
    e.setValue(a, 2);
    bind(e, 3, olat);
    e.dispatchGroups(.{ .width = a.heads, .height = a.rows }, .{ .width = 128 });
}

pub fn indexKey(p: *const Pipelines, e: mtl.ComputeEncoder, proj: Ref, w: Ref, b: Ref, inv: Ref, rows: Ref, a: IndexArgs, cache: Ref) void {
    e.setPipeline(p.index_key);
    inline for (.{ proj, w, b, inv, rows }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, cache);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 32 });
}

pub fn indexScoresBlock(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, proj: Ref, cache: Ref, rows: Ref, a: IndexArgs, score: Ref, inv: Ref, last_pos: u32, gate: ?mtl.Buffer) void {
    const prompt_mode = (a.pad0 != 0) and (a.rows >= 2);
    const dec_128 = !prompt_mode and last_pos >= 4096;
    e.setPipeline(if (prompt_mode) p.index_scores_prompt else if (dec_128) p.index_scores_dec_128 else p.index_scores_block);
    inline for (.{ q, proj, cache, rows }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 4);
    bind(e, 5, score);
    bind(e, 6, inv);
    if (prompt_mode) {
        e.dispatchGroups(.{ .width = last_pos / 512 + 1, .height = (a.rows + 1) / 2 }, .{ .width = 256 });
    } else if (dec_128) {
        fast.dispOr(e, gate, 10, .{ .width = last_pos / 128 + 1, .height = a.rows }, .{ .width = 256 });
    } else {
        fast.dispOr(e, gate, 10, .{ .width = last_pos / 32 + 1, .height = a.rows }, .{ .width = 256 });
    }
}

pub fn indexTopkOnly(p: *const Pipelines, e: mtl.ComputeEncoder, rows: Ref, a: IndexArgs, score: Ref, keys: Ref, gate: ?mtl.Buffer) void {
    e.setPipeline(p.topk);
    bind(e, 0, score);
    bind(e, 1, rows);
    e.setValue(a, 2);
    bind(e, 3, keys);
    fast.dispOr(e, gate, 14, .{ .width = a.rows }, .{ .width = 1024 });
}

pub fn indexTopk(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, proj: Ref, cache: Ref, rows: Ref, a: IndexArgs, score: Ref, inv: Ref, keys: Ref, last_pos: u32) void {
    if (last_pos >= 2048) indexScoresBlock(p, e, q, proj, cache, rows, a, score, inv, last_pos, null);
    e.barrier();
    indexTopkOnly(p, e, rows, a, score, keys, null);
}

pub fn embed(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, fp8: Ref, scale: Ref, a: GlueArgs, h: Ref, rows: u32, gate: ?mtl.Buffer) void {
    e.setPipeline(p.embed);
    inline for (.{ ids, fp8, scale }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    bind(e, 4, h);
    fast.dispOr(e, gate, 14, .{ .width = rows }, .{ .width = 256 });
}

pub fn embedBf16(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, table: Ref, a: GlueArgs, h: Ref, rows: u32, gate: ?mtl.Buffer) void {
    e.setPipeline(p.embed_bf16);
    bind(e, 0, ids);
    bind(e, 1, table);
    e.setValue(a, 2);
    bind(e, 3, h);
    const th: usize = if (((a.width | a.y_stride) & 3) == 0 and a.width >= 2048 and p.embed_bf16.maxThreads() >= 512) 512 else 256;
    fast.dispOr(e, gate, 14, .{ .width = rows }, .{ .width = th });
}

pub fn addRms(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, add: Ref, w: Ref, a: GlueArgs, y: Ref, rows: u32) void {
    e.setPipeline(p.add_rms);
    inline for (.{ h, add, w }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    bind(e, 4, y);
    const th: usize = if (((a.width | a.x_stride | a.y_stride | a.part_stride) & 3) == 0 and a.width >= 4096 and p.add_rms.maxThreads() >= 1024) 1024 else 256;
    e.dispatchGroups(.{ .width = rows }, .{ .width = th });
}

pub fn siluMul(p: *const Pipelines, e: mtl.ComputeEncoder, gu: Ref, a: GlueArgs, act: Ref, rows: u32) void {
    e.setPipeline(p.silu_mul);
    bind(e, 0, gu);
    e.setValue(a, 1);
    bind(e, 2, act);
    e.dispatchGroups(.{ .width = (a.width + 255) / 256, .height = rows }, .{ .width = 256 });
}

pub fn gather(p: *const Pipelines, e: mtl.ComputeEncoder, src: Ref, idx: Ref, width: u32, src_stride: u32, dst_stride: u32, dst: Ref, rows: u32) void {
    gatherGated(p, e, src, idx, width, src_stride, dst_stride, dst, rows, null);
}
pub fn gatherGated(p: *const Pipelines, e: mtl.ComputeEncoder, src: Ref, idx: Ref, width: u32, src_stride: u32, dst_stride: u32, dst: Ref, rows: u32, gate: ?mtl.Buffer) void {
    e.setPipeline(p.gather);
    bind(e, 0, src);
    bind(e, 1, idx);
    e.setValue([4]u32{ width, src_stride, dst_stride, 0 }, 2);
    bind(e, 3, dst);
    const th: usize = if (((width | src_stride | dst_stride) & 3) == 0 and width >= 2048 and p.gather.maxThreads() >= 512) 512 else 256;
    fast.dispOr(e, gate, 14, .{ .width = rows }, .{ .width = th });
}

pub fn moeJobs(p: *const Pipelines, e: mtl.ComputeEncoder, route_out: [9]mtl.Buffer, desc: Ref, a: JobsArgs, jobs: Ref, members: Ref, wts: Ref, sv: Ref, suh: Ref, dispatch: Ref) void {
    e.setPipeline(p.moe_jobs);
    e.setBuffer(route_out[2], 0, 0);
    e.setBuffer(route_out[3], 0, 1);
    e.setBuffer(route_out[4], 0, 2);
    e.setBuffer(route_out[0], 0, 3);
    e.setBuffer(route_out[1], 0, 4);
    bind(e, 5, desc);
    e.setValue(a, 6);
    inline for (.{ jobs, members, wts, sv, suh, dispatch }, 7..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
}

pub fn moeJobsPrompt(p: *const Pipelines, e: mtl.ComputeEncoder, pick: Ref, wts: Ref, desc: Ref, a: PromptJobsArgs, jobs: Ref, members: Ref, wts9: Ref, sv: Ref, suh: Ref, dispatch: Ref) void {
    e.setPipeline(p.moe_jobs_prompt);
    inline for (.{ pick, wts, desc }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    inline for (.{ jobs, members, wts9, sv, suh, dispatch }, 4..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 1024 });
}

/// FP8 MoE decode path (1-16 rows): r=1 fused DC+pack, r=2..8 zero-barrier half4 Xh GU, r=9..16 TG-cached GU.
pub fn fp8MoeDecode(p: *const Pipelines, e: mtl.ComputeEncoder, route_out: [9]mtl.Buffer, x: Ref, xh: Ref, gate_w: mtl.Buffer, gate_s: mtl.Buffer, up_w: mtl.Buffer, up_s: mtl.Buffer, down_w: mtl.Buffer, down_s: mtl.Buffer, a: Fp8MoeArgs, groups: Ref, members: Ref, wts9: Ref, dispatch: Ref, act: Ref, z: Ref, out: Ref, pack_info: [4]u32, pack_base: Ref, jobs_ready: bool, gate: ?mtl.Buffer) void {
    if (a.rows == 1) {
        e.setPipeline(p.fp8_moe_gu1_fast);
        bind(e, 0, x);
        inline for (.{ gate_w, gate_s, up_w, up_s, route_out[0] }, 1..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(a, 6);
        bind(e, 7, act);
        fast.dispOr(e, gate, 7, .{ .width = (a.loc + 7) / 8, .height = a.slots }, .{ .width = 128 });
        e.barrier();
        e.setPipeline(p.fp8_moe_down_combine1_fast);
        bind(e, 0, act);
        inline for (.{ down_w, down_s, route_out[0], route_out[1] }, 1..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(a, 5);
        bind(e, 6, out);
        e.setValue(pack_info, 7);
        bind(e, 8, pack_base);
        fast.dispOr(e, gate, 8, .{ .width = (a.hidden + 15) / 16, .height = 1 }, .{ .width = 128 });
        return;
    }
    if (!jobs_ready) {
        e.setPipeline(p.fp8_moe_jobs_direct);
        inline for (.{ route_out[0], route_out[1] }, 0..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(a, 2);
        inline for (.{ groups, members, wts9, dispatch }, 3..) |r, i| bind(e, i, r);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
        e.barrier();
    }
    const ci = fp8Cap(a.rows);
    e.setPipeline(p.fp8_moe_gu[ci]);
    bind(e, 0, if (a.rows <= 8) xh else x);
    inline for (.{ gate_w, gate_s, up_w, up_s }, 1..) |b, i| e.setBuffer(b, 0, i);
    inline for (.{ groups, members }, 5..) |r, i| bind(e, i, r);
    e.setValue(a, 7);
    bind(e, 8, act);
    e.dispatchIndirect(dispatch.buf, dispatch.off, .{ .width = if (a.rows <= 5) 128 else 256 });
    e.barrier();
    e.setPipeline(p.fp8_moe_down[ci]);
    bind(e, 0, act);
    inline for (.{ down_w, down_s }, 1..) |b, i| e.setBuffer(b, 0, i);
    inline for (.{ groups, members }, 3..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, z);
    e.dispatchIndirect(dispatch.buf, dispatch.off + 16, .{ .width = 256 });
    e.barrier();
    const pack_bf16 = a.rows <= 16 and pack_info[1] > 1 and (a.hidden % 4 == 0);
    e.setPipeline(if (pack_bf16) p.fp8_moe_combine_pack_bf16 else p.fp8_moe_combine);
    inline for (.{ z, wts9 }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 2);
    bind(e, 3, out);
    if (pack_bf16) {
        e.setValue(pack_info, 4);
        bind(e, 5, pack_base);
    }
    e.dispatchGroups(.{ .width = (a.hidden / 4 + 63) / 64, .height = a.rows }, .{ .width = 64 });
}

/// FP8 MoE prompt path (up to 2048 rows): 32-row tiles, half MMA gate+up+silu, bit-exact deterministic down+combine.
pub fn fp8MoePrompt(p: *const Pipelines, e: mtl.ComputeEncoder, pick: Ref, wts: Ref, x: Ref, gate_w: mtl.Buffer, gate_s: mtl.Buffer, up_w: mtl.Buffer, up_s: mtl.Buffer, down_w: mtl.Buffer, down_s: mtl.Buffer, a: Fp8MoeArgs, groups: Ref, members: Ref, wts9: Ref, dispatch: Ref, act: Ref, z: Ref, out: Ref) void {
    e.setPipeline(p.fp8_moe_prompt_jobs);
    inline for (.{ pick, wts }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 2);
    inline for (.{ groups, members, wts9, dispatch }, 3..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 1024 });
    e.barrier();
    e.setPipeline(p.fp8_moe_prompt_gu);
    bind(e, 0, x);
    inline for (.{ gate_w, gate_s, up_w, up_s }, 1..) |b, i| e.setBuffer(b, 0, i);
    inline for (.{ groups, members }, 5..) |r, i| bind(e, i, r);
    e.setValue(a, 7);
    bind(e, 8, act);
    e.dispatchIndirect(dispatch.buf, dispatch.off, .{ .width = 128 });
    e.barrier();
    e.setPipeline(p.fp8_moe_prompt_down);
    bind(e, 0, act);
    inline for (.{ down_w, down_s }, 1..) |b, i| e.setBuffer(b, 0, i);
    inline for (.{ groups, members }, 3..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, z);
    e.dispatchIndirect(dispatch.buf, dispatch.off + 16, .{ .width = 128 });
    e.barrier();
    e.setPipeline(p.fp8_moe_combine);
    inline for (.{ z, wts9 }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 2);
    bind(e, 3, out);
    e.dispatchGroups(.{ .width = (a.hidden / 4 + 63) / 64, .height = a.rows }, .{ .width = 64 });
}

pub const route_select = route.select;
pub const RouteOutputs = route.Outputs;
