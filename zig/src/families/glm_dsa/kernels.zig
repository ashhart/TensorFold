//! GLM-5.3's own Metal kernels (MLA, the DSA indexer, fp32 glue) and the core route, loaded once with their encoders.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const route = @import("../../core/moe_route.zig");

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

/// One expert's tables on this Mac for the jobs kernel (kernels/metal/glm_dsa/moe_jobs.metal's ExpertDesc).
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

pub const splits = 16;
const capacities = [_]usize{ 1, 2, 4, 8, 16 };

pub const Pipelines = struct {
    prep: mtl.Pipeline,
    absorb: [capacities.len]mtl.Pipeline,
    attend: mtl.Pipeline,
    merge: mtl.Pipeline,
    index_key: mtl.Pipeline,
    index_scores: mtl.Pipeline,
    topk: mtl.Pipeline,
    embed: mtl.Pipeline,
    add_rms: mtl.Pipeline,
    silu_mul: mtl.Pipeline,
    gather: mtl.Pipeline,
    select: mtl.Pipeline,
    moe_jobs: mtl.Pipeline,
    moe_jobs_prompt: mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, experts: u32, hidden: u32, topk: u32) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const opts = mtl.CompileOptions.mlx();
        const mla = try mtl.Library.fromSource(device, sources.glm_dsa_mla, opts);
        defer mla.deinit();
        const idx = try mtl.Library.fromSource(device, sources.glm_dsa_index, opts);
        defer idx.deinit();
        const glue = try mtl.Library.fromSource(device, sources.glm_dsa_glue, opts);
        defer glue.deinit();
        const jl = try mtl.Library.fromSource(device, sources.glm_dsa_moe_jobs, opts);
        defer jl.deinit();
        const src = try route.source(gpa, .{ .hidden = hidden, .experts = experts, .topk = topk });
        defer gpa.free(src);
        const rl = try mtl.Library.fromSource(device, src, opts);
        defer rl.deinit();
        var p: Pipelines = undefined;
        p.prep = try mtl.Pipeline.init(device, mla, "glm_mla_prep", false);
        for (capacities, &p.absorb) |c, *q| {
            var name: [32]u8 = undefined;
            q.* = try mtl.Pipeline.init(device, mla, try std.fmt.bufPrint(&name, "glm_mla_absorb_{d}", .{c}), false);
        }
        p.attend = try mtl.Pipeline.init(device, mla, "glm_mla_attend", false);
        p.merge = try mtl.Pipeline.init(device, mla, "glm_mla_merge", false);
        p.index_key = try mtl.Pipeline.init(device, idx, "glm_index_key", false);
        p.index_scores = try mtl.Pipeline.init(device, idx, "glm_index_scores", false);
        p.topk = try mtl.Pipeline.init(device, idx, "glm_index_topk", false);
        p.embed = try mtl.Pipeline.init(device, glue, "glm_embed", false);
        p.add_rms = try mtl.Pipeline.init(device, glue, "glm_add_rms", false);
        p.silu_mul = try mtl.Pipeline.init(device, glue, "glm_silu_mul", false);
        p.gather = try mtl.Pipeline.init(device, glue, "glm_gather", false);
        p.select = try mtl.Pipeline.init(device, rl, route.names[1], false);
        p.moe_jobs = try mtl.Pipeline.init(device, jl, "glm_moe_jobs", false);
        p.moe_jobs_prompt = try mtl.Pipeline.init(device, jl, "glm_moe_jobs_prompt", false);
        if (p.topk.maxThreads() < 1024 or p.moe_jobs_prompt.maxThreads() < 1024) return error.UnsupportedGlmPipeline;
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.prep.deinit();
        for (p.absorb) |q| q.deinit();
        inline for (.{ p.attend, p.merge, p.index_key, p.index_scores, p.topk, p.embed, p.add_rms, p.silu_mul, p.gather, p.select, p.moe_jobs, p.moe_jobs_prompt }) |q| q.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, i: usize, r: Ref) void {
    e.setBuffer(r.buf, r.off, i);
}

fn capacity(rows: u32) usize {
    for (capacities, 0..) |c, i| if (rows <= c) return i;
    unreachable;
}

/// q_resid and the cache's latent and rope key for every row (one threadgroup a row).
pub fn prep(p: *const Pipelines, e: mtl.ComputeEncoder, proj: Ref, w_qa: Ref, w_kva: Ref, inv: Ref, rows: Ref, a: MlaArgs, q_resid: Ref, cache: Ref) void {
    e.setPipeline(p.prep);
    inline for (.{ proj, w_qa, w_kva, inv, rows }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, q_resid);
    bind(e, 7, cache);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 256 });
}

/// Every head's latent query (W_uk^T q_nope) and roped q_rope for 1-16 rows.
pub fn absorb(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, weights: mtl.Buffer, rows: Ref, a: MlaArgs, qlat: Ref, qrope: Ref, inv: Ref) void {
    e.setPipeline(p.absorb[capacity(a.rows)]);
    bind(e, 0, q);
    e.setBuffer(weights, 0, 1);
    bind(e, 2, rows);
    e.setValue(a, 3);
    inline for (.{ qlat, qrope, inv }, 4..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 4, .height = a.heads }, .{ .width = 128 });
}

/// Each (row, head)'s splits over its key list, then the merge into the latent output.
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
    e.dispatchGroups(.{ .width = a.heads, .height = a.rows }, .{ .width = 256 });
}

/// The index cache's key for every row (a simdgroup a row).
pub fn indexKey(p: *const Pipelines, e: mtl.ComputeEncoder, proj: Ref, w: Ref, b: Ref, inv: Ref, rows: Ref, a: IndexArgs, cache: Ref) void {
    e.setPipeline(p.index_key);
    inline for (.{ proj, w, b, inv, rows }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 5);
    bind(e, 6, cache);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 32 });
}

/// Every row's top-k list in position order; scores are needed only past the all-visible prefix.
pub fn indexTopk(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, proj: Ref, cache: Ref, rows: Ref, a: IndexArgs, score: Ref, inv: Ref, keys: Ref, last_pos: u32) void {
    // last_pos is the maximum zero-based position; top-k keeps every visible key through position 2047.
    if (last_pos >= 2048) {
        e.setPipeline(p.index_scores);
        inline for (.{ q, proj, cache, rows }, 0..) |r, i| bind(e, i, r);
        e.setValue(a, 4);
        bind(e, 5, score);
        bind(e, 6, inv);
        e.dispatchGroups(.{ .width = last_pos / 256 + 1, .height = a.rows }, .{ .width = 256 });
    }
    e.barrier();
    e.setPipeline(p.topk);
    bind(e, 0, score);
    bind(e, 1, rows);
    e.setValue(a, 2);
    bind(e, 3, keys);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 1024 });
}

/// Hidden rows from token ids: an FP8 row times its scale.
pub fn embed(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, fp8: Ref, scale: Ref, a: GlueArgs, h: Ref, rows: u32) void {
    e.setPipeline(p.embed);
    inline for (.{ ids, fp8, scale }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    bind(e, 4, h);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
}

/// h += the parts (in order), then y = rms(h) w.
pub fn addRms(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, add: Ref, w: Ref, a: GlueArgs, y: Ref, rows: u32) void {
    e.setPipeline(p.add_rms);
    inline for (.{ h, add, w }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    bind(e, 4, y);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
}

/// act = silu(gate) up over `rows` rows of `a.width` channels.
pub fn siluMul(p: *const Pipelines, e: mtl.ComputeEncoder, gu: Ref, a: GlueArgs, act: Ref, rows: u32) void {
    e.setPipeline(p.silu_mul);
    bind(e, 0, gu);
    e.setValue(a, 1);
    bind(e, 2, act);
    e.dispatchGroups(.{ .width = (a.width + 255) / 256, .height = rows }, .{ .width = 256 });
}

/// Row r of dst = row idx[r] of src: `width` 32-bit words a row, strides in words.
pub fn gather(p: *const Pipelines, e: mtl.ComputeEncoder, src: Ref, idx: Ref, width: u32, src_stride: u32, dst_stride: u32, dst: Ref, rows: u32) void {
    e.setPipeline(p.gather);
    bind(e, 0, src);
    bind(e, 1, idx);
    e.setValue([4]u32{ width, src_stride, dst_stride, 0 }, 2);
    bind(e, 3, dst);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
}

/// The window's expert jobs, member lists, per-slot tables and the two projections' threadgroup counts, from the route.
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

/// A prompt window's expert jobs from its picks: tiles of 32 rows, member lists, per-slot tables, the two counts.
pub fn moeJobsPrompt(p: *const Pipelines, e: mtl.ComputeEncoder, pick: Ref, wts: Ref, desc: Ref, a: PromptJobsArgs, jobs: Ref, members: Ref, wts9: Ref, sv: Ref, suh: Ref, dispatch: Ref) void {
    e.setPipeline(p.moe_jobs_prompt);
    inline for (.{ pick, wts, desc }, 0..) |r, i| bind(e, i, r);
    e.setValue(a, 3);
    inline for (.{ jobs, members, wts9, sv, suh, dispatch }, 4..) |r, i| bind(e, i, r);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 1024 });
}

pub const route_select = route.select;
pub const RouteOutputs = route.Outputs;
