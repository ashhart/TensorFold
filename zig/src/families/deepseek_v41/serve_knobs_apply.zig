//! The served engine knobs onto a forward before its buffer plan (serve_knobs.zig reads them).
const std = @import("std");
const pf = @import("forward_prefill.zig");
const dk = @import("dsv41_kernels");
const pk = @import("serve_knobs.zig");
const Forward = @import("forward.zig").Forward;
const env = pk.env;
const gmGu2 = pk.gmGu2;
const indexBound = pk.indexBound;
const indexBudget = pk.indexBudget;
const kvNormStore = pk.kvNormStore;
const mhcPfRows = pk.mhcPfRows;
const mhcSiteRows = pk.mhcSiteRows;
const pf4k = pk.pf4k;
const pf4k_rows = pk.pf4k_rows;
const pfDense = pk.pfDense;
const pfOverlap = pk.pfOverlap;
const pfOverlapSite = pk.pfOverlapSite;
const pfTbo = pk.pfTbo;
const pfTboGmSms = pk.pfTboGmSms;
const prefetchAhead = pk.prefetchAhead;
const prefillMode = pk.prefillMode;
const prefillRows = pk.prefillRows;
const routerGroupRot = pk.routerGroupRot;
const streamRb = pk.streamRb;
const x3ldEpi = pk.x3ldEpi;

/// The knobs onto a forward before its buffer plan, on every rank; R1 is read where the forward is made (`r1`).
pub fn apply(f: *Forward, a: std.mem.Allocator, io: std.Io, rank: u32) !void {
    f.opts.kv_norm_store = kvNormStore(&env);
    f.opts.router_group_rot = routerGroupRot(&env);
    f.opts.x3ld_epi = try x3ldEpi(&env);
    f.opts.pfd = try pfDense(a, io, &env);
    // TF_DSV41_PF_4K: the forward's own options keep 2,048-row segments; 4,096-row ones use the prefill workspace
    const big = try pf4k(&env);
    if (try prefillRows(&env)) |r| if (!big) {
        f.opts.prefill_rows = r;
    };
    if (try indexBudget(&env)) |b| f.opts.index_budget = b;
    pf.state(f).ahead = prefetchAhead(&env);
    f.opts.gm_v2 = try @import("gm2pf.zig").mode(&env);
    f.opts.mhc_pf_rows = try mhcPfRows(&env);
    pf.state(f).mode = try prefillMode(&env);
    pf.state(f).k4 = big;
    // 4K needs x3gm v2 (v1's 4K tuning is not emitted) and replay (the workspace is released at the CED finish)
    if (big and (f.opts.gm_v2 == .off or pf.state(f).mode != .replay)) {
        std.log.scoped(.dsv41).err("TF_DSV41_PF_4K=1 needs TF_DSV41_GM_V2=1 (or split) and TF_DSV41_PREFILL=replay", .{});
        return error.BadKnob;
    }
    f.opts.pf_tbo = try pfTbo(&env);
    f.opts.stream_rb = try streamRb(&env);
    f.opts.index_bound = try indexBound(&env);
    // TF_DSV41_BRANCHES: the side stream exists before any capture
    if (f.branches == null) f.branches = try @import("branches.zig").install(f.gpa, f, rank);
    // TF_DSV41_PF_OVERLAP: the pieces' exchanges go on the branches' side stream (one peer: TP=2)
    f.opts.pf_overlap = try pfOverlap(&env);
    f.opts.pf_overlap_site = try pfOverlapSite(&env);
    f.opts.gm_gu2 = try gmGu2(&env);
    f.opts.mhc_site_rows = try mhcSiteRows(&env);
    if (f.opts.mhc_site_rows > 0) std.log.scoped(.dsv41).info("mhc: prefill boundary sites in pieces of about {d} rows", .{f.opts.mhc_site_rows});
    if (f.opts.gm_gu2 and f.opts.gm_v2 != .one) {
        std.log.scoped(.dsv41).err("TF_DSV41_GM_GU2=1 needs TF_DSV41_GM_V2=1 (x3gm v2's one plan a block)", .{});
        return error.BadKnob;
    }
    if (f.opts.gm_gu2) std.log.scoped(.dsv41).info("x3gm: gate / up at 128-member passes (GU2) in 4,096-row segments", .{});
    if (f.opts.pf_overlap_site and f.opts.pf_overlap < 2) {
        std.log.scoped(.dsv41).err("TF_DSV41_PF_OVERLAP_SITE=1 needs TF_DSV41_PF_OVERLAP=2..8 (the pieces it overlaps)", .{});
        return error.BadKnob;
    }
    // TF_DSV41_PF_TBO: the second micro-batch on the side stream, CED's encoder segments only; x3gm's grid capped
    if (f.opts.pf_tbo) {
        if (!f.opts.branches or pf.state(f).mode != .replay) {
            std.log.scoped(.dsv41).err("TF_DSV41_PF_TBO=1 needs TF_DSV41_BRANCHES=1 (its side stream) and TF_DSV41_PREFILL=replay", .{});
            return error.BadKnob;
        }
        dk.exl3.gm_sms_cap = try pfTboGmSms(&env);
    }
    if (f.opts.pf_overlap > 0 and (!f.opts.branches or f.opts.world != 2)) {
        std.log.scoped(.dsv41).err("TF_DSV41_PF_OVERLAP={d} needs TF_DSV41_BRANCHES=1 (its side stream) and TP=2 (world {d})", .{ f.opts.pf_overlap, f.opts.world });
        return error.BadKnob;
    }
    if (pf.state(f).mode == .replay) {
        const ced = @import("ced.zig");
        const d = ced.decoderStart(f.cfg) catch |e| {
            std.log.scoped(.dsv41).err("TF_DSV41_PREFILL=replay: no decoder layer in this model ({t}: the first ratio-1 layer must be a kv source and no later layer Engram's)", .{e});
            return e;
        };
        if (std.mem.indexOfScalar(u32, try f.backbone(a), d) == null) {
            std.log.scoped(.dsv41).err("TF_DSV41_PREFILL=replay: the forward's layers do not reach the decoder's first ({d})", .{d});
            return error.BadKnob;
        }
    }
    if (rank != 0) return;
    const log = std.log.scoped(.dsv41);
    log.info("KV norm/store fusion: {s}", .{if (f.opts.kv_norm_store) "on" else "off"});
    log.info("router grouping/rotation fusion: {s}", .{if (f.opts.router_group_rot) "on" else "off"});
    if (f.opts.index_bound) log.info("index bound: decode index scores stop at each row's own keys (_scores_b)", .{});
    if (f.opts.x3ld_epi) log.info("x3ld epilogues: gateup_epilogue / down_combine fused into the decode expert streamer (TF_DSV41_X3LD_EPI)", .{});
    if (f.opts.stream_rb == 1) log.info("stream top-k: _stream_pf (the next tile's key loads issued ahead)", .{});
    if (f.opts.stream_rb > 1) log.info("stream top-k: {d} rows a program (_stream_rb{d})", .{ f.opts.stream_rb, f.opts.stream_rb });
    if (f.opts.pf_tbo) log.info("prefill tbo: encoder segments two at a time on two streams, x3gm on at most {d} SMs (0: all){s}", .{ dk.exl3.gm_sms_cap, if (f.opts.pf_overlap > 0) "; TF_DSV41_PF_OVERLAP is off under it" else "" });
    if (f.opts.pf_overlap > 0) log.info("prefill overlap: a segment's exchanges in {d} row pieces on the side stream{s}", .{ f.opts.pf_overlap, if (f.opts.pf_overlap_site) ", every piece; the next mixing site in the same pieces, each after its own" else "" });
    log.info("serve knobs: pf_dense {s}, prefill rows {d}{s}, index budget {d} MiB, R1 {s}, prefetch ahead {s}, x3gm v2 {t}, prefill {t}, mhc pfdec {d}", .{ if (f.opts.pfd != null) "fused" else "off", if (big) pf4k_rows else @as(u32, @intCast(f.opts.prefill_rows)), if (big) " (4K: one x3gm block a segment, in the prefill's own workspace)" else "", f.opts.index_budget >> 20, if (f.opts.r1) "on" else "off", if (pf.state(f).ahead) "on" else "off", f.opts.gm_v2, pf.state(f).mode, f.opts.mhc_pf_rows });
}
