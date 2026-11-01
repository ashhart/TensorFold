//! DeepSeek-V4.1's whole Metal contract behind one hook: routes, suites and their extra gates.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;

const dsv41_route = @import("metal_dsv41_route.zig");
const dsv41_fp8_rows = @import("metal_dsv41_fp8_rows.zig");
const dsv41_fp8_mma = @import("metal_dsv41_fp8_mma.zig");
const dsv41_topk512 = @import("metal_dsv41_topk512.zig");
const dsv41_cand = @import("metal_dsv41_cand.zig");
const dsv41_attn = @import("metal_dsv41_attn.zig");
const dsv41_experts = @import("metal_dsv41_experts.zig");
const dsv41_xmma = @import("metal_dsv41_xmma.zig");
const dsv41_index = @import("metal_dsv41_index.zig");
const dsv41_engram = @import("metal_dsv41_engram.zig");
const dsv41_glue = @import("metal_dsv41_glue.zig");
const dsv41_compress = @import("metal_dsv41_compress.zig");
const dsv41_mv = @import("metal_dsv41_mv.zig");
const dsv41_vision = @import("metal_dsv41_vision.zig");
const dsv41_hc = @import("metal_dsv41_hc.zig");
const dsv41_fold = @import("metal_dsv41_fold.zig");
const dsv41_glue_tmp = @import("metal_dsv41_glue_tmp.zig");
const dsv41_glue_tmp2 = @import("metal_dsv41_glue_tmp2.zig");
const dsv41_fused = @import("metal_dsv41_fused.zig");
const dsv41_fused2 = @import("metal_dsv41_fused2.zig");
const dsv41_fused3 = @import("metal_dsv41_fused3.zig");
const dsv41_fused4 = @import("metal_dsv41_fused4.zig");
const dsv41_fused5 = @import("metal_dsv41_fused5.zig");

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const text = only orelse return true;
    return std.mem.indexOf(u8, name, text) != null;
}

/// The route probes, registered where the dsv41 registrations sat: their kernels join `kernels` in variant order.
pub const Routes = struct {
    routes: []dsv41_route.Route,
    made: usize = 0,
    a: std.mem.Allocator,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Routes {
        var r = Routes{ .routes = try a.alloc(dsv41_route.Route, std.enums.values(dsv41_route.Variant).len), .a = a };
        errdefer a.free(r.routes);
        errdefer for (r.routes[0..r.made]) |*p| p.deinit();
        for (std.enums.values(dsv41_route.Variant)) |v| {
            if (!wanted(only, dsv41_route.spec(v).name)) continue;
            r.routes[r.made] = try dsv41_route.Route.init(a, device, queue, v);
            r.made += 1;
            try kernels.append(a, r.routes[r.made - 1].kernel());
        }
        return r;
    }

    pub fn destroy(r: *Routes) void {
        for (r.routes[0..r.made]) |*p| p.deinit();
        r.a.free(r.routes);
    }
};

/// Every dsv41 suite, created and destroyed in the order main held them.
pub const Suites = struct {
    fp8: *dsv41_fp8_rows.Suite,
    fp8_mma: *dsv41_fp8_mma.Suite,
    topk: *dsv41_topk512.Suite,
    cand: *dsv41_cand.Suite,
    attn: *dsv41_attn.Suite,
    experts: *dsv41_experts.Suite,
    xmma: *dsv41_xmma.Suite,
    ix: *dsv41_index.Suite,
    engram: *dsv41_engram.Suite,
    glue: *dsv41_glue.Suite,
    compress: *dsv41_compress.Suite,
    mv: *dsv41_mv.Suite,
    vision: *dsv41_vision.Suite,
    hc: *dsv41_hc.Suite,
    fold: *dsv41_fold.Suite,
    glue_tmp: *dsv41_glue_tmp.Suite,
    glue_tmp2: *dsv41_glue_tmp2.Suite,
    fused: *dsv41_fused.Suite,
    fused2: *dsv41_fused2.Suite,
    fused3: *dsv41_fused3.Suite,
    fused4: *dsv41_fused4.Suite,
    fused5: *dsv41_fused5.Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const fp8 = try dsv41_fp8_rows.Suite.create(a, device, queue, kernels, only);
        errdefer fp8.destroy();
        const fp8_mma = try dsv41_fp8_mma.Suite.create(a, device, queue, kernels, only);
        errdefer fp8_mma.destroy();
        const topk = try dsv41_topk512.Suite.create(a, device, queue, kernels, only);
        errdefer topk.destroy();
        const cand = try dsv41_cand.Suite.create(a, device, queue, kernels, only);
        errdefer cand.destroy();
        const attn = try dsv41_attn.Suite.create(a, device, queue, kernels, only);
        errdefer attn.destroy();
        const experts = try dsv41_experts.Suite.create(a, device, queue, kernels, only);
        errdefer experts.destroy();
        const xmma = try dsv41_xmma.Suite.create(a, device, queue, kernels, only);
        errdefer xmma.destroy();
        const ix = try dsv41_index.Suite.create(a, device, queue, kernels, only);
        errdefer ix.destroy();
        const engram = try dsv41_engram.Suite.create(a, device, queue, kernels, only);
        errdefer engram.destroy();
        const glue = try dsv41_glue.Suite.create(a, device, queue, kernels, only);
        errdefer glue.destroy();
        const compress = try dsv41_compress.Suite.create(a, device, queue, kernels, only);
        errdefer compress.destroy();
        const mv = try dsv41_mv.Suite.create(a, device, queue, kernels, only);
        errdefer mv.destroy();
        const vision = try dsv41_vision.Suite.create(a, device, queue, kernels, only);
        errdefer vision.destroy();
        const hc = try dsv41_hc.Suite.create(a, device, queue, kernels, only);
        errdefer hc.destroy();
        const fold = try dsv41_fold.Suite.create(a, device, queue, kernels, only);
        errdefer fold.destroy();
        const glue_tmp = try dsv41_glue_tmp.Suite.create(a, device, queue, kernels, only);
        errdefer glue_tmp.destroy();
        const fused = try dsv41_fused.Suite.create(a, device, queue, kernels, only);
        errdefer fused.destroy();
        const fused2 = try dsv41_fused2.Suite.create(a, device, queue, kernels, only);
        errdefer fused2.destroy();
        const fused3 = try dsv41_fused3.Suite.create(a, device, queue, kernels, only);
        errdefer fused3.destroy();
        const fused4 = try dsv41_fused4.Suite.create(a, device, queue, kernels, only);
        errdefer fused4.destroy();
        const fused5 = try dsv41_fused5.Suite.create(a, device, queue, kernels, only);
        errdefer fused5.destroy();
        const glue_tmp2 = try dsv41_glue_tmp2.Suite.create(a, device, queue, kernels, only);
        errdefer glue_tmp2.destroy();
        return .{ .fp8 = fp8, .fp8_mma = fp8_mma, .topk = topk, .cand = cand, .attn = attn, .experts = experts, .xmma = xmma, .ix = ix, .engram = engram, .glue = glue, .compress = compress, .mv = mv, .vision = vision, .hc = hc, .fold = fold, .glue_tmp = glue_tmp, .fused = fused, .fused2 = fused2, .fused3 = fused3, .fused4 = fused4, .fused5 = fused5, .glue_tmp2 = glue_tmp2 };
    }

    pub fn destroy(s: *Suites) void {
        s.glue_tmp2.destroy();
        s.fused5.destroy();
        s.fused4.destroy();
        s.fused3.destroy();
        s.fused2.destroy();
        s.fused.destroy();
        s.glue_tmp.destroy();
        s.fold.destroy();
        s.hc.destroy();
        s.vision.destroy();
        s.mv.destroy();
        s.compress.destroy();
        s.glue.destroy();
        s.engram.destroy();
        s.ix.destroy();
        s.xmma.destroy();
        s.experts.destroy();
        s.attn.destroy();
        s.cand.destroy();
        s.topk.destroy();
        s.fp8_mma.destroy();
        s.fp8.destroy();
    }

    /// The dsv41 gates after runAll: equality, invariance and accuracy probes, and timings when asked for.
    pub fn gates(s: *Suites, io: std.Io, seed: u64, t: *g.Tally) !void {
        if (s.experts.probes.items.len > 0) t.failed += try s.experts.pipeline(seed);
        if (std.c.getenv("TF_DSV41_EXPERTS_TIME") != null) try s.experts.timing(io);
        if (s.xmma.probes.items.len > 0) t.failed += try s.xmma.equality(seed);
        if (std.c.getenv("TF_DSV41_XMMA_TIME") != null) try s.xmma.timing(io);
        t.failed += try s.fp8.invariance(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.fp8.timing(io);
        t.failed += try s.fp8_mma.invariance(seed);
        t.failed += try s.fp8_mma.accuracy(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.fp8_mma.timing(io);
        t.failed += try s.topk.invariance(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.topk.timing(io);
        t.failed += try s.cand.invariance(seed);
        t.failed += try s.cand.equality(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.cand.timing(io);
        try s.attn.invariance(io, seed, t);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.attn.timing(io);
        t.failed += try s.ix.invariance(seed);
        t.failed += try s.engram.invariance(seed);
        t.failed += try s.glue.invariance(seed);
        if (s.mv.probes.items.len > 0) {
            t.failed += try s.mv.invariance(seed);
            t.failed += try s.mv.equality(seed);
            t.failed += try s.mv.head(io, seed, std.c.getenv("TF_DSV41_TIME") != null);
        }
        t.failed += try s.glue.tables();
        t.failed += try s.vision.invariance(seed);
        t.failed += try s.compress.invariance(seed);
        t.failed += try s.compress.equality(seed);
        t.failed += try s.hc.invariance(seed);
        t.failed += try s.fold.invariance(seed);
        t.failed += try s.glue_tmp.invariance(seed);
        t.failed += try s.glue_tmp2.invariance(seed);
        t.failed += try s.fused.invariance(seed);
        t.failed += try s.fused3.invariance(seed);
        t.failed += try s.fused4.invariance(seed);
        t.failed += try s.fused5.invariance(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.ix.timing(io);
    }
};
