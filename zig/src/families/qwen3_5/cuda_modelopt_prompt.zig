//! ModelOpt prompt glue lives outside the MLX CUDA modules and dispatch path.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const Seg = @import("cuda_cache.zig").Seg;
pub const Glue = struct {
    gdn: cuda.Module,
    router_module: ?cuda.Module,
    chain: cuda.gdn_chain.Chain,
    router: ?cuda.moe_bf16.Fns,
    pub fn load(d: *const cuda.Driver, moe: bool) !Glue {
        var gd = try cuda.Module.load(d, cuda.kernels.qwen_gdn_bf16);
        errdefer gd.unload();
        var rt: ?cuda.Module = if (moe) try cuda.Module.load(d, cuda.kernels.qwen_router_bf16) else null;
        errdefer if (rt) |*m| m.unload();
        return .{ .gdn = gd, .router_module = rt, .chain = try cuda.gdn_chain.Chain.resolve(gd), .router = if (rt) |m| try cuda.moe_bf16.Fns.resolve(m) else null };
    }
    pub fn unload(g: *Glue) void {
        if (g.router_module) |*m| m.unload();
        g.gdn.unload();
    }
};
pub fn delta(e: anytype, layer: usize, rows: usize, norm: u64, sums: u64) !bool {
    if (e.mo) |m| {
        const sg = std.mem.bytesToValue(Seg, e.staged.bytes[@TypeOf(e.*).segs_at..][0..@sizeOf(Seg)]);
        const state = sg.rec + layer * e.g.v_heads * geometry.dk * geometry.dk * 4;
        return m.glue.chain.run(e.stream, .{ .q = e.q, .k = e.k, .v = e.v, .g = e.g_, .beta = e.beta, .state = state, .last = state, .y = e.y, .z = e.z, .norm = norm, .mix = e.mix, .sums = sums }, rows, e.g.k_heads, e.g.v_heads, e.g.eps);
    }
    try @import("cuda_prompt.zig").delta(e, layer, rows);
    return false;
}
pub fn route(e: anytype, x: u64, rows: usize, weights: u64, logits: u64, count: usize, params: u64) !void {
    if (e.mo) |m| if (m.glue.router) |f| return f.route(e.stream, x, e.g.hidden, rows, weights, logits, e.g.hidden, count, params);
    return e.moe.route(e.stream, x, e.g.hidden, rows, weights, logits, e.g.hidden, count, params);
}
