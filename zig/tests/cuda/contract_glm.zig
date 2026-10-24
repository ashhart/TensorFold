//! GLM-5.3-Flash's KV page kernels under the kernel contract: their two modules and one case for each launch.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const pages_k = @import("contract_glm_pages.zig");
const index_k = @import("contract_glm_index.zig");
const attn_k = @import("contract_glm_attn.zig");

const g = core.contract.gates;
const kk = cuda.kernels;

/// The latent write, the pooled keys, the index scores, the selected tokens' rows and the MLA chunks.
pub const Cases = struct {
    glue: cuda.Module,
    ops: cuda.Module,
    lat: pages_k.LatWrite,
    pool: pages_k.PoolKeys,
    scores: index_k.Scores,
    tokens: index_k.Tokens,
    chunks: [4]attn_k.Chunks,

    pub fn init(gpu: check.Gpu, s: cuda.Stream) !Cases {
        var glue = try cuda.Module.load(gpu.d, kk.glm_glue);
        errdefer glue.unload();
        const fns = try glm.glue.Kernels.load(glue);
        var ops = try cuda.Module.load(gpu.d, kk.glm_ops);
        errdefer ops.unload();
        var cases: Cases = .{
            .glue = glue,
            .ops = ops,
            .lat = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.lat_write },
            .pool = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.pool_keys },
            .scores = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.scores },
            .tokens = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try ops.function("tf_glm_dsa_tokens") },
            .chunks = undefined,
        };
        for (&cases.chunks, [_]u32{ 16, 16, 32, 32 }, [_]bool{ false, true, false, true }) |*x, hb, sparse|
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = if (hb == 16) fns.chunks_16 else fns.chunks_32, .merge = fns.merge, .form = .{ .hb = hb, .sparse = sparse } };
        return cases;
    }

    /// Each case's kernel in the contract's order; the cases stay where they are until deinit.
    pub fn append(cases: *Cases, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        try kernels.append(a, cases.lat.kernel());
        try kernels.append(a, cases.pool.kernel());
        try kernels.append(a, cases.scores.kernel());
        try kernels.append(a, cases.tokens.kernel());
        for (&cases.chunks) |*x| try kernels.append(a, x.kernel());
    }

    pub fn deinit(cases: *Cases) void {
        for (&cases.chunks) |*x| x.deinit();
        cases.tokens.deinit();
        cases.ops.unload();
        cases.scores.deinit();
        cases.pool.deinit();
        cases.lat.deinit();
        cases.glue.unload();
    }
};
