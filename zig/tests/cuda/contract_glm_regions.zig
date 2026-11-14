//! TF_GLM_KV_SPLIT's two regions under the contract: GLM's eight paged cases on the latent and pooled regions' tables.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");
const index = @import("contract_glm_index.zig");
const attn = @import("contract_glm_attn.zig");

/// The eight paged numerical cases on interleaved pages or on two regions; `wrong`: each one's planted control.
pub const Set = struct {
    lat: pages.LatWrite,
    pool: pages.PoolKeys,
    scores: index.Scores,
    tokens: index.Tokens,
    chunks: [4]attn.Chunks,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, f: glm.glue.Kernels, tokens: cuda.Function, regions: bool, wrong: bool) Set {
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        var out: Set = .{
            .lat = .{ .rig = rig, .f = f.lat_write, .wrong_table = wrong, .regions = regions },
            .pool = .{ .rig = rig, .f = f.pool_keys, .wrong_table = wrong, .regions = regions },
            .scores = .{ .rig = rig, .f = f.scores, .wrong_table = wrong, .regions = regions },
            .tokens = .{ .rig = rig, .f = tokens, .wrong_table = wrong, .regions = regions },
            .chunks = undefined,
        };
        for (&out.chunks, [_]u32{ 16, 16, 32, 32 }, [_]bool{ false, true, false, true }) |*x, hb, sparse|
            x.* = .{ .rig = rig, .f = if (hb == 16) f.chunks_16 else f.chunks_32, .merge = f.merge, .form = .{ .hb = hb, .sparse = sparse }, .wrong_table = wrong, .regions = regions };
        return out;
    }

    pub fn deinit(self: *Set) void {
        self.lat.deinit();
        self.pool.deinit();
        self.scores.deinit();
        self.tokens.deinit();
        for (&self.chunks) |*x| x.deinit();
    }

    /// After init's value sits at its final address: each kernel's context points into it.
    pub fn append(self: *Set, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        try kernels.append(a, self.lat.kernel());
        try kernels.append(a, self.pool.kernel());
        try kernels.append(a, self.scores.kernel());
        try kernels.append(a, self.tokens.kernel());
        for (&self.chunks) |*x| try kernels.append(a, x.kernel());
    }

    /// The eight cases on two regions, on the heap so their contexts keep their address, registered into `kernels`.
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, f: glm.glue.Kernels, tokens: cuda.Function, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const self = try a.create(Set);
        errdefer a.destroy(self);
        self.* = init(gpu, s, f, tokens, true, false);
        errdefer self.deinit();
        try self.append(a, kernels);
        return self;
    }

    pub fn destroy(self: *Set, a: std.mem.Allocator) void {
        self.deinit();
        a.destroy(self);
    }
};
