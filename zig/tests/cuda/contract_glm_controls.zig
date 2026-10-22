//! Real GLM launches with planted page-table or sparse physical-row mismatches, one per paged contract case.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const pages = @import("contract_glm_pages.zig");
const index = @import("contract_glm_index.zig");
const attn = @import("contract_glm_attn.zig");

pub const Controls = struct {
    lat: pages.LatWrite,
    pool: pages.PoolKeys,
    scores: index.Scores,
    tokens: index.Tokens,
    chunks: [4]attn.Chunks,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, f: glm.glue.Kernels, tokens: cuda.Function) Controls {
        var out: Controls = .{
            .lat = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f.lat_write, .wrong_table = true },
            .pool = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f.pool_keys, .wrong_table = true },
            .scores = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f.scores, .wrong_table = true },
            .tokens = .{ .rig = .{ .gpu = gpu, .s = s }, .f = tokens, .wrong_table = true },
            .chunks = undefined,
        };
        for (&out.chunks, [_]u32{ 16, 16, 32, 32 }, [_]bool{ false, true, false, true }) |*x, hb, sparse|
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = if (hb == 16) f.chunks_16 else f.chunks_32, .merge = f.merge, .form = .{ .hb = hb, .sparse = sparse }, .wrong_table = true };
        return out;
    }

    pub fn deinit(self: *Controls) void {
        self.lat.deinit();
        self.pool.deinit();
        self.scores.deinit();
        self.tokens.deinit();
        for (&self.chunks) |*x| x.deinit();
    }

    pub fn append(self: *Controls, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        try kernels.append(a, self.lat.kernel());
        try kernels.append(a, self.pool.kernel());
        try kernels.append(a, self.scores.kernel());
        try kernels.append(a, self.tokens.kernel());
        for (&self.chunks) |*x| try kernels.append(a, x.kernel());
    }
};
