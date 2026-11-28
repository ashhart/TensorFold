//! Real GLM launches with planted page-table or sparse physical-row mismatches, and TF_GLM_KV_HOST's gather kernels.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const Set = @import("contract_glm_regions.zig").Set;
const kv_host = @import("contract_glm_kv_host.zig");

pub const Controls = struct {
    sets: [2]Set, // on interleaved pages, then on TF_GLM_KV_SPLIT's two regions (each launch given the other's table)
    ops: ?cuda.Module, // glm_ops: TF_GLM_KV_HOST's gather kernels and their controls (null: the page cases alone)
    kvh: ?*kv_host.Set = null,
    a: ?std.mem.Allocator = null,
    regions: ?*Set = null, // TF_GLM_KV_SPLIT's eight cases on two regions, unplanted, made by append first
    f: glm.glue.Kernels,
    tokens: cuda.Function,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, f: glm.glue.Kernels, tokens: cuda.Function, ops: ?cuda.Module) Controls {
        return .{ .sets = .{ Set.init(gpu, s, f, tokens, false, true), Set.init(gpu, s, f, tokens, true, true) }, .ops = ops, .f = f, .tokens = tokens };
    }

    pub fn deinit(self: *Controls) void {
        for (&self.sets) |*x| x.deinit();
        if (self.kvh) |k| k.destroy(self.a.?);
        if (self.regions) |r| r.destroy(self.a.?);
    }

    pub fn append(self: *Controls, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        self.a = a;
        self.regions = try Set.create(a, self.sets[0].lat.rig.gpu, self.sets[0].lat.rig.s, self.f, self.tokens, kernels);
        for (&self.sets) |*x| try x.append(a, kernels);
        const ops = self.ops orelse return;
        self.kvh = try kv_host.Set.create(a, self.sets[0].lat.rig.gpu, self.sets[0].lat.rig.s, ops, kernels);
    }
};
