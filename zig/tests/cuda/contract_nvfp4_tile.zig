//! The K64 baseline and K32 candidates share byte replacements, FP64 references and a scaled-code control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const fact = @import("contract_nvfp4_fact.zig");
pub const Group = struct {
    a: std.mem.Allocator,
    tile: cuda.experts_tile.Tile,
    entries: [6]fact.Experts,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, baseline: *const cuda.experts.Nvfp4Experts, router: cuda.grouped.Router, kernels: *std.ArrayList(core.contract.gates.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.tile = try cuda.experts_tile.Tile.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)), .both);
        errdefer self.tile.unload();
        const kinds = std.enums.values(fact.Kind);
        for (&self.entries, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .kind = kinds[i % kinds.len], .k = baseline, .tile = if (i >= kinds.len) &self.tile else null, .router = router };
        errdefer for (&self.entries) |*x| x.deinit();
        for (&self.entries) |*x| try kernels.append(a, x.kernel());
        return self;
    }
    pub fn destroy(self: *Group) void {
        for (&self.entries) |*x| x.deinit();
        self.tile.unload();
        self.a.destroy(self);
    }
};
