//! K64 baseline, K32 candidates, decode waves and 1-warp CTAs: byte replacements, FP64 references, controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const fact = @import("contract_nvfp4_fact.zig");
pub const Group = struct {
    a: std.mem.Allocator,
    four: cuda.experts.Nvfp4Experts, // the decode kernels at 4 warps a CTA: the baseline, whatever the environment says
    one: cuda.experts.Nvfp4Experts, // at 1 warp a CTA (TF_NVFP4_DECODE_WARPS=1), under test
    tile: cuda.experts_tile.Tile,
    entries: [kinds.len + 3]fact.Experts, // every kind, then the K32 tile's up, down and control
    const kinds = std.enums.values(fact.Kind);
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, sms: usize, router: cuda.grouped.Router, kernels: *std.ArrayList(core.contract.gates.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.four = try cuda.experts.Nvfp4Experts.loadWarps(gpu.d, sms, 4);
        errdefer self.four.unload();
        self.one = try cuda.experts.Nvfp4Experts.loadWarps(gpu.d, sms, 1);
        errdefer self.one.unload();
        self.tile = try cuda.experts_tile.Tile.load(gpu.d, sms, .both);
        errdefer self.tile.unload();
        for (&self.entries, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .kind = if (i < kinds.len) kinds[i] else kinds[i - kinds.len], .k = &self.four, .k1 = &self.one, .tile = if (i >= kinds.len) &self.tile else null, .router = router };
        errdefer for (&self.entries) |*x| x.deinit();
        for (&self.entries) |*x| try kernels.append(a, x.kernel());
        return self;
    }
    pub fn destroy(self: *Group) void {
        for (&self.entries) |*x| x.deinit();
        self.tile.unload();
        self.one.unload();
        self.four.unload();
        self.a.destroy(self);
    }
};
