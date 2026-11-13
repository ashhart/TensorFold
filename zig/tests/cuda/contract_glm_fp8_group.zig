//! The five paged-FP8 GLM contract cases and their storage.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const attn = @import("contract_glm_attn.zig");
const pages = @import("contract_glm_fp8_pages.zig");
const g = @import("core").contract.gates;

pub const Group = struct {
    a: std.mem.Allocator,
    lat: pages.LatWrite,
    chunks: [4]attn.Chunks,

    pub fn create(
        a: std.mem.Allocator,
        gpu: check.Gpu,
        s: cuda.Stream,
        f: glm.glue.Kernels,
        kernels: *std.ArrayList(g.Kernel),
    ) !*Group {
        const self = try a.create(Group);
        self.* = .{
            .a = a,
            .lat = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f.lat_write_fp8_paged },
            .chunks = undefined,
        };
        for (&self.chunks, [_]u32{ 16, 16, 32, 32 }, [_]bool{ false, true, false, true }) |*x, hb, sparse|
            x.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .f = if (hb == 16) f.chunks_16_fp8_paged else f.chunks_32_fp8_paged,
                .merge = f.merge,
                .form = .{ .hb = hb, .sparse = sparse, .fp8 = true },
            };
        errdefer self.destroy();
        try kernels.append(a, self.lat.kernel());
        for (&self.chunks) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        self.lat.deinit();
        for (&self.chunks) |*x| x.deinit();
        self.a.destroy(self);
    }
};
