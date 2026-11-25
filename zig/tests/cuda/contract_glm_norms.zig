//! The runner's GLM norms as one set: the float32-weight forms, the BF16 replacements and full GLM-5.3's widths.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const f32w_k = @import("contract_glm_f32w.zig");

const g = core.contract.gates;
const F32w = f32w_k.F32w;

pub const Set = struct {
    f32ws: [std.enums.values(f32w_k.Which).len]F32w = undefined,
    bf16s: [f32w_k.bf16_entries.len]F32w = undefined,
    wides: [f32w_k.wide_entries.len]F32w = undefined,
    control: F32w = undefined,
    n: [4]usize = @splat(0), // how many of f32ws, bf16s, wides and control hold a kernel

    /// Every registration onto `kernels`, from the glm_glue image and the contract controls' frozen entrypoints.
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, glue: cuda.Module, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        for (std.enums.values(f32w_k.Which), &self.f32ws) |w, *x| {
            x.* = .init(.{ .gpu = gpu, .s = s }, w, try glue.function(switch (w) {
                .norm_y, .norm_sums => "tf_glm_rmsnorm_4096_f32w",
                else => "tf_glm_hc_finish_f32w",
            }));
            self.n[0] += 1;
        }
        for (f32w_k.bf16_entries, &self.bf16s) |e, *x| {
            x.* = .init(.{ .gpu = gpu, .s = s }, e.which, try glue.function(e.symbol));
            self.n[1] += 1;
            x.width = e.width;
            x.bf16_w = true;
            x.baseline = try controls.function(e.old);
        }
        for (f32w_k.wide_entries, &self.wides) |e, *x| {
            x.* = .init(.{ .gpu = gpu, .s = s }, e.which, try glue.function(e.symbol));
            self.n[2] += 1;
            x.width = e.width;
            x.bf16_w = true;
            x.row0 = e.row0;
        }
        self.control = .init(.{ .gpu = gpu, .s = s }, .norm_y, try glue.function("tf_glm_rmsnorm_4096_f32w"));
        self.n[3] = 1;
        self.control.bf16_gamma = true;
        for (&self.f32ws) |*x| try kernels.append(a, x.kernel());
        for (&self.bf16s) |*x| try kernels.append(a, x.kernel());
        for (&self.wides) |*x| try kernels.append(a, x.kernel());
        try kernels.append(a, self.control.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.f32ws[0..self.n[0]]) |*x| x.deinit();
        for (self.bf16s[0..self.n[1]]) |*x| x.deinit();
        for (self.wides[0..self.n[2]]) |*x| x.deinit();
        if (self.n[3] == 1) self.control.deinit();
    }
};
