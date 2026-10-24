//! Dory's rows kernels on the CUDA contract runner: each registration and control, owned until the run ends.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const l2_k = @import("contract_l2.zig");

const g = core.contract.gates;

pub const Dory = struct {
    l2: [std.enums.values(l2_k.Which).len]l2_k.L2 = undefined,
    l2s: usize = 0,

    /// Every Dory registration onto `kernels`; the controls' symbols from the contract_controls image `ctl`.
    pub fn add(self: *Dory, gpu: check.Gpu, s: cuda.Stream, norms: *const cuda.norms.Norms, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        for (std.enums.values(l2_k.Which), &self.l2) |w, *t| {
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .norms = norms, .ctl = if (l2_k.control(w) != null) try ctl.function(l2_k.symbol(w)) else null };
            self.l2s += 1;
        }
        for (&self.l2) |*t| try kernels.append(a, t.kernel());
    }

    pub fn deinit(self: *Dory) void {
        for (self.l2[0..self.l2s]) |*t| t.deinit();
    }
};
