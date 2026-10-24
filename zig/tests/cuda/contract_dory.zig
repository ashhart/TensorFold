//! Dory's rows kernels on the CUDA contract runner: each registration and control, owned until the run ends.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const l2_k = @import("contract_l2.zig");
const qk_k = @import("contract_qk_l2.zig");
const act_k = @import("contract_act_l2.zig");
const dense_k = @import("contract_dense_scaled.zig");

const g = core.contract.gates;

pub const Dory = struct {
    l2: [std.enums.values(l2_k.Which).len]l2_k.L2 = undefined,
    l2s: usize = 0,
    qk_rows: ?cuda.qk_rows.QkRows = null,
    qk: [qk_k.forms.len]qk_k.Qk = undefined,
    qks: usize = 0,
    act_rows: ?cuda.act_rows.ActRows = null,
    act: [std.enums.values(act_k.Which).len]act_k.Act = undefined,
    acts: usize = 0,
    dense_rows: ?cuda.dense_rows.DenseRows = null,
    dense: [dense_k.tiles.len]dense_k.Scaled = undefined,
    denses: usize = 0,

    /// Every Dory registration onto `kernels`; the controls' symbols from the contract_controls image `ctl_mod`.
    pub fn add(self: *Dory, gpu: check.Gpu, s: cuda.Stream, norms: *const cuda.norms.Norms, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        for (std.enums.values(l2_k.Which), &self.l2) |w, *t| {
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .norms = norms, .ctl = if (l2_k.control(w) != null) try ctl_mod.function(l2_k.symbol(w)) else null };
            self.l2s += 1;
        }
        for (&self.l2) |*t| try kernels.append(a, t.kernel());
        self.qk_rows = try cuda.qk_rows.QkRows.load(gpu.d);
        for (qk_k.forms, &self.qk) |f, *t| {
            const ctl = if (f.keep) |keep| try ctl_mod.function(if (keep == 2) "tf_ctl_qk_l2_drop_rope" else "tf_ctl_qk_l2_drop_norm") else null;
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .k = &self.qk_rows.?, .ctl = ctl, .arena = .init(a) };
            self.qks += 1;
        }
        for (&self.qk) |*t| try kernels.append(a, t.kernel());
        self.act_rows = try cuda.act_rows.ActRows.load(gpu.d);
        for (std.enums.values(act_k.Which), &self.act) |w, *t| {
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .k = &self.act_rows.?, .ctl = if (act_k.control(w) != null) try ctl_mod.function(act_k.symbol(w)) else null };
            self.acts += 1;
        }
        for (&self.act) |*t| try kernels.append(a, t.kernel());
        self.dense_rows = try cuda.dense_rows.DenseRows.load(gpu.d);
        for (&self.dense, 0..) |*t, i| {
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .tile = i, .g = &self.dense_rows.? };
            self.denses += 1;
        }
        for (&self.dense) |*t| try kernels.append(a, t.kernel());
    }

    pub fn deinit(self: *Dory) void {
        for (self.l2[0..self.l2s]) |*t| t.deinit();
        for (self.qk[0..self.qks]) |*t| t.deinit();
        if (self.qk_rows) |*q| q.unload();
        for (self.act[0..self.acts]) |*t| t.deinit();
        if (self.act_rows) |*q| q.unload();
        for (self.dense[0..self.denses]) |*t| t.deinit();
        if (self.dense_rows) |*q| q.unload();
    }
};
