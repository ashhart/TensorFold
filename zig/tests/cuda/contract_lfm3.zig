//! lfm3.cu under the contract: Liquid's 7-tap conv with its snapshots and its commit, each controlled (the 3-tap rigs).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const lfm2m = @import("contract_lfm2_moe.zig");

const g = core.contract.gates;

/// The kernels and their controls, registered onto `kernels`; the controls from the contract_controls image `ctl`.
pub const Set = struct {
    mod: ?cuda.Module = null,
    conv: [2]lfm2m.Conv = undefined,
    commit: [2]lfm2m.Commit = undefined,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.lfm3);
        const m = self.mod.?;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&self.conv, [_]bool{ false, true }) |*x, ctrl| x.* = .{ .rig = rig, .f = try (if (ctrl) ctl else m).function(if (ctrl) "tf_ctl_liquid_conv_7" else "tf_liquid_conv_7"), .control = ctrl, .taps = 7 };
        for (&self.commit, [_]bool{ false, true }) |*x, ctrl| x.* = .{ .rig = rig, .f = try (if (ctrl) ctl else m).function(if (ctrl) "tf_ctl_liquid_commit_7" else "tf_liquid_commit_7"), .control = ctrl, .taps = 7 };
        for (&self.conv) |*x| try kernels.append(gpu.gpa, x.kernel());
        for (&self.commit) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        const m = self.mod orelse return;
        for (&self.conv) |*x| x.deinit();
        for (&self.commit) |*x| x.deinit();
        var mm = m;
        mm.unload();
    }
};

test "the 7-tap rigs name their own kernels" {
    try std.testing.expect(std.mem.indexOf(u8, lfm2m.convSpec(7, false).name, "tf_liquid_conv_7") != null);
    try std.testing.expect(std.mem.indexOf(u8, lfm2m.commitSpec(7, true).name, "7-tap") != null);
    try std.testing.expectEqual(@as(usize, 13), lfm2m.commitSpec(7, false).served[0].k);
}
