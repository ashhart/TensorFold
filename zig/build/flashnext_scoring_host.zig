//! Flash Next's existing proposal checks and its decision capability share one production module.
const std = @import("std");
pub fn steps(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, family: *std.Build.Module, parent: *std.Build.Step) void {
    @import("flashnext_proposal_host.zig").steps(b, cuda, core, lanes, family);
    const draws = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "Flash Next nonfinite:", "Flash Next batch draws:" } }));
    b.step("test-flashnext-draw-host", "Production row draws and launch grids on the recording CUDA device").dependOn(&draws.step);
    parent.dependOn(&draws.step);
    const scoring = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{ "decision capability:", "the logsumexp" } }));
    b.step("test-flashnext-score-host", "Actual one-rank scoring capability and label reference without devices").dependOn(&scoring.step);
    parent.dependOn(&scoring.step);
}
