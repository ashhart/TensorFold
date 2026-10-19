const std = @import("std");
pub fn steps(b: *std.Build, core: *std.Build.Module) void {
    const unit = b.addTest(.{ .root_module = core, .filters = &.{"proposal vocabulary:"} });
    const run = b.addRunArtifact(unit);
    run.has_side_effects = true;
    b.step("test-proposal-vocab-host", "Immutable proposal artifact identity, IDs, padding and coverage").dependOn(&run.step);
}
