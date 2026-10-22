const std = @import("std");
pub fn steps(b: *std.Build, core: *std.Build.Module) void {
    const unit = b.addTest(.{ .root_module = core, .filters = &.{"proposal vocabulary:"} });
    const run = b.addRunArtifact(unit);
    run.has_side_effects = true;
    b.step("test-proposal-vocab-host", "Immutable proposal artifact identity, IDs, padding and coverage").dependOn(&run.step);
    const tool = b.addSystemCommand(&.{"python3"});
    tool.addFileArg(b.path("tools/test_draft_vocab_responses.py"));
    tool.addArg("-v");
    tool.setCwd(b.path("."));
    tool.setEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1");
    tool.has_side_effects = true;
    b.step("test-proposal-vocab-tool-host", "Offline committed-response vocabulary builder on synthetic receipts").dependOn(&tool.step);
}
