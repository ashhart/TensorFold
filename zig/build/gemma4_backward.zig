//! Mathematical Gemma adjoints run on the host in the default test graph, without any GPU imports.
const std = @import("std");
pub fn tests(b: *std.Build, parent: *std.Build.Step) void {
    const filter = b.option([]const u8, "gemma4-backward-filter", "One host Gemma backward regression");
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/attention_backward.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = if (filter) |text| &.{text} else &.{} }));
    const step = b.step("test-gemma4-attention-backward-host", "Gemma sliding/global attention adjoints against independent binary64 finite differences");
    step.dependOn(&run.step);
    parent.dependOn(step);
}
