//! Parse's device-free config, tensor index, and image processor tests.
const std = @import("std");
pub fn hostTests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .link_libc = true }));
    const options = b.addOptions();
    options.addOption(?[]const u8, "model_dir", b.option([]const u8, "parse-model-dir", "Existing Nemotron Parse checkpoint for host validation"));
    options.addOption(?[]const u8, "reference_dir", b.option([]const u8, "parse-reference-dir", "Existing upstream Parse processor fixture directory"));
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/families/nemotron_parse/host_test.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    });
    mod.addOptions("parse_options", options);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mod }));
    run.has_side_effects = true; // optional checkpoint data is outside the source tree
    all.dependOn(&run.step);
    b.step("test-nemotron-parse-host", "Parse config, tensor schema and image preparation without GPU work").dependOn(&run.step);
}
