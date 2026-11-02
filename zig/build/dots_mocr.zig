//! dots.mocr checkpoint and processor host checks, without loading a GPU runtime.
const std = @import("std");

pub fn hostTests(b: *std.Build, test_step: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = b.graph.host, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, b.graph.host));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .link_libc = true }));
    for ([_]std.builtin.OptimizeMode{ .debug, .fast }) |mode| {
        const family = b.createModule(.{
            .root_source_file = b.path("zig/src/families/dots_mocr/dots_mocr.zig"),
            .target = b.graph.host,
            .optimize = mode,
            .link_libc = true,
            .imports = &.{.{ .name = "core", .module = core }},
        });
        const run = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{"dots:"} }));
        run.has_side_effects = true;
        const name = if (mode == .debug) "test-dots-mocr-host" else "test-dots-mocr-fast";
        b.step(name, "dots.mocr config, tensor inventory and image processor checks; TF_DOTS_DIR checks local shards").dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}
