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

pub fn hipTests(b: *std.Build, hip: *std.Build.Module, test_step: *std.Build.Step) void {
    const family = hipModule(b, hip);
    const weights_host = b.addRunArtifact(b.addTest(.{ .root_module = family, .filters = &.{"dots HIP weights:"} }));
    b.step("test-dots-hip-weights-host", "HIP weight ownership, layout and budget refusal without GPU calls").dependOn(&weights_host.step);
    test_step.dependOn(&weights_host.step);
    const upload_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/dots_weights_test.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "dots_mocr", .module = family }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
    const upload = b.addTest(.{ .name = "tf-dots-hip-weights-test", .root_module = upload_module, .filters = &.{"dots HIP device weights:"} });
    b.step("build-dots-hip-weights-test", "Build the real checkpoint upload check without running a GPU").dependOn(&b.addInstallArtifact(upload, .{}).step);
    const upload_run = b.addRunArtifact(upload);
    upload_run.has_side_effects = true;
    b.step("test-dots-hip-weights", "Upload TF_DOTS_DIR on the selected HIP GPU after memory admission").dependOn(&upload_run.step);
}

pub fn hipModule(b: *std.Build, hip: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/dots_mocr/hip.zig"),
        .target = hip.resolved_target,
        .optimize = hip.optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = hip.import_table.get("core").? } },
    });
}
