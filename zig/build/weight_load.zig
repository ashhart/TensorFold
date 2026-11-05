const std = @import("std");
fn module(b: *std.Build, target_: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module, nemo: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/tests/cuda_weight_load.zig"), .target = target_, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen }, .{ .name = "nemotron", .module = nemo } } });
}
pub fn target(b: *std.Build, t: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module, nemo: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-cuda-weight-load", .root_module = module(b, t, optimize, cuda, qwen, nemo) });
    const install = b.addInstallArtifact(exe, .{});
    b.step("cuda-weight-load", "Build the actual Qwen/Nemotron load and final device-weight receipt tool").dependOn(&install.step);
    b.step("compile-cuda-weight-load", "Compile the load receipt tool without running it").dependOn(&exe.step);
}
pub fn host(b: *std.Build, cuda: *std.Build.Module, mods: anytype, parent: *std.Build.Step) void {
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda_weight_digest_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} }), .filters = &.{"weight digest:"} }));
    b.step("test-weight-digest-host", "Pinned device readback, tail and failure controls without a GPU").dependOn(&tests.step);
    parent.dependOn(&tests.step);
    const upload = b.step("test-weight-upload-host", "File producer custody and unchanged discrete adapters without a GPU");
    const fake = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda_weight_upload_fake.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    for ([_]struct { path: []const u8, family: *std.Build.Module }{
        .{ .path = "zig/src/families/qwen3_5/cuda_upload_test.zig", .family = mods.qwen35 },
        .{ .path = "zig/src/families/nemotron/cuda_upload_test.zig", .family = mods.nemotron },
    }) |item| {
        const m = b.createModule(.{ .root_source_file = b.path(item.path), .target = b.graph.host, .link_libc = true });
        var imports = item.family.import_table.iterator();
        while (imports.next()) |entry| m.addImport(entry.key_ptr.*, entry.value_ptr.*);
        m.addImport("upload_fake", fake);
        upload.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"weight upload:"} })).step);
    }
    parent.dependOn(upload);
    const exe = b.addExecutable(.{ .name = "weight-load-compile-host", .root_module = module(b, b.graph.host, .debug, cuda, mods.qwen35, mods.nemotron) });
    b.step("compile-weight-load-host", "Compile actual loader calls without running a driver or model").dependOn(&exe.step);
}
