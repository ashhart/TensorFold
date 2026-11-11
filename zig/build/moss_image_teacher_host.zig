//! The actual MOSS-Image case driver on an inert backend, with no shader producers or device libraries.
const std = @import("std");
pub fn steps(b: *std.Build, all: *std.Build.Step) void {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = b.graph.host });
    st.addImport("sys", b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = b.graph.host, .link_libc = true }));
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_run_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "safetensors", .module = st }} });
    const exe = b.addExecutable(.{ .name = "moss-teacher-host", .root_module = root });
    const opts = b.addOptions();
    opts.addOptionPath("probe", exe.getEmittedBin());
    const tests = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_teacher_host_test.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{.{ .name = "case_fixture", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_fixture.zig"), .target = b.graph.host }) }} });
    tests.addOptions("teacher_host_options", opts);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    const step = b.step("test-moss-teacher-host", "MOSS per-step teacher failure and inconclusive results on synthetic data");
    const unit = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_teacher.zig"), .target = b.graph.host }) }));
    step.dependOn(&unit.step);
    all.dependOn(&unit.step);
    step.dependOn(&run.step);
    all.dependOn(&run.step);
    compileCuda(b);
}

/// Type-check and link the real CUDA CLI with absent images, without loading a driver or running a model.
fn compileCuda(b: *std.Build) void {
    const cuda = @import("cuda_runtime.zig").runtime(b, b.graph.host, .Debug, &.{});
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = b.graph.host, .link_libc = true });
    const moss = b.createModule(.{ .root_source_file = b.path("zig/src/families/moss_image/cuda.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_cuda.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "moss_image", .module = moss } } });
    const exe = b.addExecutable(.{ .name = "moss-teacher-cuda-compile", .root_module = root });
    b.step("compile-moss-teacher-cuda", "Compile the MOSS CUDA teacher CLI without kernel images or execution").dependOn(&exe.step);
}

/// Compile the real Metal CLI on the host graph's empty test-only packed image, without shader producers.
pub fn compileMetal(b: *std.Build, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const tokenizer = engine.import_table.get("tokenizer") orelse b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .link_libc = true });
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "tokenizer", .module = tokenizer } } });
    const exe = b.addExecutable(.{ .name = "moss-teacher-metal-compile", .root_module = root });
    b.step("compile-moss-teacher-metal", "Compile the MOSS Metal teacher CLI without shader compilation or execution").dependOn(&exe.step);
}
