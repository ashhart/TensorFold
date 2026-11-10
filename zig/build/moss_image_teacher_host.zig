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
    generateSteps(b, all);
    compileCuda(b);
}

/// The generate driver on an inert backend: its seeded latents, schedule, both branches and the two refusals.
/// The four CPU-only family files sit beside the seam so one module holds them, as the engine module does.
fn generateSteps(b: *std.Build, all: *std.Build.Step) void {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = b.graph.host });
    st.addImport("sys", b.createModule(.{ .root_source_file = b.path("zig/src/platform/sys.zig"), .target = b.graph.host, .link_libc = true }));
    const family = "zig/src/families/moss_image/";
    const files = b.addWriteFiles();
    for ([_][]const u8{ "config.zig", "noise.zig", "noise_torch.json", "schedule.zig", "schedule_torch.json", "generate.zig" }) |name| _ = files.addCopyFile(b.path(b.fmt("{s}{s}", .{ family, name })), name);
    const seam = files.add("moss_generate_tensorfold.zig", "pub const moss_image = .{ .config = @import(\"config.zig\"), .noise = @import(\"noise.zig\"), .schedule = @import(\"schedule.zig\"), .generate = @import(\"generate.zig\") };\n");
    const tensorfold = b.createModule(.{ .root_source_file = seam, .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_generate_host.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "tensorfold", .module = tensorfold }, .{ .name = "safetensors", .module = st } } });
    const checks = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    const step = b.step("test-moss-generate-host", "MOSS generate driver on an inert backend: seed, both branches' skips, step indexing, the two refusals");
    step.dependOn(&checks.step);
    all.dependOn(&checks.step);
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
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .link_libc = true });
    const root = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "tokenizer", .module = tokenizer } } });
    const exe = b.addExecutable(.{ .name = "moss-teacher-metal-compile", .root_module = root });
    const gen = b.createModule(.{ .root_source_file = b.path("zig/tests/moss_image_generate.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "tokenizer", .module = tokenizer } } });
    const gen_exe = b.addExecutable(.{ .name = "moss-generate-metal-compile", .root_module = gen });
    const step = b.step("compile-moss-teacher-metal", "Compile the MOSS Metal teacher and generate CLIs without shader compilation or execution");
    step.dependOn(&exe.step);
    step.dependOn(&gen_exe.step);
}
