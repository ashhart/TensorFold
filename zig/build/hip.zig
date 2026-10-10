//! HIP: mock-backed admission tests join `zig build test`; GPU builds and runs are opt-in steps (-Dhipcc, -Dhip-include, -Dhip-arch).

const std = @import("std");
const caps = @import("../src/hip/caps.zig");

/// hip-host-test (part of `test`), hip-gpu-build and hip-gpu-test, hip-affine-build and hip-affine-test.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const hip_fixtures = b.addOptions();
    for ([_][]const u8{ "success", "failed", "missing" }) |kind| {
        const fixture_module = b.createModule(.{ .target = b.graph.host, .link_libc = true });
        fixture_module.addCSourceFile(.{
            .file = b.path("zig/tests/hip_mock.c"),
            .flags = if (std.mem.eql(u8, kind, "missing")) &.{ "-DOMIT_INIT", "-DINIT_RESULT=0" } else if (std.mem.eql(u8, kind, "failed")) &.{"-DINIT_RESULT=1"} else &.{"-DINIT_RESULT=0"},
        });
        const fixture = b.addLibrary(.{ .name = b.fmt("hip-mock-{s}", .{kind}), .linkage = .dynamic, .root_module = fixture_module });
        hip_fixtures.addOptionPath(kind, fixture.getEmittedBin());
    }
    const hip_test_module = b.createModule(.{
        .root_source_file = b.path("zig/src/hip/admission_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    hip_test_module.addOptions("hip_fixtures", hip_fixtures);
    const hip_tests = b.addTest(.{ .root_module = hip_test_module });
    const run_hip_tests = b.addRunArtifact(hip_tests);
    test_step.dependOn(&run_hip_tests.step);
    b.step("hip-host-test", "HIP admission tests without GPU work").dependOn(&run_hip_tests.step);
    const hipcc = b.option([]const u8, "hipcc", "HIP compiler for model-free tests") orelse "hipcc";
    const hipcc_resolved = b.findProgram(.{ .names = &.{hipcc} }) orelse hipcc;
    const hip_include = b.option([]const u8, "hip-include", "HIP header directory") orelse
        b.pathResolve(&.{ std.fs.path.dirname(hipcc_resolved) orelse "/opt/rocm/bin", "..", "include" });
    const hip_arch = b.option([]const u8, "hip-arch", "Exact GPU architecture for the probe code object") orelse "gfx1151";
    if (caps.Caps.of(hip_arch) == null or std.mem.indexOfScalar(u8, hip_arch, ':') != null)
        std.debug.panic("-Dhip-arch {s} is not an exact name in the caps table (zig/src/hip/caps.zig)", .{hip_arch});
    const hip_compile = b.addSystemCommand(&.{ hipcc, "--genco", b.fmt("--offload-arch={s}", .{hip_arch}), "-O2", "-ffp-contract=off" });
    hip_compile.addFileArg(b.path("zig/kernels/hip/runtime_tests.hip"));
    hip_compile.addArg("-o");
    const hip_object = hip_compile.addOutputFileArg("hip-runtime-probe.hsaco");
    const hip_files = b.addWriteFiles();
    _ = hip_files.addCopyFile(hip_object, "probe.hsaco");
    const hip_probe = b.createModule(.{ .root_source_file = hip_files.add("probe.zig", b.fmt("pub const arch = \"{s}\";\npub const bytes align(8) = @embedFile(\"probe.hsaco\").*;\n", .{hip_arch})) });
    const hip_gpu_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/runtime_tests.zig"), .target = target, .link_libc = true });
    hip_gpu_module.addIncludePath(.{ .cwd_relative = hip_include });
    hip_gpu_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    hip_gpu_module.addImport("hip_probe", hip_probe);
    const hip_gpu_test = b.addTest(.{ .root_module = hip_gpu_module });
    b.step("hip-gpu-build", "Compile HIP runtime tests without running GPU work").dependOn(&hip_gpu_test.step);
    b.step("hip-gpu-test", "Real HIP copies, fills and architecture-selected module launches").dependOn(&b.addRunArtifact(hip_gpu_test).step);
    const affine_compile = b.addSystemCommand(&.{ hipcc, "--genco", b.fmt("--offload-arch={s}", .{hip_arch}), "-O2", "-ffp-contract=off" });
    affine_compile.addFileArg(b.path("zig/kernels/hip/affine.hip"));
    affine_compile.addArg("-o");
    const affine_image = affine_compile.addOutputFileArg("affine.hsaco");
    const affine_files = b.addWriteFiles();
    _ = affine_files.addCopyFile(affine_image, "affine.hsaco");
    _ = affine_files.addCopyFile(b.path("zig/tests/hip_affine_g64.hex"), "golden.hex");
    _ = affine_files.addCopyFile(b.path("zig/tests/hip_affine_sensitive.hex"), "sensitive.hex");
    _ = affine_files.addCopyFile(b.path("zig/tests/hip_affine_matrix.hex"), "matrix.hex");
    const affine_data = b.createModule(.{ .root_source_file = affine_files.add("data.zig", b.fmt("pub const arch = \"{s}\";\npub const image align(8) = @embedFile(\"affine.hsaco\").*;\npub const hex = @embedFile(\"golden.hex\");\npub const sensitive = @embedFile(\"sensitive.hex\");\npub const matrix = @embedFile(\"matrix.hex\");\n", .{hip_arch})) });
    const affine_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/affine_gpu_test.zig"), .target = target, .link_libc = true });
    affine_module.addIncludePath(.{ .cwd_relative = hip_include });
    affine_module.addImport("affine_data", affine_data);
    affine_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    const affine_test = b.addTest(.{ .root_module = affine_module });
    b.step("hip-affine-build", "Compile affine golden GPU test without executing").dependOn(&affine_test.step);
    b.step("hip-affine-test", "Run exact affine golden GPU regression").dependOn(&b.addRunArtifact(affine_test).step);
}
