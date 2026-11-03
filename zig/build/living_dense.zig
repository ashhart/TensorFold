//! Dense CUDA learner checks, including the real constructor under a fake driver with deliberately invalid images.
const std = @import("std");

pub fn target(b: *std.Build, t: std.Build.ResolvedTarget, o: std.builtin.OptimizeMode, cuda: *std.Build.Module, qwen: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/cli/qwen35_learn.zig"), .target = t, .optimize = o, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "qwen35", .module = qwen } } });
    const exe = b.addExecutable(.{ .name = "tf-qwen35-learn", .root_module = m });
    b.installArtifact(exe);
    b.step("compile-qwen35-learn", "Compile actual dense model qualification without execution").dependOn(&exe.step);
}

pub fn host(b: *std.Build, cuda: *std.Build.Module, qwen: *std.Build.Module, draft_ids: *std.Build.Module, comptime runtime: anytype, comptime family: anytype, comptime n: usize) void {
    const t = b.graph.host;
    const diagnostics = b.step("test-qwen-gradient-host", "GDN gate FP64, launch selection and directional diagnostics");
    for ([_][]const u8{ "zig/gdn_gate_host_test.zig", "zig/src/cli/qwen35_slope.zig" }) |root|
        diagnostics.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path(root), .target = t, .optimize = .debug }) })).step);
    diagnostics.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"GDN gate launch:"} })).step);

    const pin = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/dense_serving_test.zig"), .target = t, .optimize = .debug }) }));
    b.step("test-qwen-serving-host", "Landed CUDA kernels and dense serving launch pins").dependOn(&pin.step);
    b.step("test-qwen-pocket-host", "Dense device publication and checked reads under CUDA failures").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen, .filters = &.{ "dense pocket:", "dense media:", "dense save:", "dense read cost:" } })).step);
    const placeholder = b.addWriteFiles().add("fake-driver-only.fatbin", "FAKE DRIVER ONLY, NEVER A DEVICE IMAGE");
    const images: [n]?std.Build.LazyPath = @splat(placeholder);
    const fake_cuda = runtime(b, t, .debug, &images);
    const contract_compile = b.addObject(.{ .name = "gdn-gate-contract-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/gdn_gate_compile.zig"), .target = t, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = fake_cuda }} }) });
    b.step("compile-gdn-gate-host", "Analyze both device GDN references with inert images").dependOn(&contract_compile.step);
    const fake = family(b, t, .debug, fake_cuda, draft_ids);
    const learn_module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/qwen35_learn.zig"), .target = t, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = fake_cuda }, .{ .name = "qwen35", .module = fake.qwen35 } } });
    const learn_compile = b.addObject(.{ .name = "qwen35-learn-host", .root_module = b.createModule(.{ .root_source_file = b.path("zig/qwen35_learn_compile.zig"), .target = t, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "tool", .module = learn_module }} }) });
    b.step("compile-qwen35-learn-host", "Analyze dense learner qualification with inert CUDA images").dependOn(&learn_compile.step);
    const constructors = b.addRunArtifact(b.addTest(.{ .root_module = fake.qwen35, .filters = &.{"dense constructor:"} }));
    b.step("test-dense-constructors-host", "Actual dense constructor ownership and completion failures").dependOn(&constructors.step);
}
