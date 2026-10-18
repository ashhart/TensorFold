//! H3's portable host checks: Metal geometry and schedule, CUDA library fixtures and compile-only libraries, no GPU.
const std = @import("std");

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const st = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/geometry_test.zig"), .target = target, .imports = &.{.{ .name = "safetensors", .module = st }} });
    const filter = b.option([]const u8, "h3-filter", "Run one H3 host regression");
    const cases = b.addTest(.{ .root_module = root, .filters = if (filter) |f| &.{f} else &.{} });
    const run = b.addRunArtifact(cases);
    b.step("test-h3-geometry", "H3 host admission and schedule regression fixtures").dependOn(&run.step);
    test_step.dependOn(&run.step);
    if (target.result.os.tag != .macos) return;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") });
    const model = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/h3.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = st }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "h3_kernels", .module = kernels } } });
    const tool = b.addExecutable(.{ .name = "tf-h3-host-compile", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/h3_dit.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "h3", .module = model }, .{ .name = "metal", .module = metal } } }) });
    b.step("compile-h3-host", "Compile the actual H3 tool without compiling or executing shaders").dependOn(&tool.step);
}

pub fn targets(b: *std.Build, cuda: *std.Build.Module, all: *std.Build.Step) void {
    const m = module(b, b.graph.host, cuda);
    const groups = [_]struct { []const u8, []const u8, []const u8 }{
        .{ "test-h3-ownership", "h3 ownership:", "H3 CUDA buffers, parents and failed loads kept or released whole" },
        .{ "test-h3-metadata", "h3 metadata:", "H3 CUDA projection shapes refused before any allocation" },
        .{ "test-h3-arithmetic", "h3 arithmetic:", "H3 CUDA byte products, sizes and offsets checked for overflow" },
        .{ "test-h3-blocks", "h3 blocks:", "H3 CUDA checkpoints with missing, gapped or partial blocks refused" },
    };
    for (groups) |g| {
        b.step(g[0], g[2]).dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{g[1]} })).step);
    }
    const selected = b.option([]const u8, "h3-cuda-filter", "Select one H3 CUDA library regression");
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = if (selected) |v| &.{v} else &.{} }));
    run.has_side_effects = true;
    all.dependOn(&run.step);
    b.step("test-h3-boundaries", "H3 C ABI and weight fixtures without GPU execution").dependOn(&run.step);
    const compile_step = b.step("compile-h3-library", "Compile complete host and Linux H3 C libraries with test-only images");
    inline for (.{ false, true }) |cross| {
        const target = if (cross) b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) else b.graph.host;
        const runtime = if (cross) @import("cuda.zig").runtime(b, target, .debug, &.{}) else cuda;
        const lib = b.addLibrary(.{ .name = if (cross) "tf_h3_linux_compile_only" else "tf_h3_host_compile_only", .linkage = .dynamic, .root_module = module(b, target, runtime) });
        compile_step.dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "host-checks" } } }).step);
    }
}

fn module(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/cuda_lib.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    m.addAnonymousImport("fatbin_h3", .{ .root_source_file = b.addWriteFiles().add("h3-compile-only.fatbin", "") });
    m.addCSourceFile(.{ .file = b.path("zig/tests/cuda/h3_abi_test.c"), .flags = &.{"-std=c11"} });
    return m;
}

/// `zig build tf-h3 -Dnvcc=...`: libtf_h3.so, MiniMax H3's blocks for a host pipeline, with its own kernel image.
pub fn library(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, h3_image: ?std.Build.LazyPath) void {
    const step = b.step("tf-h3", "libtf_h3.so: MiniMax H3 / FastH3's transformer blocks on CUDA in bf16");
    const image = h3_image orelse {
        step.dependOn(&b.addFail("tf-h3 needs -Dnvcc").step);
        return;
    };
    const cuda = @import("cuda.zig").runtime(b, target, optimize, &.{});
    const lib = b.addLibrary(.{ .name = "tf_h3", .linkage = .dynamic, .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/h3/cuda_lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = cuda }},
    }) });
    lib.root_module.addAnonymousImport("fatbin_h3", .{ .root_source_file = image });
    step.dependOn(&b.addInstallArtifact(lib, .{}).step);
}
