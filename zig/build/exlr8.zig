const std = @import("std");

/// The EXLR8 core as the checkpoint CLI imports it (node pulls read manifests and the index), from `core`'s copy.
pub fn cliImport(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module) std.Build.Module.Import {
    return .{ .name = "exlr8", .module = coreAlias(b, target, optimize, core) };
}

/// EXLR8's host tests: the reader and decoder against the reference loader's fixtures (TF_EXLR8_DIR adds the layers).
pub fn hostTests(b: *std.Build, target: std.Build.ResolvedTarget, host_tests: *std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exlr8_oracle_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    }) });
    const run = b.addRunArtifact(tests);
    run.setCwd(b.path("."));
    b.step("test-exlr8", "EXLR8 reader and decoder against the reference loader's layer fixtures").dependOn(&run.step);
    host_tests.dependOn(&run.step);
}

/// The shared test modules (FP64 references, synthetic cases, a real layer's jobs) over one `exlr8` core module.
pub fn testModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module) [3]std.Build.Module.Import {
    const ref = b.createModule(.{ .root_source_file = b.path("zig/tests/exlr8_ref.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "exlr8", .module = core }} });
    const shared: []const std.Build.Module.Import = &.{ .{ .name = "exlr8", .module = core }, .{ .name = "exlr8_ref", .module = ref } };
    return .{
        .{ .name = "exlr8_ref", .module = ref },
        .{ .name = "exlr8_cases", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/exlr8_cases.zig"), .target = target, .optimize = optimize, .imports = shared }) },
        .{ .name = "exlr8_layer", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/exlr8_layer.zig"), .target = target, .optimize = optimize, .imports = shared }) },
    };
}

/// The Metal projections' synthetic check, the real-layer bench and the projection's host tests.
pub fn metalTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, host_tests: *std.Build.Step) void {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/exlr8.zig"), .target = target, .optimize = optimize });
    const projection = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exlr8_projection.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "exlr8", .module = core } },
    });
    host_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = projection })).step);
    const shared = testModules(b, target, optimize, core);
    const dense_gpu = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exlr8_dense_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "exlr8", .module = core } },
    });
    host_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dense_gpu })).step);
    const dense_mods = denseModules(b, target, optimize, core, shared[0]);
    const dense_check = b.addExecutable(.{ .name = "tf-exlr8-dense-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/exlr8_dense_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "exlr8", .module = core }, .{ .name = "exlr8_dense_gpu", .module = dense_gpu }, .{ .name = "exlr8_projection", .module = projection }, shared[0], dense_mods[0], dense_mods[1] },
    }) });
    b.step("tf-exlr8-dense-check", "EXLR8 dense formats on Metal: decode bit-equal to the reference, FP64, row windows (TF_EXLR8_DIR adds real matrices)").dependOn(&b.addInstallArtifact(dense_check, .{}).step);
    for ([_][3][]const u8{
        .{ "tf-exlr8-check", "zig/tests/exlr8_check.zig", "EXLR8 Metal projections: codebook, FP64 bounds, row windows, frozen bytes (no models)" },
        .{ "tf-exlr8-bench", "zig/tests/exlr8_bench.zig", "EXLR8 Metal projections on a real MoE layer: FP64, row windows, then GB/s by stage" },
    }) |p| {
        const exe = b.addExecutable(.{ .name = p[0], .root_module = b.createModule(.{
            .root_source_file = b.path(p[1]),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "exlr8", .module = core }, .{ .name = "exlr8_projection", .module = projection }, shared[0], shared[1], shared[2] },
        }) });
        b.step(p[0], p[2]).dependOn(&b.addInstallArtifact(exe, .{}).step);
        if (std.mem.eql(u8, p[0], "tf-exlr8-check")) b.step("test-exlr8-gpu", "Run the EXLR8 Metal projection checks on this Mac's GPU").dependOn(&b.addRunArtifact(exe).step);
    }
}

/// The dense checks' shared modules: synthetic cases and real matrices with the reference's digests.
pub fn denseModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module, ref: std.Build.Module.Import) [2]std.Build.Module.Import {
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "exlr8", .module = core }, ref };
    return .{
        .{ .name = "exlr8_dense_cases", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/exlr8_dense_cases.zig"), .target = target, .optimize = optimize, .imports = imports }) },
        .{ .name = "exlr8_dense_real", .module = b.createModule(.{ .root_source_file = b.path("zig/tests/exlr8_dense_real.zig"), .target = target, .optimize = optimize, .imports = imports }) },
    };
}

/// The `exlr8` module where core/root.zig already holds the decoder (the CUDA side): its parts re-exported by name.
pub fn coreAlias(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module) *std.Build.Module {
    var text: []const u8 = "const e = @import(\"core\").exlr8;\n";
    for ([_][]const u8{ "codec", "experts", "dense", "job", "manifest", "node", "release", "safetensors", "xxhash64" }) |name| text = b.fmt("{s}pub const {s} = e.{s};\n", .{ text, name, name });
    const file = b.addWriteFiles().add("exlr8_alias.zig", text);
    return b.createModule(.{ .root_source_file = file, .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
}
