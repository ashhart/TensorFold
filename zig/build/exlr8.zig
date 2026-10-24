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

/// The Metal projections' synthetic check, the real-layer bench, the dense-format check and their host tests.
pub fn metalTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, host_tests: *std.Build.Step) void {
    const em = b.createModule(.{
        .root_source_file = b.path("zig/src/core/exlr8_metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } },
    });
    host_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = em })).step);
    const core = alias(b, target, optimize, em, "ex");
    const shared = testModules(b, target, optimize, core);
    const dense_mods = denseModules(b, target, optimize, core, shared[0]);
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "metal", .module = metal }, .{ .name = "exlr8", .module = core }, .{ .name = "exlr8_metal", .module = em }, shared[0], shared[1], shared[2], dense_mods[0], dense_mods[1] };
    for ([_][3][]const u8{
        .{ "tf-exlr8-check", "zig/tests/exlr8_check.zig", "EXLR8 Metal projections: codebook, FP64 bounds, row windows, frozen bytes (no models)" },
        .{ "tf-exlr8-bench", "zig/tests/exlr8_bench.zig", "EXLR8 Metal projections on a real MoE layer: FP64, row windows, then GB/s by stage" },
        .{ "tf-exlr8-dense-check", "zig/tests/exlr8_dense_check.zig", "EXLR8 dense formats on Metal: decode bit-equal to the reference, FP64, row windows (TF_EXLR8_DIR adds real matrices)" },
    }) |p| {
        const exe = b.addExecutable(.{ .name = p[0], .root_module = b.createModule(.{ .root_source_file = b.path(p[1]), .target = target, .optimize = optimize, .link_libc = true, .imports = imports }) });
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
    return alias(b, target, optimize, core, "exlr8");
}

/// An `exlr8` module whose parts are `field` of a module that already holds the decoder.
fn alias(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, core: *std.Build.Module, field: []const u8) *std.Build.Module {
    var text: []const u8 = b.fmt("const e = @import(\"core\").{s};\n", .{field});
    for ([_][]const u8{ "codec", "experts", "dense", "job", "manifest", "node", "pack", "release", "safetensors", "xxhash64" }) |name| text = b.fmt("{s}pub const {s} = e.{s};\n", .{ text, name, name });
    const file = b.addWriteFiles().add("exlr8_alias.zig", text);
    return b.createModule(.{ .root_source_file = file, .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
}

/// GLM-5.3 (glm_moe_dsa) on EXLR8: a layer on Metal against the fp32 reference forward's tensors.
pub fn glmTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module) void {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    for ([_][3][]const u8{
        .{ "tf-glm-dsa-layer-check", "zig/tests/glm_dsa_layer_check.zig", "One GLM-5.3 layer on Metal against ref_layer.py's fp32 tensors (MODEL_DIR REF_DIR)" },
        .{ "tf-glm-dsa-tp-check", "zig/tests/glm_dsa_tp_check.zig", "Two ranks of one GLM-5.3 layer on one Mac, summed, against the reference (MODEL_DIR REF_DIR)" },
        .{ "tf-glm-dsa-run", "zig/tests/glm_dsa_run.zig", "GLM-5.3 on the pair without the server: a prompt's greedy reply (MODEL_DIR SETTINGS IDS_FILE)" },
        .{ "tf-glm-dsa-bench", "zig/tests/glm_dsa_bench.zig", "One rank's share of a GLM-5.3 layer timed piece by piece on the GPU (MODEL_DIR LAYER)" },
        .{ "tf-glm-dsa-index-check", "zig/tests/glm_dsa_index_check.zig", "The DSA indexer's Metal scan on synthetic keys: the blocked kernel against the row kernel" },
        .{ "tf-exlr8-prompt-check", "zig/tests/exlr8_prompt_check.zig", "The tensor-unit prompt kernel against the lane kernel on real expert halves (MODEL_DIR)" },
        .{ "tf-glm-dsa-prompt-check", "zig/tests/glm_dsa_prompt_check.zig", "A prompt window's experts against the decode path's 16-row windows (MODEL_DIR)" },
    }) |p| {
        const exe = b.addExecutable(.{ .name = p[0], .root_module = b.createModule(.{
            .root_source_file = b.path(p[1]),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "tokenizer", .module = tokenizer } },
        }) });
        b.step(p[0], p[2]).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
}
