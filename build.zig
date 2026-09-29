const std = @import("std");
const builtin = @import("builtin");

comptime {
    const minimum = std.SemanticVersion.parse(std.mem.trim(u8, @embedFile(".zig-version"), "\r\n")) catch unreachable;
    if (builtin.zig_version.order(minimum) == .lt)
        @compileError("TensorFold requires Zig 0.17.0-dev.2248+3f6a02acd or newer; run bash scripts/fetch-zig.sh, then .zig-toolchain/zig build");
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const coverage_module = b.createModule(.{ .root_source_file = b.path("tools/upstream_coverage.zig"), .target = b.graph.host, .optimize = .safe });
    const coverage_exe = b.addExecutable(.{ .name = "upstream-coverage", .root_module = coverage_module });
    const coverage = b.addRunArtifact(coverage_exe);
    coverage.has_side_effects = true;
    b.step("check-upstream-coverage", "Reject unreviewed Mac source additions, changes and removals").dependOn(&coverage.step);
    const coverage_record = b.addRunArtifact(coverage_exe);
    coverage_record.addArg("--record-reviewed");
    coverage_record.has_side_effects = true;
    const coverage_complete = b.addRunArtifact(coverage_exe);
    coverage_complete.addArg("--require-complete");
    coverage_complete.has_side_effects = true;
    b.step("audit-native-parity", "Reject incomplete native features as well as stale implementation/test bindings").dependOn(&coverage_complete.step);
    b.step("record-upstream-coverage", "Explicitly acknowledge reviewed source changes without claiming native support").dependOn(&coverage_record.step);
    const coverage_tests = b.addRunArtifact(b.addTest(.{ .root_module = coverage_module }));
    b.step("test-upstream-coverage", "Verify drift detection for new families, changed kernels and removed paths").dependOn(&coverage_tests.step);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Optimization mode: debug, safe, fast, small") orelse .fast;
    const sync_module = b.createModule(.{
        .root_source_file = b.path("tools/sync_upstream.zig"),
        .target = b.graph.host,
        .optimize = .safe,
    });
    const sync_exe = b.addExecutable(.{ .name = "sync-upstream", .root_module = sync_module });
    const sync = b.addRunArtifact(sync_exe);
    sync.has_side_effects = true;
    b.step("sync-upstream", "Manually fast-forward fork main from upstream and rebase the clean current branch (SSH)").dependOn(&sync.step);
    const freshness = b.addRunArtifact(sync_exe);
    freshness.addArg("--check");
    freshness.has_side_effects = true;
    b.step("check-upstream", "Fetch upstream and report commits missing from fork main and the current branch").dependOn(&freshness.step);
    const sync_tests = b.addRunArtifact(b.addTest(.{ .root_module = sync_module }));
    b.step("test-sync-upstream", "Check sync worktree and remote guards without network access").dependOn(&sync_tests.step);
    const setup_module = b.createModule(.{ .root_source_file = b.path("tools/setup_native.zig"), .target = b.graph.host, .optimize = .safe });
    const setup_tests = b.addRunArtifact(b.addTest(.{ .root_module = setup_module }));
    const setup_step = b.step("test-setup", "Check setup options, installed artifacts and prerequisite versions without network access");
    setup_step.dependOn(&setup_tests.step);
    const install_module = b.createModule(.{ .root_source_file = b.path("tools/native_install.zig"), .target = b.graph.host, .optimize = .safe });
    const install_tests = b.addRunArtifact(b.addTest(.{ .root_module = install_module }));
    setup_step.dependOn(&install_tests.step);
    const prefix = b.option([]const u8, "mlx-prefix", "MLX and mlx-c install prefix") orelse "build/mlx";
    const bindings = b.addTranslateC(.{
        .root_source_file = b.path(b.fmt("{s}/include/mlx/c/mlx.h", .{prefix})),
        .target = target,
        .optimize = optimize,
    });
    bindings.addIncludePath(b.path(b.fmt("{s}/include", .{prefix})));
    const mod = b.createModule(.{
        .root_source_file = b.path("native/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const dependency_pins = std.json.parseFromSlice(std.json.Value, b.allocator, @embedFile("native/dependencies.json"), .{}) catch @panic("Invalid native dependency pins");
    const runtime_options = b.addOptions();
    runtime_options.addOption([]const u8, "mlx_version", dependency_pins.value.object.get("python").?.object.get("mlx").?.string);
    runtime_options.addOption(bool, "vision_legacy_pixel_limits", dependency_pins.value.object.get("vision_legacy_pixel_limits").?.bool);
    runtime_options.addOption([]const u8, "dflash_calibration", @embedFile("src/tensorfold/families/qwen3_5/dflash2_calibration.json"));
    mod.addOptions("native_runtime", runtime_options);
    mod.linkFramework("ImageIO", .{});
    mod.linkFramework("CoreGraphics", .{});
    mod.linkFramework("CoreFoundation", .{});
    mod.link_libcpp = true;
    mod.addIncludePath(b.path("native/vendor/jinja"));
    mod.addCSourceFiles(.{
        .root = b.path("native/vendor/jinja"),
        .files = &.{ "jinja_wrapper.cpp", "caps.cpp", "lexer.cpp", "parser.cpp", "runtime.cpp", "jinja_string.cpp", "value.cpp" },
        .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" },
    });
    const jpeg_prefix = b.option([]const u8, "jpeg-prefix", "Pillow-matched libjpeg-turbo install prefix") orelse "build/jpeg";
    mod.addObjectFile(b.path(b.fmt("{s}/lib/libturbojpeg.a", .{jpeg_prefix})));
    const dependency_check = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_runtime.py", "--mlx-prefix", prefix, "--jpeg-prefix", jpeg_prefix });
    b.step("check-dependencies", "Verify installed native/Python MLX and pinned packages against project constraints").dependOn(&dependency_check.step);
    const dependency_tests = b.addSystemCommand(&.{ ".venv/bin/python", "-m", "pytest", "-q", "tools/test_native_runtime.py" });
    b.step("test-dependencies", "Check upstream constraint changes and native dependency drift detection").dependOn(&dependency_tests.step);
    mod.addImport("mlx_c", bindings.createModule());
    const draft_vocab = b.addOptions();
    draft_vocab.addOption([]const u8, "nemotron", @embedFile("src/tensorfold/families/nemotron_h/draft_ids.txt"));
    draft_vocab.addOption([]const u8, "flash", @embedFile("src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt"));
    mod.addOptions("draft_vocab_data", draft_vocab);
    mod.addIncludePath(b.path(b.fmt("{s}/include", .{prefix})));
    mod.addLibraryPath(b.path(b.fmt("{s}/lib", .{prefix})));
    mod.addRPath(b.path(b.fmt("{s}/lib", .{prefix})));
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("mlx", .{});
    mod.linkSystemLibrary("curl", .{});
    mod.linkSystemLibrary("icucore", .{});
    const exe = b.addExecutable(.{ .name = "tensorfold", .root_module = mod });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    b.step("run", "Run the native Mac engine").dependOn(&run.step);
    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run native host-side unit tests (GPU parity: run --check-exact)").dependOn(&run_tests.step);
    const runtime_check = b.addRunArtifact(exe);
    runtime_check.addArgs(&.{ "check-runtime", "build/native-checks/runtime.json" });
    b.step("test-runtime", "Verify loaded MLX-C CPU/GPU arithmetic and record hardware capabilities").dependOn(&runtime_check.step);
    const smoke_module = b.createModule(.{ .root_source_file = b.path("tools/smoke_native.zig"), .target = b.graph.host, .optimize = .safe });
    const smoke = b.addRunArtifact(b.addExecutable(.{ .name = "metal-smoke", .root_module = smoke_module }));
    smoke.addArg("build/native-checks/runtime.json");
    smoke.step.dependOn(&runtime_check.step);
    b.step("test-metal-smoke", "Run model-free parity checks selected for the detected Metal GPU").dependOn(&smoke.step);
    setup_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = smoke_module })).step);
    const file_tests = b.addRunArtifact(exe);
    file_tests.addArgs(&.{ "check-checkpoint-files", "build/native-checks/files" });
    b.step("test-checkpoint-files", "Exercise positional reads, corrupt checkpoints and allocation failures without a GPU").dependOn(&file_tests.step);
    const metal_tests = b.step("test-metal", "Generate Python oracles and compare native Metal kernels (requires .venv)");
    const affine_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/affine", "--affine-only" });
    const affine = b.addRunArtifact(exe);
    affine.addArgs(&.{ "check-variants", "build/native-checks/affine" });
    affine.step.dependOn(&affine_fixture.step);
    b.step("test-affine", "Check all packed affine formats against upstream Metal and native dispatch").dependOn(&affine.step);
    metal_tests.dependOn(&affine.step);
    const row_attention_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/row-attention", "--row-attention" });
    const row_attention = b.addRunArtifact(exe);
    row_attention.addArgs(&.{ "check-variants", "build/native-checks/row-attention" });
    row_attention.step.dependOn(&row_attention_fixture.step);
    b.step("test-row-attention", "Compare absolute-position row/tree attention at chunk boundaries with upstream").dependOn(&row_attention.step);
    metal_tests.dependOn(&row_attention.step);
    const tensor_quant_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/tensor-quantization", "--tensor-quantization" });
    const tensor_quant = b.addRunArtifact(exe);
    tensor_quant.addArgs(&.{ "check-variants", "build/native-checks/tensor-quantization" });
    tensor_quant.step.dependOn(&tensor_quant_fixture.step);
    b.step("test-tensor-quantization", "Compare tensor-unit low-bit and byte projections with upstream and native dispatch").dependOn(&tensor_quant.step);
    const bonsai_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/bonsai", "--bonsai-only" });
    const bonsai = b.addRunArtifact(exe);
    bonsai.addArgs(&.{ "check-variants", "build/native-checks/bonsai" });
    bonsai.step.dependOn(&bonsai_fixture.step);
    b.step("test-bonsai", "Compare rotated projection, inverse embedding and dense gate kernels with upstream").dependOn(&bonsai.step);
    metal_tests.dependOn(&bonsai.step);
    const gemma_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/gemma", "--gemma-only" });
    const gemma = b.addRunArtifact(exe);
    gemma.addArgs(&.{ "check-variants", "build/native-checks/gemma" });
    gemma.step.dependOn(&gemma_fixture.step);
    b.step("test-gemma", "Compare Gemma attention, projection, normalization and expert kernels against upstream").dependOn(&gemma.step);
    metal_tests.dependOn(&gemma.step);
    const large_fixture = b.addSystemCommand(&.{ "env", "MLX_ENABLE_TF32=0", ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/large-families", "--large-families" });
    const large_kernels = b.addRunArtifact(exe);
    large_kernels.addArgs(&.{ "check-variants", "build/native-checks/large-families" });
    large_kernels.step.dependOn(&large_fixture.step);
    b.step("test-large-family-kernels", "Compare synthetic GLM/DeepSeek kernels and host dispatch; full models remain unverified").dependOn(&large_kernels.step);
    metal_tests.dependOn(&large_kernels.step);
    const ds_dense_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/deepseek-dense", "--simd-dense" });
    const ds_dense = b.addRunArtifact(exe);
    ds_dense.addArgs(&.{ "check-deepseek-dense", "build/native-checks/deepseek-dense" });
    ds_dense.step.dependOn(&ds_dense_fixture.step);
    b.step("test-deepseek-dense", "Compare calibrated DeepSeek SIMD dispatch and physical threadgroup variants").dependOn(&ds_dense.step);
    metal_tests.dependOn(&ds_dense.step);
    const ds_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", "build/native-checks/deepseek-model", "--synthetic-deepseek", "--output", "build/native-checks/deepseek-model/oracle/logits.npy", "--state-directory", "build/native-checks/deepseek-model/oracle" });
    const ds_model = b.addRunArtifact(exe);
    ds_model.addArgs(&.{ "check-deepseek-model", "build/native-checks/deepseek-model", "build/native-checks/deepseek-model/native" });
    ds_model.step.dependOn(&ds_oracle.step);
    const ds_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", "build/native-checks/deepseek-model/oracle", "build/native-checks/deepseek-model/native" });
    ds_compare.step.dependOn(&ds_model.step);
    b.step("test-deepseek-model", "Compare synthetic DeepSeek backbone and compressed cache state with upstream").dependOn(&ds_compare.step);
    const ds_wide_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", "build/native-checks/deepseek-wide", "--synthetic-deepseek-wide", "--output", "build/native-checks/deepseek-wide/oracle/logits.npy", "--state-directory", "build/native-checks/deepseek-wide/oracle" });
    const ds_wide = b.addRunArtifact(exe);
    ds_wide.addArgs(&.{ "check-deepseek-model", "build/native-checks/deepseek-wide", "build/native-checks/deepseek-wide/native" });
    ds_wide.step.dependOn(&ds_wide_oracle.step);
    const ds_wide_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", "build/native-checks/deepseek-wide/oracle", "build/native-checks/deepseek-wide/native" });
    ds_wide_compare.step.dependOn(&ds_wide.step);
    b.step("test-deepseek-wide", "Compare synthetic DeepSeek at production hidden/attention widths with fused HC and MoE").dependOn(&ds_wide_compare.step);
    const ds_packed_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", "build/native-checks/deepseek-packed", "--synthetic-deepseek-packed", "--output", "build/native-checks/deepseek-packed/oracle/logits.npy", "--state-directory", "build/native-checks/deepseek-packed/oracle" });
    const ds_packed = b.addRunArtifact(exe);
    ds_packed.addArgs(&.{ "check-deepseek-model", "build/native-checks/deepseek-packed", "build/native-checks/deepseek-packed/native" });
    ds_packed.step.dependOn(&ds_packed_oracle.step);
    const ds_packed_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", "build/native-checks/deepseek-packed/oracle", "build/native-checks/deepseek-packed/native" });
    ds_packed_compare.step.dependOn(&ds_packed.step);
    b.step("test-deepseek-packed", "Compare DeepSeek production-width BF16 packed hyper-connection and head parameters").dependOn(&ds_packed_compare.step);
    const dspark_tests = b.step("test-dspark", "Compare DSpark taps, context caches, noncausal attention, sorted experts and Markov sampling");
    const conversion_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", "build/native-checks/conversion", "--conversion-fixtures", "--output", "build/native-checks/conversion" });
    const conversion_test = b.addRunArtifact(exe);
    conversion_test.addArgs(&.{ "check-drafter-conversion", "build/native-checks/conversion" });
    conversion_test.step.dependOn(&conversion_fixture.step);
    b.step("test-drafter-conversion", "Compare native FP8/FP4 MTP and DSpark conversion byte-for-byte with upstream").dependOn(&conversion_test.step);
    var dspark_previous: ?*std.Build.Step = null;
    for (0..3) |case| {
        const dir = b.fmt("build/native-checks/dspark-{d}", .{case});
        const oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", dir, if (case == 0) "--synthetic-dspark" else if (case == 1) "--synthetic-dspark-sorted" else "--synthetic-dspark-wide", "--output", b.fmt("{s}/oracle/logits.npy", .{dir}), "--state-directory", b.fmt("{s}/oracle", .{dir}) });
        if (dspark_previous) |previous| oracle.step.dependOn(previous);
        const native = b.addRunArtifact(exe);
        native.addArgs(&.{ "check-dspark", dir, b.fmt("{s}/native", .{dir}) });
        native.step.dependOn(&oracle.step);
        const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", b.fmt("{s}/oracle", .{dir}), b.fmt("{s}/native", .{dir}) });
        compare.step.dependOn(&native.step);
        dspark_previous = &compare.step;
    }
    dspark_tests.dependOn(dspark_previous.?);
    const dflash_tests = b.step("test-dflash", "Compare standard DFlash blocks, quantization, rotary layouts and context caches");
    var dflash_previous: ?*std.Build.Step = null;
    for (0..4) |case| {
        const dir = b.fmt("build/native-checks/dflash-{d}", .{case});
        const oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", dir, "--synthetic-dflash", b.fmt("{d}", .{case}), "--output", b.fmt("{s}/oracle/logits.npy", .{dir}), "--state-directory", b.fmt("{s}/oracle", .{dir}) });
        if (dflash_previous) |previous| oracle.step.dependOn(previous);
        const native = b.addRunArtifact(exe);
        native.addArgs(&.{ "check-dflash", dir, b.fmt("{s}/native", .{dir}), b.fmt("{d}", .{case}) });
        native.step.dependOn(&oracle.step);
        const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", b.fmt("{s}/oracle", .{dir}), b.fmt("{s}/native", .{dir}) });
        compare.step.dependOn(&native.step);
        dflash_previous = &compare.step;
    }
    dflash_tests.dependOn(dflash_previous.?);
    const glm_models = b.step("test-glm-model", "Compare synthetic GLM backbone logits, mixed layouts and cache commits; full model unverified");
    var glm_previous: ?*std.Build.Step = null;
    for (0..3) |case| {
        const fixture = b.fmt("build/native-checks/glm-model-{d}", .{case});
        const oracle_dir = b.fmt("{s}/oracle", .{fixture});
        const native_dir = b.fmt("{s}/native", .{fixture});
        const oracle = b.addSystemCommand(&.{ "env", "MLX_ENABLE_TF32=0", ".venv/bin/python", "tools/native_families_reference.py", fixture, if (case == 0) "--synthetic-glm" else if (case == 1) "--synthetic-glm-layout" else "--synthetic-glm-mixed", "--trace-layers", "--output", b.fmt("{s}/logits.npy", .{oracle_dir}), "--state-directory", oracle_dir });
        if (glm_previous) |previous| oracle.step.dependOn(previous);
        const native = b.addRunArtifact(exe);
        native.addArgs(&.{ "check-glm-model", if (case == 1) b.fmt("{s}/mlxlm", .{fixture}) else fixture, native_dir });
        native.step.dependOn(&oracle.step);
        const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", oracle_dir, native_dir });
        compare.step.dependOn(&native.step);
        glm_previous = &compare.step;
        for (0..2) |mode| {
            const model_dir = if (case == 1) b.fmt("{s}/mlxlm", .{fixture}) else fixture;
            const reference_report = b.fmt("{s}/generation-{d}-oracle.json", .{ fixture, mode });
            const native_report = b.fmt("{s}/generation-{d}-native.json", .{ fixture, mode });
            const ids = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33";
            const reference = b.addSystemCommand(&.{ "env", "MLX_ENABLE_TF32=0", ".venv/bin/python", "tools/native_families_reference.py", model_dir, "--tokens", ids, "--generate", "12", "--temperature", if (mode == 0) "0" else "0.7", "--seed", "456", "--metal-sampling", "--output", reference_report });
            reference.step.dependOn(glm_previous.?);
            const completion = b.addRunArtifact(exe);
            completion.addArgs(&.{ "run", model_dir, "--tokens", ids, "--max-tokens", "12", "--temperature", if (mode == 0) "0" else "0.7", "--seed", "456", "--metal-sampling", "--mtp-drafts", "3", "--report", native_report });
            completion.step.dependOn(&reference.step);
            const reports = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-reports", reference_report, native_report });
            reports.step.dependOn(&completion.step);
            glm_previous = &reports.step;
        }
    }
    glm_models.dependOn(glm_previous.?);
    metal_tests.dependOn(glm_models);
    const simd_attention_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_attention_fixtures.py", "build/native-checks/simd-attention", "--simd" });
    const simd_attention = b.addRunArtifact(exe);
    simd_attention.addArgs(&.{ "check-attention", "build/native-checks/simd-attention" });
    simd_attention.step.dependOn(&simd_attention_fixture.step);
    b.step("test-simd-attention", "Compare SIMD chains and branches with serial MLX attention across dispatch boundaries").dependOn(&simd_attention.step);
    metal_tests.dependOn(&simd_attention.step);
    const tensor_tests = b.option(bool, "metal-tensors", "Include M5 tensor attention fixtures in test-metal") orelse false;
    if (tensor_tests) metal_tests.dependOn(&tensor_quant.step);
    const variants_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_variant_fixtures.py", "build/native-checks/variants" });
    const variants = b.addRunArtifact(exe);
    variants.addArgs(&.{ "check-variants", "build/native-checks/variants" });
    variants.step.dependOn(&variants_fixture.step);
    b.step("test-variants", "Run original optional-kernel tests and compare their launches through native embedded Metal").dependOn(&variants.step);
    const allocation_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_allocation_fixtures.py", "build/native-checks/allocations" });
    const allocation_checks = b.addRunArtifact(exe);
    allocation_checks.addArgs(&.{ "check-allocation-failures", "build/native-checks/allocations" });
    allocation_checks.step.dependOn(&allocation_fixture.step);
    b.step("test-allocation-failures", "Inject every host allocation failure at MLX ownership boundaries").dependOn(&allocation_checks.step);
    const vocab_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_draft_vocab_fixtures.py", "build/native-checks/draft-vocab" });
    const vocab_checks = b.addRunArtifact(exe);
    vocab_checks.addArgs(&.{ "check-draft-vocab", "build/native-checks/draft-vocab" });
    vocab_checks.step.dependOn(&vocab_fixture.step);
    b.step("test-draft-vocab", "Compare every original draft ID and quantized head row against Python").dependOn(&vocab_checks.step);
    const depth_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_depth_fixtures.py", "build/native-checks/draft-depth.json" });
    const depth_checks = b.addRunArtifact(exe);
    depth_checks.addArgs(&.{ "check-draft-depth", "build/native-checks/draft-depth.json" });
    depth_checks.step.dependOn(&depth_fixture.step);
    b.step("test-draft-depth", "Compare native adaptive depth decisions with original Python policy").dependOn(&depth_checks.step);
    const position_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_mtp_position_fixtures.py", "build/native-checks/mtp-positions" });
    const position_checks = b.addRunArtifact(exe);
    position_checks.addArgs(&.{ "check-mtp-positions", "build/native-checks/mtp-positions" });
    position_checks.step.dependOn(&position_fixture.step);
    b.step("test-mtp-positions", "Compare sampling positions and retained rows with original Python MTP scheduling").dependOn(&position_checks.step);
    for ([_][]const u8{ "sampling", "sparse", "attention", "ple_norm" }) |kind| {
        if (std.mem.eql(u8, kind, "attention") and !tensor_tests) continue;
        const dir = b.fmt("build/native-checks/{s}", .{kind});
        const fixture = b.addSystemCommand(&.{ ".venv/bin/python", b.fmt("tools/native_{s}_fixtures.py", .{kind}), dir });
        const check = b.addRunArtifact(exe);
        check.addArgs(&.{ b.fmt("check-{s}", .{kind}), dir });
        check.step.dependOn(&fixture.step);
        metal_tests.dependOn(&check.step);
        if (std.mem.eql(u8, kind, "ple_norm")) b.step("test-ple-norm", "Compare PLE normalization against original Python arithmetic").dependOn(&check.step);
    }
    const model_tests = b.step("test-models", "Real-model row/rollback/cache checks for all three Metal families (large RAM required)");
    const model_root = b.option([]const u8, "model-root", "Downloaded checkpoint directory for test-models") orelse "build/models";
    const lifecycle_module = b.createModule(.{ .root_source_file = b.path("tools/native_server_checks.zig"), .target = b.graph.host, .optimize = .safe });
    const lifecycle = b.addRunArtifact(b.addExecutable(.{ .name = "native-server-checks", .root_module = lifecycle_module }));
    lifecycle.addArtifactArg(exe);
    lifecycle.addArg(b.fmt("{s}/Qwen3.8-27B-MLX-4bit", .{model_root}));
    b.step("test-server-lifecycle", "Check request deadlines, stalled clients, cancellation recovery and clean SIGINT/SIGTERM shutdown with local Qwen").dependOn(&lifecycle.step);
    const chat_tests = b.step("test-chat", "Compare native Jinja prompts with upstream for all seven local tokenizers; no model weights loaded");
    const tool_fixtures = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", ".", "--tool-fixtures", "--output", "build/native-checks/tool-calls.json" });
    const tool_tests = b.addRunArtifact(exe);
    tool_tests.addArgs(&.{ "check-tool-calls", "build/native-checks/tool-calls.json" });
    tool_tests.step.dependOn(&tool_fixtures.step);
    const tool_step = b.step("test-tool-calls", "Compare native tool parsing and incremental streaming with upstream");
    tool_step.dependOn(&tool_tests.step);
    const tool_stream_tests = b.addRunArtifact(exe);
    tool_stream_tests.addArgs(&.{ "check-tool-stream", "build/native-checks/tool-stream.json" });
    tool_stream_tests.step.dependOn(&tool_fixtures.step);
    tool_step.dependOn(&tool_stream_tests.step);
    const calibration_fixture = "build/native-checks/draft-calibration.json";
    const draft_allocation_fixture = "build/native-checks/draft-allocation.json";
    const allocation_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--allocation-fixtures", "--output", draft_allocation_fixture });
    const allocation_check = b.addRunArtifact(exe);
    allocation_check.addArgs(&.{ "check-draft-allocation", draft_allocation_fixture });
    allocation_check.step.dependOn(&allocation_oracle.step);
    b.step("test-draft-allocation", "Compare shared draft budgets and acceptance chains with upstream").dependOn(&allocation_check.step);
    const capture_folder = "build/native-checks/draft-capture";
    const capture_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--capture-fixtures", "--output", capture_folder });
    const capture_check = b.addRunArtifact(exe);
    capture_check.addArgs(&.{ "check-draft-capture", capture_folder });
    capture_check.step.dependOn(&capture_oracle.step);
    b.step("test-draft-capture", "Compare native capture records with upstream and verify ordered drain and write failure handling").dependOn(&capture_check.step);
    const capture_models = b.step("test-draft-capture-model", "Verify captured committed features and target logits against upstream replay without changing generated tokens");
    var capture_prior: *std.Build.Step = &capture_check.step;
    for ([_][]const u8{ "0", "0.7", "0.7" }, 0..) |temperature, regime| {
        const prompt_ids = b.allocator.alloc([]const u8, if (regime == 1) 19 else 137) catch @panic("OOM");
        for (prompt_ids, 0..) |*id, i| id.* = b.fmt("{d}", .{1 + i % 8});
        const common = &.{ "run", b.fmt("{s}/Qwen3.8-27B-MLX-4bit", .{model_root}), "--tokens", std.mem.join(b.allocator, ",", prompt_ids) catch @panic("OOM"), "--max-tokens", "32", "--seed", "5678", "--temperature", temperature, "--top-k", "12", "--top-p", "0.8", "--metal-sampling", "--no-copy", "--drafter", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}) };
        const plain_report = b.fmt("{s}/plain-{d}.json", .{ capture_folder, regime });
        const capture_report = b.fmt("{s}/captured-{d}.json", .{ capture_folder, regime });
        const plain = b.addRunArtifact(exe);
        plain.addArgs(common);
        if (regime != 2) plain.addArg("--lane-prefill");
        plain.addArgs(&.{ "--report", plain_report });
        plain.step.dependOn(capture_prior);
        const captured = b.addRunArtifact(exe);
        captured.addArgs(common);
        if (regime != 2) captured.addArg("--lane-prefill");
        captured.addArgs(&.{ "--draft-capture", b.fmt("{s}/model", .{capture_folder}), "--report", capture_report });
        captured.step.dependOn(&plain.step);
        const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--require-rounds", "--compare-reports", plain_report, capture_report });
        compare.step.dependOn(&captured.step);
        const replay = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--model", b.fmt("{s}/Qwen3.8-27B-MLX-4bit", .{model_root}), "--verify-capture", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}), capture_report });
        replay.step.dependOn(&compare.step);
        capture_prior = &replay.step;
        if (regime == 1) {
            const failed_report = b.fmt("{s}/failed-{d}.json", .{ capture_folder, regime });
            const failed = b.addRunArtifact(exe);
            failed.addArgs(common);
            failed.addArg("--lane-prefill");
            failed.addArgs(&.{ "--draft-capture", b.fmt("{s}/fixture.json/blocked", .{capture_folder}), "--report", failed_report });
            failed.step.dependOn(capture_prior);
            const failure_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--require-rounds", "--compare-reports", plain_report, failed_report });
            failure_compare.step.dependOn(&failed.step);
            capture_prior = &failure_compare.step;
        }
    }
    capture_models.dependOn(capture_prior);
    const calibration_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--calibration-fixtures", "--output", calibration_fixture });
    const calibration_check = b.addRunArtifact(exe);
    calibration_check.addArgs(&.{ "check-draft-calibration", calibration_fixture });
    calibration_check.step.dependOn(&calibration_oracle.step);
    const calibration_fit = b.addRunArtifact(exe);
    calibration_fit.addArgs(&.{ "fit-draft-calibration", "build/native-checks/draft-calibration.samples.json", "build/native-checks/draft-calibration.fitted.json" });
    calibration_fit.step.dependOn(&calibration_check.step);
    const calibration_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-calibration", "build/native-checks/draft-calibration.samples.json", "build/native-checks/draft-calibration.fitted.json" });
    calibration_compare.step.dependOn(&calibration_fit.step);
    b.step("test-draft-calibration", "Compare calibration fits, bin boundaries and parent-preserving proposal ordering with upstream").dependOn(&calibration_compare.step);
    const calibration_models = b.step("test-draft-calibration-model", "Verify fitted calibration overrides preserve Qwen and Bonsai generation without copy proposals");
    var calibration_prior: *std.Build.Step = &calibration_compare.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "Ternary-Bonsai-2-27B-mlx-2bit" }, 0..) |name, family| {
        for ([_][]const u8{ "0", "0.7" }, 0..) |temperature, regime| {
            const common = &.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--tokens", "1,2,3,4,5,6,7,8,41,42,43,1,2,3,4,5,6,7,8", "--max-tokens", "24", "--seed", "5678", "--temperature", temperature, "--top-k", "12", "--top-p", "0.8", "--metal-sampling" };
            const serial_path = b.fmt("build/native-checks/calibrated-{d}-{d}-serial.json", .{ family, regime });
            const draft_path = b.fmt("build/native-checks/calibrated-{d}-{d}-draft.json", .{ family, regime });
            const serial = b.addRunArtifact(exe);
            serial.addArgs(common);
            serial.addArgs(&.{ "--no-drafts", "--report", serial_path });
            serial.step.dependOn(calibration_prior);
            const draft = b.addRunArtifact(exe);
            draft.addArgs(common);
            draft.addArgs(&.{ "--drafter", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}), "--no-copy", "--draft-calibration", "build/native-checks/draft-calibration.fitted.json", "--report", draft_path });
            draft.step.dependOn(&serial.step);
            const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--require-rounds", "--compare-reports", serial_path, draft_path });
            compare.step.dependOn(&draft.step);
            calibration_prior = &compare.step;
        }
    }
    calibration_models.dependOn(calibration_prior);
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "Ternary-Bonsai-2-27B-mlx-2bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP", "gemma-4-26b-a4b-it-4bit", "GLM-5.3-Flash-MLX-4bit-MTP", "DeepSeek-V4-Flash-4bit" }, 0..) |name, index| {
        const dir = b.fmt("{s}/{s}", .{ model_root, name });
        const fixture = b.fmt("build/native-checks/chat/{d}.json", .{index});
        const oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", dir, "--chat-fixtures", "--output", fixture });
        const check = b.addRunArtifact(exe);
        check.addArgs(&.{ "check-chat", dir, fixture });
        check.step.dependOn(&oracle.step);
        chat_tests.dependOn(&check.step);
    }
    const gemma_model = b.fmt("{s}/gemma-4-26b-a4b-it-4bit", .{model_root});
    const gemma_prefill_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", gemma_model, "--gemma-prefill", "--output", "build/native-checks/gemma-prefill/oracle/logits.npy", "--state-directory", "build/native-checks/gemma-prefill/oracle" });
    const gemma_prefill = b.addRunArtifact(exe);
    gemma_prefill.addArgs(&.{ "check-gemma-prefill", gemma_model, "build/native-checks/gemma-prefill/native" });
    gemma_prefill.step.dependOn(&gemma_prefill_oracle.step);
    const gemma_prefill_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", "build/native-checks/gemma-prefill/oracle", "build/native-checks/gemma-prefill/native" });
    gemma_prefill_compare.step.dependOn(&gemma_prefill.step);
    b.step("test-gemma-prefill", "Compare batched Gemma prompt arithmetic, ring wrap, caches and decode continuation").dependOn(&gemma_prefill_compare.step);
    const gemma_drafter_option = b.option([]const u8, "gemma-drafter", "Existing trained Gemma DFlash checkpoint; default is the synthetic oracle fixture");
    const gemma_drafter = gemma_drafter_option orelse "build/native-checks/dflash-3";
    const gemma_draft_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", gemma_model, "--gemma-drafter", gemma_drafter, "--output", "build/native-checks/gemma-draft/oracle/logits.npy", "--state-directory", "build/native-checks/gemma-draft/oracle" });
    if (gemma_drafter_option == null) gemma_draft_oracle.step.dependOn(dflash_previous.?);
    const gemma_draft = b.addRunArtifact(exe);
    gemma_draft.addArgs(&.{ "check-gemma-draft", gemma_model, gemma_drafter, "build/native-checks/gemma-draft/native" });
    gemma_draft.step.dependOn(&gemma_draft_oracle.step);
    const gemma_draft_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-arrays", "build/native-checks/gemma-draft/oracle", "build/native-checks/gemma-draft/native" });
    gemma_draft_compare.step.dependOn(&gemma_draft.step);
    b.step("test-gemma-draft", "Compare full Gemma taps and synthetic DFlash proposals, caches and generation").dependOn(&gemma_draft_compare.step);
    const gemma_cache = b.addRunArtifact(exe);
    gemma_cache.addArgs(&.{ "run", gemma_model, "--check-long-cache" });
    gemma_cache.step.dependOn(&gemma.step);
    var gemma_previous: *std.Build.Step = &gemma_cache.step;
    var gemma_tokens: [1156][]const u8 = undefined;
    for (&gemma_tokens, 0..) |*token, j| token.* = b.fmt("{d}", .{1000 + j});
    const gemma_long = std.mem.join(b.allocator, ",", &gemma_tokens) catch @panic("OOM");
    for (0..3) |case| {
        const base_path = b.fmt("build/native-checks/gemma-model-{d}", .{case});
        const oracle_json = b.fmt("{s}-python.json", .{base_path});
        const native_json = b.fmt("{s}-native.json", .{base_path});
        const oracle_npy = b.fmt("{s}-python.npy", .{base_path});
        const native_npy = b.fmt("{s}-native.npy", .{base_path});
        const oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", gemma_model, "--generate", "12", "--temperature", if (case == 0) "0" else "0.7", "--seed", "5678", "--output", oracle_json, "--dump-logits", oracle_npy });
        const native = b.addRunArtifact(exe);
        native.addArgs(&.{ "run", gemma_model, "--max-tokens", "12", "--temperature", if (case == 0) "0" else "0.7", "--seed", "5678", "--report", native_json, "--dump-logits", native_npy });
        if (case > 0) {
            oracle.addArg("--metal-sampling");
            native.addArg("--metal-sampling");
        }
        if (case == 2) {
            oracle.addArgs(&.{ "--tokens", gemma_long });
            native.addArgs(&.{ "--tokens", gemma_long });
        }
        oracle.step.dependOn(gemma_previous);
        native.step.dependOn(&oracle.step);
        const logits = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare", oracle_npy, native_npy });
        logits.step.dependOn(&native.step);
        const report = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-reports", oracle_json, native_json });
        report.step.dependOn(&logits.step);
        gemma_previous = &report.step;
    }
    b.step("test-gemma-model", "Compare Gemma tokenization, logits, sampling and long-context cache commits against upstream").dependOn(gemma_previous);
    const vision_model = b.fmt("{s}/Qwen3.8-27B-MLX-4bit", .{model_root});
    const image_tests = b.step("test-images", "Compare PNG/JPEG/WebP preprocessing, alpha, grayscale, CMYK and every EXIF orientation");
    const image_http_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--image-http-fixtures", "--output", "build/native-checks/image-http.json" });
    const image_http = b.addRunArtifact(exe);
    image_http.addArgs(&.{ "check-image-http", "build/native-checks/image-http.json" });
    image_http.step.dependOn(&image_http_fixture.step);
    b.step("test-image-http", "Compare image URL and public-address policies against upstream without network access").dependOn(&image_http.step);
    var image_previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "PNG", "JPEG", "WEBP" }) |format| {
        for (1..9) |orientation| {
            const dir = b.fmt("build/native-checks/images/{s}-{d}", .{ format, orientation });
            const extension = if (std.mem.eql(u8, format, "PNG")) "png" else if (std.mem.eql(u8, format, "JPEG")) "jpeg" else "webp";
            const fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--model", vision_model, "--vision-fixture", "187", "311", "--image-fixture", "--image-only", "--image-format", format, "--image-orientation", b.fmt("{d}", .{orientation}), "--output", dir });
            if (std.mem.eql(u8, format, "JPEG")) {
                if (orientation == 2) fixture.addArgs(&.{ "--image-mode", "CMYK" });
                if (orientation == 3) fixture.addArgs(&.{ "--image-mode", "L" });
            } else if (orientation % 2 == 0) fixture.addArg("--image-alpha");
            if (image_previous) |prior| fixture.step.dependOn(prior);
            const decode = b.addRunArtifact(exe);
            decode.addArgs(&.{ "check-image", b.fmt("{s}/image.{s}", .{ dir, extension }), dir });
            decode.step.dependOn(&fixture.step);
            const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare", b.fmt("{s}/pixels.npy", .{dir}), b.fmt("{s}/pixels-native.npy", .{dir}) });
            compare.step.dependOn(&decode.step);
            image_previous = &compare.step;
        }
    }
    image_tests.dependOn(image_previous.?);
    const vision_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--model", vision_model, "--vision-fixture", "187", "311", "--image-fixture", "--output", "build/native-checks/vision" });
    const vision_decode = b.addRunArtifact(exe);
    vision_decode.addArgs(&.{ "check-image", "build/native-checks/vision/image.png", "build/native-checks/vision" });
    vision_decode.step.dependOn(&vision_fixture.step);
    vision_fixture.step.dependOn(image_tests);
    const vision_pixels = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare", "build/native-checks/vision/pixels.npy", "build/native-checks/vision/pixels-native.npy" });
    vision_pixels.step.dependOn(&vision_decode.step);
    const vision_encode = b.addRunArtifact(exe);
    vision_encode.addArgs(&.{ "check-vision", vision_model, "build/native-checks/vision" });
    vision_encode.step.dependOn(&vision_pixels.step);
    const vision_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-vision", "build/native-checks/vision" });
    vision_compare.step.dependOn(&vision_encode.step);
    b.step("test-vision-encoder", "Compare PNG preprocessing and every Qwen vision encoder stage with upstream").dependOn(&vision_compare.step);
    var vision_previous: *std.Build.Step = &vision_compare.step;
    for (0..2) |backend| {
        const python_dir = b.fmt("build/native-checks/vision-prefill/{d}/python", .{backend});
        const native_dir = b.fmt("build/native-checks/vision-prefill/{d}/native", .{backend});
        const oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_qwen_prefill_reference.py", python_dir, "--model", vision_model, "--image", "build/native-checks/vision/image.png", "--generate", "4" });
        oracle.step.dependOn(vision_previous);
        const check = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_qwen_prefill_reference.py", native_dir, "--model", vision_model, "--image", "build/native-checks/vision/image.png", "--generate", "4", "--native" });
        check.addArtifactArg(exe);
        check.step.dependOn(&oracle.step);
        if (backend == 1) {
            oracle.addArg("--simd");
            check.addArg("--simd");
        }
        const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_qwen_prefill_reference.py", python_dir, "--compare", native_dir });
        compare.step.dependOn(&check.step);
        vision_previous = &compare.step;
    }
    b.step("test-vision", "Compare image-conditioned prefill, caches and continuation on tensor and SIMD backends").dependOn(vision_previous);
    const prefill_fixture = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_prefill_math.py", "build/native-checks/prefill-math" });
    const prefill_math = b.addRunArtifact(exe);
    prefill_math.addArgs(&.{ "check-prefill-math", "build/native-checks/prefill-math" });
    prefill_math.step.dependOn(&prefill_fixture.step);
    b.step("test-prefill-math", "Exhaustive BF16 activation and mixed-precision decay parity with mlx-lm").dependOn(&prefill_math.step);
    const prefill_tests = b.step("test-prefill", "Trace production Qwen/Bonsai prefill layers, caches and logits against Python");
    const prefill_family = b.option(usize, "prefill-family", "Select 0=Qwen or 1=Bonsai prefill reference") orelse 0;
    if (prefill_family > 1) @panic("prefill-family must be 0 or 1");
    const prefill_size = b.option(usize, "prefill-tokens", "Override prefill length (0 uses the English prompt)");
    const prefill_backend = b.option(usize, "prefill-backend", "Restrict prefill checks to 0=tensor or 1=SIMD");
    if (prefill_backend != null and prefill_backend.? > 1) @panic("prefill-backend must be 0 or 1");
    if (prefill_size != null and prefill_size.? > 262144) @panic("prefill-tokens must be 0..262144");
    var prefill_previous: *std.Build.Step = &prefill_math.step;
    for ([_]usize{ 0, 129, 2049, 4225 }, 0..) |default_length, case| {
        if (prefill_size != null and case > 0) continue;
        const length = prefill_size orelse default_length;
        for (0..2) |backend| {
            if (prefill_backend != null and prefill_backend.? != backend) continue;
            const base = b.fmt("build/native-checks/{s}/{d}-{d}", .{ if (prefill_family == 0) "prefill" else "bonsai-prefill", backend, length });
            const oracle_dir = b.fmt("{s}/python", .{base});
            const native_dir = b.fmt("{s}/native", .{base});
            const oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_qwen_prefill_reference.py", oracle_dir });
            const check = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_qwen_prefill_reference.py", native_dir, "--native" });
            check.addArtifactArg(exe);
            for ([_]*std.Build.Step.Run{ oracle, check }) |step| {
                step.addArgs(&.{ "--generate", "32" });
                step.addArgs(&.{ "--model", b.fmt("{s}/{s}", .{ model_root, if (prefill_family == 0) "Qwen3.8-27B-MLX-4bit" else "Ternary-Bonsai-2-27B-mlx-2bit" }) });
                if (length > 0) step.addArgs(&.{ "--length", b.fmt("{d}", .{length}) });
                if (backend == 1) step.addArg("--simd");
            }
            oracle.step.dependOn(prefill_previous);
            check.step.dependOn(&oracle.step);
            const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_qwen_prefill_reference.py", oracle_dir, "--compare", native_dir });
            compare.step.dependOn(&check.step);
            prefill_previous = &compare.step;
        }
    }
    prefill_tests.dependOn(prefill_previous);
    const ngram_checks = b.addRunArtifact(exe);
    ngram_checks.addArg("check-ngram-gpu");
    b.step("test-ngram-gpu", "Compare GPU n-gram IDs with exact CPU integer hashing").dependOn(&ngram_checks.step);
    const resident_checks = b.addRunArtifact(exe);
    resident_checks.addArgs(&.{ "check-ple-resident", b.fmt("{s}/Qwen3.8-Flash-Next-MLX-4bit-MTP", .{model_root}) });
    resident_checks.step.dependOn(&ngram_checks.step);
    b.step("test-ple-resident", "Validate all resident PLE groups and bounded loading-memory overhead").dependOn(&resident_checks.step);
    const ple_long = b.option(bool, "ple-long", "Compare resident/bounded PLE at sparse attention thresholds") orelse false;
    const ple_state = b.addRunArtifact(exe);
    ple_state.addArgs(&.{ "run", b.fmt("{s}/Qwen3.8-Flash-Next-MLX-4bit-MTP", .{model_root}), "--check-ple-state" });
    if (ple_long) ple_state.addArg("--check-long-cache");
    b.step("test-ple-state", "Compare full Flash resident/bounded logits, every retained cache prefix and EOS history").dependOn(&ple_state.step);
    const resident_runtime = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_mtp_runtime.py" });
    resident_runtime.addArtifactArg(exe);
    resident_runtime.addArgs(&.{ "--model-root", model_root, "--family", "flash", "--resident-ple", "--sampler", "metal", "--output", "build/native-checks/resident-mtp" });
    b.step("test-ple-runtime", "Compare bounded/resident Flash serial and GPU/host MTP handoff schedules").dependOn(&resident_runtime.step);
    const buffer_fixture = b.addRunArtifact(exe);
    buffer_fixture.addArg("check-kv-buffer");
    const buffer_tests = b.step("test-kv-buffers", "Check buffer donation, independent concatenation parity and snapshot ownership");
    const buffer_family = b.option(usize, "kv-family", "Restrict KV checks to 0=Qwen, 1=Nemotron or 2=Flash");
    if (buffer_family != null and buffer_family.? > 2) @panic("kv-family must be 0, 1 or 2");
    const buffer_long = b.option(bool, "kv-long", "Compare buffered/unbuffered paths across long-context attention thresholds") orelse false;
    var buffer_previous: *std.Build.Step = &buffer_fixture.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (buffer_family != null and buffer_family.? != family) continue;
        for (0..if (family == 2) @as(usize, 1) else 2) |backend| {
            for ([_][]const u8{ "--check-kv-buffers", "--check-kv-reuse" }) |flag| {
                if (buffer_long and std.mem.eql(u8, flag, "--check-kv-reuse")) continue;
                const check = b.addRunArtifact(exe);
                check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), flag });
                if (backend == 1) check.addArg("--metal-simd");
                if (buffer_long) check.addArg("--check-long-cache");
                check.step.dependOn(buffer_previous);
                buffer_previous = &check.step;
            }
        }
    }
    buffer_tests.dependOn(buffer_previous);
    const buffer_runtime = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_kv_runtime.py" });
    buffer_runtime.addArtifactArg(exe);
    buffer_runtime.addArgs(&.{ "--model-root", model_root });
    if (buffer_family) |family| buffer_runtime.addArgs(&.{ "--family", ([_][]const u8{ "qwen", "nemotron", "flash" })[family] });
    b.step("test-kv-runtime", "Compare buffered and concatenated serial/DFlash2/MTP completions and proposal streams").dependOn(&buffer_runtime.step);
    const serial_fixture = b.addRunArtifact(exe);
    serial_fixture.addArg("check-serial-pipeline");
    const serial_tests = b.step("test-serial-pipeline", "GPU serial pipeline EOS/budget ownership and real-model token/cache parity");
    const serial_family = b.option(usize, "serial-family", "Restrict serial pipeline tests to 0=Qwen, 1=Nemotron or 2=resident Flash");
    if (serial_family != null and serial_family.? > 2) @panic("serial-family must be 0, 1 or 2");
    const serial_long = b.option(bool, "serial-long", "Run serial pipeline comparisons across 10K or Flash sparse attention thresholds") orelse false;
    var serial_previous: *std.Build.Step = &serial_fixture.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (serial_family != null and serial_family.? != family) continue;
        for (0..if (family == 2) @as(usize, 1) else 2) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-serial-state" });
            if (family == 2) check.addArg("--resident-ple");
            if (backend == 1) check.addArg("--metal-simd");
            if (serial_long) check.addArg("--check-long-cache");
            check.step.dependOn(serial_previous);
            serial_previous = &check.step;
        }
    }
    serial_tests.dependOn(serial_previous);
    const serial_runtime = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_serial_runtime.py" });
    serial_runtime.addArtifactArg(exe);
    serial_runtime.addArgs(&.{ "--model-root", model_root });
    if (serial_family) |family| serial_runtime.addArgs(&.{ "--family", ([_][]const u8{ "qwen", "nemotron", "flash" })[family] });
    b.step("test-serial-runtime", "Compare synchronous/pipelined CLI completions and output-budget boundaries").dependOn(&serial_runtime.step);
    const mtp_runtime = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_mtp_runtime.py" });
    mtp_runtime.addArtifactArg(exe);
    mtp_runtime.addArgs(&.{ "--model-root", model_root });
    const mtp_family = b.option([]const u8, "mtp-family", "Restrict MTP runtime checks to nemotron or flash");
    if (mtp_family) |family| mtp_runtime.addArgs(&.{ "--family", family });
    b.step("test-mtp-runtime", "Compare full/cut vocabulary and queued/host MTP proposal streams and target output").dependOn(&mtp_runtime.step);
    var state_previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "nemotron", "flash" }, [_][]const u8{ "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |family, name, index| {
        if (mtp_family != null and !std.mem.eql(u8, family, mtp_family.?)) continue;
        for (0..if (index == 0) @as(usize, 2) else 1) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-mtp-state" });
            if (backend == 1) check.addArg("--metal-simd");
            if (state_previous) |step| check.step.dependOn(step);
            state_previous = &check.step;
        }
    }
    const state_tests = b.step("test-mtp-state", "Compare batched MTP, every prefix and continuation through 10K context");
    if (state_previous) |step| state_tests.dependOn(step);
    const simd_dir = b.addSystemCommand(&.{ "mkdir", "-p", "build/native-checks/simd-reference" });
    const simd_model = b.fmt("{s}/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", .{model_root});
    const simd_options = &.{ "--prompt", "Write a short Python function that computes the Fibonacci sequence.", "--seed", "5678", "--temperature", "0", "--top-k", "12", "--top-p", "0.8" };
    const simd_oracle = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_families_reference.py", simd_model, "--simd", "--generate", "17", "--output", "build/native-checks/simd-reference/nemotron-python.json" });
    simd_oracle.addArgs(simd_options);
    simd_oracle.step.dependOn(&simd_dir.step);
    const simd_native = b.addRunArtifact(exe);
    simd_native.addArgs(&.{ "run", simd_model, "--metal-simd", "--no-drafts", "--max-tokens", "17", "--report", "build/native-checks/simd-reference/nemotron-native.json" });
    simd_native.addArgs(simd_options);
    simd_native.step.dependOn(&simd_oracle.step);
    const simd_compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-reports", "build/native-checks/simd-reference/nemotron-python.json", "build/native-checks/simd-reference/nemotron-native.json" });
    simd_compare.step.dependOn(&simd_native.step);
    b.step("test-nemotron-simd-reference", "Compare the short code-prompt completion with original Python SIMD").dependOn(&simd_compare.step);
    const schema_tests = b.step("test-model-schemas", "Validate all downloaded tensor names/shapes/dtypes and index references without loading payloads");
    const schema_failures = b.addSystemCommand(&.{ ".venv/bin/python", "tools/test_native_schemas.py" });
    schema_failures.addArtifactArg(exe);
    schema_failures.addArgs(&.{ "--model-root", model_root });
    b.step("test-schema-failures", "Reject malformed checkpoint metadata and missing MTP through the native CLI without a GPU").dependOn(&schema_failures.step);
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "Qwen3.8-27B-DFlash2", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, [_][]const u8{ "qwen", "dflash", "nemotron", "flash" }) |name, kind| {
        const check = b.addRunArtifact(exe);
        check.addArgs(&.{ "check-model-schema", kind, b.fmt("{s}/{s}", .{ model_root, name }) });
        schema_tests.dependOn(&check.step);
    }
    const ple_tests = b.addRunArtifact(exe);
    ple_tests.addArgs(&.{ "check-ple", b.fmt("{s}/Qwen3.8-Flash-Next-MLX-4bit-MTP", .{model_root}) });
    b.step("test-ple", "Compare native positional PLE reads against MLX at all 128 shard boundaries").dependOn(&ple_tests.step);
    var previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, index| {
        for (0..if (index < 2) @as(usize, 2) else 1) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-exact" });
            if (backend == 1) check.addArg("--metal-simd");
            // Never load multiple full checkpoints concurrently on unified memory.
            if (previous) |step| check.step.dependOn(step);
            previous = &check.step;
        }
    }
    model_tests.dependOn(previous.?);
    const draft_tests = b.step("test-drafts", "Serial/draft regression matrix for all models, samplers and SIMD paths");
    const draft_family = b.option(usize, "draft-family", "Restrict draft regression to 0=Qwen, 1=Nemotron, 2=Flash, 3=Bonsai");
    const draft_scenario = b.option(usize, "draft-scenario", "Restrict draft regression to 0=greedy, 1=Metal, 2=CPU, 3=two-token CPU");
    if (draft_family != null and draft_family.? > 3) @panic("draft-family must be 0, 1, 2 or 3");
    if (draft_scenario != null and draft_scenario.? > 3) @panic("draft-scenario must be 0, 1, 2 or 3");
    const directory = b.addSystemCommand(&.{ "mkdir", "-p", "build/native-checks/drafts" });
    var prior: *std.Build.Step = &directory.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP", "Ternary-Bonsai-2-27B-mlx-2bit" }, 0..) |name, family| {
        if (draft_family != null and draft_family.? != family) continue;
        const dflash_family = family == 0 or family == 3;
        for (0..if (family != 2) @as(usize, 2) else 1) |backend| {
            for (0..4) |scenario| {
                if (draft_scenario != null and draft_scenario.? != scenario) continue;
                const count: []const u8 = if (scenario == 0) "17" else if (scenario == 3) "2" else "32";
                const common = &.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), if (family == 1) "--prompt" else "--tokens", if (family == 1) "Write a short Python function that computes the Fibonacci sequence." else "1,2,3,4,5,6,7,8,41,42,43,1,2,3,4,5,6,7,8", "--max-tokens", count, "--seed", "5678", "--temperature", if (scenario == 0) "0" else "0.7", "--top-k", "12", "--top-p", "0.8" };
                const serial_report = b.fmt("build/native-checks/drafts/{d}-{d}-{d}-serial.json", .{ family, backend, scenario });
                const serial = b.addRunArtifact(exe);
                serial.addArgs(common);
                serial.addArg("--no-drafts");
                if (backend == 1) serial.addArg("--metal-simd");
                if (scenario == 1) serial.addArg("--metal-sampling");
                serial.addArgs(&.{ "--report", serial_report });
                serial.step.dependOn(prior);
                prior = &serial.step;
                for ([_][]const u8{ "1", "3", "15" }, 0..) |budget, index| {
                    if (dflash_family and index != 0) continue;
                    const report = b.fmt("build/native-checks/drafts/{d}-{d}-{d}-{s}.json", .{ family, backend, scenario, budget });
                    const draft_run = b.addRunArtifact(exe);
                    draft_run.addArgs(common);
                    if (dflash_family) draft_run.addArgs(&.{ "--drafter", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}) }) else draft_run.addArgs(&.{ "--mtp-drafts", budget });
                    if (!dflash_family) draft_run.addArg("--fixed-drafts");
                    if (backend == 1) draft_run.addArg("--metal-simd");
                    if (scenario == 1) draft_run.addArg("--metal-sampling");
                    draft_run.addArgs(&.{ "--report", report });
                    draft_run.step.dependOn(prior);
                    const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--require-rounds", "--compare-reports", serial_report, report });
                    compare.step.dependOn(&draft_run.step);
                    prior = &compare.step;
                }
            }
        }
    }
    draft_tests.dependOn(prior);
    const cache_tests = b.step("test-cache-stress", "All accepted prefixes, randomized trees, EOS history, rejection and reset cycles");
    const long_cache_tests = b.step("test-long-cache", "Full-model cache rollback with random prompts across sparse and 10K thresholds");
    const cache_family = b.option(usize, "cache-family", "Restrict cache stress to 0=Qwen, 1=Nemotron, 2=Flash");
    if (cache_family != null and cache_family.? > 2) @panic("cache-family must be 0, 1 or 2");
    var cache_previous: ?*std.Build.Step = null;
    var long_cache_previous: ?*std.Build.Step = null;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (cache_family != null and cache_family.? != family) continue;
        for (0..if (family < 2) @as(usize, 2) else 1) |backend| {
            const check = b.addRunArtifact(exe);
            check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-cache-stress" });
            if (backend == 1) check.addArg("--metal-simd");
            if (cache_previous) |step| check.step.dependOn(step);
            cache_previous = &check.step;
            const long_check = b.addRunArtifact(exe);
            long_check.addArgs(&.{ "run", b.fmt("{s}/{s}", .{ model_root, name }), "--check-long-cache" });
            if (backend == 1) long_check.addArg("--metal-simd");
            if (long_cache_previous) |step| long_check.step.dependOn(step);
            long_cache_previous = &long_check.step;
        }
    }
    if (cache_previous) |step| cache_tests.dependOn(step);
    if (long_cache_previous) |step| long_cache_tests.dependOn(step);

    const long_tests = b.step("test-long-context", "Full-model Python/native logits and drafted continuations across attention thresholds");
    const long_family = b.option(usize, "long-family", "Restrict long-context checks to 0=Qwen, 1=Nemotron, 2=Flash");
    const long_size = b.option(usize, "long-tokens", "Override context length for one threshold or diagnostic case");
    const long_backend = b.option(usize, "long-backend", "Restrict long-context checks to 0=tensor or 1=SIMD");
    if (long_backend != null and long_backend.? > 1) @panic("long-backend must be 0 or 1");
    if (long_family == 2 and long_backend == 1) @panic("Flash has one backend; use long-backend=0");
    if (long_family != null and long_family.? > 2) @panic("long-family must be 0, 1 or 2");
    if (long_size != null and (long_size.? == 0 or long_size.? > 262128)) @panic("long-tokens must be 1..262128");
    const long_dir = b.addSystemCommand(&.{ "mkdir", "-p", "build/native-checks/long" });
    var long_prior: *std.Build.Step = &long_dir.step;
    for ([_][]const u8{ "Qwen3.8-27B-MLX-4bit", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "Qwen3.8-Flash-Next-MLX-4bit-MTP" }, 0..) |name, family| {
        if (long_family != null and long_family.? != family) continue;
        const lengths = if (family == 2) [_]usize{ 2051, 2063 } else [_]usize{ 9999, 10007 };
        for (lengths, 0..) |default_length, case| {
            if (long_size != null and case > 0) continue;
            const length = long_size orelse default_length;
            const ids = b.allocator.alloc([]const u8, length) catch @panic("OOM");
            // Four repeating tokens keep the Python PLE oracle's resident shard set
            // bounded while RoPE/cache lengths still exercise every threshold.
            for (ids, 0..) |*id, j| id.* = b.fmt("{d}", .{1000 + (j % 4) * 37});
            const prompt = std.mem.join(b.allocator, ",", ids) catch @panic("OOM");
            for (0..if (family < 2) @as(usize, 2) else 1) |backend| {
                if (long_backend != null and long_backend.? != backend) continue;
                const base = b.fmt("build/native-checks/long/{d}-{d}-{d}", .{ family, backend, length });
                const dir = b.fmt("{s}/{s}", .{ model_root, name });
                const common = &.{ "--tokens", prompt, "--seed", "5678", "--temperature", "0.7", "--top-k", "12", "--top-p", "0.8", "--metal-sampling" };
                const oracle = b.addSystemCommand(&.{ ".venv/bin/python", if (family == 0) "tools/native_reference.py" else "tools/native_families_reference.py" });
                if (family == 0) oracle.addArg("--model");
                oracle.addArg(dir);
                oracle.addArgs(common);
                oracle.addArgs(&.{ "--generate", "16", "--dump-logits", b.fmt("{s}-python.npy", .{base}), "--output", b.fmt("{s}-python.json", .{base}) });
                if (backend == 1) oracle.addArg("--simd");
                oracle.step.dependOn(long_prior);
                long_prior = &oracle.step;
                for (0..2) |drafts| {
                    const suffix = if (drafts == 0) "serial" else "draft";
                    const check = b.addRunArtifact(exe);
                    check.addArgs(&.{ "run", dir });
                    check.addArgs(common);
                    if (family == 0) check.addArg("--lane-prefill");
                    check.addArgs(&.{ "--max-tokens", "16", "--dump-logits", b.fmt("{s}-{s}.npy", .{ base, suffix }), "--report", b.fmt("{s}-{s}.json", .{ base, suffix }) });
                    if (backend == 1) check.addArg("--metal-simd");
                    if (drafts == 1) check.addArg("--no-copy");
                    if (family == 0 and drafts == 1) check.addArgs(&.{ "--drafter", b.fmt("{s}/Qwen3.8-27B-DFlash2", .{model_root}) });
                    if (family > 0) check.addArgs(if (drafts == 0) &.{"--no-drafts"} else &.{ "--mtp-drafts", "15" });
                    if (family > 0 and drafts == 1) check.addArg("--fixed-drafts");
                    check.step.dependOn(long_prior);
                    const logits = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare", b.fmt("{s}-python.npy", .{base}), b.fmt("{s}-{s}.npy", .{ base, suffix }) });
                    logits.step.dependOn(&check.step);
                    const compare = b.addSystemCommand(&.{ ".venv/bin/python", "tools/native_reference.py", "--compare-reports", b.fmt("{s}-python.json", .{base}), b.fmt("{s}-{s}.json", .{ base, suffix }) });
                    compare.step.dependOn(&logits.step);
                    long_prior = &compare.step;
                }
            }
        }
    }
    long_tests.dependOn(long_prior);
    for (b.top_level_steps.keys(), b.top_level_steps.values()) |name, step| {
        if (!std.mem.startsWith(u8, name, "test") or !hasExecutableCheck(&step.step)) continue;
        for ([_]*std.Build.Step.Run{ coverage, coverage_record, coverage_complete }) |check| check.addArgs(&.{ "--test-step", name });
    }
}

fn hasExecutableCheck(step: *std.Build.Step) bool {
    if (step.tag == .run) return true;
    for (step.dependencies.items) |dependency| if (hasExecutableCheck(dependency)) return true;
    return false;
}
