//! The Qwen3.8-27B half of the Metal build: the core modules its engine imports by name, its programs and tests.

const std = @import("std");

/// The core modules the engine imports by name for the 27B family, made once per engine module.
pub const Shared = struct {
    delta: *std.Build.Module,
    runtime_sources: *std.Build.Module,
    gdn_source: *std.Build.Module,
    bf16_topk: *std.Build.Module,
    bf16_topk_gpu: *std.Build.Module,
    tree_round: *std.Build.Module,
    tree_round_gpu: *std.Build.Module,
    tree_commit: *std.Build.Module,
    draft_ops: *std.Build.Module,
};

/// The 27B's Metal sources, embedded by name (zig/kernels/metal/qwen27).
const parts = [_][]const u8{ "xsum", "norm_xs", "norm_input", "mlp_xs", "mlp_ane", "glue", "copy", "attention_mpp", "attention_io", "attention_prompt", "prompt_glue", "prompt_state" };

/// The shared modules, imported into `engine` (whose `core` is itself) for the server and the release binary.
pub fn engineModules(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module, sources: *std.Build.Module, engine: *std.Build.Module) Shared {
    const files = b.addWriteFiles();
    var index: []const u8 = "";
    for (parts) |part| {
        _ = files.addCopyFile(b.path(b.fmt("zig/kernels/metal/qwen27/{s}.metal", .{part})), b.fmt("{s}.metal", .{part}));
        index = b.fmt("{s}pub const {s} = @embedFile(\"{s}.metal\");\n", .{ index, part, part });
    }
    _ = files.addCopyFile(b.path("zig/kernels/metal/core/deltanet.metal"), "gdn.metal");
    const bf16_topk = b.createModule(.{ .root_source_file = b.path("zig/src/core/bf16_topk.zig"), .target = target, .optimize = .ReleaseSafe });
    const tree_round = b.createModule(.{ .root_source_file = b.path("zig/src/core/tree_round.zig"), .target = target, .optimize = .ReleaseSafe });
    const s: Shared = .{
        .delta = b.createModule(.{ .root_source_file = b.path("zig/src/core/deltanet_kernels.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } }),
        .runtime_sources = b.createModule(.{ .root_source_file = files.add("qwen_runtime_sources.zig", index) }),
        .gdn_source = b.createModule(.{ .root_source_file = files.add("qwen_gdn_source.zig", "pub const text = @embedFile(\"gdn.metal\");\n") }),
        .bf16_topk = bf16_topk,
        .bf16_topk_gpu = b.createModule(.{ .root_source_file = b.path("zig/src/core/bf16_topk_gpu.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "bf16_topk", .module = bf16_topk }, .{ .name = "bf16_topk_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/bf16_topk_sources.zig") }) } } }),
        .tree_round = tree_round,
        .tree_round_gpu = b.createModule(.{ .root_source_file = b.path("zig/src/core/tree_round_gpu.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tree_round", .module = tree_round }, .{ .name = "tree_round_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/tree_round_sources.zig") }) } } }),
        .tree_commit = b.createModule(.{ .root_source_file = b.path("zig/src/core/tree_commit_gpu.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tree_round", .module = tree_round }, .{ .name = "tree_commit_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/tree_commit_sources.zig") }) } } }),
        .draft_ops = b.createModule(.{ .root_source_file = b.path("zig/src/core/draft_ops.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "draft_ops_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/draft_ops_sources.zig") }) } } }),
    };
    engine.addImport("core", engine);
    engine.addImport("core_delta", s.delta);
    engine.addImport("qwen_runtime_sources", s.runtime_sources);
    engine.addImport("bf16_topk", s.bf16_topk);
    engine.addImport("bf16_topk_gpu", s.bf16_topk_gpu);
    engine.addImport("tree_round", s.tree_round);
    engine.addImport("tree_round_gpu", s.tree_round_gpu);
    engine.addImport("tree_commit_gpu", s.tree_commit);
    engine.addImport("draft_ops", s.draft_ops);
    return s;
}

/// A program that builds into zig-out/bin under its own step.
fn program(b: *std.Build, name: []const u8, about: []const u8, path: []const u8, target: std.Build.ResolvedTarget, strip: bool, imports: []const std.Build.Module.Import) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{ .name = name, .root_module = b.createModule(.{ .root_source_file = b.path(path), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .strip = if (strip) true else null, .imports = imports }) });
    b.step(name, about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    return exe;
}

/// The core lane, shared-context and 27B programs and host tests; the 27B's module takes `core` from qwen27_core.zig.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, engine: *std.Build.Module, lanes: *std.Build.Module, s: Shared, test_step: *std.Build.Step) void {
    const core_row = b.createModule(.{ .root_source_file = b.path("zig/src/core/row_projection.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core_row })).step);
    const core_lane = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_projection.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = core_lane })).step);
    const lane_imports: []const std.Build.Module.Import = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core_lane", .module = core_lane } };
    _ = program(b, "tf-core-lane-small", "Build bounded synthetic small-width core timing and retained-byte comparison", "zig/tests/core_lane_small.zig", target, false, lane_imports);
    _ = program(b, "tf-core-lane-reg-check", "Build the reg kernel's byte check against the cooperative kernel (M5 GPU)", "zig/tests/core_lane_reg.zig", target, false, lane_imports);
    const frozen_lane = b.createModule(.{ .root_source_file = b.path("zig/tests/fixtures/fn_lane/golden.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "core_lane", .module = core_lane }} });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = frozen_lane })).step);
    const lane_check = program(b, "tf-core-lane-check", "Build the synthetic core projection checks without running a GPU", "zig/tests/core_lane_projection.zig", target, false, lane_imports);
    b.step("test-core-lane", "Run small synthetic projection checks on this Mac's tensor units, no models").dependOn(&b.addRunArtifact(lane_check).step);

    const fork_tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/recurrent_forks.zig"), .target = target, .optimize = .ReleaseSafe }) });
    test_step.dependOn(&b.addRunArtifact(fork_tests).step);
    b.step("test-recurrent-forks", "Host tests for recurrent lane lifecycle and rollback").dependOn(&b.addRunArtifact(fork_tests).step);
    const attention = b.createModule(.{ .root_source_file = b.path("zig/src/core/shared_attention.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const kv_layout = b.createModule(.{ .root_source_file = b.path("zig/src/core/shared_kv.zig"), .target = target, .optimize = .ReleaseSafe });
    const see_host = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/lanes_see_host.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } }) });
    test_step.dependOn(&b.addRunArtifact(see_host).step);
    const see = program(b, "tf-lanes-see-check", "Build synthetic shared-context checks, no GPU execution", "zig/tests/lanes_see.zig", target, false, &.{ .{ .name = "metal", .module = metal }, .{ .name = "shared_attention", .module = attention }, .{ .name = "shared_kv", .module = kv_layout } });
    b.step("test-lanes-see", "Run small shared-context, mask, position and KV append device checks").dependOn(&b.addRunArtifact(see).step);

    const qwen_core = b.createModule(.{ .root_source_file = b.path("zig/qwen27_core.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tokenizer", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true }) } } });
    const qwen27 = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen27/qwen27.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "core", .module = qwen_core }, .{ .name = "metal", .module = metal } } });
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "core", qwen_core }, .{ "core_lane", core_lane }, .{ "core_row", core_row }, .{ "bf16_topk", s.bf16_topk }, .{ "bf16_topk_gpu", s.bf16_topk_gpu }, .{ "tree_round", s.tree_round }, .{ "tree_round_gpu", s.tree_round_gpu }, .{ "tree_commit_gpu", s.tree_commit }, .{ "draft_ops", s.draft_ops }, .{ "kernel_sources", sources } }) |m| qwen_core.addImport(m[0], m[1]);
    for ([_]struct { []const u8, *std.Build.Module }{ .{ "core_lane", core_lane }, .{ "qwen_runtime_sources", s.runtime_sources }, .{ "core_delta", s.delta }, .{ "qwen_gdn_source", s.gdn_source } }) |m| qwen27.addImport(m[0], m[1]);
    for ([_]*std.Build.Module{ s.bf16_topk, s.tree_round, s.draft_ops }) |m| test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);

    const family: []const std.Build.Module.Import = &.{ .{ .name = "qwen27", .module = qwen27 }, .{ .name = "core", .module = qwen_core }, .{ .name = "metal", .module = metal } };
    _ = program(b, "tf-qwen27-run", "Build the native greedy, chunk and prompt-speed check on real weights", "zig/tests/qwen27/run.zig", target, true, &.{ .{ .name = "qwen27", .module = qwen27 }, .{ .name = "core", .module = qwen_core }, .{ .name = "metal", .module = metal }, .{ .name = "core_lane", .module = core_lane }, .{ .name = "qwen_runtime_sources", .module = s.runtime_sources }, .{ .name = "qwen_gdn_source", .module = s.gdn_source } });
    _ = program(b, "tf-qwen27-projection-cases", "Build real-input production projection precision replay", "zig/tests/qwen27/projection_cases.zig", target, true, &.{ .{ .name = "qwen27", .module = qwen27 }, .{ .name = "metal", .module = metal } });
    _ = program(b, "tf-qwen27-dflash-run", "Build the explicit draft prototype and its plain comparison on real weights", "zig/tests/qwen27/dflash_run.zig", target, true, family);
    const engine_only: []const std.Build.Module.Import = &.{ .{ .name = "tensorfold", .module = engine }, .{ .name = "metal", .module = metal } };
    _ = program(b, "tf-qwen27-synthetic", "Build bounded synthetic native hybrid chunk checks, no models", "zig/tests/qwen27/synthetic.zig", target, false, engine_only);
    _ = program(b, "tf-qwen27-dflash-synthetic", "Build model-free native draft context/forward/tree check", "zig/tests/qwen27/dflash_runtime.zig", target, false, engine_only);
    _ = program(b, "tf-qwen27-dflash-refine", "Build the drafter refinement probe (top-1 by block position, masked versus given)", "zig/tests/qwen27/dflash_refine.zig", target, false, engine_only);

    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const host = b.createModule(.{ .root_source_file = b.path("zig/src/native/qwen27_host.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine } } });
    const served: []const std.Build.Module.Import = &.{ .{ .name = "tensorfold", .module = engine }, .{ .name = "metal", .module = metal }, .{ .name = "engine_api", .module = api }, .{ .name = "qwen27_host", .module = host } };
    _ = program(b, "tf-qwen27-served-check", "Build synthetic CLI versus served-host token check", "zig/tests/qwen27/served.zig", target, false, served);
    _ = program(b, "tf-qwen27-dflash-served-check", "Build synthetic drafted CLI versus served-host token check", "zig/tests/qwen27/dflash_served.zig", target, false, served);
    _ = program(b, "tf-qwen27-lanes-check", "Build synthetic lane-core versus plain greedy token and state check", "zig/tests/qwen27/lanes_check.zig", target, false, served);
    _ = program(b, "tf-qwen27-lanes-run", "Build the real-weight serial versus lane-core A/B", "zig/tests/qwen27/lanes_run.zig", target, false, served);
    _ = program(b, "tf-qwen27-streams", "Build the real-weight check that shared rounds keep each stream's logits alone", "zig/tests/qwen27/streams.zig", target, false, served);
    _ = program(b, "tf-qwen27-reuse", "Qwen27 prompt reuse through the served host against fresh prompt passes: states and replies", "zig/tests/qwen27/reuse.zig", target, false, served);
    _ = program(b, "tf-qwen27-runtime", "Compile native Qwen interfaces without running a GPU", "zig/tests/qwen27/runtime.zig", target, false, &.{.{ .name = "qwen27", .module = qwen27 }});
    _ = program(b, "tf-qwen27-weights", "Build CPU-only Qwen checkpoint metadata validator", "zig/tests/qwen27/weights.zig", target, false, &.{ .{ .name = "qwen27", .module = qwen27 }, .{ .name = "core", .module = qwen_core } });
    _ = program(b, "tf-qwen27-dflash-inspect", "Build CPU-only target/drafter graph validator", "zig/tests/qwen27/dflash_weights.zig", target, false, &.{ .{ .name = "qwen27", .module = qwen27 }, .{ .name = "core", .module = qwen_core } });

    const qwen_tests = b.addRunArtifact(b.addTest(.{ .root_module = qwen27 }));
    test_step.dependOn(&qwen_tests.step);
    const qwen_test_step = b.step("test-qwen27", "CPU Qwen config, affine and complete tensor graph contracts");
    qwen_test_step.dependOn(&qwen_tests.step);
    const header_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true }) }));
    qwen_test_step.dependOn(&header_tests.step);
    test_step.dependOn(&header_tests.step);
    const profile_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/gpu_profile.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "metal", .module = metal }} }) }));
    qwen_test_step.dependOn(&profile_tests.step);
    test_step.dependOn(&profile_tests.step);
}
