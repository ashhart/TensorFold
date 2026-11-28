const std = @import("std");
const engines_build = @import("cuda_engines.zig");
const flashnextModule = engines_build.flashnextModule;
const glmCuda = engines_build.glmCuda;
const lfm2Module = engines_build.lfm2Module;
const stagger = @import("cuda_runtime.zig").stagger;
const kolibri1Module = @import("kolibri.zig").module;
const checkpointCli = @import("cuda.zig").checkpointCli;

/// Remaining CUDA host registrations, called in their original order after native startup checks.
pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, mods: anytype, native_modules: anytype, native: *std.Build.Module, skip_drafts: *std.Build.Step, windows_port: *std.Build.Step, server_steps: bool, build_options: *std.Build.Step.Options, step: *std.Build.Step) void {
    const h3_checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/checkpoint.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "core", .module = @import("h3_host.zig").coreModule(b, host) }} });
    // the checkpoint maps its file with mmap: not Windows
    if (host.result.os.tag != .windows) step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = h3_checkpoint })).step);
    @import("pull_tests.zig").add(b, host, .debug, native_modules.engines, mods.core, step);
    b.step("test-save-reopen-host", "Unresolved save through actual checkpoint admission").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.core, .filters = &.{"Save recovery:"} })).step);
    b.step("test-trainer-close-host", "Actual native close and real Flash Next trainer ownership sentinels").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{"Save recovery:"} })).step);
    const template, const server_tests = @import("cuda_server_host.zig").steps(b, host, cuda, mods, native_modules, build_options, windows_port, step, server_steps);
    @import("living_policy_host.zig").steps(b, host, server_tests, mods.core, mods.living, native_modules.api, mods.nemotron, step);
    // the tokenizer's and chat template's own tests run here: no other test binary reaches them
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.qwen35, mods.qwen_vision, flashnextModule(b, host, .debug, cuda, mods.core, mods.lanes, mods.qwen_vision, mods.api), lfm2Module(b, host, .debug, cuda, mods.core), mods.radio, mods.nemo_audio, mods.heat, mods.tokenizer, template, native_modules.api, native, stagger(b, host, .debug) }) |m| {
        // The native root compiles every family's open, including Muse's POSIX TP session.
        if (host.result.os.tag == .windows and m == native) continue;
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
    const selected = b.option([]const u8, "modelopt-host-filter", "One portable ModelOpt regression") orelse "modelopt host:";
    const run = b.addRunArtifact(b.addTest(.{ .root_module = mods.nemotron, .filters = &.{selected} }));
    run.has_side_effects = true;
    b.step("test-modelopt-host", "Actual Nemotron scratch and CUDA launch records without GPU execution").dependOn(&run.step);
    // the contract's attention reference (tests/cuda/contract_attn.zig): its allocations under failure
    const attn_ref = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/contract_attn.zig"), .target = host, .optimize = .debug, .imports = &.{.{ .name = "core", .module = mods.core }} });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = attn_ref })).step);
    const glm = glmCuda(b, host, .debug, cuda, mods.core, mods.lanes, mods.heat, mods.api);
    b.step("test-glm-grants-host", "GLM CUDA head-cost policy without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"shared head costs:"} })).step);
    skip_drafts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"zero grant:"} })).step);
    @import("glm_family_host.zig").steps(b, cuda, glm, mods, step);
    const prefill_exact = b.step("test-glm-prefill-exact-host", "GLM exact prefill dead-work and launch regressions without a GPU");
    prefill_exact.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"prefill exact:"} })).step);
    const embedding = b.step("test-glm-embedding-host", "GLM checkpoint and token-ID embedding boundaries without a GPU");
    embedding.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"GLM embedding:"} })).step);
    embedding.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"GLM embedding:"} })).step);
    step.dependOn(embedding);
    step.dependOn(prefill_exact);
    const q4_pdl = b.createModule(.{ .root_source_file = b.path("zig/glm_q4_pdl_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda } } });
    const q4_pdl_run = b.addRunArtifact(b.addTest(.{ .root_module = q4_pdl }));
    b.step("test-glm-q4-pdl-host", "GLM Q4 draft-head launch honors TF_GLM_PDL without a GPU").dependOn(&q4_pdl_run.step);
    if (@import("builtin").os.tag != .windows) step.dependOn(&q4_pdl_run.step); // its test sets env through libc setenv, which Windows lacks
    @import("kolibri.zig").host(b, kolibri1Module(b, host, .debug, cuda, mods.core, mods.lanes), native_modules.api, step);
    const gemma = @import("gemma4_cuda.zig").module(b, host, .debug, cuda, mods.core, mods.lanes);
    b.step("test-gemma-grants-host", "Gemma assistant and DFlash head-cost grants without a GPU").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gemma, .filters = &.{"shared head costs:"} })).step);
    @import("gemma4_cuda.zig").host(b, gemma, step);
    @import("gemma4_shared_host.zig").host(b, cuda, mods.core, mods.lanes, mods.api, gemma, step);
    skip_drafts.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gemma, .filters = &.{"zero grant:"} })).step);
    @import("cuda_small_hosts.zig").steps(b, host, cuda, mods.core, mods.lanes, step);
    @import("glm_contracts.zig").tests(b, glm, mods.lanes, cuda, mods.core);
    // the shares a GLM-5.3-Flash rank takes (cluster/canon.zig against the Python family's)
    const shares = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon_glm_test.zig"), .target = host, .optimize = .debug });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = shares })).step);
    // the mesh layer's groups, shard declarations and reduction contracts
    const mesh = @import("mesh.zig").module(b, host, .debug, @import("cuda_engines.zig").canonModule(b, host, .debug));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mesh })).step);
    // the tool-call parser's tests, run beside the server's on Macs: the GLM calls the CUDA port serves
    const parse_tests = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } } });
    _ = @import("server.zig").addTo(b, parse_tests, host, .debug);
    @import("server.zig").toolHostTests(b, parse_tests, server_tests, step);
    // Living Weights: the family-neutral learner, then the pocket in each family's shards over the core's shard files
    const living_run = b.addRunArtifact(b.addTest(.{ .root_module = mods.living }));
    const shards = b.createModule(.{ .root_source_file = b.path("zig/src/living_shards_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "living", .module = mods.living }} });
    const shards_run = b.addRunArtifact(b.addTest(.{ .root_module = shards }));
    step.dependOn(&living_run.step);
    step.dependOn(&shards_run.step);
    const living_step = b.step("test-living", "Living Weights' learner and the pocket in each family's shards, on the host");
    living_step.dependOn(&living_run.step);
    living_step.dependOn(&shards_run.step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addOptions("build_options", build_options);
    const checkpoint_cli = checkpointCli(b, host, .debug, false, native_modules.engines, mods.core);
    cli.addImport("checkpoint_cli", checkpoint_cli);
    cli.addImport("server_start", @import("server.zig").startModule(b, host, .debug, checkpoint_cli, native_modules.api, mods.core, @import("server.zig").json(b, host, .debug)));
    const memory_cli = checkpointCli(b, host, .debug, false, native_modules.engines, mods.core);
    memory_cli.root_source_file = b.path("zig/src/cli/tests.zig");
    b.step("test-cli-memory-host", "Actual checkpoint info memory tests without device or shader producers").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = memory_cli, .filters = &.{ "info prints family", "issue 934:", "plan" } })).step);
    const tfq3_run = b.addRunArtifact(b.addTest(.{ .root_module = memory_cli, .filters = &.{ "tfq3", "KR135" } }));
    const conversion_run = b.addRunArtifact(b.addTest(.{ .root_module = memory_cli, .filters = &.{"conversion pull:"} }));
    b.step("test-pull-conversion-host", "Conversion config admission and symbolic revision publication without HTTP").dependOn(&conversion_run.step);
    b.step("test-tfq3-host", "TFQ3 CLI parsing, native encoding, safety and shard inspection on the host").dependOn(&tfq3_run.step);
    b.step("test-info-config", "Info shares the checkpoint config reader and diagnostics").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = memory_cli, .filters = &.{"issue 934:"} })).step);
    const cli_run = b.addRunArtifact(b.addTest(.{ .root_module = cli }));
    b.step("test-cli-host", "Checkpoint CLI parsing and CUDA start dispatch without devices").dependOn(&cli_run.step);
    step.dependOn(&cli_run.step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = memory_cli, .filters = &.{"plan"} })).step); // plan's verdict tests in the plain `zig build test` gate
}
